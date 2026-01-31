import json
import logging
from pathlib import Path
from typing import Union, Tuple, Optional
import math

import torch
from torch import nn
import torch.nn.functional as F
import diffusers
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
import safetensors
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import transformers
from PIL import Image, ImageOps
from tqdm import tqdm

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift, iterate_safetensors
from utils.offloading import ModelOffloader


KEEP_IN_HIGH_PRECISION = ['time_text_embed', 'img_in', 'txt_in', 'norm_out', 'proj_out']

logger = logging.getLogger(__name__)


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        # Slice freqs_cis to match x's sequence length (text and image have different S)
        seq_len = x.size(1)
        freqs_cis = freqs_cis[:seq_len].unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


# I copied this because it doesn't handle encoder_hidden_states_mask, which causes high loss values when there is a lot
# of padding. When (or if) they fix it upstream, I don't want the changes to break my workaround, which is to just set
# attention_mask.
class QwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)
        
        # Save dtype before deleting tensors (needed for later conversion)
        query_dtype = joint_query.dtype
        
        # Delete individual QKV tensors after concatenation to free VRAM
        del img_query, img_key, img_value, txt_query, txt_key, txt_value

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
        
        # Delete joint QKV tensors after attention computation to free VRAM
        del joint_query, joint_key, joint_value

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(query_dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part
        
        # Delete joint_hidden_states after splitting to free VRAM
        del joint_hidden_states

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenImagePipeline(BasePipeline):
    name = 'qwen_image'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['QwenImageTransformerBlock']
    pixels_round_to_multiple = 32

    prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    prompt_template_encode_start_idx = 34
    prompt_template_encode_edit = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
    prompt_template_encode_start_idx_edit = 64

    # Size of image fed to the VLM for Qwen-Image-Edit
    vlm_image_size = 1024

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.text_encoder_offloader = None
        self.sample_prompt_embeds = None
        self.sample_prompts = None
        dtype = self.model_config['dtype']

        self.preprocess_media_file_fn = PreprocessMediaFile(self.config, support_video=True, framerate=1)

        tokenizer = transformers.Qwen2Tokenizer.from_pretrained('configs/qwen_image/tokenizer', local_files_only=True)
        processor = transformers.Qwen2VLProcessor.from_pretrained('configs/qwen_image/processor', local_files_only=True)

        if 'text_encoder_path' in self.model_config:
            text_encoder_path = self.model_config['text_encoder_path']
        else:
            text_encoder_path = Path(self.model_config['diffusers_path']) / 'text_encoder'
        text_encoder_config = transformers.Qwen2_5_VLConfig.from_pretrained('configs/qwen_image/text_encoder', local_files_only=True)
        with init_empty_weights():
            text_encoder = transformers.Qwen2_5_VLForConditionalGeneration(text_encoder_config)
        for key, tensor in iterate_safetensors(text_encoder_path):
            # The keys in the state_dict don't match the structure in the model. Annoying. Need to convert.
            key = key.replace('model.', 'language_model.')
            key = 'model.' + key
            if 'lm_head' in key:
                key = 'lm_head.weight'
            set_module_tensor_to_device(text_encoder, key, device='cpu', dtype=dtype, value=tensor)

        # TODO: make this work with ComfyUI VAE weights, which have completely different key names.
        if 'vae_path' in self.model_config:
            vae_path = self.model_config['vae_path']
        else:
            vae_path = Path(self.model_config['diffusers_path']) / 'vae'
        with open('configs/qwen_image/vae/config.json') as f:
            vae_config = json.load(f)
        with init_empty_weights():
            vae = diffusers.AutoencoderKLQwenImage.from_config(vae_config)
        for key, tensor in iterate_safetensors(vae_path):
            set_module_tensor_to_device(vae, key, device='cpu', dtype=dtype, value=tensor)
        self._move_meta_tensors_to_device(vae, 'cpu')

        self.diffusers_pipeline = diffusers.QwenImagePipeline(
            scheduler=None,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=None,
        )

        self.diffusers_pipeline.processor = processor

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(dtype)
        )
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(dtype)
        self.vae.register_buffer('latents_mean_tensor', latents_mean)
        self.vae.register_buffer('latents_std_tensor', latents_std)

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        if 'transformer_path' in self.model_config:
            transformer_path = self.model_config['transformer_path']
        else:
            transformer_path = Path(self.model_config['diffusers_path']) / 'transformer'
        with open('configs/qwen_image/transformer/config.json') as f:
            json_config = json.load(f)

        with init_empty_weights():
            transformer = diffusers.QwenImageTransformer2DModel.from_config(json_config)

        for key, tensor in iterate_safetensors(transformer_path):
            dtype_to_use = dtype if any(keyword in key for keyword in KEEP_IN_HIGH_PRECISION) or tensor.ndim == 1 else transformer_dtype
            set_module_tensor_to_device(transformer, key, device='cpu', dtype=dtype_to_use, value=tensor)

        attn_processor = QwenDoubleStreamAttnProcessor2_0()
        for block in transformer.transformer_blocks:
            block.attn.set_processor(attn_processor)

        self.diffusers_pipeline.transformer = transformer

        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder]

    def _log_vram_usage(self, context, previous_allocated=None):
        """
        Log current VRAM usage with optional delta calculation.
        
        Args:
            context: String describing the context/location of this measurement
            previous_allocated: Previous allocated memory in bytes (for delta calculation)
        """
        if not torch.cuda.is_available():
            return None
        
        allocated_bytes = torch.cuda.memory_allocated()
        allocated_gb = allocated_bytes / (1024 ** 3)
        
        if previous_allocated is not None:
            delta_bytes = allocated_bytes - previous_allocated
            delta_gb = delta_bytes / (1024 ** 3)
            print(f'[VRAM] {context}: {allocated_gb:.2f}GB allocated, {delta_gb:+.2f}GB delta')
        else:
            print(f'[VRAM] {context}: {allocated_gb:.2f}GB allocated')
        
        return allocated_bytes

    def _move_meta_tensors_to_device(self, module, target_device):
        """
        Recursively move all parameters and buffers from meta device to target device.
        This ensures that all tensors are on a real device before use, matching the behavior
        of transformer block swapping which calls block.to(device).
        
        Meta tensors cannot be moved directly with .to() because they have no actual storage.
        This function handles them by creating new tensors on the target device when needed.
        """
        target_device = torch.device(target_device) if isinstance(target_device, str) else target_device
        meta_device = torch.device('meta')
        
        # Move all parameters
        for name, param in module.named_parameters(recurse=False):
            if param.device == meta_device:
                # Meta tensors can't be moved directly, need to create new tensor
                # Initialize to zeros (safe default - if important, should have been loaded from checkpoint)
                new_param = torch.nn.Parameter(
                    torch.zeros(param.shape, device=target_device, dtype=param.dtype),
                    requires_grad=param.requires_grad
                )
                # Replace the parameter using register_parameter to ensure proper registration
                module._parameters[name] = new_param
        
        # Move all buffers
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.device == meta_device:
                # Meta buffers can't be moved directly, need to create new buffer
                # Initialize to zeros (safe default - if important, should have been loaded from checkpoint)
                new_buffer = torch.zeros(buffer.shape, device=target_device, dtype=buffer.dtype)
                module.register_buffer(name, new_buffer)
        
        # Recursively process child modules
        for child in module.children():
            self._move_meta_tensors_to_device(child, target_device)

    def cache_sample_prompts(self, prompts):
        """Cache text embeddings for sample generation prompts."""
        if prompts is None or len(prompts) == 0:
            return
        
        # Check if block swapping is enabled
        use_block_swap = (self.text_encoder_offloader is not None and 
                         self.text_encoder_offloader.blocks_to_swap is not None and 
                         self.text_encoder_offloader.blocks_to_swap > 0)
        
        if use_block_swap:
            # Use existing block swap mechanism - this calls prepare_block_devices_before_forward()
            # which ensures all blocks are properly on CUDA device
            self.prepare_text_encoder_block_swap_inference(disable_block_swap=False)
            target_device = 'cuda'  # Block swapping requires CUDA
        else:
            # No block swapping - respect text_encoder_cache_on_cpu config
            text_encoder_cache_on_cpu = self.config.get('text_encoder_cache_on_cpu', False)
            target_device = 'cpu' if text_encoder_cache_on_cpu else 'cuda'
            
            # Use same device preparation mechanism as transformer
            # Extract layers and move individually (same pattern as prepare_block_devices_before_forward when blocks_to_swap=0)
            text_encoder = self.text_encoder
            if hasattr(text_encoder, 'model') and hasattr(text_encoder.model, 'language_model'):
                language_model = text_encoder.model.language_model
                if hasattr(language_model, 'model') and hasattr(language_model.model, 'layers'):
                    layers = language_model.model.layers
                    # Move each layer individually (same as transformer when blocks_to_swap=0)
                    for layer in layers:
                        layer.to(target_device)
                    # Also move the rest of the text encoder structure
                    # Temporarily detach layers to move structure
                    language_model.model.layers = None
                    try:
                        # Try to move the structure normally first
                        try:
                            text_encoder.to(target_device)
                        except NotImplementedError as e:
                            if "meta tensor" in str(e):
                                # Some parameters/buffers on meta device - manually move them
                                # This matches the behavior of transformer block swapping
                                self._move_meta_tensors_to_device(text_encoder, target_device)
                            else:
                                raise
                    finally:
                        language_model.model.layers = layers
                else:
                    # Fallback: move entire model
                    try:
                        text_encoder.to(target_device)
                    except NotImplementedError as e:
                        if "meta tensor" in str(e):
                            # Some parameters/buffers on meta device - manually move them
                            self._move_meta_tensors_to_device(text_encoder, target_device)
                        else:
                            raise
            else:
                # Fallback: move entire model
                try:
                    text_encoder.to(target_device)
                except NotImplementedError as e:
                    if "meta tensor" in str(e):
                        # Some parameters/buffers on meta device - manually move them
                        self._move_meta_tensors_to_device(text_encoder, target_device)
                    else:
                        raise
        
        # Encode prompts
        prompt_embeds = self._get_qwen_prompt_embeds(
            prompts, 
            control_files=None, 
            device=target_device
        )
        
        # Store prompts and their embeddings
        self.sample_prompts = prompts
        self.sample_prompt_embeds = prompt_embeds

    def enable_text_encoder_block_swap(self, text_encoder_blocks_to_swap):
        if text_encoder_blocks_to_swap == 0:
            return
        
        # Extract layers from Qwen text encoder
        # Structure: text_encoder.model.language_model.model.layers
        text_encoder = self.text_encoder
        if hasattr(text_encoder, 'model') and hasattr(text_encoder.model, 'language_model'):
            language_model = text_encoder.model.language_model
            if hasattr(language_model, 'model') and hasattr(language_model.model, 'layers'):
                layers = language_model.model.layers
            else:
                raise ValueError('Could not find layers in Qwen text encoder at expected path')
        else:
            raise ValueError('Could not find language_model in Qwen text encoder')
        
        num_layers = len(layers)
        assert (
            text_encoder_blocks_to_swap <= num_layers - 2
        ), f'Cannot swap more than {num_layers - 2} text encoder blocks. Requested {text_encoder_blocks_to_swap} blocks to swap.'
        
        self.text_encoder_offloader = ModelOffloader(
            'TextEncoderBlock', layers, num_layers, text_encoder_blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        
        # Temporarily detach layers to prevent automatic GPU movement
        language_model.model.layers = None
        # Move text encoder structure to GPU (but layers will be managed separately)
        text_encoder.to('cuda')
        # Reattach layers
        language_model.model.layers = layers
        
        self.prepare_text_encoder_block_swap_training()
        print(f'Text encoder block swap enabled. Swapping {text_encoder_blocks_to_swap} blocks out of {num_layers} blocks.')

    def prepare_text_encoder_block_swap_training(self):
        if self.text_encoder_offloader is not None:
            self.text_encoder_offloader.enable_block_swap()
            self.text_encoder_offloader.set_forward_only(False)
            self.text_encoder_offloader.prepare_block_devices_before_forward()

    def prepare_text_encoder_block_swap_inference(self, disable_block_swap=False):
        if self.text_encoder_offloader is not None:
            if disable_block_swap:
                self.text_encoder_offloader.disable_block_swap()
            self.text_encoder_offloader.set_forward_only(True)
            self.text_encoder_offloader.prepare_block_devices_before_forward()

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, state_dict):
        safetensors.torch.save_file(state_dict, save_dir / 'model.safetensors', metadata={'format': 'pt'})

    def get_preprocess_media_file_fn(self):
        return self.preprocess_media_file_fn

    def get_call_vae_fn(self, vae):
        def fn(*args):
            image = args[0]
            latents = vae.encode(image.to(vae.device, vae.dtype)).latent_dist.mode()
            latents = (latents - vae.latents_mean_tensor) / vae.latents_std_tensor
            result = {'latents': latents}
            if len(args) == 2:
                control_image = args[1]
                control_latents = vae.encode(control_image.to(vae.device, vae.dtype)).latent_dist.mode()
                control_latents = (control_latents - vae.latents_mean_tensor) / vae.latents_std_tensor
                result['control_latents'] = control_latents
            return result
        return fn

    def load_image_for_vlm(self, path):
        pil_img = Image.open(path)
        height, width = pil_img.height, pil_img.width

        if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
            pil_img = pil_img.convert('RGBA')

        # add white background for transparent images
        if pil_img.mode == 'RGBA':
            canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
            canvas.alpha_composite(pil_img)
            pil_img = canvas.convert('RGB')
        else:
            pil_img = pil_img.convert('RGB')

        scale_factor = self.vlm_image_size / math.sqrt(height*width)
        return ImageOps.scale(pil_img, scale_factor)

    def _get_qwen_prompt_embeds(
        self,
        prompt,
        control_files,
        device=None,
        dtype=None,
    ):
        device = device or getattr(self, '_execution_device', torch.device('cuda'))
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Check if text encoder block swapping is enabled
        use_block_swap = self.text_encoder_offloader is not None and self.text_encoder_offloader.blocks_to_swap is not None and self.text_encoder_offloader.blocks_to_swap > 0

        if control_files is None:
            template = self.prompt_template_encode
            drop_idx = self.prompt_template_encode_start_idx
            txt = [template.format(e) for e in prompt]
            txt_tokens = self.tokenizer(
                txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            attention_mask = txt_tokens.attention_mask.to(device)
            
            if use_block_swap:
                # Manual forward pass with block swapping
                outputs = self._text_encoder_forward_with_block_swap(
                    input_ids=txt_tokens.input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            else:
                outputs = self.text_encoder(
                    input_ids=txt_tokens.input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
        else:
            template = self.prompt_template_encode_edit
            drop_idx = self.prompt_template_encode_start_idx_edit
            txt = [template.format(e) for e in prompt]
            images = [
                self.load_image_for_vlm(file)
                for file in control_files
            ]
            model_inputs = self.processor(
                text=txt,
                images=images,
                padding=True,
                return_tensors="pt",
            ).to(device)
            attention_mask = model_inputs.attention_mask.to(device)
            
            if use_block_swap:
                # Manual forward pass with block swapping
                outputs = self._text_encoder_forward_with_block_swap(
                    input_ids=model_inputs.input_ids,
                    attention_mask=attention_mask,
                    pixel_values=model_inputs.pixel_values,
                    image_grid_thw=model_inputs.image_grid_thw,
                    output_hidden_states=True,
                )
            else:
                outputs = self.text_encoder(
                    input_ids=model_inputs.input_ids,
                    attention_mask=attention_mask,
                    pixel_values=model_inputs.pixel_values,
                    image_grid_thw=model_inputs.image_grid_thw,
                    output_hidden_states=True,
                )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

        return split_hidden_states

    def _text_encoder_forward_with_block_swap(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None, output_hidden_states=False):
        """Manual forward pass through text encoder with block swapping."""
        text_encoder = self.text_encoder
        language_model = text_encoder.model.language_model
        model = language_model.model
        layers = model.layers
        
        # Get embeddings
        inputs_embeds = model.embed_tokens(input_ids)
        
        # Process vision inputs if present (for edit mode)
        if pixel_values is not None:
            vision_outputs = text_encoder.visual.forward(pixel_values, image_grid_thw)
            vision_hidden_states = vision_outputs.last_hidden_state
            # Combine text and vision embeddings
            # This is a simplified version - actual Qwen implementation may differ
            inputs_embeds = torch.cat([vision_hidden_states, inputs_embeds], dim=1)
            # Update attention mask
            vision_mask = torch.ones(vision_hidden_states.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        
        hidden_states = inputs_embeds
        hidden_states_list = [hidden_states] if output_hidden_states else []
        
        # Forward through layers with block swapping
        for layer_idx, layer in enumerate(layers):
            # Wait for block to be on GPU
            self.text_encoder_offloader.wait_for_block(layer_idx)
            
            # Forward through layer
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=None,  # Qwen handles this internally
            )
            hidden_states = layer_outputs[0]
            
            if output_hidden_states:
                hidden_states_list.append(hidden_states)
            
            # Submit block swap for next iteration
            self.text_encoder_offloader.submit_move_blocks_forward(layer_idx)
        
        # Apply final layer norm
        hidden_states = model.norm(hidden_states)
        
        # Create outputs object similar to transformers output
        class TextEncoderOutput:
            def __init__(self, last_hidden_state, hidden_states=None):
                self.last_hidden_state = last_hidden_state
                self.hidden_states = hidden_states
        
        return TextEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(hidden_states_list) if output_hidden_states else None
        )

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video, control_file: list[str] | None):
            # args are lists
            assert not any(is_video)
            prompt_embeds = self._get_qwen_prompt_embeds(caption, control_file, device=text_encoder.device)
            return {'prompt_embeds': prompt_embeds}
        return fn

    def _pack_latents(self, latents, bs, num_channels_latents, h, w):
        """Pack latents from (bs, c, f, h, w) to (bs, seq_len, hidden_dim) sequence format."""
        # This method should match what the transformer's img_in expects
        # For now, delegate to transformer if it has the method, otherwise use einops-style packing
        if hasattr(self.transformer, '_pack_latents'):
            return self.transformer._pack_latents(latents, bs, num_channels_latents, h, w)
        
        # Fallback implementation: pack 2x2 patches
        # latents shape: (bs, num_channels_latents, num_frames, h, w)
        # Pack into sequence: (bs, seq_len, hidden_dim) where hidden_dim = num_channels_latents * 4
        # and seq_len = (h//2) * (w//2) * num_frames
        
        bs_actual, c, num_frames, h_spatial, w_spatial = latents.shape
        assert bs_actual == bs
        assert c == num_channels_latents
        
        # Pack 2x2 patches: (h, w) -> (h//2, w//2) with 4x channels
        # Reshape to extract 2x2 patches: (bs, c, f, h, w) -> (bs, c, f, h//2, 2, w//2, 2)
        latents = latents.reshape(bs, c, num_frames, h_spatial // 2, 2, w_spatial // 2, 2)
        # Permute to group patch elements: (bs, c, f, h//2, w//2, 2, 2)
        latents = latents.permute(0, 2, 3, 5, 1, 4, 6)  # (bs, f, h//2, w//2, c, 2, 2)
        # Flatten patch: (bs, f, h//2, w//2, c*4)
        latents = latents.reshape(bs, num_frames, h_spatial // 2, w_spatial // 2, c * 4)
        # Flatten spatial and frame: (bs, f*(h//2)*(w//2), c*4)
        latents = latents.reshape(bs, num_frames * (h_spatial // 2) * (w_spatial // 2), c * 4)
        
        return latents

    def _unpack_latents(self, latents_packed, bs, num_channels_latents, h, w):
        """Unpack latents from (bs, seq_len, hidden_dim) back to (bs, c, f, h, w) spatial format."""
        # This method should reverse _pack_latents
        if hasattr(self.transformer, '_unpack_latents'):
            return self.transformer._unpack_latents(latents, bs, num_channels_latents, h, w)
        
        x = latents_packed.view(bs, h // 2, w // 2, num_channels_latents, 2, 2)
        # 2. Permute to correct spatial order
        # Mo ve the '2' dimensions next to height and width
        # Target: (bs, channels, height//2, 2, width//2, 2)
        x = x.permute(0, 3, 1, 4, 2, 5)
    
        # 3. Collapse into final image spatial dimensions
        # Shape: (bs, channels, height, width)
        x = x.reshape(bs, num_channels_latents, h, w)
    
        # 4. Add the temporal dimension back (needed for the VAE)
        x = x.unsqueeze(2) # (bs, channels, 1, h, w)
    
        return x

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        prompt_embeds = inputs['prompt_embeds']
        mask = inputs['mask']
        device = latents.device

        # prompt embeds are variable length
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.bool, device=device) for e in prompt_embeds]
        max_seq_len = max([e.size(0) for e in prompt_embeds])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in prompt_embeds]
        )
        prompt_embeds_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        max_text_len = prompt_embeds_mask.sum(dim=1).max().item()
        prompt_embeds = prompt_embeds[:, :max_text_len, :]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_text_len]

        bs, channels, num_frames, h, w = latents.shape

        num_channels_latents = self.transformer.config.in_channels // 4
        assert num_channels_latents == channels
        latents = self._pack_latents(latents, bs, num_channels_latents, h, w)

        if mask is not None:
            mask = mask.unsqueeze(1).expand((-1, num_channels_latents, -1, -1))  # make mask (bs, c, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # add frame dimension
            mask = self._pack_latents(mask, bs, num_channels_latents, h, w)

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
        else:
            t = dist.sample((bs,)).to(device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1
        logger.debug('prepare_inputs: target.shape=%s latents.shape=%s', target.shape, latents.shape)

        img_shapes = [(1, h // 2, w // 2)]

        if 'control_latents' in inputs:
            control_latents = inputs['control_latents'].float()
            control_latents = self._pack_latents(control_latents, bs, num_channels_latents, h, w)
            assert control_latents.shape == latents.shape, (control_latents.shape, latents.shape)
            img_seq_len = torch.tensor(x_t.shape[1], device=x_t.device).repeat((bs,))
            extra = (img_seq_len,)
            x_t = torch.cat([x_t, control_latents], dim=1)
            img_shapes.append((1, h // 2, w // 2))
        else:
            extra = tuple()

        img_shapes = torch.tensor([img_shapes], dtype=torch.int32, device=device).repeat((bs, 1, 1))
        txt_seq_lens = torch.tensor([max_text_len], dtype=torch.int32, device=device).repeat((bs,))
        img_attention_mask = torch.ones((bs, x_t.shape[1]), dtype=torch.bool, device=device)
        attention_mask = torch.cat([prompt_embeds_mask, img_attention_mask], dim=1)
        # Make broadcastable with attention weights, which are [bs, num_heads, query_len, key_value_len]
        attention_mask = attention_mask.view(bs, 1, 1, -1)
        assert attention_mask.dtype == torch.bool

        return (
            (x_t, prompt_embeds, attention_mask, t, img_shapes, txt_seq_lens) + extra,
            (target, mask),
        )

    def generate_samples(self, prompts, num_inference_steps, height, width, seed, guidance_scale=None):
        images = []
        
        # Note: guidance_scale parameter is accepted for API compatibility but not used
        # QwenImageTransformer doesn't support guidance/classifier-free guidance
        
        self.transformer.eval()
        self.vae.eval()
        
        # Block swap is left as set by the caller (e.g. train.py prepare_block_swap_inference(disable_block_swap=...)).
        # We use the layer-based forward (to_layers) so the offloader can move blocks to GPU just-in-time.
        transformer_device = next(self.transformer.img_in.parameters()).device
        base_dtype = self.model_config['dtype']
        bs = 1
        num_channels_latents = self.transformer.config.in_channels // 4
        h, w = height // 8, width // 8
        img_seq_len = (h // 2) * (w // 2)

        for prompt in prompts:
            if (hasattr(self, 'sample_prompt_embeds') and self.sample_prompt_embeds is not None and
                hasattr(self, 'sample_prompts') and self.sample_prompts is not None):
                if prompt in self.sample_prompts:
                    cached_idx = self.sample_prompts.index(prompt)
                    prompt_embeds = self.sample_prompt_embeds[cached_idx].to(transformer_device, dtype=base_dtype)
                    if prompt_embeds.dim() == 2:
                        prompt_embeds = prompt_embeds.unsqueeze(0)
                else:
                    raise RuntimeError(f'Prompt "{prompt}" not found in cache.')
            else:
                raise RuntimeError("No cached embeddings found. Call cache_sample_prompts() first.")

            txt_len = prompt_embeds.shape[1]
            generator = torch.Generator(device=transformer_device).manual_seed(seed)

            with torch.no_grad():
                # Initial noise in same format as training: (bs, c, num_frames, h, w) then pack
                latents_spatial = torch.randn(
                    (bs, num_channels_latents, 1, h, w),
                    generator=generator,
                    device=transformer_device,
                    dtype=base_dtype,
                )
                latents_seq = self._pack_latents(latents_spatial, bs, num_channels_latents, h, w)

                img_shapes_list = [(1, h // 2, w // 2)]
                img_shapes_tensor = torch.tensor([img_shapes_list], dtype=torch.int32, device=transformer_device).repeat((bs, 1, 1))
                txt_seq_lens = torch.tensor([txt_len], dtype=torch.int32, device=transformer_device).repeat((bs,))
                prompt_embeds_mask = torch.ones((bs, txt_len), dtype=torch.bool, device=transformer_device)
                img_attention_mask = torch.ones((bs, img_seq_len), dtype=torch.bool, device=transformer_device)
                attention_mask = torch.cat([prompt_embeds_mask, img_attention_mask], dim=1).view(bs, 1, 1, -1)
                extra = ()

                timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=transformer_device, dtype=base_dtype)
                layers = self.to_layers()

                for i in tqdm(range(num_inference_steps), desc="Sampling"):
                    t_curr = timesteps[i]
                    t_next = timesteps[i + 1]
                    t_tensor = t_curr.expand(bs).to(transformer_device, dtype=base_dtype)

                    model_inputs = (
                        latents_seq,
                        prompt_embeds,
                        attention_mask,
                        t_tensor,
                        img_shapes_tensor,
                        txt_seq_lens,
                    ) + extra

                    x = model_inputs
                    for layer in layers:
                        x = layer(x)
                    model_output = x

                    dt = t_next - t_curr
                    dt = dt.to(model_output.dtype)
                    if i == 0:
                        model_output_before = model_output.shape
                        expanded = model_output.size(-1) != latents_seq.size(-1)
                        ratio = latents_seq.size(-1) // model_output.size(-1) if expanded else 1
                    if model_output.size(-1) != latents_seq.size(-1):
                        model_output = model_output.repeat_interleave(latents_seq.size(-1) // model_output.size(-1), dim=-1)
                    if i == 0:
                        logger.info(
                            'generate_samples (first step): latents_seq=%s model_output_before=%s expanded=%s ratio=%s model_output_after=%s',
                            latents_seq.shape, model_output_before, expanded, ratio, model_output.shape,
                        )
                    latents_seq = latents_seq + model_output * dt

                latents_packed = latents_seq
                latents_spatial = self._unpack_latents(latents_packed, bs, num_channels_latents, h, w)
                # QwenImage VAE decode expects (batch, channels, num_frame, height, width)
                latents = latents_spatial

                vae_param = next(self.vae.parameters(), None)
                if vae_param is None:
                    raise RuntimeError('VAE has no parameters.')
                has_meta = any(p.device.type == 'meta' for p in self.vae.parameters())
                if has_meta:
                    raise RuntimeError(
                        'VAE has meta tensors; cannot move to device or decode. '
                        'Ensure the VAE is fully loaded (e.g. _move_meta_tensors_to_device after loading).'
                    )
                decode_device = transformer_device
                self.vae.to(decode_device)
                vae_dtype = vae_param.dtype
                latents = latents.to(decode_device, dtype=vae_dtype)
                #scaling_factor = getattr(self.vae.config, 'scaling_factor', None)
                #if scaling_factor is not None:
                #    latents = latents / scaling_factor
                if scaling_factor is not None:
                # Check if scaling_factor needs to be broadcast over the 'Time' dimension (dim 2)
                if isinstance(scaling_factor, torch.Tensor):
                     # If factor is (1, 4, 1, 1), it might fail against (1, 4, 1, H, W) depending on exact shape
                     # Ensure it broadcasts correctly:
                     scaling_factor = scaling_factor.to(latents.device)
                     if scaling_factor.ndim < latents.ndim:
                         # Add missing dimensions until it matches
                         while scaling_factor.ndim < latents.ndim:
                             scaling_factor = scaling_factor.unsqueeze(-1)
    
                    latents = latents / scaling_factor
                    
                image = self.vae.decode(latents, return_dict=False)[0]
                if image.dim() == 5:
                    image = image.squeeze(2)

                image = (image / 2 + 0.5).clamp(0, 1)
                if image.device.type == 'meta':
                    raise RuntimeError(
                        'VAE decode returned meta tensor; ensure the VAE is fully loaded and on a real device. '
                        'If using init_empty_weights, load all VAE weights before sample generation.'
                    )
                image = image.to('cpu').permute(0, 2, 3, 1).float().numpy()
                pil_image = Image.fromarray((image[0] * 255).astype("uint8"))
                images.append(pil_image)

        return images

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        for i, block in enumerate(transformer.transformer_blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        blocks = transformer.transformer_blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        transformer.transformer_blocks = None
        transformer.to('cuda')
        transformer.transformer_blocks = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.img_in = model.img_in
        self.txt_norm = model.txt_norm
        self.txt_in = model.txt_in
        self.time_text_embed = model.time_text_embed
        self.pos_embed = model.pos_embed

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        if torch.is_grad_enabled():
            for item in inputs:
                if torch.is_floating_point(item):
                    item.requires_grad_(True)

        hidden_states, encoder_hidden_states, attention_mask, timestep, img_shapes, txt_seq_lens, *extra = inputs

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        temb = self.time_text_embed(timestep, hidden_states)

        img_shapes = img_shapes.tolist()
        txt_seq_lens = txt_seq_lens.tolist()
        vid_freqs, txt_freqs = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        return make_contiguous(hidden_states, encoder_hidden_states, attention_mask, temb, vid_freqs, txt_freqs) + tuple(extra)

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2, device=index.device).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, attention_mask, temb, vid_freqs, txt_freqs, *extra = inputs

        self.offloader.wait_for_block(self.block_idx)
        encoder_hidden_states, hidden_states = self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=None,
            temb=temb,
            image_rotary_emb=(vid_freqs, txt_freqs),
            joint_attention_kwargs={'attention_mask': attention_mask},
        )
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(hidden_states, encoder_hidden_states, attention_mask, temb, vid_freqs, txt_freqs) + tuple(extra)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, attention_mask, temb, vid_freqs, txt_freqs, *extra = inputs
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        if len(extra) > 0:
            assert len(extra) == 1
            img_seq_len = extra[0][0].item()
            output = output[:, :img_seq_len, ...]
        logger.debug('FinalLayer.forward: output.shape=%s', output.shape)
        return output
