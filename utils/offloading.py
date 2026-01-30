# Copied from Musubi Tuner with modifications:
# https://github.com/kohya-ss/musubi-tuner/blob/main/modules/custom_offloading_utils.py

# NOTE: this is modified to work with LoRA training only, and checks for 'lora' in the parameter
# names when doing some of the swapping. It does this because for the optimizer step, all the trained
# params need to be on GPU. Musubi Tuner and sd-scripts don't have this problem because the LoRA modules
# are completely separate in those projects, while here we use PEFT which replaces the linear layers with
# LoRA modules, and therefore when moving parts of the model to/from the GPU we have to take special consideration
# of the LoRA params which are what is being trained.

from concurrent.futures import ThreadPoolExecutor
import gc
import time
from typing import Optional
import torch
import torch.nn as nn


def clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def swap_weight_devices_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    buffer_move_jobs = []  # For buffers that need to be moved (not swapped)

    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        if 'lora' in module_to_cuda_name:
            continue
        # Handle weights
        if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
            module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
            if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))
            else:
                if module_to_cuda.weight.data.device.type != device.type:
                    module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)
        
        # Handle buffers (like LayerNorm weights, biases, etc.)
        # Buffers are moved directly, not swapped
        for buffer_name in list(module_to_cuda._buffers.keys()):
            buffer = module_to_cuda._buffers.get(buffer_name)
            if buffer is not None and buffer.device.type != device.type:
                buffer_move_jobs.append((module_to_cuda, buffer_name, buffer))

    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # cuda to cpu (weights)
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.record_stream(stream)
            module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        stream.synchronize()

        # cpu to cuda (weights)
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view
        
        # Move buffers to device
        for module_to_cuda, buffer_name, buffer in buffer_move_jobs:
            module_to_cuda._buffers[buffer_name] = buffer.to(device, non_blocking=True)

    stream.synchronize()
    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value


def swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    not tested
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    # device to cpu
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

    synchronize_device()

    # cpu to device
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view

    synchronize_device()


def weights_to_device(layer: nn.Module, device: torch.device):
    """
    Move all weights and buffers in a layer to the specified device.
    This is more thorough than just calling .to() on the module when dealing with
    blocks that are part of a parent model structure.
    Recursively handles all submodules and ensures all buffers are moved.
    """
    for name, module in layer.named_modules():
        if device.type == 'cpu' and 'lora' in name:
            continue
        # Move weights
        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.device.type != device.type:
                module.weight.data = module.weight.data.to(device, non_blocking=True)
        # Move biases
        if hasattr(module, "bias") and module.bias is not None:
            if module.bias.device.type != device.type:
                module.bias.data = module.bias.data.to(device, non_blocking=True)
        # Move buffers (like LayerNorm's running_mean, running_var, or weight/bias in some cases)
        # Use named_buffers(recurse=False) to get buffers directly on this module
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer is not None and buffer.device.type != device.type:
                # Use register_buffer to properly update the buffer
                module.register_buffer(buffer_name, buffer.to(device, non_blocking=True))
        # Also check _buffers dict directly to catch any buffers not registered properly
        for buffer_name in list(module._buffers.keys()):
            buffer = module._buffers.get(buffer_name)
            if buffer is not None and buffer.device.type != device.type:
                module._buffers[buffer_name] = buffer.to(device, non_blocking=True)


def verify_module_on_device(module: nn.Module, device: torch.device, module_name: str = "") -> bool:
    """
    Verify that all parameters and buffers in a module are on the specified device.
    Returns True if all components are on device, False otherwise.
    Also moves any components found on wrong device.
    """
    all_on_device = True
    for name, submodule in module.named_modules():
        full_name = f"{module_name}.{name}" if module_name else name
        # Check parameters
        for param_name, param in submodule.named_parameters(recurse=False):
            if param.device.type != device.type:
                if 'lora' not in full_name:  # Skip LoRA params when moving to CPU
                    param.data = param.data.to(device, non_blocking=True)
                    all_on_device = False
        # Check buffers
        for buffer_name, buffer in submodule.named_buffers(recurse=False):
            if buffer is not None and buffer.device.type != device.type:
                submodule.register_buffer(buffer_name, buffer.to(device, non_blocking=True))
                all_on_device = False
        # Also check _buffers dict
        for buffer_name in list(submodule._buffers.keys()):
            buffer = submodule._buffers.get(buffer_name)
            if buffer is not None and buffer.device.type != device.type:
                submodule._buffers[buffer_name] = buffer.to(device, non_blocking=True)
                all_on_device = False
    return all_on_device


class Offloader:
    """
    common offloading class
    """

    def __init__(self, block_type: str, blocks: list[nn.Module], num_blocks: int, blocks_to_swap: int, device: torch.device, debug: bool = False):
        self.block_type = block_type
        self.blocks = blocks
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.blocks_to_swap_tmp = None
        self.device = device
        self.debug = debug

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)

    def _submit_move_blocks(self, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print(
                    f"[{self.block_type}] Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'}"
                )

            self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(f"[{self.block_type}] Moved blocks {bidx_to_cpu} and {bidx_to_cuda} in {time.perf_counter()-start_time:.2f}s")
            return bidx_to_cpu, bidx_to_cuda  # , event

        block_to_cpu = self.blocks[block_idx_to_cpu]
        block_to_cuda = self.blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda
        )

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"[{self.block_type}] Wait for block {block_idx}")
            start_time = time.perf_counter()

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        if self.debug:
            print(f"[{self.block_type}] Waited for block {block_idx}: {time.perf_counter()-start_time:.2f}s")


class ModelOffloader(Offloader):
    """
    supports forward offloading
    """

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        reentrant_activation_checkpointing: bool,
        debug: bool = False,
    ):
        super().__init__(block_type, blocks, num_blocks, blocks_to_swap, device, debug)

        self.supports_backward = supports_backward
        self.forward_only = not supports_backward  # forward only offloading: can be changed to True for inference
        self.reentrant_activation_checkpointing = reentrant_activation_checkpointing

        if self.supports_backward:
            # register backward hooks
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def disable_block_swap(self):
        self.blocks_to_swap_tmp = self.blocks_to_swap
        self.blocks_to_swap = None

    def enable_block_swap(self):
        if self.blocks_to_swap_tmp is not None:
            self.blocks_to_swap = self.blocks_to_swap_tmp

    def set_forward_only(self, forward_only: bool):
        self.forward_only = forward_only

    def __del__(self):
        if self.supports_backward:
            for handle in self.remove_handles:
                handle.remove()

    def create_backward_hook(self, block_index: int) -> Optional[callable]:
        # -1 for 0-based index
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        # create  hook
        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if self.debug:
                print(f"Backward hook for block {block_index}")

            if swapping:
                self._submit_move_blocks(block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            for block in self.blocks:
                block.to(self.device)
                # Verify all components are on device
                verify_module_on_device(block, self.device, f"block")
            return

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward")

        # Check if blocks are currently on CPU (common for text encoders)
        # If so, we need to move them incrementally to avoid OOM
        first_block_device = None
        if len(self.blocks) > 0:
            try:
                first_param = next(self.blocks[0].parameters(), None)
                if first_param is not None:
                    first_block_device = first_param.device
            except StopIteration:
                pass
        
        blocks_on_cpu = first_block_device is not None and first_block_device.type == 'cpu'
        
        if self.debug:
            print(f"[{self.block_type}] Blocks currently on CPU: {blocks_on_cpu}, blocks_to_swap: {self.blocks_to_swap}")

        # Move blocks that should stay on CUDA - do this incrementally to avoid OOM
        blocks_to_keep_on_cuda = self.blocks[0 : self.num_blocks - self.blocks_to_swap]
        for i, b in enumerate(blocks_to_keep_on_cuda):
            if blocks_on_cpu:
                # For blocks on CPU, we need to move them carefully to avoid OOM
                # Explicitly move all parameters and buffers first, then move structure
                try:
                    # Move all parameters explicitly
                    for param in b.parameters():
                        if param.device.type != self.device.type:
                            param.data = param.data.to(self.device, non_blocking=True)
                    # Move all buffers explicitly (important for LayerNorm, etc.)
                    for buffer in b.buffers():
                        if buffer.device.type != self.device.type:
                            buffer.data = buffer.data.to(self.device, non_blocking=True)
                    # Then move the block structure itself (handles any edge cases)
                    b.to(self.device)
                    # Use weights_to_device to ensure all nested components are moved
                    weights_to_device(b, self.device)
                    # Verify all components are on device
                    verify_module_on_device(b, self.device, f"block_{i}")
                    if self.debug and i % 5 == 0:
                        print(f"[{self.block_type}] Moved block {i} to CUDA")
                except torch.cuda.OutOfMemoryError as e:
                    if self.debug:
                        print(f"[{self.block_type}] OOM while moving block {i}, cleaning cache and retrying")
                    clean_memory_on_device(self.device)
                    gc.collect()
                    # Try moving weights/buffers first, then structure
                    weights_to_device(b, self.device)
                    b.to(self.device)
                    verify_module_on_device(b, self.device, f"block_{i}")
            else:
                b.to(self.device)
                weights_to_device(b, self.device)  # make sure weights and buffers are on device
                verify_module_on_device(b, self.device, f"block_{i}")

        # Handle blocks that should be swapped (kept on CPU)
        # These blocks should NOT be moved to CUDA at all
        blocks_to_swap = self.blocks[self.num_blocks - self.blocks_to_swap :]
        for i, b in enumerate(blocks_to_swap):
            block_idx = self.num_blocks - self.blocks_to_swap + i
            if blocks_on_cpu:
                # Blocks are already on CPU, ensure all weights are on CPU
                weights_to_device(b, torch.device('cpu'))
                verify_module_on_device(b, torch.device('cpu'), f"block_{block_idx}")
            else:
                # Blocks are on CUDA, move structure to CPU first, then ensure weights are on CPU
                b.to(torch.device('cpu'))
                weights_to_device(b, torch.device('cpu'))
                verify_module_on_device(b, torch.device('cpu'), f"block_{block_idx}")

        synchronize_device(self.device)
        clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        if self.reentrant_activation_checkpointing and torch.is_grad_enabled():
            # Second forward pass, don't do block swapping
            return
        self._wait_blocks_move(block_idx)
        # After waiting, ensure the block is fully on device (all params and buffers)
        # This is important because swap_weight_devices might not move all buffers
        block = self.blocks[block_idx]
        # Use weights_to_device to ensure all components are moved
        weights_to_device(block, self.device)
        # Verify all components are on device (this will move any remaining components)
        verify_module_on_device(block, self.device, f"block_{block_idx}")
        # Synchronize to ensure all moves are complete before forward pass
        synchronize_device(self.device)

    def submit_move_blocks_forward(self, block_idx: int):
        # check if blocks_to_swap is enabled
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.reentrant_activation_checkpointing and torch.is_grad_enabled():
            # Second forward pass, don't do block swapping
            return

        # if supports_backward and backward is enabled, we swap blocks more than blocks_to_swap in backward pass
        if not self.forward_only and block_idx >= self.blocks_to_swap:
            return

        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
        block_idx_to_cuda = block_idx_to_cuda % self.num_blocks  # this works for forward-only offloading
        self._submit_move_blocks(block_idx_to_cpu, block_idx_to_cuda)