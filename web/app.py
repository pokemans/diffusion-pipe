import os
import re
import json
import subprocess
import threading
import toml
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit, join_room

# Try to import GPU and system monitoring libraries
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Store active job processes
active_jobs = {}
# Store statistics per job
job_statistics = defaultdict(list)
# Store log buffers per job
job_logs = defaultdict(list)
# Store job metadata (steps_per_epoch, epochs, total_steps, gradient_accumulation_steps, iter_times, last_step_time)
job_metadata = defaultdict(dict)
# Store job queue (ordered list of job names waiting to run)
job_queue = []

# Memory management constants
MAX_LOG_LINES = 10000
PRUNE_THRESHOLD_MULTIPLIER = 1.2

# Iter time smoothing constants
ITER_TIME_WINDOW_SIZE = 30  # Number of iter times to keep for moving average
ITER_TIME_MIN_SAMPLES = 5  # Minimum samples before calculating average

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
JOBS_DIR = PROJECT_ROOT / 'jobs'

# NVML initialization state
nvml_initialized = False
nvml_gpu_handle = None
nvml_gpu_name = None


def get_job_path(job_name):
    """Get the path to a job directory."""
    return JOBS_DIR / job_name


def ensure_job_dir(job_name):
    """Ensure job directory exists."""
    job_path = get_job_path(job_name)
    job_path.mkdir(parents=True, exist_ok=True)
    return job_path


def init_nvml():
    """Initialize NVIDIA Management Library."""
    global nvml_initialized, nvml_gpu_handle, nvml_gpu_name
    
    if not NVML_AVAILABLE:
        return False
    
    if nvml_initialized:
        return True
    
    try:
        pynvml.nvmlInit()
        # Get handle for first GPU (index 0)
        nvml_gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(nvml_gpu_handle)
        # Handle both bytes and string returns (different pynvml versions)
        if isinstance(gpu_name, bytes):
            nvml_gpu_name = gpu_name.decode('utf-8')
        else:
            nvml_gpu_name = str(gpu_name)
        nvml_initialized = True
        return True
    except Exception as e:
        print(f"Failed to initialize NVML: {e}")
        nvml_initialized = False
        return False


def get_system_stats():
    """Collect system statistics including GPU, memory, and swap."""
    stats = {
        'gpu': {'available': False},
        'memory': {},
        'swap': {}
    }
    
    # GPU Stats
    if NVML_AVAILABLE and init_nvml():
        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(nvml_gpu_handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_gpu_handle)
            temperature = pynvml.nvmlDeviceGetTemperature(nvml_gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            
            memory_used_gb = memory_info.used / (1024 ** 3)
            memory_total_gb = memory_info.total / (1024 ** 3)
            memory_percent = (memory_info.used / memory_info.total) * 100 if memory_info.total > 0 else 0
            
            stats['gpu'] = {
                'available': True,
                'name': nvml_gpu_name,
                'utilization': utilization.gpu,
                'memory_used_gb': round(memory_used_gb, 1),
                'memory_total_gb': round(memory_total_gb, 1),
                'memory_percent': round(memory_percent, 1),
                'temperature': temperature
            }
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
            stats['gpu']['available'] = False
    
    # System Memory Stats
    if PSUTIL_AVAILABLE:
        try:
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024 ** 3)
            memory_total_gb = memory.total / (1024 ** 3)
            
            stats['memory'] = {
                'used_gb': round(memory_used_gb, 1),
                'total_gb': round(memory_total_gb, 1),
                'percent': round(memory.percent, 1)
            }
        except Exception as e:
            print(f"Error getting memory stats: {e}")
    
    # Swap Stats
    if PSUTIL_AVAILABLE:
        try:
            swap = psutil.swap_memory()
            swap_used_gb = swap.used / (1024 ** 3)
            swap_total_gb = swap.total / (1024 ** 3)
            
            stats['swap'] = {
                'used_gb': round(swap_used_gb, 1),
                'total_gb': round(swap_total_gb, 1),
                'percent': round(swap.percent, 1) if swap.total > 0 else 0
            }
        except Exception as e:
            print(f"Error getting swap stats: {e}")
    
    return stats


def has_running_jobs():
    """Check if any jobs are currently running."""
    for job_name, process in list(active_jobs.items()):
        if process.poll() is None:
            return True
    return False


def start_next_queued_job():
    """Start the next job in the queue if nothing is running."""
    if has_running_jobs():
        return None
    
    if not job_queue:
        return None
    
    next_job = job_queue.pop(0)
    process, error = _launch_job_internal(next_job)
    if error:
        return None
    return next_job


def prune_logs(job_name):
    """Prune log buffer if it exceeds threshold."""
    if job_name not in job_logs:
        return
    logs_list = job_logs[job_name]
    threshold = int(MAX_LOG_LINES * PRUNE_THRESHOLD_MULTIPLIER)
    if len(logs_list) > threshold:
        job_logs[job_name] = logs_list[-MAX_LOG_LINES:]


def calculate_smoothed_iter_time(iter_times):
    """
    Calculate smoothed iter time using exponential moving average.
    Gives more weight to recent values for better responsiveness while smoothing out noise.
    """
    if not iter_times:
        return 0.0
    
    if len(iter_times) == 1:
        return iter_times[0]
    
    # Use exponential moving average (EMA) for smoothing
    # Alpha controls the smoothing factor (0 < alpha <= 1)
    # Lower alpha = more smoothing, higher alpha = more responsive
    # Use adaptive alpha based on sample count for better initial stability
    sample_count = len(iter_times)
    if sample_count < 10:
        # More smoothing for fewer samples
        alpha = 0.3
    elif sample_count < 20:
        alpha = 0.4
    else:
        # More responsive for many samples
        alpha = 0.5
    
    # Calculate EMA: EMA = alpha * current + (1 - alpha) * previous_EMA
    ema = iter_times[0]
    for iter_time in iter_times[1:]:
        ema = alpha * iter_time + (1 - alpha) * ema
    
    return ema


def parse_log_line(line, job_name):
    """Parse a log line to extract statistics."""
    # Pattern: steps: 5 loss: 0.6502
    pattern = r'steps:\s*(\d+)\s+loss:\s*([\d.]+)'
    match = re.search(pattern, line)
    if match:
        step = int(match.group(1))
        loss = float(match.group(2))
        current_time = datetime.now()
        result = {'step': step, 'loss': loss, 'timestamp': current_time.isoformat()}
        
        # Initialize metadata if needed
        if job_name not in job_metadata:
            job_metadata[job_name] = {}
        metadata = job_metadata[job_name]
        
        # Calculate iter time from time between steps if we have previous step time
        if 'last_step_time' in metadata and 'last_step' in metadata:
            time_diff = (current_time - metadata['last_step_time']).total_seconds()
            steps_diff = step - metadata['last_step']
            if steps_diff > 0 and time_diff > 0:
                # Calculate iter time: time_diff covers steps_diff steps
                # steps_per_iter = gradient_accumulation_steps
                steps_per_iter = metadata.get('gradient_accumulation_steps', 1)
                if steps_per_iter > 0:
                    # Time per iter = time_diff / (steps_diff / steps_per_iter)
                    iter_time = time_diff / (steps_diff / steps_per_iter)
                    if 'iter_times' not in metadata:
                        metadata['iter_times'] = []
                    metadata['iter_times'].append(iter_time)
                    # Keep last N iter times for moving average smoothing
                    if len(metadata['iter_times']) > ITER_TIME_WINDOW_SIZE:
                        metadata['iter_times'].pop(0)
        
        metadata['last_step'] = step
        metadata['last_step_time'] = current_time
        
        # Calculate total_steps and ETA if we have the necessary data
        if 'total_steps' in metadata:
            result['total_steps'] = metadata['total_steps']
            # Calculate ETA using smoothed moving average
            if 'iter_times' in metadata and len(metadata['iter_times']) >= ITER_TIME_MIN_SAMPLES:
                avg_iter_time = calculate_smoothed_iter_time(metadata['iter_times'])
                steps_per_iter = metadata.get('gradient_accumulation_steps', 1)
                remaining_steps = metadata['total_steps'] - step
                if remaining_steps > 0 and steps_per_iter > 0:
                    eta_seconds = (remaining_steps / steps_per_iter) * avg_iter_time
                    result['eta_seconds'] = eta_seconds
        
        return result
    
    # Parse steps_per_epoch: oooooooooooooooooooooo steps_per_epoch: 131
    steps_per_epoch_pattern = r'oooooooooooooooooooooo steps_per_epoch:\s*(\d+)'
    match = re.search(steps_per_epoch_pattern, line)
    if match and job_name:
        steps_per_epoch = int(match.group(1))
        if job_name not in job_metadata:
            job_metadata[job_name] = {}
        job_metadata[job_name]['steps_per_epoch'] = steps_per_epoch
        
        # Calculate total_steps if we have epochs
        if 'epochs' in job_metadata[job_name]:
            job_metadata[job_name]['total_steps'] = steps_per_epoch * job_metadata[job_name]['epochs']
    
    # Parse iter time from deepspeed logs
    # Common patterns: "iter time: X.XXs", "iter: X.XXs", "time: X.XXs", "X.XXs/iter"
    # Also handle milliseconds: "iter time: X.XXms"
    iter_time_patterns = [
        r'iter\s+time:\s*([\d.]+)\s*(?:s|sec|seconds)',
        r'iter:\s*([\d.]+)\s*(?:s|sec|seconds)',
        r'time:\s*([\d.]+)\s*(?:s|sec|seconds).*iter',
        r'([\d.]+)\s*(?:s|sec|seconds)\s*/iter',
        r'iter\s+time:\s*([\d.]+)\s*ms',  # milliseconds
        r'([\d.]+)\s*ms\s*/iter',  # milliseconds
    ]
    for pattern in iter_time_patterns:
        match = re.search(pattern, line, re.IGNORECASE)
        if match and job_name:
            iter_time = float(match.group(1))
            # Convert milliseconds to seconds if needed
            if 'ms' in line.lower():
                iter_time = iter_time / 1000.0
            
            if job_name not in job_metadata:
                job_metadata[job_name] = {}
            if 'iter_times' not in job_metadata[job_name]:
                job_metadata[job_name]['iter_times'] = []
            # Keep last N iter times for moving average smoothing
            job_metadata[job_name]['iter_times'].append(iter_time)
            if len(job_metadata[job_name]['iter_times']) > ITER_TIME_WINDOW_SIZE:
                job_metadata[job_name]['iter_times'].pop(0)
            break
    
    return None


def stream_job_output(job_name, process):
    """Stream job output via WebSocket."""
    try:
        # Read from stdout (stderr is redirected to stdout)
        while True:
            line = process.stdout.readline()
            if not line:
                # Check if process has finished
                if process.poll() is not None:
                    break
                continue
            
            line = line.rstrip()
            if line:
                # Store log line
                job_logs[job_name].append(line)
                # Prune logs if they exceed threshold
                prune_logs(job_name)
                
                # Emit log line via WebSocket
                socketio.emit('log_line', {'line': line}, room=job_name)
                
                # Parse statistics
                stats = parse_log_line(line, job_name)
                if stats:
                    job_statistics[job_name].append(stats)
                    socketio.emit('statistics', {'stats': stats}, room=job_name)
    except Exception as e:
        socketio.emit('log_line', {'line': f'[ERROR] {str(e)}'}, room=job_name)
    finally:
        # Job finished
        if job_name in active_jobs:
            del active_jobs[job_name]
        socketio.emit('job_finished', {'job_name': job_name}, room=job_name)
        
        # Auto-start next queued job
        next_job = start_next_queued_job()
        if next_job:
            socketio.emit('job_started', {'job_name': next_job}, broadcast=True)


@app.route('/')
def index():
    """Serve the main interface."""
    return render_template('index.html')


@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all jobs in the jobs directory."""
    if not JOBS_DIR.exists():
        return jsonify({'jobs': []})
    
    jobs = [d.name for d in JOBS_DIR.iterdir() if d.is_dir()]
    return jsonify({'jobs': sorted(jobs)})


@app.route('/api/system/stats', methods=['GET'])
def get_system_stats_endpoint():
    """Get system statistics including GPU, memory, and swap."""
    try:
        stats = get_system_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/overview', methods=['GET'])
def get_jobs_overview():
    """Get overview data for all jobs, separated into running and queued lists."""
    if not JOBS_DIR.exists():
        return jsonify({'running_jobs': [], 'queued_jobs': []})
    
    running_jobs = []
    queued_jobs = []
    stopped_jobs = []
    
    jobs = [d.name for d in JOBS_DIR.iterdir() if d.is_dir()]
    
    for job_name in sorted(jobs):
        job_path = get_job_path(job_name)
        
        # Determine status
        is_running = False
        if job_name in active_jobs:
            process = active_jobs[job_name]
            if process.poll() is None:
                status = "running"
                is_running = True
            else:
                status = "finished"
        elif job_name in job_queue:
            status = "queued"
        else:
            status = "stopped"
        
        # Get statistics
        last_step = None
        last_loss = None
        moving_avg_loss = None
        step_display = None
        eta_seconds = None
        avg_iter_time = None
        
        metadata = job_metadata.get(job_name, {})
        
        if job_name in job_statistics and job_statistics[job_name]:
            last_stat = job_statistics[job_name][-1]
            last_step = last_stat['step']
            last_loss = last_stat['loss']
            
            # Calculate moving average of loss for running jobs (last 10 data points)
            if is_running and len(job_statistics[job_name]) > 0:
                recent_losses = [stat['loss'] for stat in job_statistics[job_name][-10:]]
                moving_avg_loss = sum(recent_losses) / len(recent_losses)
            
            # Format step display as "current/total" or "current/?"
            total_steps = last_stat.get('total_steps') or metadata.get('total_steps')
            if total_steps:
                step_display = f"{last_step}/{total_steps}"
            else:
                step_display = f"{last_step}/?"
            
            # Get ETA if available
            eta_seconds = last_stat.get('eta_seconds')
            if not eta_seconds and total_steps:
                # Calculate ETA if not in stat
                iter_times = metadata.get('iter_times', [])
                if len(iter_times) >= ITER_TIME_MIN_SAMPLES:
                    avg_iter_time = calculate_smoothed_iter_time(iter_times)
                    steps_per_iter = metadata.get('gradient_accumulation_steps', 1)
                    remaining_steps = total_steps - last_step
                    if remaining_steps > 0 and steps_per_iter > 0:
                        eta_seconds = (remaining_steps / steps_per_iter) * avg_iter_time
        
        # Calculate average iter time if not already calculated
        if avg_iter_time is None:
            iter_times = metadata.get('iter_times', [])
            if iter_times:
                avg_iter_time = calculate_smoothed_iter_time(iter_times)
        
        # Check if config files exist
        has_config = (job_path / 'job_config.toml').exists()
        has_dataset = (job_path / 'dataset.toml').exists()
        
        job_data = {
            'job_name': job_name,
            'status': status,
            'last_step': last_step,
            'step_display': step_display,
            'last_loss': last_loss,
            'moving_avg_loss': moving_avg_loss,
            'eta_seconds': eta_seconds,
            'avg_iter_time': avg_iter_time,
            'has_config': has_config,
            'has_dataset': has_dataset,
            'total_steps': metadata.get('total_steps'),
            'steps_per_epoch': metadata.get('steps_per_epoch'),
            'epochs': metadata.get('epochs'),
            'gradient_accumulation_steps': metadata.get('gradient_accumulation_steps')
        }
        
        if is_running:
            running_jobs.append(job_data)
        elif job_name in job_queue:
            queue_position = job_queue.index(job_name) + 1
            job_data['queue_position'] = queue_position
            queued_jobs.append(job_data)
        else:
            stopped_jobs.append(job_data)
    
    return jsonify({
        'running_jobs': running_jobs,
        'queued_jobs': queued_jobs,
        'stopped_jobs': stopped_jobs
    })


@app.route('/api/jobs/<job_name>/config', methods=['GET'])
def get_job_config(job_name):
    """Get job config TOML content."""
    config_path = get_job_path(job_name) / 'job_config.toml'
    if not config_path.exists():
        return jsonify({'error': 'Config file not found'}), 404
    
    try:
        content = config_path.read_text()
        return jsonify({'content': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_name>/config', methods=['POST'])
def save_job_config(job_name):
    """Save job config TOML content."""
    ensure_job_dir(job_name)
    config_path = get_job_path(job_name) / 'job_config.toml'
    
    try:
        data = request.get_json()
        content = data.get('content', '')
        config_path.write_text(content)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_name>/dataset', methods=['GET'])
def get_dataset_config(job_name):
    """Get dataset config TOML content."""
    config_path = get_job_path(job_name) / 'dataset.toml'
    if not config_path.exists():
        return jsonify({'error': 'Dataset config file not found'}), 404
    
    try:
        content = config_path.read_text()
        return jsonify({'content': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_name>/dataset', methods=['POST'])
def save_dataset_config(job_name):
    """Save dataset config TOML content."""
    ensure_job_dir(job_name)
    config_path = get_job_path(job_name) / 'dataset.toml'
    
    try:
        data = request.get_json()
        content = data.get('content', '')
        config_path.write_text(content)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_name>/status', methods=['GET'])
def get_job_status(job_name):
    """Get job status (running/stopped)."""
    is_running = job_name in active_jobs
    process = active_jobs.get(job_name)
    
    # Get statistics and enhance with total_steps and ETA
    # Return all statistics (not just last 100) to ensure full history available
    statistics = job_statistics[job_name] if job_name in job_statistics else []
    
    # Add total_steps and ETA to each stat if not already present
    metadata = job_metadata.get(job_name, {})
    total_steps = metadata.get('total_steps')
    for stat in statistics:
        if 'total_steps' not in stat and total_steps:
            stat['total_steps'] = total_steps
        # Calculate ETA if not present
        if 'eta_seconds' not in stat and 'step' in stat:
            step = stat['step']
            if total_steps and 'iter_times' in metadata and len(metadata['iter_times']) >= ITER_TIME_MIN_SAMPLES:
                avg_iter_time = calculate_smoothed_iter_time(metadata['iter_times'])
                steps_per_iter = metadata.get('gradient_accumulation_steps', 1)
                remaining_steps = total_steps - step
                if remaining_steps > 0 and steps_per_iter > 0:
                    stat['eta_seconds'] = (remaining_steps / steps_per_iter) * avg_iter_time
    
    status = {
        'running': is_running,
        'statistics': statistics,
        'log_count': len(job_logs.get(job_name, [])),
        'total_steps': total_steps
    }
    
    if process:
        status['returncode'] = process.poll()  # None if still running
    
    return jsonify(status)


def _launch_job_internal(job_name):
    """Internal function to launch a job (used by launch endpoint and queue system)."""
    # Check if config files exist
    job_config = get_job_path(job_name) / 'job_config.toml'
    if not job_config.exists():
        return None, 'Job config file not found'
    
    # Clear previous statistics and logs (but keep metadata like steps_per_epoch)
    job_statistics[job_name] = []
    job_logs[job_name] = []
    # Reset step tracking for iter time calculation
    if job_name in job_metadata:
        job_metadata[job_name].pop('last_step', None)
        job_metadata[job_name].pop('last_step_time', None)
    
    # Read job config to extract epochs and gradient_accumulation_steps
    try:
        with open(job_config, 'r') as f:
            config = toml.load(f)
            if job_name not in job_metadata:
                job_metadata[job_name] = {}
            if 'epochs' in config:
                job_metadata[job_name]['epochs'] = config['epochs']
            if 'gradient_accumulation_steps' in config:
                job_metadata[job_name]['gradient_accumulation_steps'] = config['gradient_accumulation_steps']
            # Calculate total_steps if we already have steps_per_epoch
            if 'steps_per_epoch' in job_metadata[job_name] and 'epochs' in job_metadata[job_name]:
                job_metadata[job_name]['total_steps'] = job_metadata[job_name]['steps_per_epoch'] * job_metadata[job_name]['epochs']
    except Exception:
        pass
    
    try:
        # Construct command - check for venv in common locations
        venv_paths = [
            PROJECT_ROOT / 'venv' / 'bin' / 'activate',
            PROJECT_ROOT / '.venv' / 'bin' / 'activate',
        ]
        
        venv_activate = None
        for path in venv_paths:
            if path.exists():
                venv_activate = path
                break
        
        config_path = job_config.absolute()
        
        # Build command string
        if venv_activate:
            cmd_str = f'source {venv_activate} && NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=1 train.py --deepspeed --config {config_path}'
        else:
            cmd_str = f'NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=1 train.py --deepspeed --config {config_path}'
        
        # Launch process with new process group so we can kill all children
        def preexec_fn():
            os.setsid()  # Create new process group
        
        process = subprocess.Popen(
            ['bash', '-c', cmd_str],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
            env=dict(os.environ),
            preexec_fn=preexec_fn
        )
        
        active_jobs[job_name] = process
        
        # Start thread to stream output
        thread = threading.Thread(target=stream_job_output, args=(job_name, process), daemon=True)
        thread.start()
        
        return process, None
    except Exception as e:
        return None, str(e)


@app.route('/api/jobs/<job_name>/launch', methods=['POST'])
def launch_job(job_name):
    """Launch a training job. If jobs are running, add to queue instead."""
    if job_name in active_jobs:
        process = active_jobs[job_name]
        if process.poll() is None:
            return jsonify({'error': 'Job is already running'}), 400
    
    # Remove from queue if present
    if job_name in job_queue:
        job_queue.remove(job_name)
    
    # Check if any jobs are running
    if has_running_jobs():
        # Add to queue
        if job_name not in job_queue:
            job_queue.append(job_name)
        queue_position = job_queue.index(job_name) + 1
        return jsonify({
            'success': True,
            'message': 'Job added to queue',
            'queued': True,
            'queue_position': queue_position
        })
    else:
        # Start immediately
        process, error = _launch_job_internal(job_name)
        if error:
            return jsonify({'error': error}), 500
        return jsonify({'success': True, 'message': 'Job launched', 'queued': False})


@app.route('/api/jobs/<job_name>/stop', methods=['POST'])
def stop_job(job_name):
    """Stop a running job."""
    if job_name not in active_jobs:
        return jsonify({'error': 'Job is not running'}), 400
    
    try:
        process = active_jobs[job_name]
        
        # Kill the entire process group (including all child processes)
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, 15)  # SIGTERM to entire process group
        except (ProcessLookupError, OSError):
            # Process group doesn't exist or process already dead
            # Fall back to killing just the process
            try:
                process.terminate()
            except ProcessLookupError:
                pass  # Process already dead
        
        # Wait a bit, then force kill if still running
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill the process group
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, 9)  # SIGKILL to entire process group
            except (ProcessLookupError, OSError):
                try:
                    process.kill()
                except ProcessLookupError:
                    pass  # Process already dead
        
        del active_jobs[job_name]
        socketio.emit('job_stopped', {'job_name': job_name}, room=job_name)
        
        return jsonify({'success': True, 'message': 'Job stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_name>/logs', methods=['GET'])
def get_job_logs(job_name):
    """Get all logs for a job."""
    logs = job_logs.get(job_name, [])
    return jsonify({'logs': logs})


@app.route('/api/jobs/<job_name>/queue/add', methods=['POST'])
def add_to_queue(job_name):
    """Explicitly add a job to the queue."""
    if job_name in active_jobs:
        process = active_jobs[job_name]
        if process.poll() is None:
            return jsonify({'error': 'Job is already running'}), 400
    
    if job_name not in job_queue:
        job_queue.append(job_name)
    
    queue_position = job_queue.index(job_name) + 1
    return jsonify({
        'success': True,
        'message': 'Job added to queue',
        'queue_position': queue_position
    })


@app.route('/api/jobs/<job_name>/queue/move_up', methods=['POST'])
def move_up_in_queue(job_name):
    """Move a job up in the queue."""
    if job_name not in job_queue:
        return jsonify({'error': 'Job is not in queue'}), 400
    
    current_index = job_queue.index(job_name)
    if current_index == 0:
        return jsonify({'error': 'Job is already at the top of the queue'}), 400
    
    # Swap with previous job
    job_queue[current_index], job_queue[current_index - 1] = job_queue[current_index - 1], job_queue[current_index]
    
    return jsonify({
        'success': True,
        'message': 'Job moved up in queue',
        'queue_position': current_index  # New position (0-indexed, but we return 1-indexed)
    })


@app.route('/api/jobs/<job_name>/queue/move_down', methods=['POST'])
def move_down_in_queue(job_name):
    """Move a job down in the queue."""
    if job_name not in job_queue:
        return jsonify({'error': 'Job is not in queue'}), 400
    
    current_index = job_queue.index(job_name)
    if current_index == len(job_queue) - 1:
        return jsonify({'error': 'Job is already at the bottom of the queue'}), 400
    
    # Swap with next job
    job_queue[current_index], job_queue[current_index + 1] = job_queue[current_index + 1], job_queue[current_index]
    
    return jsonify({
        'success': True,
        'message': 'Job moved down in queue',
        'queue_position': current_index + 2  # New position (1-indexed)
    })


@app.route('/api/jobs/<job_name>/queue/remove', methods=['POST'])
def remove_from_queue(job_name):
    """Remove a job from the queue."""
    if job_name not in job_queue:
        return jsonify({'error': 'Job is not in queue'}), 400
    
    job_queue.remove(job_name)
    return jsonify({'success': True, 'message': 'Job removed from queue'})


@app.route('/api/jobs/<job_name>/queue/start', methods=['POST'])
def start_job_from_queue(job_name):
    """Start a job immediately, removing it from queue if present."""
    if job_name in active_jobs:
        process = active_jobs[job_name]
        if process.poll() is None:
            return jsonify({'error': 'Job is already running'}), 400
    
    # Remove from queue if present
    if job_name in job_queue:
        job_queue.remove(job_name)
    
    # Check if any jobs are running
    if has_running_jobs():
        return jsonify({'error': 'Cannot start job: other jobs are running'}), 400
    
    # Start the job
    process, error = _launch_job_internal(job_name)
    if error:
        return jsonify({'error': error}), 500
    
    return jsonify({'success': True, 'message': 'Job started'})


@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    print('Client disconnected')


@socketio.on('subscribe_logs')
def handle_subscribe_logs(data):
    """Subscribe to logs for a specific job."""
    job_name = data.get('job_name')
    if job_name:
        join_room(job_name)
        emit('subscribed', {'job_name': job_name})
        
        # Send existing logs
        if job_name in job_logs:
            for line in job_logs[job_name][-1000:]:  # Last 1000 lines
                emit('log_line', {'line': line})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
