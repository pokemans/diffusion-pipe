import os
import re
import json
import subprocess
import threading
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit, join_room

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Store active job processes
active_jobs = {}
# Store statistics per job
job_statistics = defaultdict(list)
# Store log buffers per job
job_logs = defaultdict(list)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
JOBS_DIR = PROJECT_ROOT / 'jobs'


def get_job_path(job_name):
    """Get the path to a job directory."""
    return JOBS_DIR / job_name


def ensure_job_dir(job_name):
    """Ensure job directory exists."""
    job_path = get_job_path(job_name)
    job_path.mkdir(parents=True, exist_ok=True)
    return job_path


def parse_log_line(line):
    """Parse a log line to extract statistics."""
    # Pattern: steps: 5 loss: 0.6502
    pattern = r'steps:\s*(\d+)\s+loss:\s*([\d.]+)'
    match = re.search(pattern, line)
    if match:
        step = int(match.group(1))
        loss = float(match.group(2))
        return {'step': step, 'loss': loss, 'timestamp': datetime.now().isoformat()}
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
                
                # Emit log line via WebSocket
                socketio.emit('log_line', {'line': line}, room=job_name)
                
                # Parse statistics
                stats = parse_log_line(line)
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


@app.route('/api/jobs/overview', methods=['GET'])
def get_jobs_overview():
    """Get overview data for all jobs."""
    if not JOBS_DIR.exists():
        return jsonify({'jobs': []})
    
    overview = []
    jobs = [d.name for d in JOBS_DIR.iterdir() if d.is_dir()]
    
    for job_name in sorted(jobs):
        job_path = get_job_path(job_name)
        
        # Determine status
        if job_name in active_jobs:
            process = active_jobs[job_name]
            if process.poll() is None:
                status = "running"
            else:
                status = "finished"
        else:
            status = "stopped"
        
        # Get last statistics
        last_step = None
        last_loss = None
        if job_name in job_statistics and job_statistics[job_name]:
            last_stat = job_statistics[job_name][-1]
            last_step = last_stat['step']
            last_loss = last_stat['loss']
        
        # Check if config files exist
        has_config = (job_path / 'job_config.toml').exists()
        has_dataset = (job_path / 'dataset.toml').exists()
        
        overview.append({
            'job_name': job_name,
            'status': status,
            'last_step': last_step,
            'last_loss': last_loss,
            'has_config': has_config,
            'has_dataset': has_dataset
        })
    
    return jsonify({'jobs': overview})


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
    
    status = {
        'running': is_running,
        'statistics': job_statistics[job_name][-100:] if job_name in job_statistics else [],  # Last 100 points
        'log_count': len(job_logs.get(job_name, []))
    }
    
    if process:
        status['returncode'] = process.poll()  # None if still running
    
    return jsonify(status)


@app.route('/api/jobs/<job_name>/launch', methods=['POST'])
def launch_job(job_name):
    """Launch a training job."""
    if job_name in active_jobs:
        return jsonify({'error': 'Job is already running'}), 400
    
    # Check if config files exist
    job_config = get_job_path(job_name) / 'job_config.toml'
    if not job_config.exists():
        return jsonify({'error': 'Job config file not found'}), 404
    
    # Clear previous statistics and logs
    job_statistics[job_name] = []
    job_logs[job_name] = []
    
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
            # Try without venv activation (might be using system Python)
            cmd_str = f'NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=1 train.py --deepspeed --config {config_path}'
        
        # Launch process
        process = subprocess.Popen(
            ['bash', '-c', cmd_str],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
            env=dict(os.environ)  # Pass current environment
        )
        
        active_jobs[job_name] = process
        
        # Start thread to stream output
        thread = threading.Thread(target=stream_job_output, args=(job_name, process), daemon=True)
        thread.start()
        
        return jsonify({'success': True, 'message': 'Job launched'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_name>/stop', methods=['POST'])
def stop_job(job_name):
    """Stop a running job."""
    if job_name not in active_jobs:
        return jsonify({'error': 'Job is not running'}), 400
    
    try:
        process = active_jobs[job_name]
        process.terminate()
        # Wait a bit, then kill if still running
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
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
