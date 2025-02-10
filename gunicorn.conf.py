import multiprocessing
import os

# Server socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = 'uvicorn.workers.UvicornWorker'
worker_connections = 1000
timeout = 30
keepalive = 2

# Process naming
proc_name = 'sentiment-analysis-api'
pythonpath = '.'

# Logging
accesslog = '-'
errorlog = '-'
loglevel = os.getenv('LOG_LEVEL', 'info').lower()

# SSL (uncomment for HTTPS)
# keyfile = 'path/to/keyfile'
# certfile = 'path/to/certfile'

# Process management
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Server mechanics
preload_app = True
reload = False  # Set to True for development

# SSL Configuration
# Uncomment and configure these in production with proper SSL certificates
# keyfile = 'path/to/keyfile'
# certfile = 'path/to/certfile'
# ssl_version = 'TLS'
