import multiprocessing
import os
import signal

# Gunicorn config variables
workers = int(os.getenv('WORKERS', min(4, multiprocessing.cpu_count())))  # Use env var or default
worker_class = "uvicorn.workers.UvicornWorker"
bind = "0.0.0.0:8000"
keepalive = 120
timeout = int(os.getenv('TIMEOUT', 120))
graceful_timeout = 30
preload_app = True  # Load application code before forking
max_requests = int(os.getenv('MAX_REQUESTS', 1000))  # Restart workers after max requests
max_requests_jitter = int(os.getenv('MAX_REQUESTS_JITTER', 50))  # Add jitter to prevent all workers restarting at once

# For debugging and testing
reload = os.getenv("GUNICORN_RELOAD", "false").lower() == "true"
reload_engine = "auto"

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

# Server hooks
def on_starting(server):
    """Log that Gunicorn is starting."""
    server.log.info("Starting LLM Engine server")

def on_reload(server):
    """Log that Gunicorn is reloading."""
    server.log.info("Reloading LLM Engine server")

def post_fork(server, worker):
    """Clean up after forking worker process."""
    server.log.info(f"Worker spawned (pid: {worker.pid})")
    
    # Set up signal handlers
    signal.signal(signal.SIGTERM, lambda signo, frame: worker_cleanup(worker))
    signal.signal(signal.SIGINT, lambda signo, frame: worker_cleanup(worker))

def worker_exit(server, worker):
    """Clean up when worker exits."""
    worker_cleanup(worker)
    server.log.info(f"Worker exited (pid: {worker.pid})")

def worker_cleanup(worker):
    """Clean up worker resources."""
    # Close any global database connections
    try:
        from api.main import _db_connection
        if _db_connection is not None:
            _db_connection.close()
    except:
        pass
