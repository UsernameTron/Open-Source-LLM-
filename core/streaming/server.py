"""
FastAPI server for streaming inference.
"""

import logging
import asyncio
import uuid
from typing import Optional, Dict, Any
from fastapi import FastAPI, WebSocket, BackgroundTasks
from sse_starlette.sse import EventSourceResponse
from .types import (
    StreamingConfig,
    InferenceTask,
    TaskStatus,
    ProgressUpdate
)
from .websocket import WebSocketManager
from .progress import ProgressTracker

logger = logging.getLogger(__name__)

class StreamingServer:
    """Server for streaming inference API."""
    
    def __init__(self):
        """Initialize streaming server."""
        self.app = FastAPI(title="Streaming Inference API")
        self.websocket_manager = WebSocketManager()
        self.progress_tracker = ProgressTracker()
        self._setup_routes()
        logger.info("Initialized streaming server")
        
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.post("/inference/stream")
        async def stream_inference(
            prompt: str,
            config: Optional[Dict[str, Any]] = None,
            background_tasks: BackgroundTasks = None
        ):
            """
            Stream inference results via SSE.
            
            Args:
                prompt: Input prompt
                config: Optional configuration
                background_tasks: Background tasks
            """
            try:
                # Create task
                task_id = str(uuid.uuid4())
                task = InferenceTask(
                    task_id=task_id,
                    prompt=prompt,
                    config=StreamingConfig(**(config or {}))
                )
                
                # Setup response queue
                queue = asyncio.Queue()
                
                # Start inference
                background_tasks.add_task(
                    self._run_inference,
                    task,
                    queue
                )
                
                # Stream response
                async def event_generator():
                    try:
                        while True:
                            # Get next token
                            token = await queue.get()
                            if token is None:
                                break
                                
                            # Send token
                            yield {
                                'event': 'token',
                                'data': token
                            }
                            
                    except asyncio.CancelledError:
                        logger.info(f"Stream cancelled for task {task_id}")
                        await self.progress_tracker.cancel_task(task_id)
                        
                return EventSourceResponse(event_generator())
                
            except Exception as e:
                logger.error(f"Error in stream endpoint: {str(e)}")
                raise
                
        @self.app.websocket("/inference/ws/{task_id}")
        async def websocket_endpoint(
            websocket: WebSocket,
            task_id: str
        ):
            """
            WebSocket endpoint for real-time interaction.
            
            Args:
                websocket: WebSocket connection
                task_id: Task ID
            """
            await self.websocket_manager.connect(websocket, task_id)
            
        @self.app.post("/inference/start")
        async def start_inference(
            prompt: str,
            config: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Start inference task.
            
            Args:
                prompt: Input prompt
                config: Optional configuration
            """
            try:
                # Create task
                task_id = str(uuid.uuid4())
                task = InferenceTask(
                    task_id=task_id,
                    prompt=prompt,
                    config=StreamingConfig(**(config or {}))
                )
                
                # Start inference
                asyncio.create_task(self._run_inference(task))
                
                return {
                    'task_id': task_id,
                    'status': TaskStatus.PENDING.value
                }
                
            except Exception as e:
                logger.error(f"Error starting inference: {str(e)}")
                raise
                
        @self.app.get("/inference/status/{task_id}")
        async def get_status(task_id: str) -> Dict[str, Any]:
            """
            Get task status.
            
            Args:
                task_id: Task ID
            """
            task = self.progress_tracker.get_task(task_id)
            if not task:
                return {
                    'error': 'Task not found'
                }
                
            return task.to_dict()
            
        @self.app.post("/inference/cancel/{task_id}")
        async def cancel_inference(task_id: str):
            """
            Cancel inference task.
            
            Args:
                task_id: Task ID
            """
            await self.progress_tracker.cancel_task(task_id)
            return {
                'status': 'cancelled'
            }
            
    async def _run_inference(
        self,
        task: InferenceTask,
        queue: Optional[asyncio.Queue] = None
    ):
        """
        Run inference task.
        
        Args:
            task: Task to run
            queue: Optional queue for streaming tokens
        """
        try:
            # Start progress tracking
            await self.progress_tracker.start_task(
                task,
                self.websocket_manager.broadcast_progress
            )
            
            # Initialize generation
            total_tokens = min(
                task.config.max_tokens,
                2048  # Example max context
            )
            await self.progress_tracker.update_progress(
                task.task_id,
                generated_tokens=0,
                total_tokens=total_tokens
            )
            
            # Generate tokens
            for i in range(total_tokens):
                # Simulate token generation
                token = f"token_{i}"
                task.tokens.append(token)
                
                # Update progress
                await self.progress_tracker.update_progress(
                    task.task_id,
                    generated_tokens=i + 1,
                    total_tokens=total_tokens
                )
                
                # Stream token
                if queue:
                    await queue.put(token)
                await self.websocket_manager.broadcast_tokens(
                    task.task_id,
                    [token]
                )
                
                # Simulate generation time
                await asyncio.sleep(task.config.stream_interval)
                
                # Check if cancelled
                if task.status == TaskStatus.CANCELLED:
                    break
                    
            # Complete task
            await self.progress_tracker.complete_task(task.task_id)
            
            # Close stream
            if queue:
                await queue.put(None)
                
        except Exception as e:
            logger.error(f"Error in inference: {str(e)}")
            await self.progress_tracker.complete_task(
                task.task_id,
                error=str(e)
            )
            if queue:
                await queue.put(None)
                
    def start(
        self,
        host: str = "0.0.0.0",
        port: int = 8000
    ):
        """
        Start the server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)
