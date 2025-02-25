"""
Progress tracking for inference tasks.
"""

import logging
import asyncio
from typing import Dict, Set, Optional, Callable, Awaitable
from datetime import datetime
from .types import InferenceTask, ProgressUpdate, TaskStatus

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Tracks progress of inference tasks."""
    
    def __init__(self):
        """Initialize progress tracker."""
        self._tasks: Dict[str, InferenceTask] = {}
        self._callbacks: Dict[str, Set[Callable[[ProgressUpdate], Awaitable[None]]]] = {}
        self._update_tasks: Dict[str, asyncio.Task] = {}
        logger.info("Initialized progress tracker")
        
    async def start_task(
        self,
        task: InferenceTask,
        callback: Optional[Callable[[ProgressUpdate], Awaitable[None]]] = None
    ):
        """
        Start tracking a task.
        
        Args:
            task: Task to track
            callback: Optional callback for progress updates
        """
        try:
            # Initialize progress
            task.status = TaskStatus.RUNNING
            task.progress = ProgressUpdate(
                task_id=task.task_id,
                status=TaskStatus.RUNNING,
                start_time=datetime.now()
            )
            
            # Store task
            self._tasks[task.task_id] = task
            
            # Add callback
            if callback:
                if task.task_id not in self._callbacks:
                    self._callbacks[task.task_id] = set()
                self._callbacks[task.task_id].add(callback)
                
            # Start update task
            self._update_tasks[task.task_id] = asyncio.create_task(
                self._update_loop(task)
            )
            
            logger.info(f"Started tracking task {task.task_id}")
            
        except Exception as e:
            logger.error(f"Error starting task tracking: {str(e)}")
            raise
            
    async def update_progress(
        self,
        task_id: str,
        generated_tokens: int,
        total_tokens: Optional[int] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Update task progress.
        
        Args:
            task_id: Task ID
            generated_tokens: Number of tokens generated
            total_tokens: Optional total tokens
            metadata: Optional metadata
        """
        try:
            task = self._tasks.get(task_id)
            if not task:
                logger.warning(f"Task {task_id} not found")
                return
                
            # Update progress
            task.progress.generated_tokens = generated_tokens
            if total_tokens is not None:
                task.progress.total_tokens = total_tokens
            if metadata:
                task.progress.metadata.update(metadata)
                
            # Notify callbacks
            await self._notify_callbacks(task_id)
            
        except Exception as e:
            logger.error(f"Error updating progress: {str(e)}")
            raise
            
    async def complete_task(
        self,
        task_id: str,
        error: Optional[str] = None
    ):
        """
        Mark task as completed.
        
        Args:
            task_id: Task ID
            error: Optional error message
        """
        try:
            task = self._tasks.get(task_id)
            if not task:
                logger.warning(f"Task {task_id} not found")
                return
                
            # Update status
            task.status = TaskStatus.FAILED if error else TaskStatus.COMPLETED
            task.progress.status = task.status
            task.progress.end_time = datetime.now()
            task.progress.error = error
            
            # Notify callbacks
            await self._notify_callbacks(task_id)
            
            # Clean up
            if task_id in self._update_tasks:
                self._update_tasks[task_id].cancel()
                del self._update_tasks[task_id]
                
            logger.info(f"Completed task {task_id}")
            
        except Exception as e:
            logger.error(f"Error completing task: {str(e)}")
            raise
            
    async def cancel_task(self, task_id: str):
        """
        Cancel a task.
        
        Args:
            task_id: Task ID
        """
        try:
            task = self._tasks.get(task_id)
            if not task:
                logger.warning(f"Task {task_id} not found")
                return
                
            # Update status
            task.status = TaskStatus.CANCELLED
            task.progress.status = TaskStatus.CANCELLED
            task.progress.end_time = datetime.now()
            
            # Notify callbacks
            await self._notify_callbacks(task_id)
            
            # Clean up
            if task_id in self._update_tasks:
                self._update_tasks[task_id].cancel()
                del self._update_tasks[task_id]
                
            logger.info(f"Cancelled task {task_id}")
            
        except Exception as e:
            logger.error(f"Error cancelling task: {str(e)}")
            raise
            
    def get_task(self, task_id: str) -> Optional[InferenceTask]:
        """Get task by ID."""
        return self._tasks.get(task_id)
        
    async def _update_loop(self, task: InferenceTask):
        """
        Background loop for progress updates.
        
        Args:
            task: Task to update
        """
        try:
            while True:
                # Send progress update
                await self._notify_callbacks(task.task_id)
                
                # Check timeout
                if task.progress.duration and task.progress.duration > task.config.timeout:
                    await self.complete_task(
                        task.task_id,
                        error="Task exceeded timeout"
                    )
                    break
                    
                # Wait for next update
                await asyncio.sleep(task.config.progress_interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in update loop: {str(e)}")
            await self.complete_task(task.task_id, error=str(e))
            
    async def _notify_callbacks(self, task_id: str):
        """
        Notify progress callbacks.
        
        Args:
            task_id: Task ID
        """
        try:
            task = self._tasks.get(task_id)
            if not task or not task.progress:
                return
                
            callbacks = self._callbacks.get(task_id, set())
            for callback in callbacks:
                try:
                    await callback(task.progress)
                except Exception as e:
                    logger.error(f"Error in progress callback: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error notifying callbacks: {str(e)}")
            raise
