"""
WebSocket support for real-time model interaction.
"""

import logging
import json
import asyncio
from typing import Dict, Set, Optional, Callable, Awaitable
from fastapi import WebSocket, WebSocketDisconnect
from .types import InferenceTask, ProgressUpdate

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections and message handling."""
    
    def __init__(self):
        """Initialize WebSocket manager."""
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._tasks: Dict[str, InferenceTask] = {}
        self._handlers: Dict[str, Callable] = {}
        logger.info("Initialized WebSocket manager")
        
    async def connect(
        self,
        websocket: WebSocket,
        task_id: str
    ):
        """
        Connect client to task.
        
        Args:
            websocket: WebSocket connection
            task_id: Task ID to connect to
        """
        try:
            await websocket.accept()
            
            # Add connection
            if task_id not in self._connections:
                self._connections[task_id] = set()
            self._connections[task_id].add(websocket)
            
            logger.info(f"Client connected to task {task_id}")
            
            # Handle messages
            try:
                while True:
                    message = await websocket.receive_json()
                    await self._handle_message(task_id, message)
            except WebSocketDisconnect:
                self._connections[task_id].remove(websocket)
                if not self._connections[task_id]:
                    del self._connections[task_id]
                logger.info(f"Client disconnected from task {task_id}")
                
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {str(e)}")
            raise
            
    async def broadcast_tokens(
        self,
        task_id: str,
        tokens: list
    ):
        """
        Broadcast tokens to connected clients.
        
        Args:
            task_id: Task ID
            tokens: List of tokens to broadcast
        """
        if task_id not in self._connections:
            return
            
        message = {
            'type': 'tokens',
            'task_id': task_id,
            'tokens': tokens
        }
        
        await self._broadcast(task_id, message)
        
    async def broadcast_progress(
        self,
        update: ProgressUpdate
    ):
        """
        Broadcast progress update.
        
        Args:
            update: Progress update
        """
        if update.task_id not in self._connections:
            return
            
        message = {
            'type': 'progress',
            'task_id': update.task_id,
            'status': update.status.value,
            'progress': update.progress,
            'tokens_per_second': update.tokens_per_second,
            'duration': update.duration,
            'error': update.error
        }
        
        await self._broadcast(update.task_id, message)
        
    def register_handler(
        self,
        message_type: str,
        handler: Callable
    ):
        """
        Register message handler.
        
        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        self._handlers[message_type] = handler
        logger.info(f"Registered handler for {message_type}")
        
    async def _broadcast(
        self,
        task_id: str,
        message: Dict
    ):
        """
        Broadcast message to all connected clients.
        
        Args:
            task_id: Task ID
            message: Message to broadcast
        """
        if task_id not in self._connections:
            return
            
        # Send to all connections
        dead_connections = set()
        for websocket in self._connections[task_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message: {str(e)}")
                dead_connections.add(websocket)
                
        # Clean up dead connections
        for websocket in dead_connections:
            self._connections[task_id].remove(websocket)
        if not self._connections[task_id]:
            del self._connections[task_id]
            
    async def _handle_message(
        self,
        task_id: str,
        message: Dict
    ):
        """
        Handle incoming message.
        
        Args:
            task_id: Task ID
            message: Message to handle
        """
        try:
            message_type = message.get('type')
            if not message_type:
                logger.warning("Message missing type")
                return
                
            handler = self._handlers.get(message_type)
            if not handler:
                logger.warning(f"No handler for {message_type}")
                return
                
            # Call handler
            await handler(task_id, message)
            
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            raise
