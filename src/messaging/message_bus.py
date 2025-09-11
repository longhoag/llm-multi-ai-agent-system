"""Message bus implementation for inter-agent communication"""

import asyncio
from typing import Dict, Callable, List, Optional
from collections import defaultdict
from loguru import logger
from src.agents.base_agent import AgentMessage


class MessageBus:
    """Async message bus for agent communication using publish/subscribe pattern"""
    
    def __init__(self) -> None:
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        logger.info("Message bus initialized")
    
    async def subscribe(self, agent_id: str, handler: Callable) -> None:
        """Subscribe an agent to receive messages"""
        self.subscribers[agent_id].append(handler)
        logger.info(f"Agent {agent_id} subscribed to message bus")
    
    async def unsubscribe(self, agent_id: str, handler: Optional[Callable] = None) -> None:
        """Unsubscribe an agent from receiving messages"""
        if handler:
            if handler in self.subscribers[agent_id]:
                self.subscribers[agent_id].remove(handler)
        else:
            self.subscribers[agent_id].clear()
        logger.info(f"Agent {agent_id} unsubscribed from message bus")
    
    async def publish(self, message: AgentMessage) -> None:
        """Publish a message to the message queue"""
        await self.message_queue.put(message)
        logger.debug(
            f"Message published: {message.sender} -> {message.recipient} "
            f"({message.message_type})"
        )
    
    async def start(self) -> None:
        """Start the message bus processing loop"""
        self.running = True
        logger.info("Message bus started")
        
        while self.running:
            try:
                # Wait for message with timeout to allow graceful shutdown
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                await self._route_message(message)
            except asyncio.TimeoutError:
                # Continue loop to check if still running
                continue
            except Exception as e:
                logger.error(f"Error in message bus processing: {e}")
    
    async def stop(self) -> None:
        """Stop the message bus gracefully"""
        self.running = False
        logger.info("Message bus stopped")
    
    async def _route_message(self, message: AgentMessage) -> None:
        """Route message to appropriate subscribers"""
        handlers = self.subscribers.get(message.recipient, [])
        
        if not handlers:
            logger.warning(f"No handlers found for recipient: {message.recipient}")
            return
        
        # Process all handlers for the recipient
        for handler in handlers:
            try:
                response = await handler(message)
                if response:
                    await self.publish(response)
            except Exception as e:
                logger.error(
                    f"Error processing message in handler for {message.recipient}: {e}"
                )
    
    def get_subscriber_count(self) -> Dict[str, int]:
        """Get count of subscribers for monitoring"""
        return {agent_id: len(handlers) for agent_id, handlers in self.subscribers.items()}
    
    def get_queue_size(self) -> int:
        """Get current message queue size"""
        return self.message_queue.qsize()
