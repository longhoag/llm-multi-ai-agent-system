"""Base agent implementation for the multi-agent system"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
from loguru import logger

if TYPE_CHECKING:
    from src.messaging.message_bus import MessageBus


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None


class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, agent_id: str, message_bus: 'MessageBus'):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.state = {}
        self.running = False
        logger.info(f"Initializing agent: {agent_id}")
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages and optionally return a response"""
        ...

    async def start(self) -> None:
        """Start the agent and begin listening for messages"""
        self.running = True
        logger.info(f"Agent {self.agent_id} started")
        await self.message_bus.subscribe(self.agent_id, self.process_message)
    
    async def stop(self) -> None:
        """Stop the agent gracefully"""
        self.running = False
        logger.info(f"Agent {self.agent_id} stopped")
    
    async def send_message(
        self,
        recipient: str,
        message_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> None:
        """Send a message to another agent"""
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id
        )
        await self.message_bus.publish(message)
        logger.debug(f"Message sent: {self.agent_id} -> {recipient} ({message_type})")
    
    def update_state(self, key: str, value: Any) -> None:
        """Update agent internal state"""
        self.state[key] = value
        logger.debug(f"Agent {self.agent_id} state updated: {key} = {value}")
    
    def get_state(self, key: str) -> Any:
        """Get agent internal state"""
        return self.state.get(key)
