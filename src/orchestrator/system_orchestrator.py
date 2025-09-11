"""System Orchestrator for managing the multi-agent pipeline"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.messaging.message_bus import MessageBus
from src.agents.data_ingestion_agent import DataIngestionAgent
from src.config.settings import load_config, setup_logging, validate_aws_credentials
from loguru import logger


class SystemOrchestrator:
    """Orchestrates the entire multi-agent system for stock prediction"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the orchestrator with configuration"""
        self.config = config or load_config()
        
        # Setup logging
        setup_logging(self.config)
        
        # Initialize message bus
        self.message_bus = MessageBus()
        
        # Agent registry
        self.agents: Dict[str, Any] = {}
        
        # System state
        self.running = False
        self.startup_time = None
        
        # Workflow tracking
        self.active_workflows: Dict[str, Dict] = {}
        
        logger.info("System Orchestrator initialized")
    
    async def initialize(self) -> bool:
        """Initialize all system components"""
        logger.info("Initializing multi-agent system...")
        
        try:
            # Validate prerequisites
            if not await self._validate_prerequisites():
                return False
            
            # Initialize agents
            await self._initialize_agents()
            
            # Start message bus
            asyncio.create_task(self.message_bus.start())
            logger.info("Message bus started")
            
            # Start all agents
            for agent_name, agent in self.agents.items():
                await agent.start()
                logger.info(f"Agent {agent_name} started")
            
            self.startup_time = datetime.now()
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    async def _validate_prerequisites(self) -> bool:
        """Validate system prerequisites"""
        logger.info("Validating system prerequisites...")
        
        # Check AWS credentials
        if not validate_aws_credentials():
            logger.error("AWS credentials validation failed")
            return False
        
        # Add other validations as needed
        logger.info("Prerequisites validation completed")
        return True
    
    async def _initialize_agents(self) -> None:
        """Initialize all agents"""
        logger.info("Initializing agents...")
        
        # Initialize Data Ingestion Agent
        self.agents["data_ingestion"] = DataIngestionAgent(
            agent_id="data_ingestion_agent",
            message_bus=self.message_bus,
            config=self.config
        )
        logger.info("Data Ingestion Agent initialized")
        
        # TODO: Initialize other agents
        # self.agents["preprocessing"] = PreprocessingAgent(...)
        # self.agents["training"] = TrainingAgent(...)
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    async def start_stock_pipeline(
        self, 
        symbols: Optional[List[str]] = None,
        workflow_id: Optional[str] = None
    ) -> str:
        """Start the stock prediction pipeline for given symbols"""
        
        # Use default symbols if none provided
        if symbols is None:
            symbols = self.config.get("default_symbols", ["AAPL", "GOOGL", "MSFT"])
        
        # Generate workflow ID if not provided
        if workflow_id is None:
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting stock pipeline {workflow_id} for symbols: {symbols}")
        
        # Track workflow
        self.active_workflows[workflow_id] = {
            "symbols": symbols,
            "start_time": datetime.now().isoformat(),
            "status": "RUNNING",
            "completed_symbols": [],
            "failed_symbols": []
        }
        
        # Start data ingestion for all symbols
        await self.agents["data_ingestion"].send_message(
            recipient="data_ingestion_agent",
            message_type="SCHEDULE_INGESTION",
            payload={
                "symbols": symbols,
                "timeframe": "daily",
                "workflow_id": workflow_id
            },
            correlation_id=workflow_id
        )
        
        logger.info(f"Pipeline {workflow_id} initiated")
        return workflow_id
    
    async def monitor_system(self) -> None:
        """Monitor system health and performance"""
        self.running = True
        logger.info("Starting system monitoring...")
        
        while self.running:
            try:
                # Check message bus health
                queue_size = self.message_bus.get_queue_size()
                subscriber_count = self.message_bus.get_subscriber_count()
                
                # Check agent health
                agent_stats = {}
                for agent_name, agent in self.agents.items():
                    if hasattr(agent, 'get_ingestion_stats'):
                        stats = await agent.get_ingestion_stats()
                        agent_stats[agent_name] = stats
                
                # Log system health every 5 minutes
                logger.info(f"System Health - Queue: {queue_size}, Subscribers: {subscriber_count}")
                if agent_stats:
                    logger.debug(f"Agent Stats: {agent_stats}")
                
                # Check for failed workflows
                await self._check_workflow_health()
                
                # Wait before next check
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(60)  # Shorter wait on error
    
    async def _check_workflow_health(self) -> None:
        """Check health of active workflows"""
        current_time = datetime.now()
        
        for workflow_id, workflow in list(self.active_workflows.items()):
            start_time = datetime.fromisoformat(workflow["start_time"])
            elapsed_time = (current_time - start_time).total_seconds()
            
            # Check for stuck workflows (running > 1 hour)
            if workflow["status"] == "RUNNING" and elapsed_time > 3600:
                logger.warning(f"Workflow {workflow_id} may be stuck (running for {elapsed_time}s)")
                # TODO: Implement recovery logic
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "system": {
                "running": self.running,
                "startup_time": self.startup_time.isoformat() if self.startup_time else None,
                "uptime_seconds": (
                    (datetime.now() - self.startup_time).total_seconds()
                    if self.startup_time else 0
                )
            },
            "message_bus": {
                "queue_size": self.message_bus.get_queue_size(),
                "subscribers": self.message_bus.get_subscriber_count()
            },
            "agents": {
                "count": len(self.agents),
                "names": list(self.agents.keys())
            },
            "workflows": {
                "active_count": len(self.active_workflows),
                "workflows": self.active_workflows
            }
        }
        
        # Get agent-specific stats
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'get_ingestion_stats'):
                status["agents"][f"{agent_name}_stats"] = await agent.get_ingestion_stats()
        
        return status
    
    async def stop_workflow(self, workflow_id: str) -> bool:
        """Stop a specific workflow"""
        if workflow_id not in self.active_workflows:
            logger.warning(f"Workflow {workflow_id} not found")
            return False
        
        logger.info(f"Stopping workflow {workflow_id}")
        
        # Update workflow status
        self.active_workflows[workflow_id]["status"] = "STOPPED"
        self.active_workflows[workflow_id]["stop_time"] = datetime.now().isoformat()
        
        # TODO: Send stop messages to agents if needed
        
        return True
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the entire system"""
        logger.info("Starting system shutdown...")
        
        # Stop monitoring
        self.running = False
        
        # Stop all workflows
        for workflow_id in list(self.active_workflows.keys()):
            await self.stop_workflow(workflow_id)
        
        # Shutdown agents
        for agent_name, agent in self.agents.items():
            try:
                if hasattr(agent, 'shutdown'):
                    await agent.shutdown()
                else:
                    await agent.stop()
                logger.info(f"Agent {agent_name} shutdown completed")
            except Exception as e:
                logger.error(f"Error shutting down agent {agent_name}: {e}")
        
        # Stop message bus
        await self.message_bus.stop()
        logger.info("Message bus stopped")
        
        # Log final statistics
        final_stats = await self.get_system_status()
        logger.info(f"System shutdown completed. Final stats: {final_stats}")
    
    async def run_single_ingestion(self, symbol: str, timeframe: str = "daily") -> Dict[str, Any]:
        """Run a single data ingestion for testing purposes"""
        logger.info(f"Running single ingestion for {symbol} ({timeframe})")
        
        correlation_id = f"single_{symbol}_{datetime.now().strftime('%H%M%S')}"
        
        # Send ingestion request
        await self.agents["data_ingestion"].send_message(
            recipient="data_ingestion_agent",
            message_type="INGEST_REQUEST",
            payload={
                "symbol": symbol,
                "timeframe": timeframe,
                "force_refresh": True
            },
            correlation_id=correlation_id
        )
        
        # Wait a bit and return status
        await asyncio.sleep(5)
        
        stats = await self.agents["data_ingestion"].get_ingestion_stats()
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "correlation_id": correlation_id,
            "stats": stats
        }


async def main():
    """Main entry point for the system"""
    orchestrator = None
    
    try:
        # Initialize orchestrator
        orchestrator = SystemOrchestrator()
        
        # Initialize system
        if not await orchestrator.initialize():
            logger.error("System initialization failed")
            return
        
        # Start monitoring
        monitor_task = asyncio.create_task(orchestrator.monitor_system())
        
        # Start default pipeline
        symbols = ["AAPL", "GOOGL", "MSFT"]
        workflow_id = await orchestrator.start_stock_pipeline(symbols)
        
        logger.info(f"System running. Pipeline {workflow_id} started for {symbols}")
        logger.info("Press Ctrl+C to stop the system")
        
        # Keep system running
        await monitor_task
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
    finally:
        if orchestrator:
            await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
