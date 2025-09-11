"""Main entry point for the LLM Multi-Agent System"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.orchestrator.system_orchestrator import SystemOrchestrator
from src.config.settings import load_config, setup_logging
from loguru import logger


async def run_system():
    """Run the complete multi-agent system"""
    orchestrator = None
    
    try:
        logger.info("Starting LLM Multi-Agent System for Stock Prediction")
        
        # Load configuration
        config = load_config()
        setup_logging(config)
        
        # Initialize orchestrator
        orchestrator = SystemOrchestrator(config)
        
        # Initialize system
        if not await orchestrator.initialize():
            logger.error("System initialization failed")
            sys.exit(1)
        
        # Start system monitoring
        monitor_task = asyncio.create_task(orchestrator.monitor_system())
        
        # Start the stock prediction pipeline
        symbols = config.get("default_symbols", ["AAPL", "GOOGL", "MSFT"])
        workflow_id = await orchestrator.start_stock_pipeline(symbols)
        
        logger.info(f"System operational. Pipeline {workflow_id} running for symbols: {symbols}")
        logger.info("System will continue running. Press Ctrl+C to shutdown.")
        
        # Keep system running
        await monitor_task
        
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)
    finally:
        if orchestrator:
            await orchestrator.shutdown()
        logger.info("System shutdown complete")


async def run_single_test(symbol: str = "AAPL"):
    """Run a single test ingestion for development/testing"""
    orchestrator = None
    
    try:
        logger.info(f"Running single test ingestion for {symbol}")
        
        # Load configuration
        config = load_config()
        setup_logging(config)
        
        # Initialize orchestrator
        orchestrator = SystemOrchestrator(config)
        
        # Initialize system
        if not await orchestrator.initialize():
            logger.error("System initialization failed")
            sys.exit(1)
        
        # Run single ingestion test
        result = await orchestrator.run_single_ingestion(symbol)
        logger.info(f"Test completed: {result}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)
    finally:
        if orchestrator:
            await orchestrator.shutdown()


def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LLM Multi-Agent System for Stock Price Prediction"
    )
    parser.add_argument(
        "--mode",
        choices=["run", "test"],
        default="run",
        help="Run mode: 'run' for full system, 'test' for single ingestion test"
    )
    parser.add_argument(
        "--symbol",
        default="AAPL",
        help="Stock symbol for test mode (default: AAPL)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "test":
        asyncio.run(run_single_test(args.symbol))
    else:
        asyncio.run(run_system())


if __name__ == "__main__":
    main()
