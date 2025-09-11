#!/usr/bin/env python3
"""
Quick validation test for the data quality fix
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.settings import load_config
from src.messaging.message_bus import MessageBus
from src.agents.data_ingestion_agent import DataIngestionAgent
from src.agents.base_agent import AgentMessage

logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{level}</level> | {message}")

async def test_data_quality_validation():
    """Test data quality validation specifically"""
    
    load_dotenv()
    config = load_config()
    message_bus = MessageBus()
    
    try:
        # Initialize agent
        agent = DataIngestionAgent(
            agent_id="quality_test_agent",
            message_bus=message_bus,
            config=config
        )
        
        await agent.start()
        
        # Test data quality on existing S3 file (from previous test)
        quality_message = AgentMessage(
            sender="test_orchestrator",
            recipient="quality_test_agent",
            message_type="DATA_QUALITY_CHECK",
            payload={"s3_key": "raw_data/daily/AAPL/20250910_154745.json"},
            correlation_id="quality_test_001"
        )
        
        response = await agent.process_message(quality_message)
        
        if response and response.message_type == "DATA_QUALITY_RESULT":
            quality_result = response.payload.get('quality_result', {})
            logger.success("✅ Data quality validation FIXED!")
            logger.info(f"   - Valid: {quality_result.get('is_valid')}")
            logger.info(f"   - Issues: {quality_result.get('issues', [])}")
            logger.info(f"   - Confidence: {quality_result.get('confidence')}")
            return True
        else:
            logger.error("❌ Data quality validation still failing")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    
    finally:
        await agent.shutdown()
        await message_bus.stop()

if __name__ == "__main__":
    success = asyncio.run(test_data_quality_validation())
    if success:
        logger.success("🎉 Data quality validation is now working!")
    else:
        logger.error("❌ Still has issues")
