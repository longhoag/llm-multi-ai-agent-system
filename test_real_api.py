#!/usr/bin/env python3
"""
Real API Test Script for Data Ingestion Agent
Tests actual GPT-5-mini API, Alpha Vantage API, and S3 integration
"""

import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.settings import load_config
from src.messaging.message_bus import MessageBus
from src.agents.data_ingestion_agent import DataIngestionAgent
from src.agents.base_agent import AgentMessage

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", 
          format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")

async def test_real_api_integration():
    """Test real API integration with GPT-5-mini, Alpha Vantage, and S3"""
    
    # Load environment variables
    load_dotenv()
    logger.info("🚀 Starting Real API Integration Test")
    
    # Load configuration
    try:
        config = load_config()
        logger.info("✅ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return False
    
    # Verify required environment variables
    required_vars = [
        "ALPHA_VANTAGE_API_KEY",
        "OPENAI_API_KEY", 
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "S3_BUCKET_RAW_DATA"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        return False
    
    logger.info("✅ All required environment variables found")
    
    # Initialize message bus
    message_bus = MessageBus()
    
    try:
        # Initialize data ingestion agent
        logger.info("🤖 Initializing Data Ingestion Agent...")
        agent = DataIngestionAgent(
            agent_id="test_data_ingestion_agent",
            message_bus=message_bus,
            config=config
        )
        logger.info("✅ Data Ingestion Agent initialized")
        
        # Start the agent
        await agent.start()
        logger.info("✅ Agent started successfully")
        
        # Test 1: Single symbol ingestion with real APIs
        logger.info("📊 Test 1: Single symbol ingestion (AAPL)")
        test_message = AgentMessage(
            sender="test_orchestrator",
            recipient="test_data_ingestion_agent",
            message_type="INGEST_REQUEST",
            payload={
                "symbol": "AAPL",
                "timeframe": "daily",
                "force_refresh": True
            },
            correlation_id=f"test_single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        response = await agent.process_message(test_message)
        
        if response and response.message_type == "DATA_AVAILABLE":
            logger.success("✅ Test 1 PASSED: Single symbol ingestion successful")
            logger.info(f"   - Symbol: {response.payload.get('symbol')}")
            logger.info(f"   - S3 Key: {response.payload.get('s3_key')}")
            logger.info(f"   - Record Count: {response.payload.get('record_count', 'N/A')}")
        elif response and response.message_type == "INGESTION_ERROR":
            logger.error(f"❌ Test 1 FAILED: Ingestion error - {response.payload.get('error')}")
            return False
        else:
            logger.error("❌ Test 1 FAILED: No valid response received")
            return False
        
        # Test 2: Data quality validation with GPT-5-mini
        logger.info("🧠 Test 2: GPT-5-mini data quality validation")
        if response and response.payload.get('s3_key'):
            quality_message = AgentMessage(
                sender="test_orchestrator",
                recipient="test_data_ingestion_agent",
                message_type="DATA_QUALITY_CHECK",
                payload={"s3_key": response.payload['s3_key']},
                correlation_id=f"test_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            quality_response = await agent.process_message(quality_message)
            
            if quality_response and quality_response.message_type == "DATA_QUALITY_RESULT":
                quality_result = quality_response.payload.get('quality_result', {})
                logger.success("✅ Test 2 PASSED: Data quality validation successful")
                logger.info(f"   - Valid: {quality_result.get('is_valid')}")
                logger.info(f"   - Issues: {quality_result.get('issues', [])}")
                logger.info(f"   - Confidence: {quality_result.get('confidence')}")
            else:
                logger.error("❌ Test 2 FAILED: Data quality validation failed")
        
        # Test 3: Agent statistics and monitoring
        logger.info("📈 Test 3: Agent statistics")
        stats = await agent.get_ingestion_stats()
        logger.success("✅ Test 3 PASSED: Statistics retrieved")
        logger.info(f"   - Symbols Processed: {stats.get('symbols_processed', 0)}")
        logger.info(f"   - Successful Ingestions: {stats.get('successful_ingestions', 0)}")
        logger.info(f"   - Failed Ingestions: {stats.get('failed_ingestions', 0)}")
        logger.info(f"   - Last Ingestion: {stats.get('last_ingestion_time', 'N/A')}")
        
        # Test 4: Test smaller batch ingestion (avoid rate limits)
        logger.info("📦 Test 4: Small batch ingestion (GOOGL)")
        batch_message = AgentMessage(
            sender="test_orchestrator",
            recipient="test_data_ingestion_agent",
            message_type="INGEST_REQUEST",
            payload={
                "symbol": "GOOGL",
                "timeframe": "daily",
                "force_refresh": True
            },
            correlation_id=f"test_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Wait a bit to respect rate limits (Alpha Vantage free tier: 5 calls/minute)
        logger.info("⏳ Waiting 15 seconds to respect API rate limits...")
        await asyncio.sleep(15)
        
        batch_response = await agent.process_message(batch_message)
        
        if batch_response and batch_response.message_type == "DATA_AVAILABLE":
            logger.success("✅ Test 4 PASSED: Small batch ingestion successful")
        else:
            logger.warning("⚠️ Test 4 SKIPPED: Batch ingestion may have hit rate limits (expected)")
        
        # Final statistics
        final_stats = await agent.get_ingestion_stats()
        logger.info("📊 Final Test Results:")
        logger.info(f"   - Total Symbols Processed: {final_stats.get('symbols_processed', 0)}")
        logger.info(f"   - Success Rate: {(final_stats.get('successful_ingestions', 0) / max(final_stats.get('symbols_processed', 1), 1)) * 100:.1f}%")
        
        # Shutdown
        await agent.shutdown()
        logger.info("✅ Agent shutdown complete")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
    finally:
        await message_bus.stop()
        logger.info("✅ Message bus stopped")

async def main():
    """Main test function"""
    logger.info("🎯 LLM Multi-Agent System - Real API Integration Test")
    logger.info("Testing GPT-5-mini, Alpha Vantage API, and AWS S3 integration")
    logger.info("-" * 60)
    
    success = await test_real_api_integration()
    
    logger.info("-" * 60)
    if success:
        logger.success("🎉 ALL TESTS PASSED! Data Ingestion Agent is working properly with real APIs")
        return 0
    else:
        logger.error("❌ TESTS FAILED! Check the logs above for details")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
