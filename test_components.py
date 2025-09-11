#!/usr/bin/env python3
"""
Quick Component Test Script
Tests individual components before full integration
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
from loguru import logger
from src.external.alpha_vantage_client import AlphaVantageClient

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{level}</level> | {message}")

async def test_alpha_vantage():
    """Test Alpha Vantage API connection"""
    logger.info("🔌 Testing Alpha Vantage API...")
    
    try:
        
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            logger.error("❌ ALPHA_VANTAGE_API_KEY not set")
            return False
        
        client = AlphaVantageClient(api_key)
        
        # Test with a simple call
        data = await client.get_daily_time_series("AAPL", outputsize="compact")
        
        if data and "data" in data and len(data["data"]) > 0:
            logger.success(f"✅ Alpha Vantage API working! Got {len(data['data'])} records")
            await client.close()
            return True
        else:
            logger.error("❌ Alpha Vantage API returned no data")
            await client.close()
            return False
            
    except Exception as e:
        logger.error(f"❌ Alpha Vantage API test failed: {e}")
        return False

async def test_openai():
    """Test OpenAI GPT-5-mini API"""
    logger.info("🧠 Testing OpenAI GPT-5-mini API...")
    
    try:
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("❌ OPENAI_API_KEY not set")
            return False
        
        llm = ChatOpenAI(
            model="gpt-5-mini",
            temperature=1,  # GPT-5-mini only supports temperature=1
            openai_api_key=api_key
        )
        
        response = await llm.ainvoke("Say 'API test successful' if you can read this.")
        
        if response and response.content:
            logger.success(f"✅ GPT-5-mini API working! Response: {response.content}")
            return True
        else:
            logger.error("❌ GPT-5-mini API returned empty response")
            return False
            
    except Exception as e:
        logger.error(f"❌ GPT-5-mini API test failed: {e}")
        return False

async def test_s3():
    """Test AWS S3 connection"""
    logger.info("☁️ Testing AWS S3 connection...")
    
    try:
        from src.storage.s3_manager import S3Manager
        
        bucket = os.getenv('S3_BUCKET_RAW_DATA')
        if not bucket:
            logger.error("❌ S3_BUCKET_RAW_DATA not set")
            return False
        
        s3_manager = S3Manager(bucket)
        
        # Test with a small file
        test_data = {"test": "data", "timestamp": "2024-01-15"}
        test_key = "test/component_test.json"
        
        # Upload test
        success = await s3_manager.upload_json(test_data, test_key)
        if not success:
            logger.error("❌ S3 upload failed")
            return False
        
        # Download test
        downloaded = await s3_manager.download_json(test_key)
        if downloaded != test_data:
            logger.error("❌ S3 download verification failed")
            return False
        
        # Cleanup
        s3_manager.delete_object(test_key)
        
        logger.success("✅ AWS S3 connection working!")
        return True
        
    except Exception as e:
        logger.error(f"❌ AWS S3 test failed: {e}")
        return False

async def main():
    """Run component tests"""
    load_dotenv()
    
    logger.info("🔧 Component Test Suite")
    logger.info("-" * 40)
    
    tests = [
        ("Alpha Vantage API", test_alpha_vantage()),
        ("OpenAI GPT-5-mini API", test_openai()),
        ("AWS S3", test_s3())
    ]
    
    results = []
    for test_name, test_coro in tests:
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    logger.info("-" * 40)
    logger.info("📊 Test Results:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"   {test_name:<20} {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.success("🎉 All component tests passed! Ready for full integration test.")
        return 0
    else:
        logger.error("⚠️ Some components failed. Fix issues before running full test.")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
