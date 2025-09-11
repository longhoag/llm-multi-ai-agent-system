#!/usr/bin/env python3
"""
Environment validation         # Test with GPT-5-mini
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "user", "content": "Hello! Respond with just 'API test successful'"}
            ],
            max_tokens=10,
            temperature=0
        )
        
        response_text = response.choices[0].message.content.strip()
        if "successful" in response_text.lower():
            logger.success("✅ OpenAI GPT-5-mini API working!")
            return True Multi-Agent System
Tests all API keys and AWS resources are configured correctly
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import load_config, setup_logging, validate_aws_credentials
from loguru import logger


async def test_alpha_vantage_api():
    """Test Alpha Vantage API connection"""
    try:
        from src.external.alpha_vantage_client import AlphaVantageClient
        
        config = load_config()
        client = AlphaVantageClient(config["alpha_vantage_api_key"])
        
        async with client:
            # Test with a simple daily request
            data = await client.get_daily_time_series("AAPL", outputsize="compact")
            
            if data and "data" in data and len(data["data"]) > 0:
                logger.success(f"✅ Alpha Vantage API working! Got {len(data['data'])} records for AAPL")
                return True
            else:
                logger.error("❌ Alpha Vantage API returned no data")
                return False
                
    except Exception as e:
        logger.error(f"❌ Alpha Vantage API test failed: {e}")
        return False


async def test_openai_api(config: Dict[str, Any]) -> bool:
    """Test OpenAI API connection with GPT-5-mini"""
    try:
        from openai import OpenAI
        
        config = load_config()
        client = OpenAI(api_key=config["openai_api_key"])
        
        # Test with GPT-5-mini
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "user", "content": "Hello! Respond with just 'API test successful'"}
            ],
            max_tokens=10,
            temperature=0
        )
        
        response_text = response.choices[0].message.content.strip()
        if "successful" in response_text.lower():
            logger.success("✅ OpenAI GPT-5-mini API working!")
            return True
        else:
            logger.warning(f"⚠️ OpenAI API responded but unexpected content: {response_text}")
            return True  # Still working, just unexpected response
            
    except Exception as e:
        logger.error(f"❌ OpenAI API test failed: {e}")
        return False


async def test_aws_s3_buckets():
    """Test AWS S3 bucket access"""
    try:
        import boto3
        
        config = load_config()
        s3_client = boto3.client("s3", region_name=config["aws_region"])
        
        # Test each bucket
        buckets_to_test = [
            config["s3_bucket_raw_data"],
            config["s3_bucket_processed"],
            config["s3_bucket_models"]
        ]
        
        success_count = 0
        for bucket_name in buckets_to_test:
            try:
                # Try to head the bucket (check if it exists and we have access)
                s3_client.head_bucket(Bucket=bucket_name)
                logger.success(f"✅ S3 bucket '{bucket_name}' accessible")
                success_count += 1
            except Exception as e:
                logger.error(f"❌ S3 bucket '{bucket_name}' not accessible: {e}")
        
        return success_count == len(buckets_to_test)
        
    except Exception as e:
        logger.error(f"❌ AWS S3 test failed: {e}")
        return False


async def test_dynamodb_table():
    """Test DynamoDB table access"""
    try:
        import boto3
        
        config = load_config()
        dynamodb_client = boto3.client("dynamodb", region_name=config["aws_region"])
        
        table_name = config["dynamodb_table"]
        
        # Try to describe the table
        response = dynamodb_client.describe_table(TableName=table_name)
        table_status = response["Table"]["TableStatus"]
        
        if table_status == "ACTIVE":
            logger.success(f"✅ DynamoDB table '{table_name}' is active and accessible")
            return True
        else:
            logger.warning(f"⚠️ DynamoDB table '{table_name}' status: {table_status}")
            return False
            
    except Exception as e:
        logger.error(f"❌ DynamoDB test failed: {e}")
        return False


async def test_data_ingestion_agent():
    """Test the data ingestion agent initialization"""
    try:
        from src.messaging.message_bus import MessageBus
        from src.agents.data_ingestion_agent import DataIngestionAgent
        
        config = load_config()
        
        # Create message bus
        message_bus = MessageBus()
        
        # Create agent
        agent = DataIngestionAgent(
            agent_id="test_data_ingestion_agent",
            message_bus=message_bus,
            config=config
        )
        
        # Check agent state
        stats = await agent.get_ingestion_stats()
        if stats and "symbols_processed" in stats:
            logger.success("✅ Data Ingestion Agent initialized successfully")
            await agent.stop()
            await message_bus.stop()
            return True
        else:
            logger.error("❌ Data Ingestion Agent initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Data Ingestion Agent test failed: {e}")
        return False


async def main():
    """Main validation function"""
    print("🚀 LLM Multi-Agent System Environment Validation")
    print("=" * 50)
    
    try:
        # Load configuration and setup logging
        config = load_config()
        setup_logging(config)
        
        logger.info("Configuration loaded successfully from .env file")
        logger.info(f"AWS Region: {config['aws_region']}")
        logger.info(f"Raw Data Bucket: {config['s3_bucket_raw_data']}")
        logger.info(f"Default Symbols: {config['default_symbols']}")
        
        # Run all tests
        tests = [
            ("Alpha Vantage API", test_alpha_vantage_api),
            ("OpenAI GPT-5-mini API", test_openai_api),
            ("AWS Credentials", lambda: validate_aws_credentials()),
            ("S3 Buckets", test_aws_s3_buckets),
            ("DynamoDB Table", test_dynamodb_table),
            ("Data Ingestion Agent", test_data_ingestion_agent)
        ]
        
        results = []
        print("\n📋 Running Tests:")
        print("-" * 30)
        
        for test_name, test_func in tests:
            print(f"\n🧪 Testing {test_name}...")
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
                results.append((test_name, result))
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)
        
        passed = 0
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status:<8} {test_name}")
            if result:
                passed += 1
        
        print(f"\n📈 Results: {passed}/{len(results)} tests passed")
        
        if passed == len(results):
            print("\n🎉 ALL TESTS PASSED! Your environment is ready for the multi-agent system!")
            print("\n🚀 You can now run:")
            print("   poetry run python main.py --mode test --symbol AAPL")
            print("   poetry run python main.py")
        else:
            print(f"\n⚠️  {len(results) - passed} tests failed. Please check your configuration.")
            print("\n🔧 Common fixes:")
            print("   - Check your .env file has correct API keys")
            print("   - Ensure AWS credentials are valid")
            print("   - Verify S3 buckets exist and are accessible")
            print("   - Check DynamoDB table exists")
            
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
