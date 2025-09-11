#!/usr/bin/env python3
"""
Test Latest File Creation for LangGraph Workflow

This test verifies that the workflow creates both timestamped and latest.json files
in S3 when ingesting stock data using GPT-5-mini and real APIs.
"""

import os
import sys
import asyncio
import json
import boto3
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.state.workflow_state import WorkflowStateManager, WorkflowStatus
from src.nodes.workflow_nodes import data_ingestion_node


def check_s3_files(symbol: str = "AAPL", timeframe: str = "daily"):
    """Check what files exist in S3 for the given symbol"""
    
    print(f"🔍 Checking S3 files for {symbol} ({timeframe})")
    print("-" * 50)
    
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        
        bucket = os.getenv("S3_BUCKET_RAW_DATA")
        prefix = f"raw_data/{timeframe}/{symbol}/"
        
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        if 'Contents' in response:
            files = response['Contents']
            print(f"📦 Found {len(files)} files in s3://{bucket}/{prefix}")
            
            timestamped_files = []
            latest_files = []
            
            for obj in files:
                key = obj['Key']
                size = obj['Size']
                modified = obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"   📄 {key}")
                print(f"      Size: {size} bytes, Modified: {modified}")
                
                if 'latest.json' in key:
                    latest_files.append(key)
                else:
                    timestamped_files.append(key)
            
            print(f"\n📊 File Analysis:")
            print(f"   • Timestamped files: {len(timestamped_files)}")
            print(f"   • Latest files: {len(latest_files)}")
            
            return {
                "total_files": len(files),
                "timestamped_files": timestamped_files,
                "latest_files": latest_files,
                "has_latest": len(latest_files) > 0
            }
        else:
            print(f"📦 No files found in s3://{bucket}/{prefix}")
            return {
                "total_files": 0,
                "timestamped_files": [],
                "latest_files": [],
                "has_latest": False
            }
            
    except Exception as e:
        print(f"❌ Error checking S3: {e}")
        return None


async def test_workflow_file_creation():
    """Test that the workflow creates both timestamped and latest files"""
    
    print("🧪 Testing LangGraph Workflow File Creation")
    print("=" * 60)
    print("🤖 Using GPT-5-mini for intelligent processing")
    print("📈 Fetching data from Alpha Vantage API")
    print("☁️ Storing in AWS S3 with latest.json support")
    print("=" * 60)
    
    # Check current S3 state
    print("\n📂 BEFORE: Current S3 state")
    before_state = check_s3_files("AAPL", "daily")
    
    if before_state is None:
        print("❌ Cannot access S3, aborting test")
        return False
    
    try:
        # Create workflow state
        manager = WorkflowStateManager()
        initial_state = manager.create_initial_state(
            symbols=["AAPL"],
            timeframe="daily",
            prediction_horizon=30
        )
        
        print(f"\n🔄 Executing Data Ingestion Node...")
        print(f"   • Workflow ID: {initial_state['workflow_id']}")
        
        # Execute the workflow node
        result_state = data_ingestion_node(initial_state)
        
        # Check results
        ingestion_data = result_state.get("ingestion", {})
        status = ingestion_data.get("status")
        s3_keys = ingestion_data.get("s3_keys", [])
        
        print(f"\n📋 Workflow Results:")
        print(f"   • Status: {status}")
        print(f"   • S3 Keys Returned: {len(s3_keys)}")
        
        for i, key in enumerate(s3_keys, 1):
            print(f"   • Key {i}: {key}")
        
        if status != WorkflowStatus.COMPLETED:
            print(f"❌ Workflow failed with status: {status}")
            errors = ingestion_data.get("errors", [])
            for error in errors:
                print(f"   Error: {error}")
            return False
        
        # Wait a moment for S3 eventual consistency
        print(f"\n⏳ Waiting for S3 consistency...")
        await asyncio.sleep(2)
        
        # Check S3 state after workflow
        print(f"\n📂 AFTER: S3 state after workflow")
        after_state = check_s3_files("AAPL", "daily")
        
        if after_state is None:
            print("❌ Cannot verify S3 state after workflow")
            return False
        
        # Analyze the changes
        print(f"\n📊 ANALYSIS:")
        
        files_added = after_state["total_files"] - before_state["total_files"]
        print(f"   • Files added: {files_added}")
        
        if after_state["has_latest"]:
            print(f"   • ✅ latest.json file created!")
            
            # Verify latest.json content
            latest_key = after_state["latest_files"][0]
            print(f"   • Latest file: {latest_key}")
            
            # Download and verify content
            try:
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    region_name=os.getenv("AWS_REGION", "us-east-1")
                )
                
                bucket = os.getenv("S3_BUCKET_RAW_DATA")
                response = s3_client.get_object(Bucket=bucket, Key=latest_key)
                content = json.loads(response['Body'].read())
                
                print(f"\n📄 latest.json Content Verification:")
                print(f"   • Symbol: {content.get('symbol')}")
                print(f"   • Timeframe: {content.get('timeframe')}")
                print(f"   • Record count: {content.get('record_count')}")
                print(f"   • Timestamp: {content.get('timestamp')}")
                
                if 'data' in content and content['data']:
                    sample_record = content['data'][0] if isinstance(content['data'], list) else content['data']
                    if isinstance(sample_record, dict) and 'date' in sample_record:
                        print(f"   • Sample data: {sample_record['date']} - ${sample_record.get('close', 'N/A')}")
                
            except Exception as e:
                print(f"   ⚠️ Error verifying latest.json content: {e}")
                
        else:
            print(f"   • ❌ latest.json file NOT created")
        
        # Final verdict
        success = (
            status == WorkflowStatus.COMPLETED and
            files_added >= 1 and
            after_state["has_latest"]
        )
        
        print(f"\n🎯 TEST RESULT:")
        if success:
            print(f"✅ SUCCESS: Workflow creates both timestamped and latest.json files!")
            print(f"🚀 System working as expected")
        else:
            print(f"❌ FAILURE: Workflow not creating latest.json files properly")
            print(f"🔧 Investigation needed")
        
        return success
        
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test execution"""
    
    print("🔥 LangGraph Latest File Creation Test")
    print("🤖 Using GPT-4o-mini (ReAct agent compatible, cost-effective)")
    print("=" * 55)
    
    # Validate environment
    required_vars = ["OPENAI_API_KEY", "ALPHA_VANTAGE_API_KEY", "S3_BUCKET_RAW_DATA"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(f"   • {var}")
        print("\nPlease check your .env file")
        sys.exit(1)
    
    print("✅ Environment validated")
    
    # Run the test
    success = await test_workflow_file_creation()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ LangGraph workflow creates latest.json files correctly")
        print("🚀 Ready for production use!")
    else:
        print("❌ TESTS FAILED")
        print("🔧 latest.json file creation needs fixing")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
