#!/usr/bin/env python3
"""
Direct tool test to verify upload_stock_data_to_s3 works
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tools.agent_tools import initialize_tools, upload_stock_data_to_s3

def test_upload_tool_directly():
    """Test the upload_stock_data_to_s3 tool directly"""
    
    print("🧪 Testing upload_stock_data_to_s3 tool directly")
    print("=" * 50)
    
    try:
        # Initialize tools
        print("🔧 Initializing tools...")
        initialize_tools(
            alpha_vantage_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
            s3_bucket=os.getenv("S3_BUCKET_RAW_DATA"),
            openai_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Test data
        test_data = [
            {"date": "2025-09-11", "open": 230.0, "high": 235.0, "low": 229.0, "close": 233.0, "volume": 1000000},
            {"date": "2025-09-10", "open": 228.0, "high": 232.0, "low": 227.0, "close": 230.0, "volume": 950000}
        ]
        
        print("📊 Test data prepared")
        print(f"   • Records: {len(test_data)}")
        print(f"   • Sample: {test_data[0]}")
        
        # Call the tool
        print("\n🚀 Calling upload_stock_data_to_s3...")
        result = upload_stock_data_to_s3(
            data=test_data,
            symbol="TEST",
            timeframe="daily"
        )
        
        print(f"\n📋 Tool Result:")
        print(f"   • Success: {result.get('success')}")
        
        if result.get('success'):
            print(f"   • Timestamped key: {result.get('timestamped_key')}")
            print(f"   • Latest key: {result.get('latest_key')}")
            print(f"   • Bucket: {result.get('bucket')}")
            print(f"   • Record count: {result.get('record_count')}")
            
            # Verify files exist in S3
            print(f"\n🔍 Verifying files in S3...")
            import boto3
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )
            
            bucket = result.get('bucket')
            
            for key_name, key_value in [("timestamped", result.get('timestamped_key')), ("latest", result.get('latest_key'))]:
                try:
                    s3_client.head_object(Bucket=bucket, Key=key_value)
                    print(f"   ✅ {key_name} file exists: {key_value}")
                except Exception as e:
                    print(f"   ❌ {key_name} file missing: {key_value} ({e})")
            
            return True
        else:
            print(f"   • Error: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"💥 Tool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_upload_tool_directly()
    print(f"\n🎯 Tool test {'PASSED' if success else 'FAILED'}")
