"""Simple test to verify APIs work without complex tools"""

import os
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

async def test_basic_functionality():
    """Test basic API functionality"""
    print("🚀 Basic Functionality Test")
    print("=" * 40)
    
    # Test 1: Environment Variables
    print("\\n1. Testing Environment Variables...")
    required_vars = ["OPENAI_API_KEY", "ALPHA_VANTAGE_API_KEY", "S3_BUCKET_RAW_DATA"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing: {missing}")
        return False
    else:
        print("✅ All environment variables present")
    
    # Test 2: OpenAI API with GPT-5-mini
    print("\\n2. Testing OpenAI API (GPT-5-mini)...")
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model="gpt-5-mini",
            temperature=1.0,  # GPT-5-mini only supports temperature=1.0
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        response = llm.invoke("Say exactly: 'Hello from GPT-5-mini'")
        
        if response and hasattr(response, 'content') and 'GPT-5-mini' in response.content:
            print(f"✅ GPT-5-mini working: {response.content}")
        else:
            print(f"⚠️  GPT-5-mini response unexpected: {response.content}")
            
    except Exception as e:
        print(f"❌ GPT-5-mini error: {e}")
        return False
    
    # Test 3: Alpha Vantage API
    print("\\n3. Testing Alpha Vantage API...")
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path.cwd() / 'src'))
        
        from src.external.alpha_vantage_client import AlphaVantageClient
        
        client = AlphaVantageClient(api_key=os.getenv('ALPHA_VANTAGE_API_KEY'))
        data = await client.get_daily_time_series('AAPL', outputsize='compact')
        
        if data and 'data' in data and len(data['data']) > 0:
            print(f"✅ Alpha Vantage working: {len(data['data'])} records for AAPL")
        else:
            print("❌ Alpha Vantage: No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Alpha Vantage error: {e}")
        return False
    
    # Test 4: S3 Operations
    print("\\n4. Testing S3 Operations...")
    try:
        from src.storage.s3_manager import S3Manager
        
        s3_manager = S3Manager(bucket_name=os.getenv('S3_BUCKET_RAW_DATA'))
        
        # Test upload
        test_data = {
            "test": True,
            "timestamp": datetime.now().isoformat(),
            "data": [{"symbol": "TEST", "price": 100.0}]
        }
        
        test_key = f"functionality_test/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        success = await s3_manager.upload_json(test_data, test_key)
        
        if success:
            print(f"✅ S3 upload successful: {test_key}")
            
            # Test download
            downloaded = await s3_manager.download_json(test_key)
            if downloaded and downloaded.get('test') == True:
                print("✅ S3 download successful")
            else:
                print("❌ S3 download failed")
                return False
        else:
            print("❌ S3 upload failed")
            return False
            
    except Exception as e:
        print(f"❌ S3 error: {e}")
        return False
    
    # Test 5: GPT-5-mini compatibility for workflows
    print("\\n5. Testing GPT-5-mini for Direct Workflow Use...")
    try:
        # Test direct tool calling without ReAct agent
        llm = ChatOpenAI(model="gpt-5-mini", temperature=1.0)  # GPT-5-mini only supports temperature=1.0
        
        # Simulate workflow decision making
        workflow_prompt = """
        You are orchestrating a stock data ingestion workflow.
        
        Task: Analyze AAPL stock data and decide what actions to take.
        Available actions:
        1. fetch_data: Get stock data from Alpha Vantage
        2. validate_data: Check data quality  
        3. upload_data: Store data in S3
        
        Current status:
        - Alpha Vantage API: Working (100+ records available)
        - S3 Storage: Working
        - Target symbol: AAPL
        
        Provide a brief execution plan (2-3 sentences) for processing this data.
        """
        
        response = llm.invoke(workflow_prompt)
        
        if response and hasattr(response, 'content'):
            print(f"✅ GPT-5-mini workflow planning: {response.content[:100]}...")
        else:
            print("❌ GPT-5-mini workflow planning failed")
            return False
            
    except Exception as e:
        print(f"❌ GPT-5-mini workflow error: {e}")
        return False
    
    print("\\n🎉 All basic functionality tests passed!")
    print("\\n💡 GPT-5-mini Compatibility Summary:")
    print("   ✅ Basic API calls work")
    print("   ✅ Can be used for workflow orchestration")
    print("   ❌ Cannot be used with LangChain ReAct agents (no 'stop' parameter support)")
    print("   💡 Recommendation: Use direct tool calls + GPT-5-mini for orchestration")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_basic_functionality())
    exit(0 if success else 1)
