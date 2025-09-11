"""
Test GPT-4o-mini ReAct Agent Workflow
Tests the complete LangGraph workflow with ReAct agents using real APIs
"""

import os
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add src to path
import sys
sys.path.insert(0, str(Path.cwd() / 'src'))

from gpt4o_mini_react_workflow import GPT4oMiniReActWorkflow


async def test_react_agent_workflow():
    """Test the complete GPT-4o-mini ReAct agent workflow with real APIs"""
    print("🧪 Testing GPT-4o-mini ReAct Agent Workflow")
    print("=" * 60)
    
    # Verify environment variables
    required_env = ['OPENAI_API_KEY', 'ALPHA_VANTAGE_API_KEY', 'S3_BUCKET_RAW_DATA']
    missing_env = [key for key in required_env if not os.getenv(key)]
    
    if missing_env:
        print(f"❌ Missing environment variables: {missing_env}")
        return False
    
    print("✅ Environment variables configured")
    
    try:
        # Create workflow instance
        print("\n🚀 Initializing GPT-4o-mini ReAct Workflow...")
        workflow = GPT4oMiniReActWorkflow()
        print("✅ Workflow initialized successfully")
        
        # Test symbols
        test_symbols = ["AAPL", "GOOGL"]
        print(f"\n📊 Testing with symbols: {test_symbols}")
        
        # Run the complete workflow
        print("\n🔄 Executing ReAct agent workflow...")
        start_time = datetime.now()
        
        result = await workflow.run_workflow(test_symbols, timeframe="daily")
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Display results
        print(f"\n📋 Workflow Results:")
        print(f"   ⏱️  Execution Time: {execution_time:.2f} seconds")
        print(f"   ✅ Success: {result['success']}")
        
        if result['success']:
            print(f"   🆔 Workflow ID: {result['workflow_id']}")
            print(f"   📈 Symbols Processed: {result['symbols_processed']}")
            print(f"   🗂️  S3 Keys Created: {len(result.get('s3_keys', []))}")
            print(f"   📝 Node Status: {result.get('node_status', {})}")
            
            # Check for latest.json files
            s3_keys = result.get('s3_keys', [])
            latest_files = [key for key in s3_keys if 'latest.json' in key]
            print(f"   📄 Latest.json files: {len(latest_files)}")
            
            if latest_files:
                print("   📄 Latest files created:")
                for file in latest_files:
                    print(f"      • {file}")
            
            print(f"\n📝 Summary: {result.get('summary', 'No summary available')[:200]}...")
            
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
            return False
        
        print(f"\n✅ GPT-4o-mini ReAct workflow test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Workflow test failed: {e}")
        return False


async def test_individual_components():
    """Test individual components of the ReAct workflow"""
    print("\n🔧 Testing Individual ReAct Components")
    print("-" * 40)
    
    try:
        from langchain_openai import ChatOpenAI
        from src.tools.agent_tools import (
            fetch_stock_data_tool,
            validate_data_quality_tool,
            upload_stock_data_to_s3_tool,
            initialize_tools
        )
        
        # Initialize tools
        print("1. Initializing tools...")
        initialize_tools(
            alpha_vantage_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
            s3_bucket=os.getenv('S3_BUCKET_RAW_DATA'),
            openai_key=os.getenv('OPENAI_API_KEY')
        )
        print("   ✅ Tools initialized")
        
        # Test GPT-4o-mini
        print("2. Testing GPT-4o-mini...")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        response = llm.invoke("Say 'GPT-4o-mini ReAct agent ready'")
        if response and 'ready' in response.content:
            print("   ✅ GPT-4o-mini working")
        else:
            print("   ⚠️  GPT-4o-mini response unexpected")
        
        # Test fetch tool
        print("3. Testing fetch stock data tool...")
        fetch_result = await fetch_stock_data_tool._arun("AAPL", "compact")
        if fetch_result.get("success"):
            print(f"   ✅ Fetched {fetch_result.get('record_count', 0)} records for AAPL")
            
            # Test validation tool
            print("4. Testing data validation tool...")
            validation_result = validate_data_quality_tool._run(fetch_result, "AAPL")
            if validation_result.get("validation_passed"):
                print(f"   ✅ Data validation passed: {validation_result.get('overall_quality')}")
                
                # Test upload tool
                print("5. Testing S3 upload tool...")
                upload_result = await upload_stock_data_to_s3_tool._arun(fetch_result, "AAPL", "daily")
                if upload_result.get("success"):
                    print(f"   ✅ Uploaded to S3: {upload_result.get('latest_key')}")
                else:
                    print(f"   ❌ Upload failed: {upload_result.get('error')}")
            else:
                print(f"   ❌ Validation failed: {validation_result.get('error')}")
        else:
            print(f"   ❌ Fetch failed: {fetch_result.get('error')}")
        
        print("✅ Individual component tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False


async def main():
    """Run all tests for GPT-4o-mini ReAct workflow"""
    print("🧪 GPT-4o-mini ReAct Agent Test Suite")
    print("=" * 60)
    
    # Test individual components first
    component_test = await test_individual_components()
    
    if component_test:
        # Test complete workflow
        workflow_test = await test_react_agent_workflow()
        
        if workflow_test:
            print(f"\n🎉 All tests passed! GPT-4o-mini ReAct workflow is working correctly.")
        else:
            print(f"\n⚠️  Component tests passed but workflow test failed.")
    else:
        print(f"\n❌ Component tests failed. Check configuration.")


if __name__ == "__main__":
    asyncio.run(main())
