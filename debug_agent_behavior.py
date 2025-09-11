#!/usr/bin/env python3
"""Debug script to test agent behavior with tools"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from src.tools.agent_tools import DATA_INGESTION_TOOLS, initialize_tools

load_dotenv()

async def debug_agent():
    """Test agent behavior step by step"""
    
    print("🔧 DEBUGGING AGENT BEHAVIOR")
    print("=" * 50)
    
    # Initialize tools
    initialize_tools(
        alpha_vantage_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
        s3_bucket=os.getenv('S3_BUCKET_RAW_DATA'),
        openai_key=os.getenv('OPENAI_API_KEY')
    )
    
    print(f"✅ Tools initialized: {len(DATA_INGESTION_TOOLS)} tools")
    for i, tool in enumerate(DATA_INGESTION_TOOLS):
        print(f"   {i+1}. {getattr(tool, 'name', 'Unknown')}")
    
    # Create agent
    print("\n🤖 Creating ReAct agent...")
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Use gpt-4o-mini for agent compatibility
        temperature=1.0,
        api_key=os.getenv('OPENAI_API_KEY')
    )
    
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, DATA_INGESTION_TOOLS, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=DATA_INGESTION_TOOLS, 
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True
    )
    
    # Test with a simple task
    test_prompt = """
You are a data ingestion specialist. Your task is to:

1. Fetch stock data for AAPL using fetch_stock_data tool
2. Upload the data to S3 using upload_stock_data_to_s3 tool (this creates both timestamped and latest.json files)

IMPORTANT: You MUST use the upload_stock_data_to_s3 tool for uploading. 

Please complete these steps and report the S3 keys created.
"""
    
    print("\n🎯 Testing agent with specific instructions...")
    print(f"Prompt: {test_prompt}")
    
    try:
        result = agent_executor.invoke({"input": test_prompt})
        
        print("\n📋 AGENT RESULT:")
        print(f"Output: {result['output']}")
        
        # Check if any files were actually created
        print("\n🔍 Checking S3 for new files...")
        import boto3
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        bucket = os.getenv('S3_BUCKET_RAW_DATA')
        response = s3.list_objects_v2(Bucket=bucket, Prefix='raw_data/daily/AAPL/')
        
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(minutes=5)
        
        new_files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['LastModified'].replace(tzinfo=None) > cutoff:
                    new_files.append(obj['Key'])
        
        if new_files:
            print(f"✅ NEW FILES CREATED: {new_files}")
        else:
            print("❌ NO NEW FILES CREATED")
            
    except Exception as e:
        print(f"\n❌ AGENT ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_agent())
