#!/usr/bin/env python3
"""
Clean, GPT-5-mini compatible workflow implementation
Direct tool orchestration without ReAct agents
"""

import os
import asyncio
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add src to path  
import sys
sys.path.insert(0, str(Path.cwd() / 'src'))

from src.external.alpha_vantage_client import AlphaVantageClient
from src.storage.s3_manager import S3Manager
from langchain_openai import ChatOpenAI

class GPT5MiniStockWorkflow:
    """Clean, GPT-5-mini compatible stock data workflow"""
    
    def __init__(self):
        # Initialize clients
        self.alpha_vantage = AlphaVantageClient(api_key=os.getenv('ALPHA_VANTAGE_API_KEY'))
        self.s3_manager = S3Manager(bucket_name=os.getenv('S3_BUCKET_RAW_DATA'))
        self.llm = ChatOpenAI(
            model="gpt-5-mini", 
            temperature=1.0,  # Only supported value for GPT-5-mini
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
    async def fetch_stock_data(self, symbol: str, timeframe: str = "daily") -> dict:
        """Fetch stock data from Alpha Vantage"""
        try:
            data = await self.alpha_vantage.get_daily_time_series(symbol, outputsize='compact')
            
            if data and 'data' in data:
                return {
                    "success": True,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "record_count": len(data["data"]),
                    "data": data["data"],
                    "metadata": data.get("metadata", {}),
                    "fetch_timestamp": datetime.now().isoformat()
                }
            else:
                return {"success": False, "error": "No data received", "symbol": symbol}
                
        except Exception as e:
            return {"success": False, "error": str(e), "symbol": symbol}
    
    def validate_data_with_gpt5(self, data: dict, symbol: str) -> dict:
        """Use GPT-5-mini to validate data quality"""
        validation_prompt = f"""
        Analyze this stock data for {symbol} and assess its quality:
        
        - Records available: {data.get('record_count', 0)}
        - Data type: {data.get('timeframe', 'unknown')}
        - Sample data available: {bool(data.get('data'))}
        
        Provide a brief quality assessment (1-2 sentences) and rate as: EXCELLENT, GOOD, FAIR, or POOR.
        """
        
        try:
            response = self.llm.invoke(validation_prompt)
            return {
                "validation_passed": True,
                "assessment": response.content,
                "validator": "GPT-5-mini",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "validation_passed": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def upload_to_s3_with_latest(self, data: dict, symbol: str, timeframe: str = "daily") -> dict:
        """Upload data to S3 with both timestamped and latest versions"""
        try:
            # Generate keys
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_key = f"raw_data/{timeframe}/{symbol}/{timestamp}.json"
            latest_key = f"raw_data/{timeframe}/{symbol}/latest.json"
            
            # Upload both versions
            timestamped_success = await self.s3_manager.upload_json(data, timestamped_key)
            latest_success = await self.s3_manager.upload_json(data, latest_key)
            
            if timestamped_success and latest_success:
                return {
                    "success": True,
                    "timestamped_key": timestamped_key,
                    "latest_key": latest_key,
                    "bucket": self.s3_manager.bucket_name,
                    "upload_timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"Upload failed - timestamped: {timestamped_success}, latest: {latest_success}"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def process_symbol(self, symbol: str, timeframe: str = "daily") -> dict:
        """Process a single stock symbol through the complete workflow"""
        print(f"🔄 Processing {symbol}...")
        
        # Step 1: Fetch data
        fetch_result = await self.fetch_stock_data(symbol, timeframe)
        if not fetch_result.get("success"):
            return {"symbol": symbol, "status": "failed", "step": "fetch", "error": fetch_result.get("error")}
        
        print(f"   ✅ Fetched {fetch_result.get('record_count', 0)} records")
        
        # Step 2: Validate with GPT-5-mini  
        validation_result = self.validate_data_with_gpt5(fetch_result, symbol)
        print(f"   🤖 GPT-5-mini assessment: {validation_result.get('assessment', 'N/A')[:60]}...")
        
        # Step 3: Upload to S3
        upload_result = await self.upload_to_s3_with_latest(fetch_result, symbol, timeframe)
        if not upload_result.get("success"):
            return {"symbol": symbol, "status": "failed", "step": "upload", "error": upload_result.get("error")}
        
        print(f"   ☁️  Uploaded to S3: {upload_result.get('latest_key')}")
        
        return {
            "symbol": symbol,
            "status": "completed",
            "fetch_result": fetch_result,
            "validation_result": validation_result,
            "upload_result": upload_result
        }
    
    async def run_workflow(self, symbols: list, timeframe: str = "daily") -> dict:
        """Run the complete workflow for multiple symbols"""
        print(f"🚀 Starting GPT-5-mini Stock Data Workflow")
        print(f"📊 Processing symbols: {symbols}")
        print(f"⏰ Timeframe: {timeframe}")
        print("=" * 50)
        
        start_time = datetime.now()
        results = {}
        
        for symbol in symbols:
            results[symbol] = await self.process_symbol(symbol, timeframe)
        
        # Generate summary with GPT-5-mini
        successful_symbols = [s for s, r in results.items() if r.get("status") == "completed"]
        failed_symbols = [s for s, r in results.items() if r.get("status") == "failed"]
        
        summary_prompt = f"""
        Summarize this stock data ingestion workflow results:
        
        Symbols processed: {len(symbols)}
        Successful: {len(successful_symbols)} ({successful_symbols})
        Failed: {len(failed_symbols)} ({failed_symbols})
        
        Provide a brief executive summary (2-3 sentences) of the workflow execution.
        """
        
        try:
            summary_response = self.llm.invoke(summary_prompt)
            workflow_summary = summary_response.content
        except Exception:
            workflow_summary = f"Processed {len(symbols)} symbols: {len(successful_symbols)} successful, {len(failed_symbols)} failed"
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\\n" + "=" * 50)
        print("📋 WORKFLOW SUMMARY")
        print("=" * 50)
        print(f"🎯 {workflow_summary}")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"✅ Success rate: {len(successful_symbols)}/{len(symbols)} ({len(successful_symbols)/len(symbols)*100:.1f}%)")
        
        return {
            "workflow_id": f"gpt5_workflow_{start_time.strftime('%Y%m%d_%H%M%S')}",
            "symbols_processed": symbols,
            "results": results,
            "summary": workflow_summary,
            "execution_time": duration,
            "success_rate": len(successful_symbols) / len(symbols),
            "completed_at": end_time.isoformat()
        }


async def main():
    """Test the GPT-5-mini compatible workflow"""
    workflow = GPT5MiniStockWorkflow()
    
    # Test with AAPL
    result = await workflow.run_workflow(['AAPL'], 'daily')
    
    # Verify latest.json was created
    print("\\n🔍 Verifying latest.json file creation...")
    
    try:
        import boto3
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        # Check for latest.json
        bucket = os.getenv('S3_BUCKET_RAW_DATA')
        latest_key = 'raw_data/daily/AAPL/latest.json'
        
        s3.head_object(Bucket=bucket, Key=latest_key)
        print("✅ latest.json file confirmed in S3!")
        
    except Exception as e:
        print(f"❌ latest.json verification failed: {e}")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\\n🎉 Workflow completed with {result['success_rate']*100:.1f}% success rate")
