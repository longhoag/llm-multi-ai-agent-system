#!/usr/bin/env python3
"""
🎯 DEMO: Data Processing Agent with LLM Intelligence
==================================================

This demo showcases the complete data processing pipeline:
1. Generate 1000 synthetic stock records
2. Use LLM agent to intelligently process data via AWS Glue
3. Monitor job execution and validate results
4. Demonstrate production-ready multi-agent system

Requirements:
- AWS Glue job configured
- S3 buckets set up
- Environment variables configured
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dotenv import load_dotenv
import boto3
import pandas as pd
import numpy as np

from src.tools.agent_tools import initialize_tools
from src.nodes.workflow_nodes import preprocessing_node
from src.state.workflow_state import StockPredictionWorkflowState, WorkflowStatus

# Load environment variables
load_dotenv()

class DataProcessingDemo:
    """Demo class for showcasing the data processing agent"""
    
    def __init__(self):
        self.demo_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.s3_client = None
        self.setup_aws_clients()
        
    def setup_aws_clients(self):
        """Initialize AWS clients with environment credentials"""
        self.s3_client = boto3.client(
            's3',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        print("✅ AWS clients initialized")
    
    def generate_synthetic_stock_data(self, symbol: str = "DEMO", num_records: int = 1000) -> List[Dict]:
        """Generate realistic synthetic stock data for demo"""
        print(f"📊 Generating {num_records} synthetic stock records for {symbol}...")
        
        # Start with a base price and generate realistic OHLCV data
        base_price = 150.0
        records = []
        
        # Generate data for the last N trading days
        start_date = datetime.now() - timedelta(days=num_records + 100)
        
        for i in range(num_records):
            date = start_date + timedelta(days=i)
            
            # Skip weekends (rough approximation)
            if date.weekday() >= 5:
                continue
                
            if len(records) >= num_records:
                break
            
            # Generate realistic price movements
            daily_change = np.random.normal(0, 0.02)  # 2% daily volatility
            base_price *= (1 + daily_change)
            base_price = max(base_price, 1.0)  # Prevent negative prices
            
            # Generate OHLC with realistic relationships
            close = round(base_price, 2)
            open_price = round(close * (1 + np.random.normal(0, 0.005)), 2)
            
            # High and low should respect open/close
            high_mult = 1 + abs(np.random.normal(0, 0.01))
            low_mult = 1 - abs(np.random.normal(0, 0.01))
            
            high = round(max(open_price, close) * high_mult, 2)
            low = round(min(open_price, close) * low_mult, 2)
            
            # Generate realistic volume
            volume = int(np.random.lognormal(15, 0.5))  # Log-normal distribution for volume
            
            record = {
                "date": date.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume
            }
            records.append(record)
        
        print(f"✅ Generated {len(records)} realistic stock records")
        print(f"   📈 Price range: ${min(r['low'] for r in records):.2f} - ${max(r['high'] for r in records):.2f}")
        print(f"   📊 Volume range: {min(r['volume'] for r in records):,} - {max(r['volume'] for r in records):,}")
        
        return records
    
    def upload_synthetic_data_to_s3(self, records: List[Dict], symbol: str) -> str:
        """Upload synthetic data to S3 in NDJSON format"""
        print(f"📤 Uploading {len(records)} records to S3...")
        
        # Convert to NDJSON format (one JSON object per line)
        ndjson_content = "\n".join(json.dumps(record) for record in records)
        
        # Define S3 path
        bucket = "longhhoang-stock-data-raw"
        key = f"demo_data/daily/{symbol}/{self.demo_id}_synthetic_data.json"
        
        try:
            # Upload to S3
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=ndjson_content.encode('utf-8'),
                ContentType='application/json'
            )
            
            s3_path = f"s3://{bucket}/{key}"
            print(f"✅ Uploaded to: {s3_path}")
            print(f"   📁 Size: {len(ndjson_content.encode('utf-8')):,} bytes")
            
            return s3_path
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            raise
    
    async def run_intelligent_processing(self, input_s3_path: str, symbol: str) -> Dict[str, Any]:
        """Use LLM agent to intelligently process data via AWS Glue"""
        print(f"\n🤖 Starting LLM-powered data processing agent...")
        
        # Initialize tools for the agent
        initialize_tools(
            alpha_vantage_key=os.getenv('ALPHA_VANTAGE_API_KEY', 'demo'),
            s3_bucket=os.getenv('S3_BUCKET_RAW', 'longhhoang-stock-data-raw'),
            openai_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Create workflow state for the agent
        workflow_state = StockPredictionWorkflowState(
            workflow_id=self.demo_id,
            correlation_id=f"demo_correlation_{self.demo_id}",
            status=WorkflowStatus.RUNNING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            symbols=[symbol],
            timeframe="daily",
            prediction_horizon=30,
            ingestion={},  # Will be populated by agent
            preprocessing={},  # Will be populated by agent
            training={},  # Will be populated by agent
            workflow_config={
                "demo_mode": True,
                "input_path": input_s3_path,
                "output_path": f"s3://longhhoang-stock-data-processed/demo_output/daily/{symbol}/{self.demo_id}/",
                "processing_type": "feature_engineering"
            },
            global_errors=[],
            predictions={},
            confidence_scores={},
            model_metadata={}
        )
        
        print(f"🎯 LLM Agent Configuration:")
        print(f"   • Input: {input_s3_path}")
        print(f"   • Output: {workflow_state['workflow_config']['output_path']}")
        print(f"   • Symbol: {symbol}")
        print(f"   • Demo ID: {self.demo_id}")
        
        try:
            # Let the LLM agent process the data
            print(f"\n⚡ Executing LLM agent workflow...")
            result_state = preprocessing_node(workflow_state)
            
            return {
                "success": True,
                "workflow_state": result_state,
                "demo_id": self.demo_id
            }
            
        except Exception as e:
            print(f"❌ Agent processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "demo_id": self.demo_id
            }
    
    def validate_processed_output(self, output_path: str) -> Dict[str, Any]:
        """Validate the processed output from Glue job"""
        print(f"\n🔍 Validating processed output...")
        
        try:
            # Extract bucket and prefix from S3 path
            s3_path_parts = output_path.replace("s3://", "").split("/", 1)
            bucket = s3_path_parts[0]
            prefix = s3_path_parts[1] if len(s3_path_parts) > 1 else ""
            
            # List objects in the output path
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return {"success": False, "error": "No output files found"}
            
            # Get the first output file
            output_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.json')]
            
            if not output_files:
                return {"success": False, "error": "No JSON output files found"}
            
            # Download and analyze the first output file
            output_file = output_files[0]
            file_response = self.s3_client.get_object(Bucket=bucket, Key=output_file['Key'])
            content = file_response['Body'].read().decode('utf-8')
            
            # Parse NDJSON
            lines = content.strip().split('\n')
            records = [json.loads(line) for line in lines if line.strip()]
            
            if not records:
                return {"success": False, "error": "No records in output file"}
            
            # Analyze the processed data
            sample_record = records[0]
            
            # Count technical indicators
            original_fields = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            technical_indicators = [k for k in sample_record.keys() if k not in original_fields]
            
            validation_result = {
                "success": True,
                "total_records": len(records),
                "output_files": len(output_files),
                "technical_indicators": len(technical_indicators),
                "indicators_list": technical_indicators,
                "sample_record": sample_record,
                "output_size_bytes": sum(obj['Size'] for obj in output_files),
                "output_files_paths": [f"s3://{bucket}/{obj['Key']}" for obj in output_files]
            }
            
            print(f"✅ Validation successful!")
            print(f"   📊 Records processed: {validation_result['total_records']:,}")
            print(f"   🎯 Technical indicators: {validation_result['technical_indicators']}")
            print(f"   📁 Output size: {validation_result['output_size_bytes']:,} bytes")
            print(f"   🔧 Indicators: {', '.join(technical_indicators[:5])}{'...' if len(technical_indicators) > 5 else ''}")
            
            return validation_result
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_complete_demo(self):
        """Run the complete demo from start to finish"""
        print("🚀 STARTING DATA PROCESSING AGENT DEMO")
        print("=" * 60)
        
        demo_results = {
            "demo_id": self.demo_id,
            "timestamp": datetime.now().isoformat(),
            "steps": {},
            "success": False
        }
        
        try:
            # Step 1: Generate synthetic data
            print("\n📊 STEP 1: Generating Synthetic Data")
            print("-" * 40)
            records = self.generate_synthetic_stock_data("DEMO", 1000)
            demo_results["steps"]["data_generation"] = {
                "success": True,
                "records_generated": len(records)
            }
            
            # Step 2: Upload to S3
            print("\n📤 STEP 2: Uploading to S3")
            print("-" * 40)
            input_s3_path = self.upload_synthetic_data_to_s3(records, "DEMO")
            demo_results["steps"]["s3_upload"] = {
                "success": True,
                "s3_path": input_s3_path
            }
            
            # Step 3: LLM Agent Processing
            print("\n🤖 STEP 3: LLM Agent Processing")
            print("-" * 40)
            processing_result = await self.run_intelligent_processing(input_s3_path, "DEMO")
            demo_results["steps"]["llm_processing"] = processing_result
            
            if not processing_result["success"]:
                print(f"❌ Demo failed at LLM processing step")
                return demo_results
            
            # Step 4: Validate output
            print("\n🔍 STEP 4: Validating Results")
            print("-" * 40)
            output_path = processing_result["workflow_state"]["workflow_config"]["output_path"]
            validation_result = self.validate_processed_output(output_path)
            demo_results["steps"]["validation"] = validation_result
            
            if validation_result["success"]:
                demo_results["success"] = True
                print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
            else:
                print("\n❌ Demo validation failed")
                
        except Exception as e:
            print(f"\n💥 Demo failed with error: {e}")
            demo_results["error"] = str(e)
        
        # Final summary
        print("\n" + "=" * 60)
        print("📋 DEMO SUMMARY")
        print("=" * 60)
        
        if demo_results["success"]:
            validation = demo_results["steps"]["validation"]
            print("✅ Status: SUCCESS")
            print(f"🆔 Demo ID: {self.demo_id}")
            print(f"📊 Records processed: {validation['total_records']:,}")
            print(f"🎯 Technical indicators: {validation['technical_indicators']}")
            print(f"📁 Output size: {validation['output_size_bytes']:,} bytes")
            print(f"🕒 Processing time: Complete")
            print("\n🚀 The Data Processing Agent is PRODUCTION READY!")
        else:
            print("❌ Status: FAILED")
            print(f"💥 Error: {demo_results.get('error', 'Unknown error')}")
        
        return demo_results

async def main():
    """Main demo execution function"""
    demo = DataProcessingDemo()
    results = await demo.run_complete_demo()
    return results

if __name__ == "__main__":
    print("🎯 Data Processing Agent Demo")
    print("Demonstrating LLM-powered multi-agent stock data processing")
    print()
    
    # Run the complete demo
    results = asyncio.run(main())
    
    # Print final status
    if results["success"]:
        print("\n🎊 Demo completed successfully! The system is ready for production.")
    else:
        print("\n⚠️ Demo encountered issues. Check the output above for details.")
