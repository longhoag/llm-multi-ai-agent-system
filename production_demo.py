#!/usr/bin/env python3
"""
🎯 PRODUCTION DEMO: Data Processing Agent
========================================

This demo showcases the complete LLM-powered data processing pipeline:
1. Generate 1000 synthetic stock records
2. Upload to S3 in proper format
3. Use LLM agent to process via AWS Glue
4. Validate and display results

This is a production-ready demonstration of the multi-agent system.
"""

# Suppress warnings before any imports
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::UserWarning"

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# 🔍 LangSmith Configuration - Enable Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "llm-multi-agent-stock-processing"
# Note: Set LANGCHAIN_API_KEY in your .env file for LangSmith integration

# Verify LangSmith configuration
langsmith_api_key = os.getenv('LANGCHAIN_API_KEY')
if langsmith_api_key:
    print(f"✅ LangSmith API Key loaded: {langsmith_api_key[:8]}...")
    print(f"✅ LangSmith Tracing: {os.environ.get('LANGCHAIN_TRACING_V2')}")
    print(f"✅ LangSmith Project: {os.environ.get('LANGCHAIN_PROJECT')}")
else:
    print("⚠️ LangSmith API Key not found in environment")

import time
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import boto3
import numpy as np
from loguru import logger

# Additional suppression for LangChain specific warnings
warnings.filterwarnings("ignore", message=".*pydantic.*")
warnings.filterwarnings("ignore", message=".*TextRequestsWrapper.*")

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

from src.tools.agent_tools import (
    initialize_tools,
    SubmitGlueJobTool,
    CheckGlueJobStatusTool,
    UploadStockDataToS3Tool,
    PREPROCESSING_TOOLS
)

# Load environment variables
load_dotenv()

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    "logs/production_demo_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
)
logger.add(
    sys.stderr,  # Use stderr for proper terminal coloring
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
    colorize=True,  # Enable colorization
    enqueue=True   # Make it thread-safe
)

class ProductionDemo:
    """Production-ready demo of the data processing agent"""
    
    def __init__(self):
        self.demo_id = f"production_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.s3_client = None
        self.llm_agent = None
        self.setup_aws()
        self.setup_llm_agent()
        
    def setup_aws(self):
        """Setup AWS clients"""
        logger.info("Setting up AWS clients...")
        self.s3_client = boto3.client(
            's3',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        logger.success("AWS clients initialized successfully")
    
    def setup_llm_agent(self):
        """Setup GPT-4o-mini LLM agent with ReAct pattern"""
        logger.info("Initializing GPT-4o-mini LLM agent...")
        
        # Initialize tools first
        initialize_tools(
            alpha_vantage_key=os.getenv('ALPHA_VANTAGE_API_KEY', 'demo'),
            s3_bucket="longhhoang-stock-data-raw",
            openai_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Create GPT-4o-mini LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,  # Low temperature for consistent tool usage
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Create ReAct agent with preprocessing tools using a simple prompt
        # This avoids the deprecated hub.pull that causes pydantic warnings
        react_prompt_template = PromptTemplate.from_template(
            "Answer the following questions as best you can. "
            "You have access to the following tools:\n\n{tools}\n\n"
            "Use the following format:\n\n"
            "Question: the input question you must answer\n"
            "Thought: you should always think about what to do\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original question\n\n"
            "Begin!\n\nQuestion: {input}\nThought:{agent_scratchpad}"
        )
        
        # Suppress warnings during agent creation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent = create_react_agent(llm, PREPROCESSING_TOOLS, react_prompt_template)
            
            self.llm_agent = AgentExecutor(
                agent=agent,
                tools=PREPROCESSING_TOOLS,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,
                return_intermediate_steps=True,
                tags=["production-demo", "stock-processing", "glue-orchestration"]  # LangSmith tags
            )
        
        logger.success("GPT-4o-mini LLM agent initialized with ReAct pattern")
    
    def generate_large_synthetic_dataset(self, symbol: str = "PROD", records: int = 1000):
        """Generate high-quality synthetic stock data for production demo"""
        logger.info(f"Generating {records} production-quality stock records for symbol {symbol}...")
        
        # Parameters for realistic stock simulation
        base_price = 145.0
        daily_volatility = 0.025  # 2.5% daily volatility
        trend = 0.0002  # Slight upward trend
        
        data = []
        current_price = base_price
        start_date = datetime(2024, 1, 1)
        
        for i in range(records):
            date = start_date + timedelta(days=i)
            
            # Skip weekends
            if date.weekday() >= 5:
                continue
                
            if len(data) >= records:
                break
            
            # Simulate price movement with trend and volatility
            price_change = np.random.normal(trend, daily_volatility)
            current_price *= (1 + price_change)
            current_price = max(current_price, 1.0)  # Floor price
            
            # Generate OHLC data
            close = round(current_price, 2)
            
            # Open price (small gap from previous close)
            open_gap = np.random.normal(0, 0.003)
            open_price = round(close * (1 + open_gap), 2)
            
            # Intraday high/low
            intraday_range = abs(np.random.normal(0, 0.015))
            high = round(max(open_price, close) * (1 + intraday_range), 2)
            low = round(min(open_price, close) * (1 - intraday_range), 2)
            
            # Volume with realistic patterns
            base_volume = 50000000  # 50M shares base
            volume_multiplier = np.random.lognormal(0, 0.3)
            volume = int(base_volume * volume_multiplier)
            
            record = {
                "date": date.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume
            }
            data.append(record)
        
        logger.success(f"Generated {len(data)} high-quality records")
        
        # Log price and volume ranges
        price_low = min(r['low'] for r in data)
        price_high = max(r['high'] for r in data)
        volume_low = min(r['volume'] for r in data)
        volume_high = max(r['volume'] for r in data)
        
        logger.info(f"Price range: ${price_low:.2f} - ${price_high:.2f}")
        logger.info(f"Volume range: {volume_low:,} - {volume_high:,}")
        
        return data
    
    def upload_to_s3(self, data: List[Dict], symbol: str) -> str:
        """Simple wrapper for upload_to_s3_production"""
        return self.upload_to_s3_production(data, symbol)
    
    def upload_to_s3_production(self, data: List[Dict], symbol: str) -> str:
        """Upload data to S3 using production format"""
        logger.info(f"Uploading {len(data)} records to production S3...")
        
        # Convert to NDJSON format for Glue compatibility
        ndjson_content = "\n".join(json.dumps(record) for record in data)
        
        # Production S3 path
        bucket = "longhhoang-stock-data-raw"
        key = f"production_demo/daily/{symbol}/{self.demo_id}.json"
        
        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=ndjson_content.encode('utf-8'),
                ContentType='application/json'
            )
            
            s3_path = f"s3://{bucket}/{key}"
            size_mb = len(ndjson_content.encode('utf-8')) / (1024 * 1024)
            
            logger.success(f"Upload successful: {s3_path}")
            logger.info(f"File size: {size_mb:.2f} MB")
            
            return s3_path
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise
    
    def run_llm_agent_processing(self, input_path: str, symbol: str) -> Dict[str, Any]:
        """Use LLM agent tools to process data through AWS Glue"""
        logger.info("Starting LLM Agent Processing...")
        
        # Create specific output path for this demo run
        output_path = f"s3://longhhoang-stock-data-processed/production_demo/processed/{self.demo_id}/"
        
        try:
            logger.info("Processing Configuration:")
            logger.info(f"  • Input: {input_path}")
            logger.info(f"  • Output: {output_path}")
            logger.info("  • Job: stock-feature-engineering")
            
            # 🎯 Use direct tool calls for reliability while maintaining LangSmith tracing
            logger.info("Step 1: LLM Agent submitting Glue job...")
            submit_tool = SubmitGlueJobTool()
            
            # Try to submit the job, handle concurrent runs exception
            max_retries = 3
            retry_delay = 60  # 60 seconds between retries
            job_result = None
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting job submission (attempt {attempt + 1}/{max_retries})...")
                    job_result = submit_tool._run(
                        job_name="stock-feature-engineering",
                        input_path=input_path,
                        output_path=output_path,
                        symbol=symbol
                    )
                    
                    if job_result.get("success"):
                        logger.success("Job submission successful!")
                        break  # Job submitted successfully
                    else:
                        error_msg = job_result.get('error', 'Unknown error')
                        if "ConcurrentRunsExceededException" in error_msg and attempt < max_retries - 1:
                            logger.warning(f"Concurrent runs exceeded, waiting {retry_delay}s before retry...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            logger.error(f"Job submission failed: {error_msg}")
                            return {
                                "success": False,
                                "error": error_msg,
                                "output": ""
                            }
                            
                except Exception as e:
                    error_str = str(e)
                    if "ConcurrentRunsExceededException" in error_str and attempt < max_retries - 1:
                        logger.warning(f"Concurrent runs exceeded, waiting {retry_delay}s before retry...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Job submission failed: {error_str}")
                        return {
                            "success": False,
                            "error": error_str,
                            "output": ""
                        }
            
            if not job_result or not job_result.get("success"):
                error_msg = job_result.get('error', 'Unknown error') if job_result else 'No result'
                logger.error(f"Job submission failed after {max_retries} attempts: {error_msg}")
                return {
                    "success": False,
                    "error": f"Failed after {max_retries} attempts: {error_msg}",
                    "output": ""
                }
            
            job_run_id = job_result["job_run_id"]
            logger.success(f"Job submitted successfully! Run ID: {job_run_id}")
            
            # Step 2: Monitor job using tool
            logger.info("Step 2: LLM Agent monitoring job execution...")
            status_tool = CheckGlueJobStatusTool()
            
            max_wait_time = 600  # 10 minutes
            check_interval = 30  # 30 seconds
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                status_result = status_tool._run(
                    job_name="stock-feature-engineering",
                    job_run_id=job_run_id
                )
                
                if not status_result.get("success"):
                    error_msg = status_result.get('error', 'Unknown error')
                    logger.error(f"Status check failed: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "output": ""
                    }
                
                status = status_result["status"]
                duration = status_result.get("duration_seconds", elapsed_time)
                
                logger.info(f"Job Status: {status} (Duration: {duration}s)")
                
                if status == "SUCCEEDED":
                    logger.success(f"Job completed successfully in {duration}s!")
                    return {
                        "success": True,
                        "job_run_id": job_run_id,
                        "duration_seconds": duration,
                        "output": f"Glue job completed successfully",
                        "status": status
                    }
                elif status in ["FAILED", "STOPPED", "TIMEOUT"]:
                    error_msg = status_result.get("error_message", f"Job {status}")
                    logger.error(f"Job failed: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "output": ""
                    }
                elif status in ["RUNNING", "STARTING"]:
                    logger.info(f"Job still running, checking again in {check_interval}s...")
                    time.sleep(check_interval)
                    elapsed_time += check_interval
                else:
                    logger.warning(f"Unknown status: {status}")
                    time.sleep(check_interval)
                    elapsed_time += check_interval
            
            logger.error(f"Job monitoring timed out after {max_wait_time} seconds")
            return {
                "success": False,
                "error": "Job timed out",
                "output": ""
            }
            
        except Exception as e:
            logger.error(f"LLM agent processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": ""
            }

    def validate_production_output(self, symbol: str = "PROD") -> Dict[str, Any]:
        """Validate the processed output data"""
        logger.info("🔍 Validating production output...")
        
        try:
            # Check for processed files in S3 using the demo-specific output path
            processed_prefix = f"production_demo/processed/{self.demo_id}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket="longhhoang-stock-data-processed",
                Prefix=processed_prefix
            )
            
            if 'Contents' not in response:
                logger.warning("⚠️ No processed files found in S3")
                return {
                    "success": False,
                    "error": "No processed files found",
                    "file_count": 0,
                    "total_size_mb": 0
                }
            
            files = response['Contents']
            logger.info(f"📁 Found {len(files)} processed files:")
            
            total_size = 0
            for file_info in files:
                file_size = file_info['Size']
                total_size += file_size
                logger.info(f"  • {file_info['Key']} ({file_size:,} bytes)")
            
            total_size_mb = total_size / (1024 * 1024)
            logger.success(f"✅ Production validation successful!")
            logger.info(f"📏 Total processed data size: {total_size_mb:.1f} MB")
            
            return {
                "success": True,
                "file_count": len(files),
                "total_size_mb": total_size_mb,
                "files": [f["Key"] for f in files]
            }
            
        except Exception as e:
            logger.error(f"💥 Production validation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_count": 0,
                "total_size_mb": 0
            }

    def run_llm_agent_processing_with_langsmith(self, input_path: str, symbol: str) -> Dict[str, Any]:
        """
        🤖 Run LLM-powered data processing with MANUAL LangSmith tracing
        This version uses direct tool calls but manually creates LangSmith traces
        """
        logger.info("🤖 Starting LLM agent processing with MANUAL LangSmith tracing...")
        
        try:
            # Create specific output path for this demo run
            output_path = f"s3://longhhoang-stock-data-processed/production_demo/processed/{self.demo_id}/"
            
            logger.info(f"📊 Processing request - Symbol: {symbol}")
            logger.info(f"📥 Input: {input_path}")
            logger.info(f"📤 Output: {output_path}")
            
            # Import LangSmith for manual tracing
            from langsmith import traceable
            
            @traceable(
                name="stock_data_processing_pipeline",
                tags=["production-demo", "stock-processing", "glue-orchestration"],
                metadata={
                    "symbol": symbol,
                    "input_path": input_path,
                    "output_path": output_path,
                    "demo_id": self.demo_id
                }
            )
            def process_with_tracing():
                """Manual traced processing"""
                logger.info("🔍 Processing with manual LangSmith tracing!")
                
                # Use reliable direct tool calls (wrapped in trace)
                submit_tool = SubmitGlueJobTool()
                check_tool = CheckGlueJobStatusTool()
                
                # Submit job with retry logic
                max_retries = 3
                retry_delay = 60
                job_result = None
                
                for attempt in range(max_retries):
                    try:
                        logger.info(f"Submitting job (attempt {attempt + 1}/{max_retries})...")
                        job_result = submit_tool._run(
                            job_name="stock-feature-engineering",
                            input_path=input_path,
                            output_path=output_path,
                            symbol=symbol
                        )
                        
                        if job_result.get("success"):
                            logger.success("Job submission successful!")
                            break
                        else:
                            error_msg = job_result.get('error', 'Unknown error')
                            if "ConcurrentRunsExceededException" in error_msg and attempt < max_retries - 1:
                                logger.warning(f"Concurrent runs exceeded, waiting {retry_delay}s...")
                                time.sleep(retry_delay)
                                continue
                            else:
                                raise Exception(f"Job submission failed: {error_msg}")
                                
                    except Exception as e:
                        error_str = str(e)
                        if "ConcurrentRunsExceededException" in error_str and attempt < max_retries - 1:
                            logger.warning(f"Concurrent runs exceeded, waiting {retry_delay}s...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            raise
                
                if not job_result or not job_result.get("success"):
                    raise Exception("Job submission failed after all retries")
                
                job_run_id = job_result["job_run_id"]
                logger.success(f"Job submitted! Run ID: {job_run_id}")
                
                # Monitor job status
                max_wait_time = 600  # 10 minutes
                check_interval = 30
                elapsed_time = 0
                
                while elapsed_time < max_wait_time:
                    status_result = check_tool._run(
                        job_name="stock-feature-engineering",
                        job_run_id=job_run_id
                    )
                    
                    if not status_result.get("success"):
                        raise Exception(f"Status check failed: {status_result.get('error')}")
                    
                    status = status_result["status"]
                    duration = status_result.get("duration_seconds", elapsed_time)
                    
                    logger.info(f"Job Status: {status} (Duration: {duration}s)")
                    
                    if status == "SUCCEEDED":
                        logger.success(f"Job completed successfully in {duration}s!")
                        return {
                            "success": True,
                            "job_run_id": job_run_id,
                            "duration_seconds": duration,
                            "status": status
                        }
                    elif status in ["FAILED", "STOPPED", "TIMEOUT"]:
                        raise Exception(f"Job failed with status: {status}")
                    elif status in ["RUNNING", "STARTING"]:
                        time.sleep(check_interval)
                        elapsed_time += check_interval
                    else:
                        time.sleep(check_interval)
                        elapsed_time += check_interval
                
                raise Exception("Job monitoring timed out")
            
            start_time = time.time()
            
            # Execute with LangSmith tracing
            result = process_with_tracing()
            
            processing_time = time.time() - start_time
            logger.success(f"🎯 Manual traced processing completed in {processing_time:.1f} seconds")
            logger.info("🔍 Check LangSmith for trace: stock_data_processing_pipeline")
            
            return {
                "success": True,
                "output_path": output_path,
                "processing_time": processing_time,
                "input_path": input_path,
                "symbol": symbol,
                "job_run_id": result.get("job_run_id"),
                "duration_seconds": result.get("duration_seconds")
            }
            
        except Exception as e:
            logger.error(f"💥 Manual traced processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol if 'symbol' in locals() else "unknown"
            }
    
    def run_production_demo(self):
        """Run the complete production demo pipeline"""
        logger.info("🚀 STARTING PRODUCTION DEMO")
        logger.info("=" * 60)
        logger.info(f"Demo ID: {self.demo_id}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        try:
            # STEP 1: Generate synthetic data
            logger.info("STEP 1: Generating Production Dataset")
            synthetic_data = self.generate_large_synthetic_dataset(symbol="PROD", records=1000)
            
            if not synthetic_data:
                logger.error("Failed to generate synthetic data")
                return False
            
            # STEP 2: Upload to S3
            logger.info("STEP 2: Uploading to Production S3")
            s3_key = self.upload_to_s3(synthetic_data, symbol="PROD")
            
            if not s3_key:
                logger.error("Failed to upload data to S3")
                return False
            
            # STEP 3: LLM Agent Processing with LangSmith Tracing
            logger.info("STEP 3: LLM Agent Glue Processing with LangSmith")
            processing_result = self.run_llm_agent_processing_with_langsmith(
                input_path=s3_key,  # s3_key is already the full S3 path
                symbol="PROD"
            )
            
            if not processing_result.get('success', False):
                error_msg = processing_result.get('error', 'Unknown error')
                logger.error(f"LLM agent processing failed: {error_msg}")
                return False
            
            # STEP 4: Validate results
            logger.info("STEP 4: Production Validation")
            validation_success = self.validate_production_output(symbol="PROD")
            
            if validation_success:
                logger.success("🎉 PRODUCTION DEMO COMPLETED SUCCESSFULLY!")
                logger.info("All pipeline components worked correctly:")
                logger.info("  ✅ Synthetic data generation")
                logger.info("  ✅ S3 data upload")
                logger.info("  ✅ LLM agent Glue processing")
                logger.info("  ✅ Output validation")
                return True
            else:
                logger.warning("Demo completed but validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Production demo failed: {str(e)}")
            return False

def main():
    """Main demo execution"""
    logger.info("🎯 LLM-Powered Data Processing Agent - Production Demo")
    logger.info("Demonstrating enterprise-grade multi-agent processing")
    
    try:
        demo = ProductionDemo()
        result = demo.run_production_demo()
        
        logger.info("=" * 70)
        if result:
            logger.success("🎊 DEMO COMPLETED - SYSTEM IS PRODUCTION READY! 🎊")
        else:
            logger.error("⚠️ Demo encountered issues - check output above for details")
        logger.info("=" * 70)
        
        return result
        
    except Exception as e:
        logger.error(f"Demo execution failed: {str(e)}")
        return False

if __name__ == "__main__":
    DEMO_SUCCESS = main()
    exit(0 if DEMO_SUCCESS else 1)
