"""LangGraph tools for stock data pipeline agents"""

import os
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

from langchain.tools import BaseTool

from src.external.alpha_vantage_client import AlphaVantageClient
from src.storage.s3_manager import S3Manager

# Global tool instances (initialized via initialize_tools)
_alpha_vantage_client = None
_s3_manager = None
_s3_manager_processed = None


class FetchStockDataInput(BaseModel):
    """Input for fetch_stock_data_tool"""
    symbol: str = Field(description="Stock symbol (e.g., AAPL, GOOGL)")
    outputsize: str = Field(default="compact", description="Data size: 'compact' or 'full'")


class FetchStockDataTool(BaseTool):
    """Tool to fetch stock data from Alpha Vantage API"""
    name: str = "fetch_stock_data"
    description: str = "Fetch daily stock data from Alpha Vantage API for a given symbol"
    args_schema: type = FetchStockDataInput

    def _run(self, symbol: str, outputsize: str = "compact") -> Dict[str, Any]:
        """Synchronous execution with robust async handling"""
        import nest_asyncio
        nest_asyncio.apply()
        
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                # Create new event loop if closed
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function
            if loop.is_running():
                # If loop is running, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(symbol, outputsize))
                    return future.result()
            else:
                return loop.run_until_complete(self._arun(symbol, outputsize))
                
        except Exception as e:
            return {"success": False, "error": f"Execution error: {str(e)}"}

    async def _arun(self, symbol: str, outputsize: str = "compact") -> Dict[str, Any]:
        """Fetch stock data asynchronously"""
        if not _alpha_vantage_client:
            return {"error": "Alpha Vantage client not initialized"}
        
        try:
            data = await _alpha_vantage_client.get_daily_time_series(symbol, outputsize=outputsize)
            
            if data and 'data' in data:
                return {
                    "success": True,
                    "symbol": symbol,
                    "record_count": len(data["data"]),
                    "data": data["data"],
                    "metadata": data.get("metadata", {}),
                    "fetch_timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False, 
                    "error": "No data received - possible rate limit or API issue", 
                    "symbol": symbol,
                    "message": "Alpha Vantage free tier allows 25 requests per day. Consider upgrading if limit exceeded."
                }
                
        except Exception as e:
            return {"success": False, "error": str(e), "symbol": symbol}


# ==============================================================================
# PREPROCESSING TOOLS FOR AWS GLUE INTEGRATION
# ==============================================================================

class DownloadFromS3Input(BaseModel):
    """Input for download_from_s3_tool"""
    key: str = Field(description="S3 key to download from")
    bucket: str = Field(default="", description="S3 bucket name (optional, uses default if empty)")


class DownloadFromS3Tool(BaseTool):
    """Tool to download data from S3"""
    name: str = "download_from_s3"
    description: str = "Download JSON data from S3 for preprocessing"
    args_schema: type = DownloadFromS3Input

    def _run(self, key: str, bucket: str = "") -> Dict[str, Any]:
        """Synchronous execution with robust async handling"""
        import nest_asyncio
        nest_asyncio.apply()
        
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                # Create new event loop if closed
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function
            if loop.is_running():
                # If loop is running, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(key, bucket))
                    return future.result()
            else:
                return loop.run_until_complete(self._arun(key, bucket))
                
        except Exception as e:
            return {"success": False, "error": f"Execution error: {str(e)}"}

    async def _arun(self, key: str, bucket: str = "") -> Dict[str, Any]:
        """Download data from S3 asynchronously"""
        if not _s3_manager:
            return {"success": False, "error": "S3 manager not initialized"}
        
        try:
            data = await _s3_manager.download_json(key)
            
            if data:
                return {
                    "success": True,
                    "key": key,
                    "data": data,
                    "download_timestamp": datetime.now().isoformat()
                }
            else:
                return {"success": False, "error": f"No data found at key: {key}"}
                
        except Exception as e:
            return {"success": False, "error": str(e), "key": key}


class SubmitGlueJobInput(BaseModel):
    """Input for submit_glue_job_tool"""
    job_name: str = Field(description="Name of the Glue job to submit")
    input_path: str = Field(description="S3 path to input data")
    output_path: str = Field(description="S3 path for output data")
    symbol: str = Field(description="Stock symbol being processed")


class SubmitGlueJobTool(BaseTool):
    """Tool to submit AWS Glue job for data preprocessing"""
    name: str = "submit_glue_job"
    description: str = "Submit AWS Glue job to process stock data with feature engineering"
    args_schema: type = SubmitGlueJobInput

    def _run(self, job_name: str, input_path: str, output_path: str, symbol: str) -> Dict[str, Any]:
        """Synchronous execution"""
        try:
            import boto3
            import os
            
            # Initialize Glue client
            glue = boto3.client(
                'glue',
                region_name=os.getenv('AWS_REGION', 'us-east-1'),
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            
            # Job arguments for Glue job - must match PySpark script expectations
            job_args = {
                '--INPUT_PATH': input_path,
                '--OUTPUT_PATH': output_path,
                '--SYMBOL': symbol,
                '--job-bookmark-option': 'job-bookmark-disable'
            }
            
            # Submit Glue job
            response = glue.start_job_run(
                JobName=job_name,
                Arguments=job_args
            )
            
            job_run_id = response['JobRunId']
            
            return {
                "success": True,
                "job_name": job_name,
                "job_run_id": job_run_id,
                "input_path": input_path,
                "output_path": output_path,
                "symbol": symbol,
                "submit_timestamp": datetime.now().isoformat(),
                "status": "RUNNING"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "job_name": job_name,
                "symbol": symbol
            }

    async def _arun(self, job_name: str, input_path: str, output_path: str, symbol: str) -> Dict[str, Any]:
        """Async version"""
        return self._run(job_name, input_path, output_path, symbol)


class CheckGlueJobStatusInput(BaseModel):
    """Input for check_glue_job_status_tool"""
    job_name: str = Field(description="Name of the Glue job")
    job_run_id: str = Field(description="Job run ID to check status")


class CheckGlueJobStatusTool(BaseTool):
    """Tool to check AWS Glue job status"""
    name: str = "check_glue_job_status"
    description: str = "Check the status of a running AWS Glue job"
    args_schema: type = CheckGlueJobStatusInput

    def _run(self, job_name: str, job_run_id: str) -> Dict[str, Any]:
        """Check Glue job status"""
        try:
            import boto3
            import os
            
            # Initialize Glue client
            glue = boto3.client(
                'glue',
                region_name=os.getenv('AWS_REGION', 'us-east-1'),
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            
            # Get job run details
            response = glue.get_job_run(
                JobName=job_name,
                RunId=job_run_id
            )
            
            job_run = response['JobRun']
            
            return {
                "success": True,
                "job_name": job_name,
                "job_run_id": job_run_id,
                "status": job_run['JobRunState'],
                "started_on": job_run.get('StartedOn', '').isoformat() if job_run.get('StartedOn') else None,
                "completed_on": job_run.get('CompletedOn', '').isoformat() if job_run.get('CompletedOn') else None,
                "execution_time": job_run.get('ExecutionTime', 0),
                "error_message": job_run.get('ErrorMessage', ''),
                "check_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "job_name": job_name,
                "job_run_id": job_run_id
            }

    async def _arun(self, job_name: str, job_run_id: str) -> Dict[str, Any]:
        """Async version"""
        return self._run(job_name, job_run_id)


class UploadProcessedDataInput(BaseModel):
    """Input for upload_processed_data_tool"""
    data: Dict[str, Any] = Field(description="Processed data to upload")
    symbol: str = Field(description="Stock symbol")
    timeframe: str = Field(default="daily", description="Data timeframe")


class UploadProcessedDataTool(BaseTool):
    """Tool to upload processed data to S3"""
    name: str = "upload_processed_data"
    description: str = "Upload processed stock data with features to S3"
    args_schema: type = UploadProcessedDataInput

    def _run(self, data: Dict[str, Any], symbol: str, timeframe: str = "daily") -> Dict[str, Any]:
        """Synchronous execution with robust async handling"""
        import nest_asyncio
        nest_asyncio.apply()
        
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                # Create new event loop if closed
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function
            if loop.is_running():
                # If loop is running, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(data, symbol, timeframe))
                    return future.result()
            else:
                return loop.run_until_complete(self._arun(data, symbol, timeframe))
                
        except Exception as e:
            return {"success": False, "error": f"Execution error: {str(e)}"}

    async def _arun(self, data: Dict[str, Any], symbol: str, timeframe: str = "daily") -> Dict[str, Any]:
        """Upload processed data to S3 asynchronously"""
        if not _s3_manager_processed:
            return {"success": False, "error": "Processed data S3 manager not initialized"}
        
        try:
            # Generate keys for processed data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_key = f"processed_data/{timeframe}/{symbol}/{timestamp}.json"
            latest_key = f"processed_data/{timeframe}/{symbol}/latest.json"
            
            # Upload both versions
            timestamped_success = await _s3_manager_processed.upload_json(data, timestamped_key)
            latest_success = await _s3_manager_processed.upload_json(data, latest_key)
            
            if timestamped_success and latest_success:
                return {
                    "success": True,
                    "symbol": symbol,
                    "timestamped_key": timestamped_key,
                    "latest_key": latest_key,
                    "upload_timestamp": datetime.now().isoformat(),
                    "message": f"Successfully uploaded processed {symbol} data to S3"
                }
            else:
                return {
                    "success": False,
                    "error": f"Upload failed: timestamped={timestamped_success}, latest={latest_success}",
                    "symbol": symbol
                }
                
        except Exception as e:
            return {"success": False, "error": str(e), "symbol": symbol}


class ValidateDataQualityInput(BaseModel):
    """Input for validate_data_quality_tool"""
    data: Dict[str, Any] = Field(description="Stock data to validate")
    symbol: str = Field(description="Stock symbol being validated")


class ValidateDataQualityTool(BaseTool):
    """Tool to validate stock data quality"""
    name: str = "validate_data_quality"
    description: str = "Validate the quality and completeness of stock data"
    args_schema: type = ValidateDataQualityInput

    def _run(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Validate data quality"""
        try:
            # Basic validation checks
            validation_results = {
                "symbol": symbol,
                "validation_timestamp": datetime.now().isoformat(),
                "checks": {}
            }
            
            # Check if data exists
            if not data or not data.get("success"):
                validation_results["checks"]["data_exists"] = False
                validation_results["overall_quality"] = "POOR"
                validation_results["issues"] = ["No valid data found"]
                return validation_results
            
            # Check record count
            record_count = data.get("record_count", 0)
            validation_results["checks"]["sufficient_records"] = record_count >= 10
            validation_results["record_count"] = record_count
            
            # Check data structure
            has_data = bool(data.get("data"))
            validation_results["checks"]["data_structure"] = has_data
            
            # Check for required fields in sample data
            if has_data and isinstance(data["data"], list) and len(data["data"]) > 0:
                sample_record = data["data"][0]
                required_fields = ["open", "high", "low", "close", "volume"]
                has_required_fields = all(field in sample_record for field in required_fields)
                validation_results["checks"]["required_fields"] = has_required_fields
            else:
                validation_results["checks"]["required_fields"] = False
            
            # Overall quality assessment
            passed_checks = sum(1 for check in validation_results["checks"].values() if check)
            total_checks = len(validation_results["checks"])
            
            if passed_checks == total_checks:
                validation_results["overall_quality"] = "EXCELLENT"
            elif passed_checks >= total_checks * 0.75:
                validation_results["overall_quality"] = "GOOD"
            elif passed_checks >= total_checks * 0.5:
                validation_results["overall_quality"] = "FAIR"
            else:
                validation_results["overall_quality"] = "POOR"
            
            validation_results["validation_passed"] = validation_results["overall_quality"] in ["EXCELLENT", "GOOD"]
            
            return validation_results
            
        except Exception as e:
            return {
                "symbol": symbol,
                "validation_passed": False,
                "error": str(e),
                "overall_quality": "ERROR",
                "validation_timestamp": datetime.now().isoformat()
            }

    async def _arun(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Async version"""
        return self._run(data, symbol)


class UploadToS3Input(BaseModel):
    """Input for upload_stock_data_to_s3 tool"""
    data: Dict[str, Any] = Field(description="Stock data to upload")
    symbol: str = Field(description="Stock symbol")
    timeframe: str = Field(default="daily", description="Data timeframe (daily, weekly, etc.)")


class UploadStockDataToS3Tool(BaseTool):
    """Tool to upload stock data to S3 with both timestamped and latest versions"""
    name: str = "upload_stock_data_to_s3"
    description: str = "Upload stock data to S3 with both timestamped and latest.json versions"
    args_schema: type = UploadToS3Input

    def _run(self, data: Dict[str, Any], symbol: str, timeframe: str = "daily") -> Dict[str, Any]:
        """Synchronous execution with robust async handling"""
        import nest_asyncio
        nest_asyncio.apply()
        
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                # Create new event loop if closed
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function
            if loop.is_running():
                # If loop is running, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(data, symbol, timeframe))
                    return future.result()
            else:
                return loop.run_until_complete(self._arun(data, symbol, timeframe))
                
        except Exception as e:
            return {"success": False, "error": f"Execution error: {str(e)}"}

    async def _arun(self, data: Dict[str, Any], symbol: str, timeframe: str = "daily") -> Dict[str, Any]:
        """Upload data to S3 with dual file creation"""
        if not _s3_manager:
            return {"error": "S3 manager not initialized"}
        
        try:
            # Generate keys
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_key = f"raw_data/{timeframe}/{symbol}/{timestamp}.json"
            latest_key = f"raw_data/{timeframe}/{symbol}/latest.json"
            
            # Upload both versions
            timestamped_success = await _s3_manager.upload_json(data, timestamped_key)
            latest_success = await _s3_manager.upload_json(data, latest_key)
            
            if timestamped_success and latest_success:
                return {
                    "success": True,
                    "symbol": symbol,
                    "timestamped_key": timestamped_key,
                    "latest_key": latest_key,
                    "upload_timestamp": datetime.now().isoformat(),
                    "message": f"Successfully uploaded {symbol} data to S3 with both timestamped and latest versions"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to upload: timestamped={timestamped_success}, latest={latest_success}",
                    "symbol": symbol
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }


# Tool instances
fetch_stock_data_tool = FetchStockDataTool()
validate_data_quality_tool = ValidateDataQualityTool()
upload_stock_data_to_s3_tool = UploadStockDataToS3Tool()

class CreateFeaturesInput(BaseModel):
    """Input for create_features_tool"""
    data: Dict[str, Any] = Field(description="Raw stock data to process")
    symbol: str = Field(description="Stock symbol")


class CreateFeaturesTool(BaseTool):
    """Tool to create features locally (alternative to Glue job)"""
    name: str = "create_features"
    description: str = "Create technical indicators and features from raw stock data locally"
    args_schema: type = CreateFeaturesInput

    def _run(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Create features from raw stock data"""
        try:
            raw_records = data.get("data", [])
            if not raw_records:
                return {"success": False, "error": "No data records found"}
            
            # Sort by date to ensure proper order
            sorted_records = sorted(raw_records, key=lambda x: x["date"])
            
            processed_records = []
            for i, record in enumerate(sorted_records):
                processed_record = record.copy()
                
                # Calculate simple technical indicators
                if i >= 5:  # Need at least 5 days for moving averages
                    # 5-day moving average
                    recent_closes = [sorted_records[j]["close"] for j in range(i-4, i+1)]
                    processed_record["sma_5"] = round(sum(recent_closes) / 5, 2)
                    
                    # Price return (1-day)
                    if i > 0:
                        prev_close = sorted_records[i-1]["close"]
                        processed_record["return_1d"] = round((record["close"] - prev_close) / prev_close, 4)
                    
                    # Volatility (5-day standard deviation of returns)
                    if i >= 5:
                        recent_returns = []
                        for j in range(i-4, i+1):
                            if j > 0:
                                curr_close = sorted_records[j]["close"]
                                prev_close = sorted_records[j-1]["close"]
                                ret = (curr_close - prev_close) / prev_close
                                recent_returns.append(ret)
                        
                        if recent_returns:
                            mean_return = sum(recent_returns) / len(recent_returns)
                            variance = sum((r - mean_return) ** 2 for r in recent_returns) / len(recent_returns)
                            processed_record["volatility_5d"] = round(variance ** 0.5, 4)
                
                # Volume features
                processed_record["volume_normalized"] = record["volume"] / 1000000  # In millions
                
                processed_records.append(processed_record)
            
            # Create feature metadata
            features_created = {
                "technical_indicators": ["sma_5"],
                "price_features": ["return_1d", "volatility_5d"],
                "volume_features": ["volume_normalized"],
                "total_features": 4
            }
            
            result = {
                "symbol": symbol,
                "data": processed_records,
                "features": features_created,
                "processing_metadata": {
                    "original_records": len(raw_records),
                    "processed_records": len(processed_records),
                    "features_added": features_created["total_features"],
                    "processing_timestamp": datetime.now().isoformat(),
                    "method": "local_processing"
                }
            }
            
            return {
                "success": True,
                "processed_data": result,
                "features_created": features_created,
                "record_count": len(processed_records)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "symbol": symbol}

    async def _arun(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Async version"""
        return self._run(data, symbol)


# Tool instances for preprocessing
download_from_s3_tool = DownloadFromS3Tool()
submit_glue_job_tool = SubmitGlueJobTool()
check_glue_job_status_tool = CheckGlueJobStatusTool()
upload_processed_data_tool = UploadProcessedDataTool()
create_features_tool = CreateFeaturesTool()

# Tool collections for different agents
DATA_INGESTION_TOOLS = [
    fetch_stock_data_tool,
    validate_data_quality_tool,
    upload_stock_data_to_s3_tool
]

PREPROCESSING_TOOLS = [
    download_from_s3_tool,
    submit_glue_job_tool,
    check_glue_job_status_tool,
    upload_processed_data_tool,
    create_features_tool
]

TRAINING_TOOLS = [
    # Add training tools here  
]


def initialize_tools(alpha_vantage_key: str, s3_bucket: str, openai_key: str = None):
    """Initialize global tool instances with API keys and configurations"""
    global _alpha_vantage_client, _s3_manager, _s3_manager_processed
    
    _alpha_vantage_client = AlphaVantageClient(api_key=alpha_vantage_key)
    _s3_manager = S3Manager(bucket_name=s3_bucket)
    
    # Initialize processed data S3 manager with separate bucket if available
    processed_bucket = os.getenv('S3_BUCKET_PROCESSED', s3_bucket)
    _s3_manager_processed = S3Manager(bucket_name=processed_bucket)
    
    return True


def get_all_tools() -> List[BaseTool]:
    """Get all available tools"""
    return DATA_INGESTION_TOOLS + PREPROCESSING_TOOLS + TRAINING_TOOLS
