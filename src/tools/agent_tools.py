"""LangGraph tools for stock data pipeline agents"""

import os
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

from langchain.tools import BaseTool
from langgraph.prebuilt import ToolNode

from src.external.alpha_vantage_client import AlphaVantageClient
from src.storage.s3_manager import S3Manager

# Global tool instances (initialized via initialize_tools)
_alpha_vantage_client = None
_s3_manager = None


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
        """Synchronous execution"""
        return asyncio.run(self._arun(symbol, outputsize))

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
                return {"success": False, "error": "No data received", "symbol": symbol}
                
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
        """Synchronous execution"""
        return asyncio.run(self._arun(data, symbol, timeframe))

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

# Tool collections for different agents
DATA_INGESTION_TOOLS = [
    fetch_stock_data_tool,
    validate_data_quality_tool,
    upload_stock_data_to_s3_tool
]

PREPROCESSING_TOOLS = [
    # Add preprocessing tools here
]

TRAINING_TOOLS = [
    # Add training tools here  
]

# Tool nodes for LangGraph
data_ingestion_tool_node = ToolNode(DATA_INGESTION_TOOLS)


def initialize_tools(alpha_vantage_key: str, s3_bucket: str, openai_key: str = None):
    """Initialize global tool instances with API keys and configurations"""
    global _alpha_vantage_client, _s3_manager
    
    _alpha_vantage_client = AlphaVantageClient(api_key=alpha_vantage_key)
    _s3_manager = S3Manager(bucket_name=s3_bucket)
    
    return True


def get_all_tools() -> List[BaseTool]:
    """Get all available tools"""
    return DATA_INGESTION_TOOLS + PREPROCESSING_TOOLS + TRAINING_TOOLS
