"""Configuration management for the multi-agent system"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    
    # Validate required environment variables
    required_vars = [
        "ALPHA_VANTAGE_API_KEY",
        "OPENAI_API_KEY",
        "S3_BUCKET_RAW_DATA",
        "AWS_REGION"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    config = {
        # API Keys
        "alpha_vantage_api_key": os.getenv("ALPHA_VANTAGE_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        
        # AWS Configuration
        "aws_region": os.getenv("AWS_REGION", "us-east-1"),
        "s3_bucket_raw_data": os.getenv("S3_BUCKET_RAW_DATA"),
        "s3_bucket_processed": os.getenv("S3_BUCKET_PROCESSED"),
        "s3_bucket_models": os.getenv("S3_BUCKET_MODELS"),
        "dynamodb_table": os.getenv("DYNAMODB_TABLE", "agent-states"),
        
        # Agent Configuration
        "max_retries": int(os.getenv("MAX_RETRIES", "3")),
        "retry_delay": int(os.getenv("RETRY_DELAY", "60")),
        "rate_limit_delay": float(os.getenv("RATE_LIMIT_DELAY", "12.0")),
        
        # Stock Symbols Configuration
        "default_symbols": _parse_symbol_list(
            os.getenv("DEFAULT_SYMBOLS", "AAPL,GOOGL,MSFT,TSLA,AMZN")
        ),
        
        # LLM Configuration
        "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0")),
        "llm_max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1000")),
        
        # System Configuration
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "message_bus_timeout": float(os.getenv("MESSAGE_BUS_TIMEOUT", "1.0")),
        
        # Development/Testing
        "development_mode": os.getenv("DEVELOPMENT_MODE", "false").lower() == "true",
        "mock_apis": os.getenv("MOCK_APIS", "false").lower() == "true",
    }
    
    logger.info("Configuration loaded successfully")
    logger.debug(f"Configuration (sensitive data masked): {_mask_sensitive_config(config)}")
    
    return config


def _parse_symbol_list(symbol_string: str) -> List[str]:
    """Parse comma-separated symbol list"""
    return [symbol.strip().upper() for symbol in symbol_string.split(",") if symbol.strip()]


def _mask_sensitive_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Mask sensitive information in config for logging"""
    sensitive_keys = ["api_key", "secret", "password", "token"]
    
    masked_config = {}
    for key, value in config.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            masked_config[key] = "***MASKED***" if value else None
        else:
            masked_config[key] = value
    
    return masked_config


def setup_logging(config: Dict[str, Any]) -> None:
    """Configure logging based on configuration"""
    log_level = config.get("log_level", "INFO").upper()
    
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sink=lambda message: print(message, end=""),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True
    )
    
    # Add file logger for production
    if not config.get("development_mode", False):
        logger.add(
            "logs/multi_agent_system_{time:YYYY-MM-DD}.log",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="1 day",
            retention="30 days",
            compression="zip"
        )
    
    logger.info(f"Logging configured with level: {log_level}")


def validate_aws_credentials() -> bool:
    """Validate AWS credentials are available"""
    try:
        import boto3
        
        # Try to create S3 client
        s3_client = boto3.client("s3")
        
        # Test credentials by listing buckets
        s3_client.list_buckets()
        
        logger.info("AWS credentials validated successfully")
        return True
    except Exception as e:
        logger.error(f"AWS credentials validation failed: {e}")
        return False


def validate_openai_credentials(api_key: str) -> bool:
    """Validate OpenAI API key"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        # Test with a simple completion using GPT-5-mini
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        
        logger.info("OpenAI credentials validated successfully")
        return True
    except Exception as e:
        logger.error(f"OpenAI credentials validation failed: {e}")
        return False


def get_environment_info() -> Dict[str, str]:
    """Get environment information for debugging"""
    return {
        "python_version": os.sys.version,
        "working_directory": os.getcwd(),
        "environment_variables": len(os.environ),
        "aws_region": os.getenv("AWS_REGION", "not_set"),
        "development_mode": os.getenv("DEVELOPMENT_MODE", "false"),
    }
