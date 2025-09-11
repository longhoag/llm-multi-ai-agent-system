"""Test configuration and fixtures"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from src.messaging.message_bus import MessageBus
from src.agents.data_ingestion_agent import DataIngestionAgent


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Mock configuration for testing"""
    return {
        "alpha_vantage_api_key": "test_av_key",
        "openai_api_key": "test_openai_key",
        "s3_bucket_raw_data": "test-raw-bucket",
        "s3_bucket_processed": "test-processed-bucket",
        "aws_region": "us-east-1",
        "max_retries": 2,
        "retry_delay": 1,
        "rate_limit_delay": 0.1,
        "default_symbols": ["AAPL", "GOOGL"],
        "development_mode": True,
        "mock_apis": True
    }


@pytest.fixture
async def mock_message_bus():
    """Mock message bus for testing"""
    message_bus = MessageBus()
    yield message_bus
    await message_bus.stop()


@pytest.fixture
def mock_s3_manager():
    """Mock S3 manager for testing"""
    s3_manager = Mock()
    s3_manager.upload_json = AsyncMock(return_value=True)
    s3_manager.download_json = AsyncMock(return_value={"test": "data"})
    s3_manager.list_objects = Mock(return_value=[])
    return s3_manager


@pytest.fixture
def mock_alpha_vantage_client():
    """Mock Alpha Vantage client for testing"""
    client = Mock()
    client.get_daily_time_series = AsyncMock(return_value={
        "symbol": "AAPL",
        "data": [
            {
                "date": "2024-01-01",
                "symbol": "AAPL",
                "open": 100.0,
                "high": 105.0,
                "low": 98.0,
                "close": 103.0,
                "volume": 1000000
            }
        ],
        "metadata": {},
        "fetch_timestamp": "2024-01-01T10:00:00"
    })
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    llm = Mock()
    llm.agenerate = AsyncMock(return_value=Mock(generations=[
        Mock(text="HIGH priority ingestion strategy recommended")
    ]))
    return llm


@pytest.fixture
async def data_ingestion_agent(mock_config, mock_message_bus):
    """Create a data ingestion agent for testing"""
    agent = DataIngestionAgent(
        agent_id="test_data_ingestion_agent",
        message_bus=mock_message_bus,
        config=mock_config
    )
    yield agent
    await agent.stop()
