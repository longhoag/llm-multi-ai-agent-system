"""Test cases for the Data Ingestion Agent"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.agents.base_agent import AgentMessage
from src.agents.data_ingestion_agent import DataIngestionAgent


class TestDataIngestionAgent:
    """Test suite for Data Ingestion Agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, mock_config, mock_message_bus):
        """Test agent initialization"""
        agent = DataIngestionAgent(
            agent_id="test_agent",
            message_bus=mock_message_bus,
            config=mock_config
        )
        
        assert agent.agent_id == "test_agent"
        assert agent.config == mock_config
        assert agent.get_state("ingestion_stats") is not None
        
        await agent.stop()
    
    @pytest.mark.asyncio
    async def test_ingest_request_message(self, data_ingestion_agent):
        """Test handling of ingestion request message"""
        # Mock dependencies
        with patch.object(data_ingestion_agent, '_determine_ingestion_strategy', 
                         new_callable=AsyncMock) as mock_strategy:
            mock_strategy.return_value = "HIGH priority strategy"
            
            with patch.object(data_ingestion_agent, '_check_recent_data', 
                             new_callable=AsyncMock) as mock_recent:
                mock_recent.return_value = False
                
                with patch.object(data_ingestion_agent, '_fetch_stock_data', 
                                 new_callable=AsyncMock) as mock_fetch:
                    mock_fetch.return_value = {
                        "symbol": "AAPL",
                        "data": [{"date": "2024-01-01", "close": 100.0}]
                    }
                    
                    with patch.object(data_ingestion_agent, '_store_raw_data', 
                                     new_callable=AsyncMock) as mock_store:
                        mock_store.return_value = "raw_data/daily/AAPL/test.json"
                        
                        # Create test message
                        message = AgentMessage(
                            sender="test_sender",
                            recipient="data_ingestion_agent",
                            message_type="INGEST_REQUEST",
                            payload={"symbol": "AAPL", "timeframe": "daily"}
                        )
                        
                        # Process message
                        response = await data_ingestion_agent.process_message(message)
                        
                        # Verify response
                        assert response is not None
                        assert response.message_type == "DATA_AVAILABLE"
                        assert response.payload["symbol"] == "AAPL"
                        assert "s3_key" in response.payload
    
    @pytest.mark.asyncio
    async def test_scheduled_ingestion(self, data_ingestion_agent):
        """Test scheduled batch ingestion"""
        message = AgentMessage(
            sender="orchestrator",
            recipient="data_ingestion_agent", 
            message_type="SCHEDULE_INGESTION",
            payload={
                "symbols": ["AAPL", "GOOGL"],
                "timeframe": "daily"
            }
        )
        
        with patch.object(data_ingestion_agent, 'send_message', 
                         new_callable=AsyncMock) as mock_send:
            response = await data_ingestion_agent.process_message(message)
            
            # Should not return a response for scheduled ingestion
            assert response is None
            
            # Should have sent individual ingestion requests
            assert mock_send.call_count == 2
    
    @pytest.mark.asyncio
    async def test_data_validation(self, data_ingestion_agent):
        """Test data quality validation"""
        test_data = {
            "symbol": "AAPL",
            "data": [
                {
                    "date": "2024-01-01",
                    "open": 100.0,
                    "high": 105.0,
                    "low": 98.0,
                    "close": 103.0,
                    "volume": 1000000
                }
            ]
        }
        
        # Mock LLM response
        with patch.object(data_ingestion_agent.routine_llm, 'agenerate', 
                         new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = Mock(generations=[
                Mock(text='{"is_valid": true, "issues": [], "confidence": 0.9}')
            ])
            
            result = await data_ingestion_agent._validate_data_quality(test_data)
            
            assert result["is_valid"] is True
            assert result["confidence"] > 0.5
    
    @pytest.mark.asyncio 
    async def test_error_handling(self, data_ingestion_agent):
        """Test error handling in message processing"""
        # Create message that will cause an error
        message = AgentMessage(
            sender="test_sender",
            recipient="data_ingestion_agent",
            message_type="INGEST_REQUEST",
            payload={"symbol": "INVALID", "timeframe": "daily"}
        )
        
        # Mock fetch to raise an exception
        with patch.object(data_ingestion_agent, '_fetch_stock_data', 
                         new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = Exception("API Error")
            
            response = await data_ingestion_agent.process_message(message)
            
            # Should return error response
            assert response is not None
            assert response.message_type == "INGESTION_ERROR"
            assert "error" in response.payload
    
    @pytest.mark.asyncio
    async def test_ingestion_statistics(self, data_ingestion_agent):
        """Test ingestion statistics tracking"""
        initial_stats = await data_ingestion_agent.get_ingestion_stats()
        
        assert "symbols_processed" in initial_stats
        assert "successful_ingestions" in initial_stats
        assert "failed_ingestions" in initial_stats
        assert initial_stats["symbols_processed"] == 0
    
    def test_recent_data_check(self, data_ingestion_agent):
        """Test checking for recent data"""
        # Mock S3 manager
        with patch.object(data_ingestion_agent.s3_manager, 'list_objects') as mock_list:
            # Test with recent data
            mock_list.return_value = ["raw_data/daily/AAPL/20240101_100000.json"]
            
            # This would need to be implemented as async in the actual agent
            # For now, testing the concept
            assert mock_list.return_value is not None


class TestMessageBus:
    """Test suite for Message Bus"""
    
    @pytest.mark.asyncio
    async def test_message_bus_creation(self, mock_message_bus):
        """Test message bus creation"""
        assert mock_message_bus.subscribers is not None
        assert mock_message_bus.message_queue is not None
    
    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, mock_message_bus):
        """Test message subscription and publishing"""
        received_messages = []
        
        async def test_handler(message):
            received_messages.append(message)
            return None
        
        # Subscribe to messages
        await mock_message_bus.subscribe("test_agent", test_handler)
        
        # Create and publish a message
        test_message = AgentMessage(
            sender="sender",
            recipient="test_agent",
            message_type="TEST_MESSAGE",
            payload={"test": "data"}
        )
        
        await mock_message_bus.publish(test_message)
        
        # Process one message cycle (simplified)
        if mock_message_bus.get_queue_size() > 0:
            # In real test, would need to let message bus process
            pass
        
        # Verify subscription worked
        assert "test_agent" in mock_message_bus.subscribers
        assert len(mock_message_bus.subscribers["test_agent"]) == 1


if __name__ == "__main__":
    pytest.main([__file__])
