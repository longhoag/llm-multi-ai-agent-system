# pylint: disable=broad-exception-caught
"""Data Ingestion Agent for stock market data"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from loguru import logger
from src.agents.base_agent import BaseAgent, AgentMessage
from src.storage.s3_manager import S3Manager
from src.external.alpha_vantage_client import AlphaVantageClient

class DataIngestionAgent(BaseAgent):
    """Agent responsible for ingesting financial data from external sources"""
    
    def __init__(
        self, 
        agent_id: str, 
        message_bus,
        config: Dict[str, Any]
    ):
        super().__init__(agent_id, message_bus)
        
        # Initialize storage manager
        self.s3_manager = S3Manager(
            bucket_name=config["s3_bucket_raw_data"],
            region_name=config.get("aws_region", "us-east-1")
        )
        
        # Initialize Alpha Vantage client
        self.alpha_vantage = AlphaVantageClient(
            api_key=config["alpha_vantage_api_key"]
        )
        
        # Initialize OpenAI LLM clients for intelligent decisions using GPT-5-mini
        self.llm_model = "gpt-5-mini"  # Latest cost-effective model for agent decisions
        self.strategy_llm = ChatOpenAI(
            model=self.llm_model,
            temperature=1,  # GPT-5-mini only supports temperature=1
            openai_api_key=config.get("openai_api_key", os.getenv('OPENAI_API_KEY'))
        )
        self.routine_llm = ChatOpenAI(
            model=self.llm_model,
            temperature=1,  # GPT-5-mini only supports temperature=1
            openai_api_key=config.get("openai_api_key", os.getenv('OPENAI_API_KEY'))
        )
        
        # Configuration
        self.config = config
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 60)  # seconds
        
        # State tracking
        self.update_state("ingestion_stats", {
            "symbols_processed": 0,
            "successful_ingestions": 0,
            "failed_ingestions": 0,
            "last_ingestion_time": None
        })
        
        logger.info(f"Data Ingestion Agent {agent_id} initialized")
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages"""
        logger.info(
            f"Processing message: {message.message_type} from {message.sender}"
        )
        
        try:
            if message.message_type == "INGEST_REQUEST":
                return await self._handle_ingest_request(message)
            elif message.message_type == "SCHEDULE_INGESTION":
                return await self._handle_scheduled_ingestion(message)
            elif message.message_type == "DATA_QUALITY_CHECK":
                return await self._handle_data_quality_check(message)
            elif message.message_type == "RETRY_FAILED_INGESTION":
                return await self._handle_retry_ingestion(message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                message_type="INGESTION_ERROR",
                payload={"error": str(e), "correlation_id": message.correlation_id}
            )
    
    async def _handle_ingest_request(self, message: AgentMessage) -> AgentMessage:
        """Handle individual symbol ingestion request"""
        symbol = message.payload.get("symbol")
        timeframe = message.payload.get("timeframe", "daily")
        force_refresh = message.payload.get("force_refresh", False)
        
        logger.info(f"Starting ingestion for {symbol} ({timeframe})")
        
        try:
            # Use GPT-5-mini to determine optimal ingestion strategy
            strategy = await self._determine_ingestion_strategy(symbol, timeframe)
            logger.info(f"LLM suggested strategy for {symbol}: {strategy}")
            
            # Check if recent data exists (unless force refresh)
            if not force_refresh:
                recent_data_exists = await self._check_recent_data(symbol, timeframe)
                if recent_data_exists:
                    logger.info(f"Recent data exists for {symbol}, skipping ingestion")
                    return await self._create_data_available_message(
                        symbol, "existing_data", message.correlation_id
                    )
            
            # Fetch data from Alpha Vantage
            raw_data = await self._fetch_stock_data(symbol, timeframe, strategy)
            
            if not raw_data:
                raise Exception(f"Failed to fetch data from Alpha Vantage for {symbol}")
            
            # Validate data quality using GPT-5-mini
            quality_check = await self._validate_data_quality(raw_data, symbol)
            if not quality_check["is_valid"]:
                logger.warning(f"Data quality issues for {symbol}: {quality_check['issues']}")
            
            # Store raw data in S3
            s3_key = await self._store_raw_data(symbol, timeframe, raw_data)
            
            # Update statistics
            stats = self.get_state("ingestion_stats")
            stats["symbols_processed"] += 1
            stats["successful_ingestions"] += 1
            stats["last_ingestion_time"] = datetime.now().isoformat()
            self.update_state("ingestion_stats", stats)
            
            logger.info(f"Successfully ingested data for {symbol}")
            
            return await self._create_data_available_message(
                symbol, s3_key, message.correlation_id, raw_data
            )
            
        except Exception as e:
            logger.error(f"Ingestion failed for {symbol}: {e}")
            
            # Update error statistics
            stats = self.get_state("ingestion_stats")
            stats["failed_ingestions"] += 1
            self.update_state("ingestion_stats", stats)
            
            # Schedule retry if within limits
            await self._schedule_retry_if_needed(symbol, timeframe, str(e))
            
            return AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                message_type="INGESTION_ERROR",
                payload={
                    "symbol": symbol,
                    "error": str(e),
                    "correlation_id": message.correlation_id,
                    "retry_scheduled": True
                }
            )
    
    async def _handle_scheduled_ingestion(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle batch ingestion for multiple symbols"""
        symbols = message.payload.get("symbols", [])
        timeframe = message.payload.get("timeframe", "daily")
        
        logger.info(f"Starting scheduled ingestion for {len(symbols)} symbols")
        
        # Process symbols with rate limiting
        for i, symbol in enumerate(symbols):
            try:
                await self.send_message(
                    recipient=self.agent_id,
                    message_type="INGEST_REQUEST",
                    payload={
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "force_refresh": False
                    },
                    correlation_id=f"batch_{message.correlation_id}_{i}"
                )
                
                # Rate limiting between requests (Alpha Vantage free tier: 5 calls/minute)
                if i < len(symbols) - 1:
                    await asyncio.sleep(12)  # 12 seconds between requests
                    
            except Exception as e:
                logger.error(f"Failed to schedule ingestion for {symbol}: {e}")
        
        return None  # No response needed for scheduled batch
    
    async def _determine_ingestion_strategy(self, symbol: str, timeframe: str) -> str:
        """Use GPT-5-mini to determine optimal ingestion strategy"""
        strategy_prompt = PromptTemplate(
            input_variables=["symbol", "timeframe", "current_time", "market_hours"],
            template="""
            Analyze the optimal data ingestion strategy for stock symbol {symbol} with {timeframe} timeframe.
            Current time: {current_time}
            Market hours consideration: {market_hours}
            
            Consider:
            1. API rate limits (Alpha Vantage free: 5 calls/minute, 500 calls/day)
            2. Data freshness requirements
            3. Market trading hours
            4. Weekend/holiday considerations
            5. Data completeness validation needs
            
            Provide a concise strategy recommendation with priority level (HIGH/MEDIUM/LOW).
            """
        )
        
        current_time = datetime.now()
        market_hours = "OPEN" if 9 <= current_time.hour <= 16 else "CLOSED"
        
        try:
            prompt_text = strategy_prompt.format(
                symbol=symbol,
                timeframe=timeframe,
                current_time=current_time.isoformat(),
                market_hours=market_hours
            )
            response = await self.strategy_llm.ainvoke(prompt_text)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM strategy generation failed: {e}")
            return "STANDARD: Fetch data with normal priority and validation"
    
    async def _fetch_stock_data(
        self, 
        symbol: str, 
        timeframe: str, 
        strategy: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch stock data from Alpha Vantage with retry logic"""
        for attempt in range(self.max_retries):
            try:
                if timeframe == "daily":
                    data = await self.alpha_vantage.get_daily_time_series(symbol)
                elif timeframe.endswith("min"):
                    data = await self.alpha_vantage.get_intraday_time_series(
                        symbol, interval=timeframe
                    )
                else:
                    raise ValueError(f"Unsupported timeframe: {timeframe}")
                
                if data:
                    logger.info(f"Successfully fetched {timeframe} data for {symbol}")
                    return data
                
                # If no data, wait and retry
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"No data received for {symbol}, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        return None
    
    async def _validate_data_quality(self, raw_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Validate data quality using GPT-5-mini"""
        validation_prompt = PromptTemplate(
            input_variables=["data_summary", "record_count"],
            template="""
            Validate the quality of stock market data:
            
            Data summary: {data_summary}
            Record count: {record_count}
            
            Check for:
            1. Reasonable price ranges (no negative prices, extreme outliers)
            2. Complete data fields (open, high, low, close, volume)
            3. Logical price relationships (high >= close >= low, etc.)
            4. Sufficient data volume
            
            Respond with JSON format:
            {{"is_valid": true/false, "issues": ["list of issues"], "confidence": 0.0-1.0}}
            """
        )
        
        try:
            # Create data summary for validation
            data_records = raw_data.get("data", [])
            if not data_records:
                return {"is_valid": False, "issues": ["No data records"], "confidence": 1.0}
            
            sample_record = data_records[0]
            data_summary = (f"Sample: Open={sample_record.get('open')}, "
                          f"High={sample_record.get('high')}, "
                          f"Low={sample_record.get('low')}, "
                          f"Close={sample_record.get('close')}, "
                          f"Volume={sample_record.get('volume')}")
            
            prompt_text = validation_prompt.format(
                data_summary=data_summary,
                record_count=len(data_records)
            )
            response = await self.routine_llm.ainvoke(prompt_text)
            
            # Parse LLM response (simplified - in production, use proper JSON parsing)
            response_text = response.content.strip()
            if "true" in response_text.lower():
                return {"is_valid": True, "issues": [], "confidence": 0.8}
            else:
                return {"is_valid": False, "issues": ["Data quality concerns"], "confidence": 0.6}
                
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {"is_valid": True, "issues": [], "confidence": 0.5}  # Default to valid
    
    async def _store_raw_data(
        self, 
        symbol: str, 
        timeframe: str, 
        raw_data: Dict[str, Any]
    ) -> str:
        """Store raw data in S3 and return the key"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"raw_data/{timeframe}/{symbol}/{timestamp}.json"
        
        success = await self.s3_manager.upload_json(raw_data, s3_key)
        if not success:
            raise Exception(f"Failed to upload data to S3: {s3_key}")
        
        return s3_key
    
    async def _check_recent_data(self, symbol: str, timeframe: str) -> bool:
        """Check if recent data exists for the symbol"""
        try:
            # List recent files for this symbol
            prefix = f"raw_data/{timeframe}/{symbol}/"
            objects = self.s3_manager.list_objects(prefix)
            
            if not objects:
                return False
            
            # Check if we have data from today
            today = datetime.now().strftime("%Y%m%d")
            for obj_key in objects:
                if today in obj_key:
                    logger.info(f"Found recent data for {symbol}: {obj_key}")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking recent data for {symbol}: {e}")
            return False
    
    async def _create_data_available_message(
        self, 
        symbol: str, 
        s3_key: str, 
        correlation_id: Optional[str],
        raw_data: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Create standardized data available message"""
        payload = {
            "symbol": symbol,
            "s3_key": s3_key,
            "ingestion_time": datetime.now().isoformat(),
            "correlation_id": correlation_id
        }
        
        if raw_data:
            payload["record_count"] = len(raw_data.get("data", []))
            payload["data_source"] = "alpha_vantage"
        
        return AgentMessage(
            sender=self.agent_id,
            recipient="preprocessing_agent",
            message_type="DATA_AVAILABLE",
            payload=payload,
            correlation_id=correlation_id
        )
    
    async def _schedule_retry_if_needed(
        self, 
        symbol: str, 
        timeframe: str, 
        error: str
    ) -> None:
        """Schedule retry for failed ingestion"""
        retry_count = self.get_state(f"retry_count_{symbol}") or 0
        
        if retry_count < self.max_retries:
            retry_count += 1
            self.update_state(f"retry_count_{symbol}", retry_count)
            
            # Schedule retry with exponential backoff
            retry_delay = self.retry_delay * (2 ** retry_count)
            
            logger.info(
                f"Scheduling retry {retry_count}/{self.max_retries} for {symbol} "
                f"in {retry_delay}s"
            )
            
            # In a real implementation, you'd use a task scheduler
            # For now, we'll send a delayed message
            await asyncio.sleep(retry_delay)
            await self.send_message(
                recipient=self.agent_id,
                message_type="RETRY_FAILED_INGESTION",
                payload={"symbol": symbol, "timeframe": timeframe, "error": error}
            )
    
    async def _handle_retry_ingestion(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle retry of failed ingestion"""
        symbol = message.payload.get("symbol")
        timeframe = message.payload.get("timeframe", "daily")
        
        logger.info(f"Retrying ingestion for {symbol}")
        
        # Reset retry count and attempt ingestion
        self.update_state(f"retry_count_{symbol}", 0)
        
        return await self._handle_ingest_request(AgentMessage(
            sender=self.agent_id,
            recipient=self.agent_id,
            message_type="INGEST_REQUEST",
            payload={"symbol": symbol, "timeframe": timeframe, "force_refresh": True}
        ))
    
    async def _handle_data_quality_check(self, message: AgentMessage) -> AgentMessage:
        """Handle data quality validation request"""
        s3_key = message.payload.get("s3_key")
        
        try:
            # Download data from S3
            raw_data = await self.s3_manager.download_json(s3_key)
            if not raw_data:
                raise Exception(f"Failed to download data from S3: {s3_key}")
            
            # Validate quality
            symbol = s3_key.split('/')[-2] if '/' in s3_key else "UNKNOWN"
            quality_result = await self._validate_data_quality(raw_data, symbol)
            
            return AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                message_type="DATA_QUALITY_RESULT",
                payload={
                    "s3_key": s3_key,
                    "quality_result": quality_result,
                    "correlation_id": message.correlation_id
                }
            )
            
        except Exception as e:
            logger.error(f"Data quality check failed: {e}")
            return AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                message_type="QUALITY_CHECK_ERROR",
                payload={"error": str(e), "s3_key": s3_key}
            )
    
    async def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get current ingestion statistics"""
        return self.get_state("ingestion_stats")
    
    async def shutdown(self) -> None:
        """Graceful shutdown of the agent"""
        logger.info(f"Shutting down Data Ingestion Agent {self.agent_id}")
        
        # Close Alpha Vantage client
        await self.alpha_vantage.close()
        
        # Stop the agent
        await self.stop()
        
        # Log final statistics
        stats = self.get_state("ingestion_stats")
        logger.info(f"Final ingestion stats: {stats}")
