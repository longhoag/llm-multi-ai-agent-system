"""Alpha Vantage API client for financial data"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import aiohttp
import asyncio
from loguru import logger


class AlphaVantageClient:
    """Client for Alpha Vantage financial data API"""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info("AlphaVantage client initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, params: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Make HTTP request to Alpha Vantage API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        params["apikey"] = self.api_key
        
        try:
            async with self.session.get(self.BASE_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check for API error messages
                    if "Error Message" in data:
                        logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                        return None
                    
                    if "Note" in data:
                        logger.warning(f"Alpha Vantage API note: {data['Note']}")
                        return None
                    
                    return data
                else:
                    logger.error(f"HTTP error {response.status}: {await response.text()}")
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in API request: {e}")
            return None
    
    async def get_daily_time_series(
        self, 
        symbol: str, 
        outputsize: str = "compact"
    ) -> Optional[Dict[str, Any]]:
        """Get daily time series data for a stock symbol"""
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol.upper(),
            "outputsize": outputsize
        }
        
        logger.info(f"Fetching daily data for {symbol}")
        data = await self._make_request(params)
        
        if data and "Time Series (Daily)" in data:
            # Transform data to more usable format
            time_series = data["Time Series (Daily)"]
            transformed_data = []
            
            for date_str, values in time_series.items():
                record = {
                    "date": date_str,
                    "symbol": symbol.upper(),
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "volume": int(values["5. volume"]),
                    "fetch_timestamp": datetime.now().isoformat()
                }
                transformed_data.append(record)
            
            logger.info(f"Successfully fetched {len(transformed_data)} records for {symbol}")
            return {
                "symbol": symbol.upper(),
                "data": transformed_data,
                "metadata": data.get("Meta Data", {}),
                "fetch_timestamp": datetime.now().isoformat()
            }
        
        logger.error(f"Failed to fetch data for {symbol}")
        return None
    
    async def get_intraday_time_series(
        self, 
        symbol: str, 
        interval: str = "5min",
        outputsize: str = "compact"
    ) -> Optional[Dict[str, Any]]:
        """Get intraday time series data for a stock symbol"""
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol.upper(),
            "interval": interval,
            "outputsize": outputsize
        }
        
        logger.info(f"Fetching {interval} intraday data for {symbol}")
        data = await self._make_request(params)
        
        if data and f"Time Series ({interval})" in data:
            time_series = data[f"Time Series ({interval})"]
            transformed_data = []
            
            for datetime_str, values in time_series.items():
                record = {
                    "datetime": datetime_str,
                    "symbol": symbol.upper(),
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "volume": int(values["5. volume"]),
                    "fetch_timestamp": datetime.now().isoformat()
                }
                transformed_data.append(record)
            
            logger.info(f"Successfully fetched {len(transformed_data)} intraday records for {symbol}")
            return {
                "symbol": symbol.upper(),
                "interval": interval,
                "data": transformed_data,
                "metadata": data.get("Meta Data", {}),
                "fetch_timestamp": datetime.now().isoformat()
            }
        
        logger.error(f"Failed to fetch intraday data for {symbol}")
        return None
    
    async def get_company_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company overview/fundamental data"""
        params = {
            "function": "OVERVIEW",
            "symbol": symbol.upper()
        }
        
        logger.info(f"Fetching company overview for {symbol}")
        data = await self._make_request(params)
        
        if data and "Symbol" in data:
            data["fetch_timestamp"] = datetime.now().isoformat()
            logger.info(f"Successfully fetched company overview for {symbol}")
            return data
        
        logger.error(f"Failed to fetch company overview for {symbol}")
        return None
    
    async def get_multiple_symbols_daily(
        self, 
        symbols: List[str],
        delay_between_requests: float = 12.0  # Alpha Vantage free tier: 5 calls per minute
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Fetch daily data for multiple symbols with rate limiting"""
        results = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Fetching data for {symbol} ({i+1}/{len(symbols)})")
            results[symbol] = await self.get_daily_time_series(symbol)
            
            # Rate limiting - wait between requests except for the last one
            if i < len(symbols) - 1:
                logger.debug(f"Waiting {delay_between_requests}s before next request")
                await asyncio.sleep(delay_between_requests)
        
        return results
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            logger.info("Alpha Vantage client session closed")
