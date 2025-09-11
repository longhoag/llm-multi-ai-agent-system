#!/usr/bin/env python3
"""
Verify the ingested data in S3
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.storage.s3_manager import S3Manager

logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{level}</level> | {message}")

async def verify_s3_data():
    """Verify the data that was uploaded to S3"""
    
    load_dotenv()
    bucket = os.getenv('S3_BUCKET_RAW_DATA')
    
    if not bucket:
        logger.error("S3_BUCKET_RAW_DATA not configured")
        return
    
    s3_manager = S3Manager(bucket)
    
    try:
        # List all objects in the raw_data/daily directory
        logger.info("📁 Listing S3 objects in raw_data/daily/...")
        objects = s3_manager.list_objects("raw_data/daily/")
        
        if not objects:
            logger.warning("No objects found in raw_data/daily/")
            return
        
        logger.success(f"Found {len(objects)} files:")
        for obj in objects:
            logger.info(f"   📄 {obj}")
        
        # Download and examine one of the files
        if objects:
            sample_file = objects[0]  # Take the first file
            logger.info(f"📄 Examining file: {sample_file}")
            
            data = await s3_manager.download_json(sample_file)
            
            if data:
                logger.success("✅ Successfully downloaded data!")
                logger.info(f"   📊 Keys in data: {list(data.keys())}")
                
                if 'data' in data:
                    records = data['data']
                    logger.info(f"   📈 Number of records: {len(records)}")
                    
                    if records:
                        sample_record = records[0]
                        logger.info(f"   📋 Sample record keys: {list(sample_record.keys())}")
                        logger.info(f"   💰 Sample prices - Open: {sample_record.get('open')}, Close: {sample_record.get('close')}")
                        logger.info(f"   📅 Sample date: {sample_record.get('date')}")
                
                if 'metadata' in data:
                    metadata = data['metadata']
                    logger.info(f"   ℹ️  Metadata: symbol={metadata.get('symbol')}, timeframe={metadata.get('timeframe')}")
            else:
                logger.error("❌ Failed to download data")
        
    except Exception as e:
        logger.error(f"❌ Error verifying S3 data: {e}")

if __name__ == "__main__":
    asyncio.run(verify_s3_data())
