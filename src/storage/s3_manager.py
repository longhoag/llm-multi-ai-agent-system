"""S3 storage manager for data persistence"""

import json
from io import StringIO, BytesIO
from typing import Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd
from loguru import logger


class S3Manager:
    """Manages S3 operations for the multi-agent system"""
    
    def __init__(self, bucket_name: str, region_name: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.s3_client = boto3.client("s3", region_name=region_name)
        logger.info(f"S3Manager initialized for bucket: {bucket_name}")
    
    async def upload_json(self, data: Dict[str, Any], key: str) -> bool:
        """Upload JSON data to S3"""
        try:
            json_string = json.dumps(data, indent=2, default=str)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json_string,
                ContentType="application/json"
            )
            logger.info(f"JSON uploaded to S3: s3://{self.bucket_name}/{key}")
            return True
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to upload JSON to S3: {e}")
            return False
    
    async def download_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Download JSON data from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            content = response["Body"].read().decode("utf-8")
            data = json.loads(content)
            logger.info(f"JSON downloaded from S3: s3://{self.bucket_name}/{key}")
            return data
        except (ClientError, NoCredentialsError, json.JSONDecodeError) as e:
            logger.error(f"Failed to download JSON from S3: {e}")
            return None
    
    async def upload_dataframe(self, df: pd.DataFrame, key: str, file_format: str = "parquet") -> bool:
        """Upload pandas DataFrame to S3"""
        try:
            if file_format.lower() == "parquet":
                buffer = BytesIO()
                df.to_parquet(buffer, index=False)
                content_type = "application/octet-stream"
            elif file_format.lower() == "csv":
                buffer = StringIO()
                df.to_csv(buffer, index=False)
                content_type = "text/csv"
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            buffer.seek(0)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer.getvalue(),
                ContentType=content_type
            )
            logger.info(f"DataFrame uploaded to S3: s3://{self.bucket_name}/{key}")
            return True
        except (ClientError, NoCredentialsError, ValueError) as e:
            logger.error(f"Failed to upload DataFrame to S3: {e}")
            return False
    
    async def download_dataframe(
        self, key: str, file_format: str = "parquet"
    ) -> Optional[pd.DataFrame]:
        """Download pandas DataFrame from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            content = response["Body"].read()
            
            if file_format.lower() == "parquet":
                buffer = BytesIO(content)
                df = pd.read_parquet(buffer)
            elif file_format.lower() == "csv":
                buffer = StringIO(content.decode("utf-8"))
                df = pd.read_csv(buffer)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            logger.info(f"DataFrame downloaded from S3: s3://{self.bucket_name}/{key}")
            return df
        except (ClientError, NoCredentialsError, ValueError) as e:
            logger.error(f"Failed to download DataFrame from S3: {e}")
            return None
    
    async def upload_text(self, text: str, key: str) -> bool:
        """Upload text content to S3"""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=text,
                ContentType="text/plain"
            )
            logger.info(f"Text uploaded to S3: s3://{self.bucket_name}/{key}")
            return True
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to upload text to S3: {e}")
            return False
    
    async def download_text(self, key: str) -> Optional[str]:
        """Download text content from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            content = response["Body"].read().decode("utf-8")
            logger.info(f"Text downloaded from S3: s3://{self.bucket_name}/{key}")
            return content
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to download text from S3: {e}")
            return None
    
    def list_objects(self, prefix: str = "") -> list:
        """List objects in S3 bucket with optional prefix"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            objects = response.get("Contents", [])
            keys = [obj["Key"] for obj in objects]
            logger.info(f"Listed {len(keys)} objects with prefix '{prefix}'")
            return keys
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to list objects: {e}")
            return []
    
    def delete_object(self, key: str) -> bool:
        """Delete object from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"Object deleted from S3: s3://{self.bucket_name}/{key}")
            return True
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to delete object: {e}")
            return False
    
    def object_exists(self, key: str) -> bool:
        """Check if object exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError:
            return False
