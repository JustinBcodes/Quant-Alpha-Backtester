"""
AWS utilities for S3 operations.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Union

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    AWS_S3_BUCKET
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_to_s3(
    local_path: Union[str, Path], 
    s3_path: str, 
    bucket: Optional[str] = None
) -> bool:
    """
    Save a local file or directory to S3.
    
    Args:
        local_path: Local file or directory path
        s3_path: S3 key prefix
        bucket: S3 bucket name (defaults to AWS_S3_BUCKET)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize S3 client
        s3 = boto3.client(
            's3', 
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        # Use default bucket if not specified
        bucket = bucket or AWS_S3_BUCKET
        
        if not bucket:
            logger.error("No S3 bucket specified")
            return False
            
        local_path = Path(local_path)
        
        if local_path.is_file():
            # Upload single file
            s3_key = os.path.join(s3_path, local_path.name)
            s3.upload_file(str(local_path), bucket, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
            return True
            
        elif local_path.is_dir():
            # Upload directory contents
            success = True
            for item in local_path.glob('**/*'):
                if item.is_file():
                    # Calculate relative path to maintain directory structure
                    rel_path = item.relative_to(local_path)
                    s3_key = os.path.join(s3_path, str(rel_path))
                    
                    try:
                        s3.upload_file(str(item), bucket, s3_key)
                        logger.info(f"Uploaded {item} to s3://{bucket}/{s3_key}")
                    except Exception as e:
                        logger.error(f"Error uploading {item}: {str(e)}")
                        success = False
            
            return success
        else:
            logger.error(f"Path not found: {local_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error saving to S3: {str(e)}")
        return False

class S3Handler:
    """Handler for S3 operations."""
    
    def __init__(self):
        """Initialize the S3 handler."""
        self.s3 = boto3.client(
            's3', 
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        self.bucket = AWS_S3_BUCKET
    
    def save_dataframe(self, df: pd.DataFrame, key: str) -> None:
        """
        Save DataFrame to S3 as parquet.
        
        Args:
            df: DataFrame to save
            key: S3 key
        """
        try:
            # Convert to parquet bytes
            parquet_buffer = df.to_parquet()
            
            # Upload to S3
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=parquet_buffer
            )
            logger.info(f"Saved DataFrame to s3://{self.bucket}/{key}")
            
        except Exception as e:
            logger.error(f"Error saving DataFrame to S3: {str(e)}")
            raise
    
    def load_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from S3 parquet file.
        
        Args:
            key: S3 key
            
        Returns:
            DataFrame if file exists, None otherwise
        """
        try:
            # Get object from S3
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=key
            )
            
            # Read parquet
            df = pd.read_parquet(response['Body'])
            logger.info(f"Loaded DataFrame from s3://{self.bucket}/{key}")
            
            return df
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.info(f"No data found at s3://{self.bucket}/{key}")
                return None
            else:
                logger.error(f"Error loading DataFrame from S3: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Error loading DataFrame from S3: {str(e)}")
            raise
    
    def save_json(self, data: Dict, key: str) -> None:
        """
        Save dictionary to S3 as JSON.
        
        Args:
            data: Dictionary to save
            key: S3 key
        """
        try:
            # Convert to JSON string
            json_str = json.dumps(data)
            
            # Upload to S3
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json_str
            )
            logger.info(f"Saved JSON to s3://{self.bucket}/{key}")
            
        except Exception as e:
            logger.error(f"Error saving JSON to S3: {str(e)}")
            raise
    
    def load_json(self, key: str) -> Optional[Dict]:
        """
        Load dictionary from S3 JSON file.
        
        Args:
            key: S3 key
            
        Returns:
            Dictionary if file exists, None otherwise
        """
        try:
            # Get object from S3
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=key
            )
            
            # Read JSON
            data = json.loads(response['Body'].read().decode('utf-8'))
            logger.info(f"Loaded JSON from s3://{self.bucket}/{key}")
            
            return data
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.info(f"No data found at s3://{self.bucket}/{key}")
                return None
            else:
                logger.error(f"Error loading JSON from S3: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Error loading JSON from S3: {str(e)}")
            raise
    
    def list_objects(
        self,
        prefix: str = '',
        suffix: str = ''
    ) -> List[str]:
        """
        List objects in S3 bucket.
        
        Args:
            prefix: Object key prefix
            suffix: Object key suffix
            
        Returns:
            List of object keys
        """
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            objects = [
                obj['Key'] for obj in response.get('Contents', [])
                if obj['Key'].endswith(suffix)
            ]
            
            return objects
            
        except Exception as e:
            logger.error(f"Error listing objects in S3: {str(e)}")
            return []
    
    def delete_object(
        self,
        key: str
    ) -> bool:
        """
        Delete object from S3.
        
        Args:
            key: S3 object key
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.s3.delete_object(
                Bucket=self.bucket,
                Key=key
            )
            
            logger.info(f"Deleted object s3://{self.bucket}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting object from S3: {str(e)}")
            return False

if __name__ == "__main__":
    # Example usage
    s3 = S3Handler()
    
    # Save test data
    df = pd.DataFrame({'test': [1, 2, 3]})
    s3.save_dataframe(df, 'test.parquet')
    
    # Load test data
    loaded_df = s3.load_dataframe('test.parquet')
    print("\nLoaded DataFrame:")
    print(loaded_df)
    
    # Clean up
    s3.delete_object('test.parquet') 