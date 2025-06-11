import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

from utils.budget_check import FREE_TIER_LIMITS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class S3Manager:
    """Class to manage S3 storage within Free Tier limits."""
    
    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region: str = 'us-east-1'
    ):
        """
        Initialize the S3Manager.
        
        Args:
            aws_access_key_id: AWS access key
            aws_secret_access_key: AWS secret key
            region: AWS region
        """
        self.session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region
        )
        self.s3 = self.session.client('s3')
        self.s3_resource = self.session.resource('s3')
    
    def get_bucket_size(self, bucket_name: str) -> Tuple[float, int]:
        """
        Get bucket size and object count.
        
        Args:
            bucket_name: S3 bucket name
            
        Returns:
            Tuple of (size in GB, object count)
        """
        try:
            total_size = 0
            total_objects = 0
            
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        total_size += obj['Size']
                        total_objects += 1
            
            # Convert to GB
            size_gb = total_size / (1024 * 1024 * 1024)
            
            return size_gb, total_objects
            
        except Exception as e:
            logger.error(f"Error getting bucket size: {str(e)}")
            return 0.0, 0
    
    def check_storage_limits(self, bucket_name: str) -> Dict:
        """
        Check if bucket usage is within Free Tier limits.
        
        Args:
            bucket_name: S3 bucket name
            
        Returns:
            Dictionary with usage metrics
        """
        size_gb, object_count = self.get_bucket_size(bucket_name)
        
        # Check against limits
        storage_percent = (size_gb / FREE_TIER_LIMITS['s3']['storage_gb']) * 100
        requests_percent = (object_count / FREE_TIER_LIMITS['s3']['requests']) * 100
        
        return {
            'size_gb': size_gb,
            'object_count': object_count,
            'storage_percent': storage_percent,
            'requests_percent': requests_percent,
            'is_safe': storage_percent < 80 and requests_percent < 80
        }
    
    def upload_file(
        self,
        bucket_name: str,
        file_path: str,
        object_key: Optional[str] = None,
        check_limits: bool = True
    ) -> bool:
        """
        Upload a file to S3.
        
        Args:
            bucket_name: S3 bucket name
            file_path: Local file path
            object_key: S3 object key (defaults to filename)
            check_limits: Whether to check storage limits before upload
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get file size
            file_size = Path(file_path).stat().st_size
            file_size_gb = file_size / (1024 * 1024 * 1024)
            
            # Check if upload would exceed limits
            if check_limits:
                current_size, _ = self.get_bucket_size(bucket_name)
                if current_size + file_size_gb > FREE_TIER_LIMITS['s3']['storage_gb']:
                    logger.error("Upload would exceed Free Tier storage limit")
                    return False
            
            # Upload file
            if object_key is None:
                object_key = Path(file_path).name
            
            self.s3.upload_file(file_path, bucket_name, object_key)
            logger.info(f"Uploaded {file_path} to s3://{bucket_name}/{object_key}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            return False
    
    def download_file(
        self,
        bucket_name: str,
        object_key: str,
        file_path: str
    ) -> bool:
        """
        Download a file from S3.
        
        Args:
            bucket_name: S3 bucket name
            object_key: S3 object key
            file_path: Local file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.s3.download_file(bucket_name, object_key, file_path)
            logger.info(f"Downloaded s3://{bucket_name}/{object_key} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            return False
    
    def delete_file(
        self,
        bucket_name: str,
        object_key: str
    ) -> bool:
        """
        Delete a file from S3.
        
        Args:
            bucket_name: S3 bucket name
            object_key: S3 object key
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.s3.delete_object(Bucket=bucket_name, Key=object_key)
            logger.info(f"Deleted s3://{bucket_name}/{object_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            return False
    
    def list_files(
        self,
        bucket_name: str,
        prefix: str = '',
        suffix: str = ''
    ) -> List[Dict]:
        """
        List files in S3 bucket.
        
        Args:
            bucket_name: S3 bucket name
            prefix: Object key prefix
            suffix: Object key suffix
            
        Returns:
            List of file information dictionaries
        """
        try:
            files = []
            
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['Key'].endswith(suffix):
                            files.append({
                                'key': obj['Key'],
                                'size': obj['Size'],
                                'last_modified': obj['LastModified']
                            })
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return []
    
    def cleanup_old_files(
        self,
        bucket_name: str,
        days: int = 30,
        prefix: str = ''
    ) -> List[str]:
        """
        Clean up old files from S3.
        
        Args:
            bucket_name: S3 bucket name
            days: Age threshold in days
            prefix: Object key prefix
            
        Returns:
            List of deleted object keys
        """
        deleted = []
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        try:
            files = self.list_files(bucket_name, prefix=prefix)
            
            for file in files:
                if file['last_modified'] < cutoff:
                    if self.delete_file(bucket_name, file['key']):
                        deleted.append(file['key'])
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error cleaning up old files: {str(e)}")
            return deleted

if __name__ == "__main__":
    # Example usage
    manager = S3Manager()
    
    # Check storage limits
    bucket_name = "your-bucket-name"
    limits = manager.check_storage_limits(bucket_name)
    print("\nStorage Limits:")
    print(f"  Size: {limits['size_gb']:.2f} GB ({limits['storage_percent']:.1f}% of limit)")
    print(f"  Objects: {limits['object_count']} ({limits['requests_percent']:.1f}% of limit)")
    
    # List files
    files = manager.list_files(bucket_name)
    print("\nFiles in bucket:")
    for file in files:
        print(f"  {file['key']}: {file['size'] / 1024:.1f} KB")
    
    # Clean up old files
    deleted = manager.cleanup_old_files(bucket_name)
    if deleted:
        print(f"\nDeleted {len(deleted)} old files:")
        for key in deleted:
            print(f"  {key}") 