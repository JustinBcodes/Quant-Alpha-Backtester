import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Free Tier Limits
FREE_TIER_LIMITS = {
    's3': {
        'storage_gb': 5,
        'requests': 20000,
        'data_transfer_gb': 1
    },
    'lambda': {
        'requests': 1000000,
        'compute_seconds': 400000
    },
    'ec2': {
        'hours': 750,
        'instance_type': ['t2.micro', 't3.micro']
    },
    'api_gateway': {
        'requests': 1000000
    }
}

class BudgetMonitor:
    """Class to monitor AWS Free Tier usage."""
    
    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region: str = 'us-east-1'
    ):
        """
        Initialize the BudgetMonitor.
        
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
        
        # Initialize AWS clients
        self.s3 = self.session.client('s3')
        self.lambda_client = self.session.client('lambda')
        self.ec2 = self.session.client('ec2')
        self.cloudwatch = self.session.client('cloudwatch')
        
    def check_s3_usage(self) -> Dict:
        """
        Check S3 storage usage.
        
        Returns:
            Dictionary with usage metrics
        """
        try:
            # Get bucket sizes
            total_size = 0
            total_objects = 0
            
            for bucket in self.s3.list_buckets()['Buckets']:
                bucket_name = bucket['Name']
                
                # Get bucket size
                paginator = self.s3.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=bucket_name):
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            total_size += obj['Size']
                            total_objects += 1
            
            # Convert to GB
            size_gb = total_size / (1024 * 1024 * 1024)
            
            # Check against limits
            storage_percent = (size_gb / FREE_TIER_LIMITS['s3']['storage_gb']) * 100
            requests_percent = (total_objects / FREE_TIER_LIMITS['s3']['requests']) * 100
            
            return {
                'storage_gb': size_gb,
                'storage_percent': storage_percent,
                'total_objects': total_objects,
                'requests_percent': requests_percent,
                'is_safe': storage_percent < 80 and requests_percent < 80
            }
            
        except Exception as e:
            logger.error(f"Error checking S3 usage: {str(e)}")
            return {'error': str(e)}
    
    def check_lambda_usage(self) -> Dict:
        """
        Check Lambda usage.
        
        Returns:
            Dictionary with usage metrics
        """
        try:
            # Get Lambda metrics from CloudWatch
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)
            
            # Get invocation count
            invocations = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Invocations',
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,
                Statistics=['Sum']
            )
            
            total_invocations = sum(point['Sum'] for point in invocations['Datapoints'])
            invocations_percent = (total_invocations / FREE_TIER_LIMITS['lambda']['requests']) * 100
            
            return {
                'total_invocations': total_invocations,
                'invocations_percent': invocations_percent,
                'is_safe': invocations_percent < 80
            }
            
        except Exception as e:
            logger.error(f"Error checking Lambda usage: {str(e)}")
            return {'error': str(e)}
    
    def check_ec2_usage(self) -> Dict:
        """
        Check EC2 usage.
        
        Returns:
            Dictionary with usage metrics
        """
        try:
            # Get running instances
            instances = self.ec2.describe_instances()
            
            total_hours = 0
            non_compliant = []
            
            for reservation in instances['Reservations']:
                for instance in reservation['Instances']:
                    # Check instance type
                    if instance['InstanceType'] not in FREE_TIER_LIMITS['ec2']['instance_type']:
                        non_compliant.append(instance['InstanceId'])
                    
                    # Calculate running hours
                    launch_time = instance['LaunchTime']
                    if instance['State']['Name'] == 'running':
                        hours = (datetime.utcnow() - launch_time).total_seconds() / 3600
                        total_hours += hours
            
            hours_percent = (total_hours / FREE_TIER_LIMITS['ec2']['hours']) * 100
            
            return {
                'total_hours': total_hours,
                'hours_percent': hours_percent,
                'non_compliant_instances': non_compliant,
                'is_safe': hours_percent < 80 and not non_compliant
            }
            
        except Exception as e:
            logger.error(f"Error checking EC2 usage: {str(e)}")
            return {'error': str(e)}
    
    def check_all_usage(self) -> Dict:
        """
        Check all AWS service usage.
        
        Returns:
            Dictionary with usage metrics for all services
        """
        return {
            's3': self.check_s3_usage(),
            'lambda': self.check_lambda_usage(),
            'ec2': self.check_ec2_usage()
        }
    
    def log_usage(self, log_file: str = 'cloudwatch.log') -> None:
        """
        Log usage metrics to file.
        
        Args:
            log_file: Path to log file
        """
        usage = self.check_all_usage()
        
        with open(log_file, 'a') as f:
            f.write(f"\n=== AWS Usage Report - {datetime.now()} ===\n")
            
            for service, metrics in usage.items():
                f.write(f"\n{service.upper()}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value}\n")
            
            f.write("\n" + "="*50 + "\n")

if __name__ == "__main__":
    # Example usage
    monitor = BudgetMonitor()
    
    # Check all usage
    usage = monitor.check_all_usage()
    print("\nAWS Usage Report:")
    for service, metrics in usage.items():
        print(f"\n{service.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    # Log usage
    monitor.log_usage() 