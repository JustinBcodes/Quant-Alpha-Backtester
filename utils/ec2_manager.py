import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from utils.budget_check import FREE_TIER_LIMITS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EC2Manager:
    """Class to manage EC2 instances within Free Tier limits."""
    
    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region: str = 'us-east-1'
    ):
        """
        Initialize the EC2Manager.
        
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
        self.ec2 = self.session.client('ec2')
        self.ec2_resource = self.session.resource('ec2')
    
    def list_instances(self) -> List[Dict]:
        """
        List all EC2 instances.
        
        Returns:
            List of instance information dictionaries
        """
        try:
            instances = []
            for instance in self.ec2_resource.instances.all():
                # Get instance details
                instance_info = {
                    'id': instance.id,
                    'type': instance.instance_type,
                    'state': instance.state['Name'],
                    'launch_time': instance.launch_time,
                    'running_hours': self._get_running_hours(instance),
                    'is_free_tier': instance.instance_type in FREE_TIER_LIMITS['ec2']['instance_type']
                }
                instances.append(instance_info)
            
            return instances
            
        except Exception as e:
            logger.error(f"Error listing instances: {str(e)}")
            return []
    
    def _get_running_hours(self, instance) -> float:
        """
        Calculate running hours for an instance.
        
        Args:
            instance: EC2 instance object
            
        Returns:
            Running hours
        """
        if instance.state['Name'] == 'running':
            return (datetime.utcnow() - instance.launch_time).total_seconds() / 3600
        return 0
    
    def create_instance(
        self,
        instance_type: str = 't2.micro',
        image_id: str = 'ami-0c55b159cbfafe1f0',  # Amazon Linux 2
        key_name: Optional[str] = None,
        security_group_ids: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Create a new EC2 instance.
        
        Args:
            instance_type: EC2 instance type
            image_id: AMI ID
            key_name: SSH key pair name
            security_group_ids: List of security group IDs
            
        Returns:
            Instance ID if successful, None otherwise
        """
        # Check if instance type is Free Tier compliant
        if instance_type not in FREE_TIER_LIMITS['ec2']['instance_type']:
            logger.error(f"Instance type {instance_type} is not Free Tier compliant")
            return None
        
        try:
            # Create instance
            response = self.ec2.run_instances(
                ImageId=image_id,
                InstanceType=instance_type,
                MinCount=1,
                MaxCount=1,
                KeyName=key_name,
                SecurityGroupIds=security_group_ids or [],
                TagSpecifications=[
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {
                                'Key': 'Name',
                                'Value': 'quant-alpha-instance'
                            },
                            {
                                'Key': 'Project',
                                'Value': 'quant-alpha-aws'
                            }
                        ]
                    }
                ]
            )
            
            instance_id = response['Instances'][0]['InstanceId']
            logger.info(f"Created instance {instance_id}")
            
            return instance_id
            
        except Exception as e:
            logger.error(f"Error creating instance: {str(e)}")
            return None
    
    def stop_instance(self, instance_id: str) -> bool:
        """
        Stop an EC2 instance.
        
        Args:
            instance_id: EC2 instance ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.ec2.stop_instances(InstanceIds=[instance_id])
            logger.info(f"Stopped instance {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping instance {instance_id}: {str(e)}")
            return False
    
    def terminate_instance(self, instance_id: str) -> bool:
        """
        Terminate an EC2 instance.
        
        Args:
            instance_id: EC2 instance ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.ec2.terminate_instances(InstanceIds=[instance_id])
            logger.info(f"Terminated instance {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error terminating instance {instance_id}: {str(e)}")
            return False
    
    def cleanup_idle_instances(self, max_idle_hours: float = 1.0) -> List[str]:
        """
        Clean up idle instances.
        
        Args:
            max_idle_hours: Maximum idle hours before cleanup
            
        Returns:
            List of terminated instance IDs
        """
        terminated = []
        
        try:
            instances = self.list_instances()
            
            for instance in instances:
                if instance['state'] == 'running' and instance['running_hours'] > max_idle_hours:
                    if self.terminate_instance(instance['id']):
                        terminated.append(instance['id'])
            
            return terminated
            
        except Exception as e:
            logger.error(f"Error cleaning up idle instances: {str(e)}")
            return terminated
    
    def get_instance_status(self, instance_id: str) -> Optional[Dict]:
        """
        Get instance status.
        
        Args:
            instance_id: EC2 instance ID
            
        Returns:
            Dictionary with instance status
        """
        try:
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            instance = response['Reservations'][0]['Instances'][0]
            
            return {
                'id': instance['InstanceId'],
                'state': instance['State']['Name'],
                'type': instance['InstanceType'],
                'launch_time': instance['LaunchTime'],
                'running_hours': self._get_running_hours(instance),
                'is_free_tier': instance['InstanceType'] in FREE_TIER_LIMITS['ec2']['instance_type']
            }
            
        except Exception as e:
            logger.error(f"Error getting instance status: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage
    manager = EC2Manager()
    
    # List instances
    instances = manager.list_instances()
    print("\nCurrent Instances:")
    for instance in instances:
        print(f"  {instance['id']}: {instance['type']} ({instance['state']})")
    
    # Clean up idle instances
    terminated = manager.cleanup_idle_instances()
    if terminated:
        print(f"\nTerminated {len(terminated)} idle instances:")
        for instance_id in terminated:
            print(f"  {instance_id}") 