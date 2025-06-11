import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

from utils.budget_check import BudgetMonitor
from utils.ec2_manager import EC2Manager
from utils.s3_manager import S3Manager
from utils.lambda_manager import LambdaManager
from utils.api_gateway_manager import APIGatewayManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AWSOrchestrator:
    """Class to orchestrate AWS services within Free Tier limits."""
    
    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region: str = 'us-east-1'
    ):
        """
        Initialize the AWSOrchestrator.
        
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
        
        # Initialize service managers
        self.budget_monitor = BudgetMonitor(aws_access_key_id, aws_secret_access_key, region)
        self.ec2_manager = EC2Manager(aws_access_key_id, aws_secret_access_key, region)
        self.s3_manager = S3Manager(aws_access_key_id, aws_secret_access_key, region)
        self.lambda_manager = LambdaManager(aws_access_key_id, aws_secret_access_key, region)
        self.api_gateway_manager = APIGatewayManager(aws_access_key_id, aws_secret_access_key, region)
    
    def check_all_usage(self) -> Dict:
        """
        Check usage across all AWS services.
        
        Returns:
            Dictionary with usage metrics for all services
        """
        try:
            # Get usage metrics from all services
            s3_usage = self.s3_manager.check_storage_limits("your-bucket-name")
            lambda_usage = self.lambda_manager.get_function_usage("your-function-name")
            api_usage = self.api_gateway_manager.get_api_usage("your-api-id")
            ec2_usage = self.ec2_manager.list_instances()
            
            # Calculate overall usage
            total_usage = {
                's3': s3_usage,
                'lambda': lambda_usage,
                'api_gateway': api_usage,
                'ec2': {
                    'running_instances': len([i for i in ec2_usage if i['state'] == 'running']),
                    'total_instances': len(ec2_usage)
                }
            }
            
            # Check if any service is approaching limits
            is_safe = all([
                s3_usage['is_safe'],
                lambda_usage['is_safe'],
                api_usage['is_safe'],
                len([i for i in ec2_usage if i['state'] == 'running']) <= 1
            ])
            
            total_usage['is_safe'] = is_safe
            return total_usage
            
        except Exception as e:
            logger.error(f"Error checking usage: {str(e)}")
            return {'is_safe': False}
    
    def deploy_backtest_pipeline(
        self,
        bucket_name: str,
        function_name: str,
        api_name: str,
        code_path: str
    ) -> Tuple[bool, Dict]:
        """
        Deploy a complete backtest pipeline.
        
        Args:
            bucket_name: S3 bucket name
            function_name: Lambda function name
            api_name: API Gateway name
            code_path: Path to function code
            
        Returns:
            Tuple of (success, deployment info)
        """
        try:
            # Check usage before deployment
            usage = self.check_all_usage()
            if not usage['is_safe']:
                logger.error("Deployment would exceed Free Tier limits")
                return False, {'error': 'Free Tier limits exceeded'}
            
            # Create S3 bucket if needed
            try:
                self.s3_manager.s3.create_bucket(Bucket=bucket_name)
                logger.info(f"Created S3 bucket {bucket_name}")
            except ClientError as e:
                if e.response['Error']['Code'] != 'BucketAlreadyExists':
                    raise
            
            # Create Lambda function
            lambda_role = "arn:aws:iam::your-account-id:role/lambda-role"  # Replace with your role
            lambda_success = self.lambda_manager.create_function(
                function_name=function_name,
                handler="app.handler",
                runtime="python3.9",
                role=lambda_role,
                code_path=code_path
            )
            
            if not lambda_success:
                return False, {'error': 'Failed to create Lambda function'}
            
            # Create API Gateway
            api_id = self.api_gateway_manager.create_api(
                name=api_name,
                description="Backtest API"
            )
            
            if not api_id:
                return False, {'error': 'Failed to create API Gateway'}
            
            # Create API resources and methods
            root_resource_id = self.api_gateway_manager.api_gateway.get_resources(
                restApiId=api_id
            )['items'][0]['id']
            
            resource_id = self.api_gateway_manager.create_resource(
                api_id=api_id,
                parent_id=root_resource_id,
                path_part="backtest"
            )
            
            if not resource_id:
                return False, {'error': 'Failed to create API resource'}
            
            # Create Lambda integration
            lambda_uri = f"arn:aws:apigateway:{self.session.region_name}:lambda:path/2015-03-31/functions/arn:aws:lambda:{self.session.region_name}:your-account-id:function:{function_name}/invocations"
            
            method_success = self.api_gateway_manager.create_method(
                api_id=api_id,
                resource_id=resource_id,
                http_method="POST",
                integration_type="AWS_PROXY",
                integration_uri=lambda_uri
            )
            
            if not method_success:
                return False, {'error': 'Failed to create API method'}
            
            # Deploy API
            deployment_id = self.api_gateway_manager.deploy_api(
                api_id=api_id,
                stage_name="prod"
            )
            
            if not deployment_id:
                return False, {'error': 'Failed to deploy API'}
            
            # Return deployment info
            deployment_info = {
                'bucket_name': bucket_name,
                'function_name': function_name,
                'api_id': api_id,
                'deployment_id': deployment_id,
                'endpoint': f"https://{api_id}.execute-api.{self.session.region_name}.amazonaws.com/prod/backtest"
            }
            
            return True, deployment_info
            
        except Exception as e:
            logger.error(f"Error deploying pipeline: {str(e)}")
            return False, {'error': str(e)}
    
    def cleanup_resources(self) -> Dict:
        """
        Clean up all AWS resources.
        
        Returns:
            Dictionary with cleanup results
        """
        try:
            results = {
                's3': [],
                'lambda': [],
                'api_gateway': [],
                'ec2': []
            }
            
            # Clean up S3 buckets
            buckets = self.s3_manager.s3.list_buckets()['Buckets']
            for bucket in buckets:
                try:
                    # Delete all objects first
                    self.s3_manager.s3.delete_objects(
                        Bucket=bucket['Name'],
                        Delete={'Objects': [{'Key': obj['Key']} for obj in self.s3_manager.s3.list_objects_v2(Bucket=bucket['Name']).get('Contents', [])]}
                    )
                    # Delete bucket
                    self.s3_manager.s3.delete_bucket(Bucket=bucket['Name'])
                    results['s3'].append(bucket['Name'])
                except Exception as e:
                    logger.error(f"Error cleaning up S3 bucket {bucket['Name']}: {str(e)}")
            
            # Clean up Lambda functions
            functions = self.lambda_manager.list_functions()
            for function in functions:
                try:
                    self.lambda_manager.delete_function(function['name'])
                    results['lambda'].append(function['name'])
                except Exception as e:
                    logger.error(f"Error cleaning up Lambda function {function['name']}: {str(e)}")
            
            # Clean up API Gateways
            apis = self.api_gateway_manager.list_apis()
            for api in apis:
                try:
                    self.api_gateway_manager.delete_api(api['id'])
                    results['api_gateway'].append(api['name'])
                except Exception as e:
                    logger.error(f"Error cleaning up API Gateway {api['name']}: {str(e)}")
            
            # Clean up EC2 instances
            instances = self.ec2_manager.list_instances()
            for instance in instances:
                try:
                    self.ec2_manager.terminate_instance(instance['id'])
                    results['ec2'].append(instance['id'])
                except Exception as e:
                    logger.error(f"Error cleaning up EC2 instance {instance['id']}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error cleaning up resources: {str(e)}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Example usage
    orchestrator = AWSOrchestrator()
    
    # Check usage
    usage = orchestrator.check_all_usage()
    print("\nAWS Usage:")
    print(f"  S3: {usage['s3']['storage_percent']:.1f}% storage, {usage['s3']['requests_percent']:.1f}% requests")
    print(f"  Lambda: {usage['lambda']['invocations_percent']:.1f}% invocations")
    print(f"  API Gateway: {usage['api_gateway']['requests_percent']:.1f}% requests")
    print(f"  EC2: {usage['ec2']['running_instances']} running instances")
    print(f"  Overall Status: {'Safe' if usage['is_safe'] else 'Warning'}")
    
    # Deploy pipeline
    success, info = orchestrator.deploy_backtest_pipeline(
        bucket_name="your-bucket",
        function_name="your-function",
        api_name="your-api",
        code_path="path/to/code"
    )
    
    if success:
        print("\nDeployment successful:")
        print(f"  Endpoint: {info['endpoint']}")
    else:
        print(f"\nDeployment failed: {info['error']}")
    
    # Clean up resources
    results = orchestrator.cleanup_resources()
    print("\nCleanup Results:")
    for service, resources in results.items():
        if resources:
            print(f"  {service}: {len(resources)} resources cleaned up") 