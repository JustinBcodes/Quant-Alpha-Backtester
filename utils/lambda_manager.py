import logging
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
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

class LambdaManager:
    """Class to manage AWS Lambda functions within Free Tier limits."""
    
    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region: str = 'us-east-1'
    ):
        """
        Initialize the LambdaManager.
        
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
        self.lambda_client = self.session.client('lambda')
        self.cloudwatch = self.session.client('cloudwatch')
    
    def get_function_usage(self, function_name: str) -> Dict:
        """
        Get Lambda function usage metrics.
        
        Args:
            function_name: Lambda function name
            
        Returns:
            Dictionary with usage metrics
        """
        try:
            # Get invocation count
            invocations = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Invocations',
                Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
                StartTime=datetime.utcnow() - timedelta(days=30),
                EndTime=datetime.utcnow(),
                Period=86400,  # Daily
                Statistics=['Sum']
            )
            
            # Get duration
            duration = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Duration',
                Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
                StartTime=datetime.utcnow() - timedelta(days=30),
                EndTime=datetime.utcnow(),
                Period=86400,  # Daily
                Statistics=['Average']
            )
            
            # Calculate metrics
            total_invocations = sum(point['Sum'] for point in invocations['Datapoints'])
            avg_duration = sum(point['Average'] for point in duration['Datapoints']) / len(duration['Datapoints']) if duration['Datapoints'] else 0
            
            # Check against limits
            invocations_percent = (total_invocations / FREE_TIER_LIMITS['lambda']['invocations']) * 100
            
            return {
                'total_invocations': total_invocations,
                'avg_duration_ms': avg_duration,
                'invocations_percent': invocations_percent,
                'is_safe': invocations_percent < 80
            }
            
        except Exception as e:
            logger.error(f"Error getting function usage: {str(e)}")
            return {
                'total_invocations': 0,
                'avg_duration_ms': 0,
                'invocations_percent': 0,
                'is_safe': True
            }
    
    def create_function(
        self,
        function_name: str,
        handler: str,
        runtime: str,
        role: str,
        code_path: str,
        timeout: int = 30,
        memory_size: int = 128,
        environment: Optional[Dict] = None
    ) -> bool:
        """
        Create a new Lambda function.
        
        Args:
            function_name: Function name
            handler: Function handler
            runtime: Runtime (e.g., 'python3.9')
            role: IAM role ARN
            code_path: Path to function code
            timeout: Function timeout in seconds
            memory_size: Memory allocation in MB
            environment: Environment variables
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if function exists
            try:
                self.lambda_client.get_function(FunctionName=function_name)
                logger.error(f"Function {function_name} already exists")
                return False
            except ClientError:
                pass
            
            # Create deployment package
            zip_path = f"{function_name}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(code_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, code_path)
                        zipf.write(file_path, arcname)
            
            # Read deployment package
            with open(zip_path, 'rb') as f:
                code_bytes = f.read()
            
            # Create function
            self.lambda_client.create_function(
                FunctionName=function_name,
                Runtime=runtime,
                Role=role,
                Handler=handler,
                Code={'ZipFile': code_bytes},
                Timeout=timeout,
                MemorySize=memory_size,
                Environment={'Variables': environment} if environment else None
            )
            
            # Clean up
            os.remove(zip_path)
            
            logger.info(f"Created Lambda function {function_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating function: {str(e)}")
            return False
    
    def update_function(
        self,
        function_name: str,
        code_path: str,
        environment: Optional[Dict] = None
    ) -> bool:
        """
        Update Lambda function code and configuration.
        
        Args:
            function_name: Function name
            code_path: Path to updated code
            environment: Updated environment variables
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create deployment package
            zip_path = f"{function_name}_update.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(code_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, code_path)
                        zipf.write(file_path, arcname)
            
            # Read deployment package
            with open(zip_path, 'rb') as f:
                code_bytes = f.read()
            
            # Update function code
            self.lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=code_bytes
            )
            
            # Update configuration if needed
            if environment:
                self.lambda_client.update_function_configuration(
                    FunctionName=function_name,
                    Environment={'Variables': environment}
                )
            
            # Clean up
            os.remove(zip_path)
            
            logger.info(f"Updated Lambda function {function_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating function: {str(e)}")
            return False
    
    def delete_function(self, function_name: str) -> bool:
        """
        Delete a Lambda function.
        
        Args:
            function_name: Function name
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.lambda_client.delete_function(FunctionName=function_name)
            logger.info(f"Deleted Lambda function {function_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting function: {str(e)}")
            return False
    
    def invoke_function(
        self,
        function_name: str,
        payload: Dict,
        invocation_type: str = 'RequestResponse'
    ) -> Optional[Dict]:
        """
        Invoke a Lambda function.
        
        Args:
            function_name: Function name
            payload: Function input payload
            invocation_type: Invocation type
            
        Returns:
            Optional[Dict]: Function response
        """
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType=invocation_type,
                Payload=payload
            )
            
            if invocation_type == 'RequestResponse':
                return response['Payload'].read()
            return None
            
        except Exception as e:
            logger.error(f"Error invoking function: {str(e)}")
            return None
    
    def list_functions(self) -> List[Dict]:
        """
        List all Lambda functions.
        
        Returns:
            List of function information dictionaries
        """
        try:
            functions = []
            paginator = self.lambda_client.get_paginator('list_functions')
            
            for page in paginator.paginate():
                for function in page['Functions']:
                    functions.append({
                        'name': function['FunctionName'],
                        'runtime': function['Runtime'],
                        'memory_size': function['MemorySize'],
                        'timeout': function['Timeout'],
                        'last_modified': function['LastModified']
                    })
            
            return functions
            
        except Exception as e:
            logger.error(f"Error listing functions: {str(e)}")
            return []

if __name__ == "__main__":
    # Example usage
    manager = LambdaManager()
    
    # List functions
    functions = manager.list_functions()
    print("\nLambda Functions:")
    for function in functions:
        print(f"  {function['name']}: {function['runtime']}")
    
    # Check function usage
    function_name = "your-function-name"
    usage = manager.get_function_usage(function_name)
    print(f"\nFunction Usage for {function_name}:")
    print(f"  Invocations: {usage['total_invocations']} ({usage['invocations_percent']:.1f}% of limit)")
    print(f"  Average Duration: {usage['avg_duration_ms']:.1f} ms") 