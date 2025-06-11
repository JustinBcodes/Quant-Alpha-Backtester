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

class APIGatewayManager:
    """Class to manage API Gateway within Free Tier limits."""
    
    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region: str = 'us-east-1'
    ):
        """
        Initialize the APIGatewayManager.
        
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
        self.api_gateway = self.session.client('apigateway')
        self.cloudwatch = self.session.client('cloudwatch')
    
    def get_api_usage(self, api_id: str) -> Dict:
        """
        Get API Gateway usage metrics.
        
        Args:
            api_id: API Gateway ID
            
        Returns:
            Dictionary with usage metrics
        """
        try:
            # Get request count
            requests = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/ApiGateway',
                MetricName='Count',
                Dimensions=[
                    {'Name': 'ApiId', 'Value': api_id},
                    {'Name': 'Stage', 'Value': 'prod'}
                ],
                StartTime=datetime.utcnow() - timedelta(days=30),
                EndTime=datetime.utcnow(),
                Period=86400,  # Daily
                Statistics=['Sum']
            )
            
            # Get latency
            latency = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/ApiGateway',
                MetricName='Latency',
                Dimensions=[
                    {'Name': 'ApiId', 'Value': api_id},
                    {'Name': 'Stage', 'Value': 'prod'}
                ],
                StartTime=datetime.utcnow() - timedelta(days=30),
                EndTime=datetime.utcnow(),
                Period=86400,  # Daily
                Statistics=['Average']
            )
            
            # Calculate metrics
            total_requests = sum(point['Sum'] for point in requests['Datapoints'])
            avg_latency = sum(point['Average'] for point in latency['Datapoints']) / len(latency['Datapoints']) if latency['Datapoints'] else 0
            
            # Check against limits
            requests_percent = (total_requests / FREE_TIER_LIMITS['api_gateway']['requests']) * 100
            
            return {
                'total_requests': total_requests,
                'avg_latency_ms': avg_latency,
                'requests_percent': requests_percent,
                'is_safe': requests_percent < 80
            }
            
        except Exception as e:
            logger.error(f"Error getting API usage: {str(e)}")
            return {
                'total_requests': 0,
                'avg_latency_ms': 0,
                'requests_percent': 0,
                'is_safe': True
            }
    
    def create_api(
        self,
        name: str,
        description: str = '',
        endpoint_type: str = 'REGIONAL'
    ) -> Optional[str]:
        """
        Create a new API Gateway.
        
        Args:
            name: API name
            description: API description
            endpoint_type: API endpoint type
            
        Returns:
            Optional[str]: API ID if successful, None otherwise
        """
        try:
            response = self.api_gateway.create_rest_api(
                name=name,
                description=description,
                endpointConfiguration={
                    'types': [endpoint_type]
                }
            )
            
            api_id = response['id']
            logger.info(f"Created API Gateway {name} with ID {api_id}")
            return api_id
            
        except Exception as e:
            logger.error(f"Error creating API: {str(e)}")
            return None
    
    def create_resource(
        self,
        api_id: str,
        parent_id: str,
        path_part: str
    ) -> Optional[str]:
        """
        Create a new API resource.
        
        Args:
            api_id: API ID
            parent_id: Parent resource ID
            path_part: Resource path part
            
        Returns:
            Optional[str]: Resource ID if successful, None otherwise
        """
        try:
            response = self.api_gateway.create_resource(
                restApiId=api_id,
                parentId=parent_id,
                pathPart=path_part
            )
            
            resource_id = response['id']
            logger.info(f"Created resource {path_part} with ID {resource_id}")
            return resource_id
            
        except Exception as e:
            logger.error(f"Error creating resource: {str(e)}")
            return None
    
    def create_method(
        self,
        api_id: str,
        resource_id: str,
        http_method: str,
        authorization_type: str = 'NONE',
        integration_type: str = 'AWS_PROXY',
        integration_uri: Optional[str] = None
    ) -> bool:
        """
        Create a new API method.
        
        Args:
            api_id: API ID
            resource_id: Resource ID
            http_method: HTTP method
            authorization_type: Authorization type
            integration_type: Integration type
            integration_uri: Integration URI
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create method
            self.api_gateway.put_method(
                restApiId=api_id,
                resourceId=resource_id,
                httpMethod=http_method,
                authorizationType=authorization_type
            )
            
            # Create integration if URI provided
            if integration_uri:
                self.api_gateway.put_integration(
                    restApiId=api_id,
                    resourceId=resource_id,
                    httpMethod=http_method,
                    type=integration_type,
                    integrationHttpMethod='POST',
                    uri=integration_uri
                )
            
            logger.info(f"Created {http_method} method for resource {resource_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating method: {str(e)}")
            return False
    
    def deploy_api(
        self,
        api_id: str,
        stage_name: str = 'prod',
        description: str = ''
    ) -> Optional[str]:
        """
        Deploy API to a stage.
        
        Args:
            api_id: API ID
            stage_name: Stage name
            description: Deployment description
            
        Returns:
            Optional[str]: Deployment ID if successful, None otherwise
        """
        try:
            # Create deployment
            response = self.api_gateway.create_deployment(
                restApiId=api_id,
                stageName=stage_name,
                description=description
            )
            
            deployment_id = response['id']
            logger.info(f"Deployed API {api_id} to stage {stage_name}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Error deploying API: {str(e)}")
            return None
    
    def delete_api(self, api_id: str) -> bool:
        """
        Delete an API Gateway.
        
        Args:
            api_id: API ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.api_gateway.delete_rest_api(restApiId=api_id)
            logger.info(f"Deleted API Gateway {api_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting API: {str(e)}")
            return False
    
    def list_apis(self) -> List[Dict]:
        """
        List all API Gateways.
        
        Returns:
            List of API information dictionaries
        """
        try:
            apis = []
            paginator = self.api_gateway.get_paginator('get_rest_apis')
            
            for page in paginator.paginate():
                for api in page['items']:
                    apis.append({
                        'id': api['id'],
                        'name': api['name'],
                        'description': api.get('description', ''),
                        'created_date': api['createdDate']
                    })
            
            return apis
            
        except Exception as e:
            logger.error(f"Error listing APIs: {str(e)}")
            return []

if __name__ == "__main__":
    # Example usage
    manager = APIGatewayManager()
    
    # List APIs
    apis = manager.list_apis()
    print("\nAPI Gateways:")
    for api in apis:
        print(f"  {api['name']} ({api['id']})")
    
    # Check API usage
    api_id = "your-api-id"
    usage = manager.get_api_usage(api_id)
    print(f"\nAPI Usage for {api_id}:")
    print(f"  Requests: {usage['total_requests']} ({usage['requests_percent']:.1f}% of limit)")
    print(f"  Average Latency: {usage['avg_latency_ms']:.1f} ms") 