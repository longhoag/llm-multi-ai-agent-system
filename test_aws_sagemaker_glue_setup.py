"""
AWS SageMaker and Glue Setup Verification Test
Tests all AWS services configured for LangGraph preprocessing node
"""

import os
import boto3
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from botocore.exceptions import ClientError, NoCredentialsError

# Load environment
load_dotenv()

class AWSServicesTestSuite:
    """Test suite for AWS SageMaker and Glue setup verification"""
    
    def __init__(self):
        self.results = {}
        self.aws_region = os.getenv('AWS_REGION')
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        self.results[test_name] = {"success": success, "details": details}
        status = "✅" if success else "❌"
        print(f"   {status} {test_name}: {details}")
    
    def test_aws_credentials(self):
        """Test 1: AWS Credentials and Basic Connectivity"""
        print("\n🔐 Test 1: AWS Credentials")
        print("-" * 40)
        
        try:
            # Test basic AWS connectivity
            sts = boto3.client(
                'sts',
                region_name=self.aws_region,
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key
            )
            
            # Get caller identity
            identity = sts.get_caller_identity()
            account_id = identity['Account']
            user_arn = identity['Arn']
            
            self.log_test("AWS Credentials", True, f"Account: {account_id}")
            self.log_test("AWS Identity", True, f"User: {user_arn}")
            
            return True
            
        except NoCredentialsError:
            self.log_test("AWS Credentials", False, "Invalid or missing credentials")
            return False
        except Exception as e:
            self.log_test("AWS Credentials", False, f"Error: {str(e)}")
            return False
    
    def test_s3_buckets(self):
        """Test 2: S3 Buckets Access"""
        print("\n🪣 Test 2: S3 Buckets")
        print("-" * 40)
        
        try:
            s3 = boto3.client(
                's3',
                region_name=self.aws_region,
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key
            )
            
            # Test buckets from .env
            buckets_to_test = [
                os.getenv('S3_BUCKET_RAW_DATA'),
                os.getenv('S3_BUCKET_PROCESSED'),
                os.getenv('S3_BUCKET_MODELS'),
                os.getenv('S3_BUCKET_GLUE_SCRIPTS'),
                os.getenv('S3_BUCKET_GLUE_TEMP')
            ]
            
            existing_buckets = [bucket['Name'] for bucket in s3.list_buckets()['Buckets']]
            
            for bucket_name in buckets_to_test:
                if bucket_name:
                    if bucket_name in existing_buckets:
                        # Test bucket access by trying to list objects
                        try:
                            s3.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
                            self.log_test(f"S3 Bucket: {bucket_name}", True, "Accessible")
                        except ClientError as e:
                            if e.response['Error']['Code'] == 'NoSuchBucket':
                                self.log_test(f"S3 Bucket: {bucket_name}", False, "Bucket does not exist")
                            else:
                                self.log_test(f"S3 Bucket: {bucket_name}", False, f"Access denied: {e}")
                    else:
                        self.log_test(f"S3 Bucket: {bucket_name}", False, "Bucket not found")
            
            return True
            
        except Exception as e:
            self.log_test("S3 Access", False, f"Error: {str(e)}")
            return False
    
    def test_sagemaker_access(self):
        """Test 3: SageMaker Access and Role"""
        print("\n🤖 Test 3: SageMaker Setup")
        print("-" * 40)
        
        try:
            sagemaker = boto3.client(
                'sagemaker',
                region_name=self.aws_region,
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key
            )
            
            # Test SageMaker API access
            response = sagemaker.list_processing_jobs(MaxResults=1)
            self.log_test("SageMaker API", True, f"Connected to SageMaker in {self.aws_region}")
            
            # Test SageMaker role
            sagemaker_role_arn = os.getenv('SAGEMAKER_ROLE_ARN')
            if sagemaker_role_arn:
                # Test if role exists and is accessible
                iam = boto3.client(
                    'iam',
                    region_name=self.aws_region,
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key
                )
                
                try:
                    role_name = sagemaker_role_arn.split('/')[-1]
                    role = iam.get_role(RoleName=role_name)
                    self.log_test("SageMaker Role", True, f"Role exists: {role_name}")
                except ClientError:
                    self.log_test("SageMaker Role", False, f"Role not accessible: {role_name}")
            else:
                self.log_test("SageMaker Role", False, "SAGEMAKER_ROLE_ARN not set")
            
            # Test instance type availability
            instance_type = os.getenv('SAGEMAKER_INSTANCE_TYPE', 'ml.m5.large')
            self.log_test("Instance Type", True, f"Configured: {instance_type}")
            
            return True
            
        except Exception as e:
            self.log_test("SageMaker Access", False, f"Error: {str(e)}")
            return False
    
    def test_glue_access(self):
        """Test 4: AWS Glue Access and Database"""
        print("\n🔧 Test 4: AWS Glue Setup")
        print("-" * 40)
        
        try:
            glue = boto3.client(
                'glue',
                region_name=self.aws_region,
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key
            )
            
            # Test Glue API access
            databases = glue.get_databases()
            self.log_test("Glue API", True, f"Connected to Glue in {self.aws_region}")
            
            # Test specific database
            database_name = os.getenv('GLUE_DATABASE_NAME', 'stock_prediction_db')
            database_names = [db['Name'] for db in databases['DatabaseList']]
            
            if database_name in database_names:
                # Get database details
                db_detail = glue.get_database(Name=database_name)
                self.log_test("Glue Database", True, f"Database '{database_name}' exists")
            else:
                self.log_test("Glue Database", False, f"Database '{database_name}' not found")
                print(f"   📋 Available databases: {database_names}")
            
            # Test Glue role
            glue_role_arn = os.getenv('GLUE_ROLE_ARN')
            if glue_role_arn:
                iam = boto3.client(
                    'iam',
                    region_name=self.aws_region,
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key
                )
                
                try:
                    role_name = glue_role_arn.split('/')[-1]
                    role = iam.get_role(RoleName=role_name)
                    self.log_test("Glue Role", True, f"Role exists: {role_name}")
                except ClientError:
                    self.log_test("Glue Role", False, f"Role not accessible: {role_name}")
            else:
                self.log_test("Glue Role", False, "GLUE_ROLE_ARN not set")
            
            return True
            
        except Exception as e:
            self.log_test("Glue Access", False, f"Error: {str(e)}")
            return False
    
    def test_environment_variables(self):
        """Test 5: Environment Variables Completeness"""
        print("\n📋 Test 5: Environment Variables")
        print("-" * 40)
        
        required_vars = [
            'AWS_REGION',
            'AWS_ACCESS_KEY_ID', 
            'AWS_SECRET_ACCESS_KEY',
            'S3_BUCKET_PROCESSED',
            'SAGEMAKER_ROLE_ARN',
            'GLUE_ROLE_ARN',
            'GLUE_DATABASE_NAME'
        ]
        
        missing_vars = []
        for var in required_vars:
            value = os.getenv(var)
            if value:
                self.log_test(f"ENV: {var}", True, "Set")
            else:
                missing_vars.append(var)
                self.log_test(f"ENV: {var}", False, "Missing")
        
        if not missing_vars:
            self.log_test("Environment Complete", True, "All required variables set")
            return True
        else:
            self.log_test("Environment Complete", False, f"Missing: {missing_vars}")
            return False
    
    def test_preprocessing_node_readiness(self):
        """Test 6: Preprocessing Node Readiness"""
        print("\n🚀 Test 6: LangGraph Preprocessing Node Readiness")
        print("-" * 40)
        
        try:
            # Test that we can import all necessary components
            import sys
            sys.path.insert(0, str(Path.cwd() / 'src'))
            
            # Test imports for preprocessing node
            from src.nodes.workflow_nodes import preprocessing_node
            self.log_test("Preprocessing Node Import", True, "Can import preprocessing_node")
            
            # Test tool imports
            from src.tools.agent_tools import PREPROCESSING_TOOLS
            self.log_test("Preprocessing Tools", True, f"Found {len(PREPROCESSING_TOOLS)} tools")
            
            # Test state import
            from src.state.workflow_state import StockPredictionWorkflowState
            self.log_test("Workflow State", True, "Can import workflow state")
            
            return True
            
        except ImportError as e:
            self.log_test("Module Import", False, f"Import error: {e}")
            return False
        except Exception as e:
            self.log_test("Preprocessing Readiness", False, f"Error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all AWS setup tests"""
        print("🧪 AWS SAGEMAKER & GLUE SETUP VERIFICATION")
        print("Testing: SageMaker, Glue, S3, IAM Roles for LangGraph Preprocessing Node")
        print("=" * 60)
        
        # Run all tests
        test_methods = [
            self.test_environment_variables,
            self.test_aws_credentials,
            self.test_s3_buckets,
            self.test_sagemaker_access,
            self.test_glue_access,
            self.test_preprocessing_node_readiness
        ]
        
        all_passed = True
        for test_method in test_methods:
            success = test_method()
            if not success:
                all_passed = False
        
        # Final report
        print("\n" + "=" * 60)
        print("📋 FINAL AWS SETUP VERIFICATION REPORT")
        print("=" * 60)
        
        passed_tests = sum(1 for result in self.results.values() if result["success"])
        total_tests = len(self.results)
        success_rate = (passed_tests / total_tests) * 100
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        print(f"\n📊 Test Results:")
        for test_name, result in self.results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"   {status} - {test_name}")
            if result["details"]:
                print(f"           {result['details']}")
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if all_passed:
            print("🎉 ALL TESTS PASSED! AWS setup is ready for LangGraph preprocessing node.")
            print("✅ SageMaker and Glue integration ready")
            print("✅ Can proceed with preprocessing node implementation")
        else:
            print("⚠️  Some tests failed. Please fix the issues above before proceeding.")
            print("🔧 Check your .env file and AWS setup")
        
        return all_passed


async def main():
    """Main test execution function"""
    test_suite = AWSServicesTestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
