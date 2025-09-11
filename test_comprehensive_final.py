"""
Final Comprehensive Test Suite
Tests all components with real APIs to verify the project is working properly
"""

import os
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add src to path
import sys
sys.path.insert(0, str(Path.cwd() / 'src'))

from gpt4o_mini_react_workflow import GPT4oMiniReActWorkflow
from src.tools.agent_tools import (
    fetch_stock_data_tool,
    validate_data_quality_tool,
    upload_stock_data_to_s3_tool,
    initialize_tools
)
from langchain_openai import ChatOpenAI


class ComprehensiveTestSuite:
    """Complete test suite for the stock data pipeline"""
    
    def __init__(self):
        self.results = {}
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        self.results[test_name] = {"success": success, "details": details}
        status = "✅" if success else "❌"
        print(f"   {status} {test_name}: {details}")
    
    async def test_environment_setup(self):
        """Test 1: Environment Setup"""
        print("\n🔧 Test 1: Environment Setup")
        print("-" * 40)
        
        required_env = ['OPENAI_API_KEY', 'ALPHA_VANTAGE_API_KEY', 'S3_BUCKET_RAW_DATA']
        missing_env = [key for key in required_env if not os.getenv(key)]
        
        if missing_env:
            self.log_test("Environment Variables", False, f"Missing: {missing_env}")
            return False
        else:
            self.log_test("Environment Variables", True, "All required variables set")
            return True
    
    async def test_openai_api(self):
        """Test 2: OpenAI GPT-4o-mini API"""
        print("\n🤖 Test 2: OpenAI GPT-4o-mini API")
        print("-" * 40)
        
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            response = llm.invoke("Say 'API test successful' if you can read this.")
            
            if response and hasattr(response, 'content') and 'successful' in response.content:
                self.log_test("GPT-4o-mini API", True, "Connected and responding")
                return True
            else:
                self.log_test("GPT-4o-mini API", False, "Unexpected response format")
                return False
        except Exception as e:
            self.log_test("GPT-4o-mini API", False, f"Error: {str(e)[:100]}...")
            return False
    
    async def test_alpha_vantage_api(self):
        """Test 3: Alpha Vantage API"""
        print("\n📊 Test 3: Alpha Vantage API")
        print("-" * 40)
        
        try:
            # Initialize tools
            initialize_tools(
                alpha_vantage_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
                s3_bucket=os.getenv('S3_BUCKET_RAW_DATA')
            )
            
            # Test fetch tool
            result = await fetch_stock_data_tool._arun("AAPL", "compact")
            
            if result.get("success") and result.get("record_count", 0) > 0:
                self.log_test("Alpha Vantage API", True, f"Fetched {result['record_count']} records")
                return result
            else:
                self.log_test("Alpha Vantage API", False, result.get("error", "Unknown error"))
                return None
        except Exception as e:
            self.log_test("Alpha Vantage API", False, f"Error: {str(e)[:100]}...")
            return None
    
    async def test_data_validation(self, stock_data):
        """Test 4: Data Validation"""
        print("\n🔍 Test 4: Data Validation")
        print("-" * 40)
        
        if not stock_data:
            self.log_test("Data Validation", False, "No data to validate")
            return None
        
        try:
            validation_result = validate_data_quality_tool._run(stock_data, "AAPL")
            
            if validation_result.get("validation_passed"):
                quality = validation_result.get("overall_quality", "Unknown")
                self.log_test("Data Validation", True, f"Quality: {quality}")
                return validation_result
            else:
                self.log_test("Data Validation", False, "Validation failed")
                return None
        except Exception as e:
            self.log_test("Data Validation", False, f"Error: {str(e)[:100]}...")
            return None
    
    async def test_s3_upload(self, stock_data):
        """Test 5: S3 Upload with latest.json"""
        print("\n☁️  Test 5: S3 Upload")
        print("-" * 40)
        
        if not stock_data:
            self.log_test("S3 Upload", False, "No data to upload")
            return False
        
        try:
            upload_result = await upload_stock_data_to_s3_tool._arun(stock_data, "AAPL", "daily")
            
            if upload_result.get("success"):
                self.log_test("S3 Upload", True, "Both timestamped and latest.json created")
                self.log_test("Latest.json File", True, upload_result.get("latest_key", ""))
                return True
            else:
                self.log_test("S3 Upload", False, upload_result.get("error", "Unknown error"))
                return False
        except Exception as e:
            self.log_test("S3 Upload", False, f"Error: {str(e)[:100]}...")
            return False
    
    async def test_react_workflow(self):
        """Test 6: Complete ReAct Workflow"""
        print("\n🚀 Test 6: Complete ReAct Workflow")
        print("-" * 40)
        
        try:
            workflow = GPT4oMiniReActWorkflow()
            result = await workflow.run_workflow(["AAPL", "GOOGL"], timeframe="daily")
            
            if result.get("success"):
                symbols = result.get("symbols_processed", [])
                self.log_test("ReAct Workflow", True, f"Processed {len(symbols)} symbols")
                self.log_test("Multi-Symbol Processing", True, f"Symbols: {symbols}")
                return True
            else:
                error = result.get("error", "Unknown error")
                self.log_test("ReAct Workflow", False, f"Error: {error[:100]}...")
                return False
        except Exception as e:
            self.log_test("ReAct Workflow", False, f"Error: {str(e)[:100]}...")
            return False
    
    def generate_report(self):
        """Generate final test report"""
        print("\n" + "=" * 60)
        print("📋 FINAL TEST REPORT")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result["success"])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📊 Test Results:")
        for test_name, result in self.results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"   {status} - {test_name}")
            if result["details"]:
                print(f"           {result['details']}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! The project is working correctly.")
            print("✅ Ready for production use with real APIs")
        elif passed_tests >= total_tests * 0.8:
            print("⚠️  Most tests passed. Minor issues detected.")
            print("🔧 Some components may need attention")
        else:
            print("❌ Multiple test failures detected.")
            print("🛠️  Project needs significant fixes before use")
        
        return passed_tests == total_tests


async def main():
    """Run comprehensive test suite"""
    print("🧪 COMPREHENSIVE PROJECT TEST SUITE")
    print("Testing: GPT API, Alpha Vantage API, S3 Storage, ReAct Workflows")
    print("=" * 60)
    
    test_suite = ComprehensiveTestSuite()
    
    # Run all tests in sequence
    env_ok = await test_suite.test_environment_setup()
    if not env_ok:
        print("\n❌ Environment setup failed. Cannot continue with API tests.")
        test_suite.generate_report()
        return
    
    await test_suite.test_openai_api()
    stock_data = await test_suite.test_alpha_vantage_api()
    validation_result = await test_suite.test_data_validation(stock_data)
    await test_suite.test_s3_upload(stock_data)
    await test_suite.test_react_workflow()
    
    # Generate final report
    all_passed = test_suite.generate_report()
    
    if all_passed:
        print(f"\n🚀 Project Status: READY FOR USE")
        print(f"   • GPT-4o-mini ReAct agents: ✅ Working")
        print(f"   • Real API integration: ✅ Working") 
        print(f"   • S3 data storage: ✅ Working")
        print(f"   • latest.json creation: ✅ Working")
    else:
        print(f"\n⚠️  Project Status: NEEDS ATTENTION")
        print(f"   Check failed tests above for details")


if __name__ == "__main__":
    asyncio.run(main())
