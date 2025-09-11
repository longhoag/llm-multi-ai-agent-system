#!/usr/bin/env python3
"""
Comprehensive functionality test for the LLM Multi-Agent System
Tests: ChatGPT API, Alpha Vantage API, S3 ingestion, and GPT-5-mini compatibility
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

# Import our components
from src.external.alpha_vantage_client import AlphaVantageClient
from src.storage.s3_manager import S3Manager
from src.tools.agent_tools import initialize_tools, DATA_INGESTION_TOOLS
from src.nodes.workflow_nodes import data_ingestion_node
from src.state.workflow_state import DataIngestionState, WorkflowStatus

# Load environment
load_dotenv()

class ComprehensiveTester:
    """Test suite for the multi-agent system functionality"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        
    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        self.results[test_name] = {"success": success, "details": details}
        
    def log_error(self, test_name: str, error: Exception):
        """Log test error"""
        error_msg = f"{type(error).__name__}: {str(error)}"
        self.log_result(test_name, False, error_msg)
        self.errors.append(f"{test_name}: {error_msg}")
    
    async def test_environment_variables(self):
        """Test 1: Verify all required environment variables"""
        print("\\n🧪 Test 1: Environment Variables")
        print("-" * 40)
        
        required_vars = [
            "OPENAI_API_KEY",
            "ALPHA_VANTAGE_API_KEY", 
            "S3_BUCKET_RAW_DATA",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
                
        if missing_vars:
            self.log_result("Environment Variables", False, f"Missing: {missing_vars}")
            return False
        else:
            self.log_result("Environment Variables", True, "All required variables present")
            return True
    
    async def test_alpha_vantage_api(self):
        """Test 2: Alpha Vantage API connectivity"""
        print("\\n🧪 Test 2: Alpha Vantage API")
        print("-" * 40)
        
        try:
            client = AlphaVantageClient(api_key=os.getenv('ALPHA_VANTAGE_API_KEY'))
            
            # Test API call
            data = await client.get_daily_time_series('AAPL', outputsize='compact')
            
            if data and 'data' in data and len(data['data']) > 0:
                record_count = len(data['data'])
                self.log_result("Alpha Vantage API", True, f"Retrieved {record_count} records for AAPL")
                return True
            else:
                self.log_result("Alpha Vantage API", False, "No data returned")
                return False
                
        except Exception as e:
            self.log_error("Alpha Vantage API", e)
            return False
    
    async def test_s3_connectivity(self):
        """Test 3: S3 connectivity and operations"""
        print("\\n🧪 Test 3: S3 Connectivity")
        print("-" * 40)
        
        try:
            s3_manager = S3Manager(bucket_name=os.getenv('S3_BUCKET_RAW_DATA'))
            
            # Test upload
            test_data = {
                "test": True,
                "timestamp": datetime.now().isoformat(),
                "data": [{"symbol": "TEST", "price": 100.0}]
            }
            
            test_key = f"test_functionality/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            success = await s3_manager.upload_json(test_data, test_key)
            
            if success:
                self.log_result("S3 Upload", True, f"Test file uploaded: {test_key}")
                
                # Test download
                downloaded = await s3_manager.download_json(test_key)
                if downloaded and downloaded.get('test') == True:
                    self.log_result("S3 Download", True, "Test file downloaded and verified")
                    return True
                else:
                    self.log_result("S3 Download", False, "Downloaded data doesn't match")
                    return False
            else:
                self.log_result("S3 Upload", False, "Upload failed")
                return False
                
        except Exception as e:
            self.log_error("S3 Operations", e)
            return False
    
    async def test_openai_api_basic(self):
        """Test 4: Basic OpenAI API connectivity"""
        print("\\n🧪 Test 4: OpenAI API (Basic)")
        print("-" * 40)
        
        try:
            from langchain_openai import ChatOpenAI
            
            # Test with GPT-4o-mini (known to work)
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=os.getenv('OPENAI_API_KEY')
            )
            
            response = llm.invoke("Say 'Hello from GPT-4o-mini'")
            
            if response and hasattr(response, 'content') and 'GPT-4o-mini' in response.content:
                self.log_result("OpenAI API (GPT-4o-mini)", True, f"Response: {response.content[:50]}...")
                return True
            else:
                self.log_result("OpenAI API (GPT-4o-mini)", False, "Unexpected response format")
                return False
                
        except Exception as e:
            self.log_error("OpenAI API (GPT-4o-mini)", e)
            return False
    
    async def test_gpt5_mini_compatibility(self):
        """Test 5: GPT-5-mini compatibility (critical test)"""
        print("\\n🧪 Test 5: GPT-5-mini Compatibility")
        print("-" * 40)
        
        try:
            from langchain_openai import ChatOpenAI
            
            # Test basic GPT-5-mini call
            llm = ChatOpenAI(
                model="gpt-5-mini",
                temperature=0.1,
                api_key=os.getenv('OPENAI_API_KEY')
            )
            
            response = llm.invoke("Say 'Hello from GPT-5-mini'")
            
            if response and hasattr(response, 'content') and 'GPT-5-mini' in response.content:
                self.log_result("GPT-5-mini Basic Call", True, f"Response: {response.content[:50]}...")
                
                # Test with ReAct agent (this might fail)
                try:
                    from langchain.agents import create_react_agent, AgentExecutor
                    from langchain import hub
                    
                    # Simple tools for testing
                    test_tools = [tool for tool in DATA_INGESTION_TOOLS if hasattr(tool, 'name')][:2]
                    
                    prompt = hub.pull("hwchase17/react")
                    agent = create_react_agent(llm, test_tools, prompt)
                    agent_executor = AgentExecutor(
                        agent=agent,
                        tools=test_tools,
                        max_iterations=3,
                        handle_parsing_errors=True
                    )
                    
                    result = agent_executor.invoke({"input": "Just say hello, don't use any tools"})
                    
                    self.log_result("GPT-5-mini ReAct Agent", True, "ReAct agent works with GPT-5-mini")
                    return True
                    
                except Exception as agent_error:
                    error_msg = str(agent_error)
                    if "'stop' is not supported" in error_msg:
                        self.log_result("GPT-5-mini ReAct Agent", False, "GPT-5-mini doesn't support 'stop' parameter for ReAct agents")
                    else:
                        self.log_result("GPT-5-mini ReAct Agent", False, f"Agent error: {error_msg[:100]}...")
                    return False
            else:
                self.log_result("GPT-5-mini Basic Call", False, "Unexpected response format")
                return False
                
        except Exception as e:
            self.log_error("GPT-5-mini Basic Call", e)
            return False
    
    async def test_tools_initialization(self):
        """Test 6: Tools initialization and functionality"""
        print("\\n🧪 Test 6: Tools Initialization")
        print("-" * 40)
        
        try:
            # Initialize tools
            initialize_tools(
                alpha_vantage_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
                s3_bucket=os.getenv('S3_BUCKET_RAW_DATA'),
                openai_key=os.getenv('OPENAI_API_KEY')
            )
            
            # Check available tools
            tool_count = len(DATA_INGESTION_TOOLS)
            tool_names = [getattr(tool, 'name', 'Unknown') for tool in DATA_INGESTION_TOOLS]
            
            # Check for our critical upload tool
            has_upload_tool = any('upload_stock_data_to_s3' in name for name in tool_names)
            
            if has_upload_tool:
                self.log_result("Tools Initialization", True, f"{tool_count} tools loaded including upload_stock_data_to_s3")
                return True
            else:
                self.log_result("Tools Initialization", False, f"Missing upload_stock_data_to_s3 tool. Available: {tool_names}")
                return False
                
        except Exception as e:
            self.log_error("Tools Initialization", e)
            return False
    
    async def test_workflow_execution(self):
        """Test 7: Full workflow execution"""
        print("\\n🧪 Test 7: Workflow Execution")
        print("-" * 40)
        
        try:
            # Create test state
            state = DataIngestionState(
                workflow_id=f'test_workflow_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                correlation_id='test-correlation-123',
                status=WorkflowStatus.RUNNING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                agent_id='data_ingestion_agent',
                node_name='data_ingestion',
                errors=[],
                retry_count=0,
                max_retries=3,
                symbols=['AAPL'],
                timeframe='daily',
                force_refresh=True,
                raw_data={},
                s3_keys=[],
                ingestion_stats={},
                data_quality={},
                validation_passed=False
            )
            
            # Execute workflow node
            result_state = await data_ingestion_node(state)
            
            if result_state['status'] == WorkflowStatus.COMPLETED:
                s3_keys = result_state.get('s3_keys', [])
                key_count = len(s3_keys)
                
                # Check if latest.json was created
                has_latest = any('latest.json' in key for key in s3_keys)
                
                if has_latest:
                    self.log_result("Workflow Execution", True, f"Completed with {key_count} S3 keys including latest.json")
                    return True
                else:
                    self.log_result("Workflow Execution", False, f"Completed but no latest.json file created. Keys: {s3_keys}")
                    return False
            else:
                errors = result_state.get('errors', [])
                self.log_result("Workflow Execution", False, f"Failed with status: {result_state['status']}, errors: {errors}")
                return False
                
        except Exception as e:
            self.log_error("Workflow Execution", e)
            return False
    
    async def run_all_tests(self):
        """Run all tests and provide summary"""
        print("🚀 LLM Multi-Agent System - Comprehensive Functionality Test")
        print("=" * 60)
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Run tests in order
        await self.test_environment_variables()
        await self.test_alpha_vantage_api()
        await self.test_s3_connectivity()
        await self.test_openai_api_basic()
        await self.test_gpt5_mini_compatibility()
        await self.test_tools_initialization()
        await self.test_workflow_execution()
        
        # Summary
        print("\\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.errors:
            print("\\n🚨 Critical Issues Found:")
            for error in self.errors:
                print(f"   • {error}")
        
        # GPT-5-mini specific analysis
        gpt5_basic = self.results.get("GPT-5-mini Basic Call", {}).get("success", False)
        gpt5_react = self.results.get("GPT-5-mini ReAct Agent", {}).get("success", False)
        
        print("\\n🔍 GPT-5-mini Compatibility Analysis:")
        print(f"   • Basic API calls: {'✅ Supported' if gpt5_basic else '❌ Not supported'}")
        print(f"   • ReAct Agent compatibility: {'✅ Supported' if gpt5_react else '❌ Not supported'}")
        
        if gpt5_basic and not gpt5_react:
            print("\\n💡 RECOMMENDATION:")
            print("   GPT-5-mini works for basic calls but NOT for ReAct agents.")
            print("   For LangGraph workflows with agents, use GPT-4o-mini instead.")
            print("   Alternative: Implement custom agent logic without ReAct framework.")
        
        return passed_tests == total_tests

async def main():
    """Main test execution"""
    tester = ComprehensiveTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\\n🎉 All tests passed! System is fully functional.")
        return 0
    else:
        print("\\n⚠️  Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
