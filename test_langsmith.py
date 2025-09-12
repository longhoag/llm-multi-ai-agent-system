#!/usr/bin/env python3
"""
Test LangSmith integration with simple agent
"""

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import warnings
from loguru import logger
from langchain._api import LangChainDeprecationWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", message=".*pydantic.*")
warnings.filterwarnings("ignore", message=".*langchain_core.pydantic_v1.*")

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)

# 🔍 LangSmith Configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "llm-multi-agent-stock-processing"

# Verify LangSmith configuration
langsmith_api_key = os.getenv('LANGCHAIN_API_KEY')
if langsmith_api_key:
    logger.success(f"✅ LangSmith API Key loaded: {langsmith_api_key[:8]}...")
    logger.success(f"✅ LangSmith Tracing: {os.environ.get('LANGCHAIN_TRACING_V2')}")
    logger.success(f"✅ LangSmith Project: {os.environ.get('LANGCHAIN_PROJECT')}")
else:
    logger.error("❌ LangSmith API Key not found in environment")
    logger.info("🔧 Please add LANGCHAIN_API_KEY to your .env file")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub

def simple_tool(query: str) -> str:
    """Simple test tool that just echoes the input"""
    return f"Processed: {query}"

def test_langsmith_tracing():
    """Test LangSmith tracing with simple agent"""
    logger.info("🧪 Testing LangSmith tracing...")
    
    try:
        # Create LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        # Create simple tool
        tools = [
            Tool(
                name="simple_tool",
                func=simple_tool,
                description="A simple test tool that processes input"
            )
        ]
        
        # Create prompt
        prompt = PromptTemplate(
            template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""",
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
        )
        
        # Create agent
        agent = create_react_agent(llm, tools, prompt)
        
        # Create agent executor with LangSmith tags
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            tags=["test", "langsmith-verification"]
        )
        
        # Test the agent - this should show up in LangSmith!
        logger.info("🚀 Running test agent (check LangSmith for traces)...")
        result = agent_executor.invoke({
            "input": "Use the simple_tool to process the text 'Hello LangSmith!'"
        })
        
        logger.success("✅ Agent test completed!")
        logger.info(f"🤖 Agent result: {result.get('output', 'No output')}")
        logger.info("🔍 Check LangSmith dashboard for trace visibility!")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 LangSmith test failed: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("🔍 LangSmith Integration Test")
    logger.info("=" * 50)
    
    success = test_langsmith_tracing()
    
    if success:
        logger.success("🎉 LangSmith test completed successfully!")
        logger.info("📊 Check your LangSmith dashboard at: https://smith.langchain.com/")
        logger.info("🔍 Look for project: llm-multi-agent-stock-processing")
    else:
        logger.error("❌ LangSmith test failed")
        sys.exit(1)
