"""
GPT-4o-mini ReAct Agent Workflow for Stock Data Pipeline
Replaces GPT-5-mini direct tool calling with proper LangGraph ReAct agents
"""

import os
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List

# Load environment
load_dotenv()

# Add src to path  
import sys
sys.path.insert(0, str(Path.cwd() / 'src'))

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from src.tools.agent_tools import (
    DATA_INGESTION_TOOLS,
    initialize_tools
)
from src.state.workflow_state import StockPredictionWorkflowState


class GPT4oMiniReActWorkflow:
    """Stock data workflow using GPT-4o-mini ReAct agents with LangGraph"""
    
    def __init__(self):
        """Initialize workflow with GPT-4o-mini ReAct agents"""
        # Initialize tools
        initialize_tools(
            alpha_vantage_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
            s3_bucket=os.getenv('S3_BUCKET_RAW_DATA'),
            openai_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Create GPT-4o-mini LLM for ReAct agents
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.7,  # ReAct agents work well with moderate temperature
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Create ReAct agents for different workflow stages
        self.data_ingestion_agent = create_react_agent(
            self.llm, 
            DATA_INGESTION_TOOLS,
            state_modifier="You are a stock data ingestion specialist. Use tools to fetch, validate, and store stock data."
        )
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow with ReAct agents"""
        workflow = StateGraph(StockPredictionWorkflowState)
        
        # Add nodes with unique names that don't conflict with state keys
        workflow.add_node("data_ingestion", self._data_ingestion_node)
        workflow.add_node("data_preprocessing", self._preprocessing_node)  
        workflow.add_node("model_training", self._training_node)
        
        # Add edges
        workflow.add_edge(START, "data_ingestion")
        workflow.add_edge("data_ingestion", "data_preprocessing")
        workflow.add_edge("data_preprocessing", "model_training")
        workflow.add_edge("model_training", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    async def _data_ingestion_node(self, state: StockPredictionWorkflowState) -> StockPredictionWorkflowState:
        """Data ingestion node using GPT-4o-mini ReAct agent"""
        print(f"🚀 Starting ReAct data ingestion for symbols: {state['symbols']}")
        
        try:
            symbols = state["symbols"]
            timeframe = state.get("timeframe", "daily")
            
            # Create ReAct agent prompt
            agent_prompt = f"""
            You are a stock data ingestion specialist. Your task is to fetch and store stock market data for multiple symbols.
            
            **Current Task:**
            - Symbols to process: {symbols}
            - Timeframe: {timeframe}
            - Workflow ID: {state['workflow_id']}
            
            **Instructions:**
            For each symbol in the list, perform these steps in order:
            
            1. **Fetch Data**: Use the fetch_stock_data tool to get stock data
            2. **Validate Quality**: Use the validate_data_quality tool to check data quality  
            3. **Upload to S3**: Use the upload_stock_data_to_s3 tool to store the data
            
            **Important Notes:**
            - Process each symbol completely before moving to the next
            - The upload_stock_data_to_s3 tool creates both timestamped and latest.json files automatically
            - Provide a summary of what was accomplished for each symbol
            - If any step fails, note the failure but continue with remaining symbols
            
            Start with the first symbol: {symbols[0] if symbols else 'N/A'}
            """
            
            # Execute ReAct agent
            result = await self.data_ingestion_agent.ainvoke({
                "messages": [{"role": "user", "content": agent_prompt}]
            })
            
            # Extract information from agent result
            ingestion_results = self._parse_agent_results(result, symbols)
            
            # Update state
            updated_state = state.copy()
            updated_state.update({
                "raw_data": ingestion_results.get("raw_data", {}),
                "s3_keys": ingestion_results.get("s3_keys", []),
                "ingestion_summary": ingestion_results.get("summary", ""),
                "node_status": {
                    **state.get("node_status", {}),
                    "ingestion": "completed"
                }
            })
            
            print(f"✅ ReAct data ingestion completed for {len(symbols)} symbols")
            return updated_state
            
        except Exception as e:
            print(f"❌ ReAct data ingestion failed: {e}")
            updated_state = state.copy()
            updated_state.update({
                "node_status": {
                    **state.get("node_status", {}),
                    "ingestion": "failed"
                },
                "errors": state.get("errors", []) + [f"Ingestion error: {str(e)}"]
            })
            return updated_state
    
    async def _preprocessing_node(self, state: StockPredictionWorkflowState) -> StockPredictionWorkflowState:
        """Preprocessing node using GPT-4o-mini ReAct agent"""
        print("🔄 Starting ReAct preprocessing...")
        
        # For now, just pass through - implement preprocessing ReAct agent later
        updated_state = state.copy()
        updated_state.update({
            "processed_data": state.get("raw_data", {}),
            "node_status": {
                **state.get("node_status", {}),
                "preprocessing": "completed"
            }
        })
        
        print("✅ ReAct preprocessing completed (pass-through)")
        return updated_state
    
    async def _training_node(self, state: StockPredictionWorkflowState) -> StockPredictionWorkflowState:
        """Training node using GPT-4o-mini ReAct agent"""
        print("🎯 Starting ReAct training...")
        
        # For now, just pass through - implement training ReAct agent later
        updated_state = state.copy()
        updated_state.update({
            "model_results": {"status": "completed", "model_type": "placeholder"},
            "node_status": {
                **state.get("node_status", {}),
                "training": "completed"
            }
        })
        
        print("✅ ReAct training completed (pass-through)")
        return updated_state
    
    def _parse_agent_results(self, result: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
        """Parse ReAct agent results into structured format"""
        try:
            # Extract messages from ReAct agent result
            messages = result.get("messages", [])
            if not messages:
                return {"raw_data": {}, "s3_keys": [], "summary": "No agent messages found"}
            
            # Get the last message content (agent's final response)
            last_message = messages[-1]
            agent_response = last_message.content if hasattr(last_message, 'content') else str(last_message)
            
            # Simple parsing - in production, use structured output
            parsed_results = {
                "raw_data": {},
                "s3_keys": [],
                "summary": agent_response,
                "symbols_processed": symbols,
                "agent_response": agent_response
            }
            
            # Extract S3 keys from agent response (basic string parsing)
            for symbol in symbols:
                if f"raw_data/daily/{symbol}" in agent_response:
                    parsed_results["s3_keys"].extend([
                        f"raw_data/daily/{symbol}/latest.json",
                        f"raw_data/daily/{symbol}/"  # Timestamped file pattern
                    ])
                    parsed_results["raw_data"][symbol] = {"status": "processed"}
            
            return parsed_results
            
        except Exception as e:
            return {
                "raw_data": {},
                "s3_keys": [],
                "summary": f"Failed to parse agent results: {str(e)}",
                "error": str(e)
            }
    
    async def run_workflow(self, symbols: List[str], timeframe: str = "daily") -> Dict[str, Any]:
        """Run the complete ReAct agent workflow"""
        print(f"🚀 Starting GPT-4o-mini ReAct Workflow")
        print(f"   📊 Symbols: {symbols}")
        print(f"   ⏱️  Timeframe: {timeframe}")
        
        # Create initial state
        initial_state = StockPredictionWorkflowState(
            workflow_id=f"react_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbols=symbols,
            timeframe=timeframe,
            node_status={},
            created_at=datetime.now().isoformat()
        )
        
        try:
            # Execute workflow with proper config for checkpointer
            config = {"configurable": {"thread_id": "test_workflow"}}
            result = await self.workflow.ainvoke(initial_state, config=config)
            
            print(f"✅ GPT-4o-mini ReAct workflow completed successfully")
            return {
                "success": True,
                "workflow_id": result["workflow_id"],
                "symbols_processed": result["symbols"],
                "s3_keys": result.get("s3_keys", []),
                "summary": result.get("ingestion_summary", ""),
                "node_status": result.get("node_status", {}),
                "execution_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ GPT-4o-mini ReAct workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow_id": initial_state["workflow_id"]
            }


async def main():
    """Test the GPT-4o-mini ReAct workflow"""
    print("🧪 Testing GPT-4o-mini ReAct Agent Workflow")
    
    # Create workflow
    workflow = GPT4oMiniReActWorkflow()
    
    # Test symbols
    test_symbols = ["AAPL", "GOOGL"]
    
    # Run workflow
    result = await workflow.run_workflow(test_symbols)
    
    print(f"\n📋 Final Results:")
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Workflow ID: {result['workflow_id']}")
        print(f"   Symbols Processed: {result['symbols_processed']}")
        print(f"   S3 Keys: {len(result.get('s3_keys', []))}")
        print(f"   Node Status: {result.get('node_status', {})}")
    else:
        print(f"   Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())
