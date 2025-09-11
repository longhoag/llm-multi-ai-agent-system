#!/usr/bin/env python3
"""
LangGraph Stock Prediction Pipeline

Simple entry point for running the LangGraph-based stock prediction workflow.
This script demonstrates how to execute the workflow with real APIs.

Usage:
    python run_workflow.py AAPL GOOGL MSFT

Prerequisites:
    - Set environment variables: OPENAI_API_KEY, ALPHA_VANTAGE_API_KEY, S3_BUCKET_RAW_DATA
    - Configure AWS credentials
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.state.workflow_state import WorkflowStateManager, WorkflowStatus
from src.nodes.workflow_nodes import (
    data_ingestion_node,
    preprocessing_node, 
    training_node
)


def check_environment():
    """Check if required environment variables are set"""
    required_vars = [
        "OPENAI_API_KEY",
        "ALPHA_VANTAGE_API_KEY",
        "S3_BUCKET_RAW_DATA"
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(f"   • {var}")
        print("\nSet these variables before running:")
        print("export OPENAI_API_KEY='your_key'")
        print("export ALPHA_VANTAGE_API_KEY='your_key'")
        print("export S3_BUCKET_RAW_DATA='your_bucket'")
        return False
    
    return True


def create_workflow():
    """Create the LangGraph workflow"""
    # Define workflow
    workflow = StateGraph(dict)
    
    # Add nodes with unique names that don't conflict with state keys
    workflow.add_node("data_ingestion", data_ingestion_node)
    workflow.add_node("data_preprocessing", preprocessing_node)
    workflow.add_node("model_training", training_node)
    
    # Define routing logic
    def route_after_ingestion(state):
        ingestion_status = state.get("ingestion", {}).get("status")
        if ingestion_status == WorkflowStatus.COMPLETED:
            return "data_preprocessing"
        return END
    
    def route_after_preprocessing(state):
        preprocessing_status = state.get("preprocessing", {}).get("status")
        if preprocessing_status == WorkflowStatus.COMPLETED:
            return "model_training"
        return END
    
    # Add edges
    workflow.add_conditional_edges(
        "data_ingestion",
        route_after_ingestion,
        {
            "data_preprocessing": "data_preprocessing",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "data_preprocessing", 
        route_after_preprocessing,
        {
            "model_training": "model_training",
            END: END
        }
    )
    
    workflow.add_edge("model_training", END)
    workflow.set_entry_point("data_ingestion")
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


async def run_stock_prediction(symbols: List[str], prediction_horizon: int = 30):
    """Run the complete stock prediction pipeline"""
    
    print(f"🚀 Starting Stock Prediction Pipeline")
    print(f"📈 Symbols: {', '.join(symbols)}")
    print(f"🔮 Prediction Horizon: {prediction_horizon} days")
    print("=" * 60)
    
    # Create workflow
    app = create_workflow()
    
    # Initialize state
    manager = WorkflowStateManager()
    initial_state = manager.create_initial_state(
        symbols=symbols,
        timeframe="daily",
        prediction_horizon=prediction_horizon
    )
    
    # Execute workflow
    workflow_id = f"stock_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = {"configurable": {"thread_id": workflow_id}}
    
    print(f"🔄 Executing workflow: {workflow_id}")
    
    try:
        final_state = app.invoke(initial_state, config)
        
        # Show results
        print("\n📊 WORKFLOW RESULTS")
        print("=" * 60)
        
        # Ingestion results
        ingestion = final_state.get("ingestion", {})
        print(f"📥 Data Ingestion: {ingestion.get('status', 'Unknown')}")
        if ingestion.get("s3_keys"):
            print(f"   📁 Files stored: {len(ingestion['s3_keys'])}")
        
        # Preprocessing results  
        preprocessing = final_state.get("preprocessing", {})
        print(f"🔧 Data Preprocessing: {preprocessing.get('status', 'Unknown')}")
        if preprocessing.get("feature_metadata"):
            features = preprocessing["feature_metadata"].get("features", 0)
            print(f"   🎯 Features engineered: {features}")
        
        # Training results
        training = final_state.get("training", {})
        print(f"🤖 Model Training: {training.get('status', 'Unknown')}")
        if training.get("training_metrics"):
            accuracy = training["training_metrics"].get("accuracy")
            if accuracy:
                print(f"   📈 Model accuracy: {accuracy:.2%}")
        
        print(f"\n✅ Workflow completed successfully!")
        return final_state
        
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        return None


async def main():
    """Main entry point"""
    
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python run_workflow.py SYMBOL1 [SYMBOL2] [SYMBOL3] ...")
        print("Example: python run_workflow.py AAPL GOOGL MSFT")
        sys.exit(1)
    
    symbols = sys.argv[1:]
    
    # Validate environment
    if not check_environment():
        sys.exit(1)
    
    # Run workflow
    try:
        result = await run_stock_prediction(symbols)
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n🛑 Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
