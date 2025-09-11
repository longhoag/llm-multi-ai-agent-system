"""GPT-5-mini compatible workflow nodes without ReAct agents"""

from typing import Dict, Any, List
from datetime import datetime
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from src.state.workflow_state import (
    StockPredictionWorkflowState, 
    WorkflowStatus,
    WorkflowStateManager
)
from src.tools.agent_tools import (
    DATA_INGESTION_TOOLS,
    initialize_tools,
    fetch_stock_data_tool,
    upload_stock_data_to_s3,
    validate_data_quality_tool
)


async def data_ingestion_node_gpt5_compatible(state: StockPredictionWorkflowState) -> StockPredictionWorkflowState:
    """
    GPT-5-mini compatible data ingestion node that doesn't use ReAct agents.
    Uses direct tool calls and GPT-5-mini for orchestration.
    
    Core tasks:
    - Fetching stock data from Alpha Vantage
    - Validating data quality
    - Storing data in S3 with both timestamped and latest.json files
    """
    logger.info(f"Starting GPT-5-mini compatible data ingestion for workflow {state['workflow_id']}")
    
    try:
        # Initialize tools first
        import os
        initialize_tools(
            alpha_vantage_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
            s3_bucket=os.getenv('S3_BUCKET_RAW_DATA'),
            openai_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Update node status
        state_manager = WorkflowStateManager()
        state = state_manager.update_node_state(
            state, "ingestion", {"status": WorkflowStatus.RUNNING}
        )
        
        # Create GPT-5-mini client for orchestration
        llm = ChatOpenAI(model="gpt-5-mini", temperature=0.3)
        
        # Process each symbol
        symbols = state["symbols"]
        timeframe = state["timeframe"]
        all_s3_keys = []
        ingestion_results = {}
        
        for symbol in symbols:
            logger.info(f"Processing symbol: {symbol}")
            
            # Step 1: Fetch stock data
            logger.info(f"Fetching data for {symbol}")
            fetch_result = fetch_stock_data_tool(symbol=symbol, timeframe=timeframe)
            
            if not fetch_result.get("success"):
                logger.error(f"Failed to fetch data for {symbol}: {fetch_result.get('error')}")
                continue
            
            # Step 2: Validate data quality using GPT-5-mini
            logger.info(f"Validating data quality for {symbol}")
            
            # Create a validation prompt for GPT-5-mini
            validation_prompt = f"""
            You are a data quality analyst. Analyze the following stock data for {symbol} and provide a quality assessment.
            
            Data summary:
            - Symbol: {symbol}
            - Timeframe: {timeframe}
            - Record count: {fetch_result.get('record_count', 0)}
            - Sample data: {str(fetch_result.get('data', [])[:3]) if fetch_result.get('data') else 'No data'}
            
            Provide a brief assessment of data quality (1-2 sentences) and whether it's suitable for analysis.
            """
            
            validation_response = llm.invoke([
                SystemMessage(content="You are a data quality expert for financial data."),
                HumanMessage(content=validation_prompt)
            ])
            
            # Step 3: Upload to S3 with both timestamped and latest versions
            logger.info(f"Uploading {symbol} data to S3")
            
            upload_result = upload_stock_data_to_s3(
                data=fetch_result,
                symbol=symbol,
                timeframe=timeframe
            )
            
            if upload_result.get("success"):
                all_s3_keys.extend([
                    upload_result.get("timestamped_key"),
                    upload_result.get("latest_key")
                ])
                
                ingestion_results[symbol] = {
                    "fetch_result": fetch_result,
                    "upload_result": upload_result,
                    "validation_assessment": validation_response.content,
                    "status": "completed"
                }
                
                logger.info(f"Successfully processed {symbol} - S3 keys: {upload_result.get('timestamped_key')}, {upload_result.get('latest_key')}")
            else:
                logger.error(f"Failed to upload {symbol} data: {upload_result.get('error')}")
                ingestion_results[symbol] = {
                    "status": "failed",
                    "error": upload_result.get("error")
                }
        
        # Generate final summary using GPT-5-mini
        summary_prompt = f"""
        Summarize the data ingestion results for the following symbols: {symbols}
        
        Results:
        {str(ingestion_results)}
        
        Provide a concise summary of what was accomplished, including:
        - Number of symbols processed successfully
        - Total S3 keys created
        - Any issues encountered
        """
        
        summary_response = llm.invoke([
            SystemMessage(content="You are a workflow summarization assistant."),
            HumanMessage(content=summary_prompt)
        ])
        
        # Update state with results
        state = state_manager.update_node_state(state, "ingestion", {
            "status": WorkflowStatus.COMPLETED,
            "raw_data": ingestion_results,
            "s3_keys": all_s3_keys,
            "ingestion_stats": {
                "symbols_processed": len(ingestion_results),
                "successful_symbols": len([r for r in ingestion_results.values() if r.get("status") == "completed"]),
                "total_s3_keys": len(all_s3_keys),
                "workflow_summary": summary_response.content
            }
        })
        
        logger.success(f"GPT-5-mini data ingestion completed for {len(symbols)} symbols")
        return state
        
    except Exception as e:
        logger.error(f"GPT-5-mini data ingestion failed: {e}")
        
        state = state_manager.update_node_state(state, "ingestion", {
            "status": WorkflowStatus.FAILED,
            "errors": [{"error": str(e), "error_type": "execution_error", "timestamp": datetime.now().isoformat(), "node_name": "ingestion"}]
        })
        
        return state


def create_gpt5_mini_workflow():
    """
    Create a GPT-5-mini compatible workflow that doesn't use ReAct agents.
    This approach uses direct tool calls and GPT-5-mini for orchestration.
    """
    from langgraph.graph import StateGraph, END
    
    # Create workflow
    workflow = StateGraph(StockPredictionWorkflowState)
    
    # Add GPT-5-mini compatible nodes
    workflow.add_node("ingestion", data_ingestion_node_gpt5_compatible)
    
    # Define workflow flow
    workflow.set_entry_point("ingestion")
    workflow.add_edge("ingestion", END)
    
    return workflow.compile()


async def test_gpt5_mini_workflow():
    """Test the GPT-5-mini compatible workflow"""
    from src.state.workflow_state import DataIngestionState, WorkflowStatus
    
    # Create test state
    state = DataIngestionState(
        workflow_id=f'gpt5_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        correlation_id='gpt5-test-123',
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
    
    # Run the GPT-5-mini compatible workflow
    result = await data_ingestion_node_gpt5_compatible(state)
    
    return result
