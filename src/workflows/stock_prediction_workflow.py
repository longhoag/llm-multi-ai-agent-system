"""Main LangGraph workflow for stock prediction pipeline"""

from typing import Dict, Any, List
from datetime import datetime
from loguru import logger

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.state.workflow_state import (
    StockPredictionWorkflowState,
    WorkflowStateManager,
    WorkflowStatus
)
from src.nodes.workflow_nodes import (
    data_ingestion_node,
    preprocessing_node,
    training_node
)
from src.tools.agent_tools import initialize_tools
from src.config.settings import config_manager


def create_stock_prediction_workflow():
    """
    Create and configure the LangGraph workflow for stock prediction.
    
    Returns:
        Compiled LangGraph workflow ready for execution
    """
    # Create workflow graph
    workflow = StateGraph(StockPredictionWorkflowState)
    
    # Add nodes
    workflow.add_node("ingestion", data_ingestion_node)
    workflow.add_node("preprocessing", preprocessing_node)
    workflow.add_node("training", training_node)
    
    # Add edges
    workflow.add_edge("ingestion", "preprocessing")
    workflow.add_edge("preprocessing", "training")
    workflow.add_edge("training", END)
    
    # Set entry point
    workflow.set_entry_point("ingestion")
    
    # Add memory for checkpointing
    memory = MemorySaver()
    
    # Compile workflow
    app = workflow.compile(checkpointer=memory)
    
    logger.info("LangGraph stock prediction workflow created successfully")
    return app


def create_advanced_workflow_with_routing():
    """
    Create advanced workflow with conditional routing based on data quality and business logic.
    
    Returns:
        Compiled LangGraph workflow with intelligent routing
    """
    
    def should_proceed_to_preprocessing(state: StockPredictionWorkflowState) -> str:
        """Conditional routing logic for preprocessing"""
        ingestion_state = state["ingestion"]
        
        # Check if ingestion was successful
        if ingestion_state["status"] != WorkflowStatus.COMPLETED:
            logger.warning("Ingestion failed, skipping preprocessing")
            return END
        
        # Check data quality
        if not ingestion_state.get("validation_passed", False):
            logger.warning("Data quality validation failed, skipping preprocessing")
            return END
        
        # Check if we have sufficient data
        if len(ingestion_state.get("s3_keys", [])) == 0:
            logger.warning("No data available, skipping preprocessing")
            return END
        
        logger.info("Data quality checks passed, proceeding to preprocessing")
        return "preprocessing"
    
    def should_proceed_to_training(state: StockPredictionWorkflowState) -> str:
        """Conditional routing logic for training"""
        preprocessing_state = state["preprocessing"]
        
        # Check if preprocessing was successful
        if preprocessing_state["status"] != WorkflowStatus.COMPLETED:
            logger.warning("Preprocessing failed, skipping training")
            return END
        
        # Check data completeness
        completeness = preprocessing_state.get("data_completeness", 0.0)
        if completeness < 0.8:  # Require 80% data completeness
            logger.warning(f"Data completeness too low ({completeness:.2%}), skipping training")
            return END
        
        # Check if we have processed data
        processed_keys = preprocessing_state.get("processed_s3_keys", [])
        if len(processed_keys) == 0:
            logger.warning("No processed data available, skipping training")
            return END
        
        logger.info("Preprocessing checks passed, proceeding to training")
        return "training"
    
    # Create workflow graph with conditional routing
    workflow = StateGraph(StockPredictionWorkflowState)
    
    # Add nodes
    workflow.add_node("ingestion", data_ingestion_node)
    workflow.add_node("preprocessing", preprocessing_node)
    workflow.add_node("training", training_node)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "ingestion",
        should_proceed_to_preprocessing,
        {
            "preprocessing": "preprocessing",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "preprocessing",
        should_proceed_to_training,
        {
            "training": "training",
            END: END
        }
    )
    
    workflow.add_edge("training", END)
    
    # Set entry point
    workflow.set_entry_point("ingestion")
    
    # Add memory for checkpointing and persistence
    memory = MemorySaver()
    
    # Compile workflow
    app = workflow.compile(checkpointer=memory)
    
    logger.info("Advanced LangGraph workflow with routing created successfully")
    return app


class StockPredictionOrchestrator:
    """Main orchestrator for stock prediction workflows using LangGraph"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize orchestrator with configuration"""
        self.config = config or config_manager.config
        self.state_manager = WorkflowStateManager()
        
        # Initialize tools with configuration
        initialize_tools(
            alpha_vantage_key=self.config["alpha_vantage_api_key"],
            s3_bucket=self.config["s3_bucket_raw_data"],
            openai_key=self.config["openai_api_key"]
        )
        
        # Create workflows
        self.basic_workflow = create_stock_prediction_workflow()
        self.advanced_workflow = create_advanced_workflow_with_routing()
        
        logger.info("Stock Prediction Orchestrator initialized")
    
    async def run_basic_workflow(
        self,
        symbols: List[str],
        timeframe: str = "daily",
        prediction_horizon: int = 30
    ) -> Dict[str, Any]:
        """
        Run basic stock prediction workflow.
        
        Args:
            symbols: List of stock symbols to process
            timeframe: Data timeframe (daily, 5min, etc.)
            prediction_horizon: Number of days to predict
        
        Returns:
            Workflow results with predictions and metadata
        """
        logger.info(f"Starting basic workflow for symbols: {symbols}")
        
        # Create initial state
        initial_state = self.state_manager.create_initial_state(
            symbols=symbols,
            timeframe=timeframe,
            prediction_horizon=prediction_horizon
        )
        
        try:
            # Execute workflow
            config = {"configurable": {"thread_id": initial_state["workflow_id"]}}
            final_state = await self.basic_workflow.ainvoke(initial_state, config)
            
            # Extract results
            results = {
                "workflow_id": final_state["workflow_id"],
                "status": final_state["status"],
                "predictions": final_state["predictions"],
                "confidence_scores": final_state["confidence_scores"],
                "model_metadata": final_state["model_metadata"],
                "execution_time": (
                    final_state["updated_at"] - final_state["created_at"]
                ).total_seconds(),
                "nodes_completed": self._count_completed_nodes(final_state)
            }
            
            logger.success(f"Basic workflow completed: {results['workflow_id']}")
            return results
            
        except Exception as e:
            logger.error(f"Basic workflow failed: {e}")
            return {
                "workflow_id": initial_state["workflow_id"],
                "status": "failed",
                "error": str(e),
                "predictions": {},
                "confidence_scores": {},
                "model_metadata": {}
            }
    
    async def run_advanced_workflow(
        self,
        symbols: List[str],
        timeframe: str = "daily",
        prediction_horizon: int = 30
    ) -> Dict[str, Any]:
        """
        Run advanced workflow with intelligent routing and quality checks.
        
        Args:
            symbols: List of stock symbols to process
            timeframe: Data timeframe (daily, 5min, etc.)  
            prediction_horizon: Number of days to predict
        
        Returns:
            Workflow results with predictions and detailed execution info
        """
        logger.info(f"Starting advanced workflow for symbols: {symbols}")
        
        # Create initial state
        initial_state = self.state_manager.create_initial_state(
            symbols=symbols,
            timeframe=timeframe,
            prediction_horizon=prediction_horizon
        )
        
        try:
            # Execute workflow with checkpointing
            config = {"configurable": {"thread_id": initial_state["workflow_id"]}}
            final_state = await self.advanced_workflow.ainvoke(initial_state, config)
            
            # Extract detailed results
            results = {
                "workflow_id": final_state["workflow_id"],
                "status": final_state["status"],
                "predictions": final_state["predictions"],
                "confidence_scores": final_state["confidence_scores"],
                "model_metadata": final_state["model_metadata"],
                "execution_time": (
                    final_state["updated_at"] - final_state["created_at"]
                ).total_seconds(),
                "nodes_completed": self._count_completed_nodes(final_state),
                "ingestion_results": {
                    "status": final_state["ingestion"]["status"],
                    "symbols_processed": len(final_state["ingestion"]["s3_keys"]),
                    "data_quality_passed": final_state["ingestion"]["validation_passed"]
                },
                "preprocessing_results": {
                    "status": final_state["preprocessing"]["status"],
                    "data_completeness": final_state["preprocessing"]["data_completeness"],
                    "features_created": len(final_state["preprocessing"]["feature_metadata"])
                },
                "training_results": {
                    "status": final_state["training"]["status"],
                    "model_type": final_state["training"]["model_type"],
                    "training_metrics": final_state["training"]["training_metrics"]
                }
            }
            
            logger.success(f"Advanced workflow completed: {results['workflow_id']}")
            return results
            
        except Exception as e:
            logger.error(f"Advanced workflow failed: {e}")
            return {
                "workflow_id": initial_state["workflow_id"],
                "status": "failed",
                "error": str(e),
                "predictions": {},
                "confidence_scores": {},
                "model_metadata": {}
            }
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a running workflow"""
        # In production, this would query the checkpointer/database
        return {
            "workflow_id": workflow_id,
            "status": "running",
            "current_node": "unknown",
            "progress": "unknown"
        }
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows and their statuses"""
        # In production, this would query the persistence layer
        return []
    
    def _count_completed_nodes(self, state: StockPredictionWorkflowState) -> int:
        """Count how many nodes completed successfully"""
        completed = 0
        if state["ingestion"]["status"] == WorkflowStatus.COMPLETED:
            completed += 1
        if state["preprocessing"]["status"] == WorkflowStatus.COMPLETED:
            completed += 1
        if state["training"]["status"] == WorkflowStatus.COMPLETED:
            completed += 1
        return completed


# Convenience functions for direct workflow usage
async def run_stock_prediction(
    symbols: List[str],
    timeframe: str = "daily",
    prediction_horizon: int = 30,
    use_advanced_routing: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run stock prediction workflow.
    
    Args:
        symbols: Stock symbols to analyze
        timeframe: Data timeframe
        prediction_horizon: Days to predict
        use_advanced_routing: Use advanced workflow with quality checks
    
    Returns:
        Prediction results
    """
    orchestrator = StockPredictionOrchestrator()
    
    if use_advanced_routing:
        return await orchestrator.run_advanced_workflow(
            symbols, timeframe, prediction_horizon
        )
    else:
        return await orchestrator.run_basic_workflow(
            symbols, timeframe, prediction_horizon
        )
