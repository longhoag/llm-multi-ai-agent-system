"""LangGraph state management for multi-agent workflows"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class AgentState(TypedDict):
    """Base state for agent workflow nodes"""
    # Core workflow information
    workflow_id: str
    correlation_id: str
    status: WorkflowStatus
    created_at: datetime
    updated_at: datetime
    
    # Agent-specific data
    agent_id: str
    node_name: str
    
    # Error handling
    errors: List[Dict[str, Any]]
    retry_count: int
    max_retries: int


class DataIngestionState(AgentState):
    """State for data ingestion workflow node"""
    # Input parameters
    symbols: List[str]
    timeframe: str
    force_refresh: bool
    
    # Processing results
    raw_data: Dict[str, Any]
    s3_keys: List[str]
    ingestion_stats: Dict[str, Any]
    
    # Quality validation
    data_quality: Dict[str, Any]
    validation_passed: bool
    
    # LLM strategy
    ingestion_strategy: str
    strategy_confidence: float


class PreprocessingState(AgentState):
    """State for preprocessing workflow node"""
    # Input from ingestion
    input_s3_keys: List[str]
    
    # Processing configuration
    preprocessing_config: Dict[str, Any]
    feature_engineering_rules: List[str]
    
    # Output results
    processed_s3_keys: List[str]
    feature_metadata: Dict[str, Any]
    data_statistics: Dict[str, Any]
    
    # Quality metrics
    processing_quality: Dict[str, Any]
    data_completeness: float


class TrainingState(AgentState):
    """State for model training workflow node"""
    # Input from preprocessing
    training_data_keys: List[str]
    
    # Model configuration
    model_type: str
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    
    # Training results
    model_artifacts: Dict[str, str]
    training_metrics: Dict[str, Any]
    model_performance: Dict[str, Any]
    
    # SageMaker integration
    sagemaker_job_name: str
    training_status: str


class StockPredictionWorkflowState(TypedDict):
    """Complete workflow state for stock prediction pipeline"""
    # Workflow metadata
    workflow_id: str
    correlation_id: str
    status: WorkflowStatus
    created_at: datetime
    updated_at: datetime
    
    # Input parameters
    symbols: List[str]
    timeframe: str
    prediction_horizon: int
    
    # Node states
    ingestion: DataIngestionState
    preprocessing: PreprocessingState  
    training: TrainingState
    
    # Global workflow data
    workflow_config: Dict[str, Any]
    global_errors: List[Dict[str, Any]]
    
    # Final results
    predictions: Dict[str, Any]
    confidence_scores: Dict[str, float]
    model_metadata: Dict[str, Any]


class WorkflowStateManager:
    """Manages workflow state persistence and retrieval"""
    
    def __init__(self, dynamodb_table: str = None):
        self.dynamodb_table = dynamodb_table
    
    def create_initial_state(
        self,
        symbols: List[str],
        timeframe: str = "daily",
        prediction_horizon: int = 30
    ) -> StockPredictionWorkflowState:
        """Create initial workflow state"""
        now = datetime.now()
        workflow_id = f"stock_prediction_{now.strftime('%Y%m%d_%H%M%S')}"
        correlation_id = f"corr_{workflow_id}"
        
        return StockPredictionWorkflowState(
            # Workflow metadata
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            status=WorkflowStatus.PENDING,
            created_at=now,
            updated_at=now,
            
            # Input parameters
            symbols=symbols,
            timeframe=timeframe,
            prediction_horizon=prediction_horizon,
            
            # Node states (initialized empty)
            ingestion=DataIngestionState(
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                status=WorkflowStatus.PENDING,
                created_at=now,
                updated_at=now,
                agent_id="data_ingestion",
                node_name="ingestion",
                errors=[],
                retry_count=0,
                max_retries=3,
                symbols=symbols,
                timeframe=timeframe,
                force_refresh=False,
                raw_data={},
                s3_keys=[],
                ingestion_stats={},
                data_quality={},
                validation_passed=False,
                ingestion_strategy="",
                strategy_confidence=0.0
            ),
            preprocessing=PreprocessingState(
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                status=WorkflowStatus.PENDING,
                created_at=now,
                updated_at=now,
                agent_id="preprocessing",
                node_name="preprocessing",
                errors=[],
                retry_count=0,
                max_retries=3,
                input_s3_keys=[],
                preprocessing_config={},
                feature_engineering_rules=[],
                processed_s3_keys=[],
                feature_metadata={},
                data_statistics={},
                processing_quality={},
                data_completeness=0.0
            ),
            training=TrainingState(
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                status=WorkflowStatus.PENDING,
                created_at=now,
                updated_at=now,
                agent_id="training",
                node_name="training",
                errors=[],
                retry_count=0,
                max_retries=3,
                training_data_keys=[],
                model_type="",
                hyperparameters={},
                training_config={},
                model_artifacts={},
                training_metrics={},
                model_performance={},
                sagemaker_job_name="",
                training_status=""
            ),
            
            # Global workflow data
            workflow_config={},
            global_errors=[],
            
            # Final results
            predictions={},
            confidence_scores={},
            model_metadata={}
        )
    
    def update_node_state(
        self,
        state: StockPredictionWorkflowState,
        node_name: str,
        updates: Dict[str, Any]
    ) -> StockPredictionWorkflowState:
        """Update specific node state within workflow"""
        state["updated_at"] = datetime.now()
        
        if node_name == "ingestion":
            state["ingestion"].update(updates)
            state["ingestion"]["updated_at"] = datetime.now()
        elif node_name == "preprocessing":
            state["preprocessing"].update(updates)
            state["preprocessing"]["updated_at"] = datetime.now()
        elif node_name == "training":
            state["training"].update(updates)
            state["training"]["updated_at"] = datetime.now()
        
        return state
    
    def add_error(
        self,
        state: StockPredictionWorkflowState,
        error: str,
        node_name: str = None,
        error_type: str = "general"
    ) -> StockPredictionWorkflowState:
        """Add error to workflow or node state"""
        error_record = {
            "error": error,
            "error_type": error_type,
            "timestamp": datetime.now().isoformat(),
            "node_name": node_name
        }
        
        if node_name:
            if node_name == "ingestion":
                state["ingestion"]["errors"].append(error_record)
            elif node_name == "preprocessing":
                state["preprocessing"]["errors"].append(error_record)
            elif node_name == "training":
                state["training"]["errors"].append(error_record)
        else:
            state["global_errors"].append(error_record)
        
        return state
    
    def get_workflow_status(self, state: StockPredictionWorkflowState) -> WorkflowStatus:
        """Determine overall workflow status based on node states"""
        ingestion_status = state["ingestion"]["status"]
        preprocessing_status = state["preprocessing"]["status"] 
        training_status = state["training"]["status"]
        
        if training_status == WorkflowStatus.COMPLETED:
            return WorkflowStatus.COMPLETED
        elif any(status == WorkflowStatus.FAILED for status in [ingestion_status, preprocessing_status, training_status]):
            return WorkflowStatus.FAILED
        elif any(status == WorkflowStatus.RUNNING for status in [ingestion_status, preprocessing_status, training_status]):
            return WorkflowStatus.RUNNING
        else:
            return WorkflowStatus.PENDING
