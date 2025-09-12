# Copilot Instructions for llm-multi-ai-agent-system

## Project Overview
This repository implements a multi-agent LLM pipeline for stock price prediction using AWS services. Agents are specialized for data ingestion, preprocessing, and model training, communicating via Python async messaging within a single app for simplicity.

## Architecture & Major Components
- **LangGraph Workflow System**:
  - **Graph-based Orchestration**: Uses LangGraph StateGraph for declarative workflow management
  - **Node-based Agents**: Each agent is a LangGraph node with state management and tool integration
  - **Built-in Persistence**: Automatic workflow state persistence to DynamoDB with checkpointing
- **Agent Nodes**:
  - **Data Ingestion Node**: Fetches financial data (Alpha Vantage API), stores in AWS S3. Uses LangGraph tools for `boto3`, AWS Glue integration.
  - **Preprocessing Node**: Cleans, transforms, and feature-engineers data. Uses SageMaker Processing tools, AWS Glue, Pandas, Scikit-learn.
  - **Training Node**: Selects algorithms, launches SageMaker training jobs (XGBoost, PyTorch, TensorFlow), tracks metrics.
- **State Management**: LangGraph StateGraph manages workflow state with automatic persistence and rollback capabilities.
- **LLM Framework**: Uses **LangGraph + GPT-4o-mini** for intelligent workflow routing, ReAct agent patterns, and dynamic decision-making.

## Production Demo Success (September 2025)
✅ **WORKING STATE ACHIEVED**: The production demo (`production_demo.py`) successfully demonstrates:
- Synthetic data generation (715 records)
- S3 upload/download operations
- LLM-powered AWS Glue job orchestration
- Feature engineering with Spark/Glue
- Complete end-to-end validation
- Professional colorized logging with loguru

## Critical Bugs & Fixes Documentation

### 🔧 Bug #1: AWS Glue ConcurrentRunsExceededException
**Problem**: Glue jobs failed with "ConcurrentRunsExceededException" when multiple jobs were running
**Root Cause**: AWS Glue has concurrent execution limits per job definition
**Solution**: Implemented robust retry logic with exponential backoff
```python
max_retries = 3
retry_delay = 60  # seconds
for attempt in range(max_retries):
    try:
        # Submit Glue job
        if "ConcurrentRunsExceededException" in error_str and attempt < max_retries - 1:
            logger.warning(f"Concurrent runs exceeded, waiting {retry_delay}s before retry...")
            time.sleep(retry_delay)
            continue
```
**Prevention**: Always implement retry logic for AWS Glue job submissions

### 🔧 Bug #2: Empty Output Path in Glue Jobs
**Problem**: Glue jobs failed with "IllegalArgumentException: Can not create a Path from an empty string"
**Root Cause**: Output path was set to bucket root (`s3://bucket/`) instead of specific directory
**Solution**: Use demo-specific output paths with proper directory structure
```python
# ❌ WRONG - causes empty path errors
output_path = "s3://longhhoang-stock-data-processed/"

# ✅ CORRECT - specific directory path
output_path = f"s3://longhhoang-stock-data-processed/production_demo/processed/{demo_id}/"
```
**Prevention**: Always use specific S3 paths with trailing directory structure for Glue outputs

### 🔧 Bug #3: Missing Loguru Colorization
**Problem**: Console logs appeared without colors despite loguru configuration
**Root Cause**: Default loguru sink uses stdout, but colorization requires stderr
**Solution**: Configure loguru with sys.stderr and explicit colorize=True
```python
# ❌ WRONG - no colors
logger.add(sys.stdout, colorize=True)

# ✅ CORRECT - beautiful colors
logger.remove()  # Remove default
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
)
```
**Prevention**: Always use sys.stderr for colorized console output with loguru

### 🔧 Bug #4: LangChain Pydantic Deprecation Warnings
**Problem**: Excessive deprecation warnings cluttering output
**Root Cause**: LangChain internal modules still using pydantic v1 compatibility layer
**Solution**: Comprehensive warning suppression for development
```python
import warnings
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", message=".*pydantic.*")
warnings.filterwarnings("ignore", message=".*langchain_core.pydantic_v1.*")
```
**Prevention**: Implement warning filters early in main application modules

### 🔧 Bug #5: Deprecated LangChain Hub Usage
**Problem**: `hub.pull()` method deprecated, causing PromptTemplate issues
**Root Cause**: LangChain hub integration changed in recent versions
**Solution**: Use direct PromptTemplate creation instead of hub
```python
# ❌ WRONG - deprecated
prompt = hub.pull("hwchase17/react")

# ✅ CORRECT - direct template
prompt = PromptTemplate(
    template="""Answer the following questions as best you can...""",
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
)
```
**Prevention**: Avoid hub dependencies, use direct template definitions

## LangGraph Workflow Coordination Model
The system uses **LangGraph StateGraph** for advanced workflow orchestration:

- **Graph-Based Workflows**: Declarative workflow definition using LangGraph StateGraph with nodes, edges, and conditional routing
- **State-Driven Communication**: Agents communicate through shared state objects with automatic persistence and versioning
- **Intelligent Routing**: GPT-4o-mini powered conditional edges for dynamic workflow branching based on data quality, system load, and business logic
- **Built-in Orchestration**: LangGraph handles workflow execution, state management, error handling, and rollback automatically
- **ReAct Agent Patterns**: Advanced reasoning-action loops for complex decision-making within workflow nodes
- **Workflow Persistence**: Automatic checkpointing and state persistence to DynamoDB for fault tolerance and resume capabilities

## Developer Workflows
- **Dependency Management**: Use [Poetry](https://python-poetry.org/) for Python dependencies. Always update `pyproject.toml` with LangGraph, LangSmith and run `poetry install`.
- **Logging**: Use `loguru` for logging across all workflow nodes. **CRITICAL**: Configure with `sys.stderr` and `colorize=True` for proper colorization.
- **AWS Integration**: Use `boto3` through LangGraph tools for AWS SDK calls. Wrap AWS operations as tools for agent nodes.
- **Testing**: Place tests in a `tests/` directory. Use `pytest` for both unit tests and LangGraph workflow tests.
- **Workflow Development**: Define workflows in `src/workflows/`. Use LangGraph StateGraph for complex orchestration patterns.
- **Scripts & Notebooks**: Store exploratory notebooks in `notebooks/`. Workflow scripts and graph definitions go in `src/workflows/`.

## Project-Specific Conventions
- **Workflow Nodes**: Each agent should be a LangGraph node function with clear input/output state contracts. Use tools for external API calls.
- **External Data**: Always fetch data via the Data Ingestion node; do not bypass workflow patterns for experiments.
- **Glue/Spark Scripts**: Generate via LLM when possible; store in `scripts/glue/`. **CRITICAL**: Use specific S3 output paths, not bucket roots.
- **Model Training**: Use SageMaker jobs through LangGraph training node. Local training is for prototyping only.
- **State Management**: Use LangGraph built-in state persistence to DynamoDB with automatic checkpointing and versioning.
- **LLM Model Selection**: Use **GPT-4o-mini** for ReAct agents and LangChain workflows with intelligent tool integration and dynamic decision-making.
- **Workflow Patterns**: Define workflows using LangGraph StateGraph. Use conditional edges for dynamic routing. **ALWAYS** implement retry logic for AWS services.

## Proven Production Patterns

### ✅ Robust AWS Glue Job Submission
```python
max_retries = 3
retry_delay = 60  # seconds
for attempt in range(max_retries):
    try:
        result = submit_tool._run(
            job_name="stock-feature-engineering",
            input_path=input_path,
            output_path=f"s3://bucket/specific/path/{demo_id}/",  # SPECIFIC PATH
            symbol=symbol
        )
        if result.get("success"):
            break
    except Exception as e:
        if "ConcurrentRunsExceededException" in str(e) and attempt < max_retries - 1:
            logger.warning(f"Retry in {retry_delay}s...")
            time.sleep(retry_delay)
            continue
        raise
```

### ✅ Professional Loguru Configuration
```python
import sys
from loguru import logger

# Remove default handler
logger.remove()

# Add colorized console output
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)

# Add file logging with rotation
logger.add(
    "logs/demo_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    compression="gz",
    level="DEBUG"
)
```

### ✅ LangChain Warning Suppression
```python
import warnings
from langchain._api import LangChainDeprecationWarning

# Suppress development warnings
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", message=".*pydantic.*")
warnings.filterwarnings("ignore", message=".*langchain_core.pydantic_v1.*")
```

### ✅ Direct PromptTemplate Creation
```python
from langchain.prompts import PromptTemplate

# Use direct templates instead of hub.pull()
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
```

## Troubleshooting Guide

### 🚨 AWS Glue Issues
**Symptom**: `ConcurrentRunsExceededException`
**Solution**: Implement retry logic with 60s delays, max 3 attempts

**Symptom**: `IllegalArgumentException: Can not create a Path from an empty string`
**Solution**: Use specific S3 paths: `s3://bucket/path/to/specific/directory/`

**Symptom**: Job status stuck in RUNNING
**Solution**: Check CloudWatch logs, verify IAM permissions, ensure script exists in S3

### 🚨 Logging Issues
**Symptom**: No console colors in loguru
**Solution**: Use `sys.stderr` instead of `sys.stdout`, set `colorize=True`

**Symptom**: Excessive LangChain warnings
**Solution**: Add comprehensive warning filters at application startup

### 🚨 LLM Agent Issues
**Symptom**: `hub.pull()` deprecation warnings
**Solution**: Use direct PromptTemplate creation instead of hub dependencies

**Symptom**: Tool calling failures
**Solution**: Verify tool schemas match exactly, check API key configuration

## Integration Points & Patterns
- **Alpha Vantage API**: Use official client or REST calls; wrap as LangGraph tools for ingestion nodes.
- **AWS Services**: Prefer SDK calls over CLI. For ETL, use Glue jobs; for streaming, use Kinesis. All integrated as LangGraph tools.
- **LangGraph**: Primary framework for workflow orchestration, state management, and agent coordination.
- **LLM Integration**: Configure ChatOpenAI with **GPT-4o-mini** for ReAct agents and intelligent workflow routing.
- **State Flow**: Use LangGraph StateGraph for workflow execution with automatic persistence and conditional routing.

## Key Files & Directories
- `README.md`: High-level architecture and workflow responsibilities.
- `pyproject.toml`: Poetry dependency management with LangGraph.
- `production_demo.py`: **WORKING PRODUCTION DEMO** - Complete end-to-end LLM agent demonstration
- `src/workflows/`: LangGraph workflow definitions and state graphs.
- `src/nodes/`: Individual workflow nodes (agents as functions).
- `src/tools/agent_tools.py`: **CRITICAL** - AWS Glue and S3 integration tools with retry logic
- `src/state/`: State management schemas and persistence logic.
- `scripts/glue/`: **WORKING** ETL scripts integrated as LangGraph tools.
- `tests/`: Test suite with LangGraph workflow tests.
- `notebooks/`: Exploratory analysis.
- `.github/copilot-instructions.md`: **THIS FILE** - Complete bug fixes and patterns documentation

## Production Demo Architecture (`production_demo.py`)
The working production demo demonstrates:

1. **Synthetic Data Generation**: Creates 715 realistic stock records
2. **S3 Upload**: Uploads data to `s3://longhhoang-stock-data-raw/production_demo/`
3. **LLM Agent Processing**: GPT-4o-mini calls AWS Glue tools with ReAct pattern
4. **AWS Glue Execution**: Feature engineering with Spark/Glue (`scripts/glue/stock_feature_engineering.py`)
5. **Output Validation**: Verifies processed data in `s3://longhhoang-stock-data-processed/`

### Demo Class Structure
```python
class ProductionDemo:
    def __init__(self):
        self.setup_logging()      # Professional loguru configuration
        self.setup_aws_clients()  # S3, Glue clients with retry logic
        self.setup_llm_agent()    # GPT-4o-mini with ReAct pattern
        
    async def run_demo(self):
        data = self.generate_large_synthetic_dataset(1000)  # 715 actual records
        upload_result = self.upload_to_s3_production(data)
        processing_result = self.run_llm_agent_processing(upload_result)
        validation_result = self.validate_production_output()
```

## Example Patterns
- **Working Production Demo Pattern**:
  ```python
  from production_demo import ProductionDemo
  import asyncio
  
  # Complete working example
  async def main():
      demo = ProductionDemo()
      await demo.run_demo()
      
  # Results: 715 records → S3 → Glue processing → 376KB processed data
  ```

- **LangGraph workflow with GPT-4o-mini ReAct agents**:
  ```python
  from langgraph.graph import StateGraph, END
  from langgraph.prebuilt import create_react_agent
  from langchain_openai import ChatOpenAI
  
  # Define workflow state
  class WorkflowState(TypedDict):
      symbols: List[str]
      raw_data: Dict[str, Any]
      processed_data: Dict[str, Any]
      model_results: Dict[str, Any]
  
  # Create LangGraph workflow
  workflow = StateGraph(WorkflowState)
  
  # GPT-4o-mini for ReAct agents with tool integration
  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
  
  # Add nodes with ReAct agent integration
  workflow.add_node("ingestion", data_ingestion_node)
  workflow.add_node("preprocessing", preprocessing_node)
  workflow.add_node("training", training_node)
  
  # Add edges with GPT-4o-mini powered routing
  workflow.add_conditional_edges("ingestion", route_next_step)
  workflow.add_edge("preprocessing", "training")
  
  # Compile with GPT-4o-mini for intelligent routing
  app = workflow.compile()
  ```

- **Professional Logging Setup**:
  ```python
  import sys
  from loguru import logger
  
  # CRITICAL: Use sys.stderr for colors
  logger.remove()
  logger.add(
      sys.stderr,
      colorize=True,
      format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
  )
  logger.success("✅ Beautiful colorized logging!")
  ```

- **AWS Glue Integration with Retry Logic**:
  ```python
  # PROVEN WORKING PATTERN
  max_retries = 3
  retry_delay = 60
  
  for attempt in range(max_retries):
      try:
          result = submit_tool._run(
              job_name="stock-feature-engineering",
              input_path="s3://bucket/input/path/file.json",
              output_path="s3://bucket/output/specific/directory/",  # SPECIFIC PATH
              symbol="PROD"
          )
          break
      except Exception as e:
          if "ConcurrentRunsExceededException" in str(e):
              time.sleep(retry_delay)
              continue
          raise
  ```

- **Message structure**:
  ```python
  from dataclasses import dataclass
  
  @dataclass
  class AgentMessage:
      sender: str
      recipient: str
      message_type: str
      payload: dict
      correlation_id: str = None
  ```

- **Coordination pattern**:
  ```python
  from src.coordination.agent_coordinator import AgentCoordinator
  
  coordinator = AgentCoordinator(message_bus, config)
  await coordinator.coordinate_workflow("workflow_123", ["AAPL", "GOOGL"])
  ```

- **AWS SDK usage**:
  ```python
  import boto3
  s3 = boto3.client('s3')
  s3.upload_file(...)
  ```

---

**This documentation represents the WORKING STATE as of September 2025. All patterns and fixes have been battle-tested in production demo. Use these exact patterns to avoid the documented bugs.**