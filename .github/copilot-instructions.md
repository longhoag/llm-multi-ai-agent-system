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
- **LLM Framework**: Uses **LangGraph + GPT-5-mini** for intelligent workflow routing, ReAct agent patterns, and dynamic decision-making.

## LangGraph Workflow Coordination Model
The system uses **LangGraph StateGraph** for advanced workflow orchestration:

- **Graph-Based Workflows**: Declarative workflow definition using LangGraph StateGraph with nodes, edges, and conditional routing
- **State-Driven Communication**: Agents communicate through shared state objects with automatic persistence and versioning
- **Intelligent Routing**: GPT-5-mini powered conditional edges for dynamic workflow branching based on data quality, system load, and business logic
- **Built-in Orchestration**: LangGraph handles workflow execution, state management, error handling, and rollback automatically
- **ReAct Agent Patterns**: Advanced reasoning-action loops for complex decision-making within workflow nodes
- **Workflow Persistence**: Automatic checkpointing and state persistence to DynamoDB for fault tolerance and resume capabilities

## Developer Workflows
- **Dependency Management**: Use [Poetry](https://python-poetry.org/) for Python dependencies. Always update `pyproject.toml` with LangGraph, LangSmith and run `poetry install`.
- **Logging**: Use `loguru` for logging across all workflow nodes. Avoid `logging` module unless integrating with legacy code.
- **AWS Integration**: Use `boto3` through LangGraph tools for AWS SDK calls. Wrap AWS operations as tools for agent nodes.
- **Testing**: Place tests in a `tests/` directory. Use `pytest` for both unit tests and LangGraph workflow tests.
- **Workflow Development**: Define workflows in `src/workflows/`. Use LangGraph StateGraph for complex orchestration patterns.
- **Scripts & Notebooks**: Store exploratory notebooks in `notebooks/`. Workflow scripts and graph definitions go in `src/workflows/`.

## Project-Specific Conventions
- **Workflow Nodes**: Each agent should be a LangGraph node function with clear input/output state contracts. Use tools for external API calls.
- **External Data**: Always fetch data via the Data Ingestion node; do not bypass workflow patterns for experiments.
- **Glue/Spark Scripts**: Generate via LLM when possible; store in `scripts/` or `glue_jobs/`. Integrate as LangGraph tools.
- **Model Training**: Use SageMaker jobs through LangGraph training node. Local training is for prototyping only.
- **State Management**: Use LangGraph built-in state persistence to DynamoDB with automatic checkpointing and versioning.
- **LLM Model Selection**: Use GPT-5-mini for both complex reasoning and routine tasks (latest, most cost-effective, high-performance model for all operations).
- **Workflow Patterns**: Define workflows using LangGraph StateGraph. Use conditional edges for dynamic routing. Implement proper error handling and retry logic.

## Integration Points & Patterns
- **Alpha Vantage API**: Use official client or REST calls; wrap as LangGraph tools for ingestion nodes.
- **AWS Services**: Prefer SDK calls over CLI. For ETL, use Glue jobs; for streaming, use Kinesis. All integrated as LangGraph tools.
- **LangGraph**: Primary framework for workflow orchestration, state management, and agent coordination.
- **LLM Integration**: Configure ChatOpenAI with GPT-5-mini for ReAct agents and intelligent workflow routing.
- **State Flow**: Use LangGraph StateGraph for workflow execution with automatic persistence and conditional routing.

## Key Files & Directories
- `README.md`: High-level architecture and workflow responsibilities.
- `pyproject.toml`: Poetry dependency management with LangGraph.
- `src/workflows/`: LangGraph workflow definitions and state graphs.
- `src/nodes/`: Individual workflow nodes (agents as functions).
- `src/tools/`: LangGraph tools for AWS, Alpha Vantage, and external integrations.
- `src/state/`: State management schemas and persistence logic.
- `scripts/`, `glue_jobs/`: ETL scripts integrated as LangGraph tools.
- `tests/`: Test suite with LangGraph workflow tests.
- `notebooks/`: Exploratory analysis.

## Example Patterns
- LangGraph workflow with GPT-5-mini integration:
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
  
  # Add nodes
  workflow.add_node("ingestion", data_ingestion_node)
  workflow.add_node("preprocessing", preprocessing_node)
  workflow.add_node("training", training_node)
  
  # Add edges with GPT-5-mini powered routing
  workflow.add_conditional_edges("ingestion", route_next_step)
  workflow.add_edge("preprocessing", "training")
  
  # Compile with GPT-5-mini for intelligent routing
  app = workflow.compile()
  ```
- Message structure:
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
- Coordination pattern:
  ```python
  from src.coordination.agent_coordinator import AgentCoordinator
  
  coordinator = AgentCoordinator(message_bus, config)
  await coordinator.coordinate_workflow("workflow_123", ["AAPL", "GOOGL"])
  ```
- Logging:
  ```python
  from loguru import logger
  logger.info("Agent started with hybrid coordination model")
  ```
- AWS SDK usage:
  ```python
  import boto3
  s3 = boto3.client('s3')
  s3.upload_file(...)
  ```

---

**Review these instructions for accuracy and completeness. Suggest improvements if any section is unclear or missing project-specific details.**