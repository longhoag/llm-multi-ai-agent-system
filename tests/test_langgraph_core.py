"""
Core LangGraph Integration Tests

This module contains the essential tests for validating the LangGraph
migration and workflow functionality.
"""

import pytest
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.state.workflow_state import WorkflowStatus, WorkflowStateManager
from src.tools.agent_tools import (
    fetch_stock_data_tool,
    upload_to_s3_tool,
    validate_data_quality_tool
)


class TestLangGraphCore:
    """Core LangGraph functionality tests"""
    
    def test_langgraph_imports(self):
        """Test that LangGraph imports are working"""
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.prebuilt import create_react_agent
        assert True  # If we get here, imports worked
    
    def test_workflow_creation(self):
        """Test basic workflow creation"""
        workflow = StateGraph(dict)
        
        def test_node(state):
            return {"result": "processed"}
        
        workflow.add_node("test", test_node)
        workflow.add_edge("test", END)
        workflow.set_entry_point("test")
        
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        
        result = app.invoke({"input": "test"}, {"configurable": {"thread_id": "test"}})
        assert result["result"] == "processed"
    
    def test_state_management(self):
        """Test workflow state management"""
        manager = WorkflowStateManager()
        state = manager.create_initial_state(
            symbols=["AAPL"],
            timeframe="daily", 
            prediction_horizon=30
        )
        
        assert state["symbols"] == ["AAPL"]
        assert state["timeframe"] == "daily"
        assert state["prediction_horizon"] == 30
        assert state["status"] == WorkflowStatus.RUNNING
    
    def test_workflow_status_enum(self):
        """Test WorkflowStatus enum values"""
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running" 
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
    
    def test_agent_tools_available(self):
        """Test that agent tools are properly defined"""
        tools = [fetch_stock_data_tool, upload_to_s3_tool, validate_data_quality_tool]
        
        for tool in tools:
            assert hasattr(tool, "__call__")  # Can be called
            assert hasattr(tool, "__name__")   # Has a name


class TestWorkflowIntegration:
    """Integration tests for complete workflows"""
    
    def test_conditional_routing_workflow(self):
        """Test workflow with conditional routing"""
        workflow = StateGraph(dict)
        
        def ingestion_node(state):
            state["ingestion_complete"] = True
            return state
        
        def processing_node(state):
            state["processing_complete"] = True
            return state
        
        def should_process(state):
            return "processing" if state.get("ingestion_complete") else END
        
        workflow.add_node("ingestion", ingestion_node)
        workflow.add_node("processing", processing_node)
        
        workflow.add_conditional_edges(
            "ingestion",
            should_process,
            {
                "processing": "processing",
                END: END
            }
        )
        workflow.add_edge("processing", END)
        workflow.set_entry_point("ingestion")
        
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        
        result = app.invoke({}, {"configurable": {"thread_id": "test-routing"}})
        
        assert result["ingestion_complete"] is True
        assert result["processing_complete"] is True
    
    def test_workflow_nodes_importable(self):
        """Test that workflow nodes can be imported"""
        from src.nodes.workflow_nodes import (
            data_ingestion_node,
            preprocessing_node,
            training_node
        )
        
        # Nodes should be callable
        assert callable(data_ingestion_node)
        assert callable(preprocessing_node) 
        assert callable(training_node)


@pytest.mark.asyncio
async def test_async_workflow_execution():
    """Test async workflow execution patterns"""
    
    async def async_node(state):
        # Simulate async operation
        import asyncio
        await asyncio.sleep(0.01)
        state["async_complete"] = True
        return state
    
    workflow = StateGraph(dict)
    workflow.add_node("async_test", async_node)
    workflow.add_edge("async_test", END)
    workflow.set_entry_point("async_test")
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    # Note: LangGraph workflows are executed synchronously even with async nodes
    result = app.invoke({}, {"configurable": {"thread_id": "async-test"}})
    assert result["async_complete"] is True
