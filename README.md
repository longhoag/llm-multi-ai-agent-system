# LLM Multi-Agent System for Stock Price Prediction

A sophisticated multi-agent system that uses Large Language Models (LLMs) and AWS services to predict stock prices through intelligent data ingestion, preprocessing, and machine learning workflows.

## 🏗️ Architecture Overview

The system implements a hybrid coordination model with three specialized agents:

### **Data Ingestion Agent** 
- **Responsibilities**: Fetch real financial data from Alpha Vantage API and other sources
- **AWS Tools**: S3 for storage, Glue for ETL, Kinesis for streaming
- **LLM Role**: GPT-5-mini powered strategy optimization and data validation

### **Preprocessing Agent** (Coming Soon)
- **Responsibilities**: Clean, transform, and feature-engineer raw financial data
- **AWS Tools**: SageMaker Processing, AWS Glue for large-scale ETL jobs  
- **LLM Role**: Generate Pandas/Scikit-learn code and feature engineering strategies

### **Training Agent** (Coming Soon)
- **Responsibilities**: Select algorithms, launch training jobs, track metrics
- **AWS Tools**: SageMaker Training Jobs (XGBoost, PyTorch, TensorFlow)
- **LLM Role**: Algorithm selection and hyperparameter optimization

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+** with Poetry installed
2. **AWS Account** with programmatic access
3. **API Keys**: Alpha Vantage (free) and OpenAI

### Installation

1. **Clone and setup**:
```bash
git clone <repository-url>
cd llm-multi-ai-agent-system
poetry install
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your actual API keys and AWS credentials
```

3. **Set up AWS S3 buckets**:
```bash
# Create S3 buckets (replace with your unique names)
aws s3 mb s3://your-project-stock-data-raw
aws s3 mb s3://your-project-stock-data-processed  
aws s3 mb s3://your-project-ml-models
```

4. **Run the system**:
```bash
# Full system
poetry run python main.py

# Test single ingestion
poetry run python main.py --mode test --symbol AAPL
```

## 📁 Project Structure

```
src/
├── agents/
│   ├── base_agent.py              # Base agent class
│   └── data_ingestion_agent.py    # Data ingestion implementation
├── messaging/
│   └── message_bus.py             # Async message bus
├── storage/
│   └── s3_manager.py              # S3 operations manager
├── external/
│   └── alpha_vantage_client.py    # Alpha Vantage API client
├── config/
│   └── settings.py                # Configuration management
└── orchestrator/
    └── system_orchestrator.py     # System coordination
```

## 🔧 Configuration

Key environment variables in `.env`:

```bash
# Required API Keys
ALPHA_VANTAGE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# AWS Configuration  
AWS_REGION=us-east-1
S3_BUCKET_RAW_DATA=your-raw-data-bucket

# System Settings
DEFAULT_SYMBOLS=AAPL,GOOGL,MSFT,TSLA,AMZN
LOG_LEVEL=INFO
DEVELOPMENT_MODE=true
```

## 🧪 Testing

Run the test suite:

```bash
# All tests
poetry run pytest

# Specific test file
poetry run pytest tests/test_data_ingestion_agent.py

# With coverage
poetry run pytest --cov=src
```

## 🏃‍♂️ Usage Examples

### Run Data Ingestion for Specific Symbols

```python
from src.orchestrator.system_orchestrator import SystemOrchestrator

async def custom_ingestion():
    orchestrator = SystemOrchestrator()
    await orchestrator.initialize()
    
    # Start pipeline for specific symbols
    workflow_id = await orchestrator.start_stock_pipeline(
        symbols=["AAPL", "GOOGL", "TSLA"]
    )
    
    print(f"Pipeline {workflow_id} started")
    await orchestrator.shutdown()
```

### Monitor System Status

```python
# Get comprehensive system status
status = await orchestrator.get_system_status()
print(status)

# Get agent statistics
stats = await orchestrator.agents["data_ingestion"].get_ingestion_stats()
print(f"Ingestion stats: {stats}")
```

## 🔄 System Coordination

The system uses a **hybrid coordination model**:

- **Peer-to-Peer**: Agents communicate directly via async message bus
- **Hierarchical**: System Orchestrator provides high-level coordination
- **GPT-5-mini Powered**: Dynamic coordination decisions based on system state
- **Message-Driven**: All communication uses structured messages with correlation IDs

## 📊 Data Flow

```
Alpha Vantage API → Data Ingestion Agent → S3 Raw Data
                                       ↓
S3 Raw Data → Preprocessing Agent → S3 Processed Data  
                                  ↓
S3 Processed Data → Training Agent → SageMaker → S3 Models
```

## 🛠️ Development

### Adding New Agents

1. Inherit from `BaseAgent`
2. Implement `process_message()` method
3. Register in `SystemOrchestrator`
4. Add configuration and tests

### Message Types

Standard message types:
- `INGEST_REQUEST`: Request data ingestion
- `DATA_AVAILABLE`: Notify data is ready
- `SCHEDULE_INGESTION`: Batch ingestion request
- `INGESTION_ERROR`: Error notification

## 📈 Monitoring & Logging

- **Logging**: Structured logging with `loguru`
- **Metrics**: Agent performance statistics
- **Health Checks**: System and agent health monitoring
- **Error Handling**: Comprehensive error tracking and recovery

## 🚨 Error Handling

- **Retry Logic**: Exponential backoff for API failures
- **Circuit Breaker**: Protection against cascading failures  
- **Graceful Degradation**: Continue operation with partial failures
- **Recovery**: Automatic retry and manual intervention options

## 🔐 Security

- **API Keys**: Stored in environment variables
- **AWS IAM**: Least privilege access policies
- **S3 Encryption**: Server-side encryption enabled
- **Network**: VPC endpoints for secure AWS communication

## 📋 TODO / Roadmap

- [ ] Implement Preprocessing Agent
- [ ] Implement Training Agent  
- [ ] Add real-time data streaming with Kinesis
- [ ] Implement model deployment and serving
- [ ] Add web dashboard for monitoring
- [ ] Add more data sources (Yahoo Finance, IEX)
- [ ] Implement backtesting capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions or issues:
1. Check the [GitHub Issues](https://github.com/your-repo/issues)
2. Review the `.github/copilot-instructions.md` for AI coding guidelines
3. Check system logs in the `logs/` directory





