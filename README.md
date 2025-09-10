# llm-multi-ai-agent-system

Description of the project: I want to build a LLM multi-agent system pipeline to predict stock prices. The agents are equiped with AWS tools consisting of:

Data ingestion agent: ingest real financial data from Alpha Vantage API and other real free sources
- Responsibilities: Move data from external sources → AWS S3.
- WS Tools: boto3 for S3, AWS Glue for ETL, Kinesis for streaming.
- LLM Role: Can generate Glue/Spark scripts for schema detection.

Preprocessing Agent
- Responsibilities: Clean, transform, feature-engineer.
- AWS Tools: SageMaker Processing, AWS Glue (for ETL-scale jobs).
- LLM Role: Generates Pandas/Scikit-learn code → runs as SageMaker job.

Training Agent
- Responsibilities: Select algorithm, launch training job, track metrics.
- AWS Tools: SageMaker Training Jobs (XGBoost, PyTorch, TensorFlow).
- LLM Role: Decides algorithm & hyperparameters (or uses Autopilot).


Some tech for multi-agent:
- LLM Framework: use langchain
- Communication: Python async messaging inside one app.
- Execution: Each agent runs AWS SDK (boto3) calls or generates scripts.
- Persistence: S3 for data, DynamoDB or RDS for agent states.

The project should use, as default, peotry for file dependecies and loguru for logging.
