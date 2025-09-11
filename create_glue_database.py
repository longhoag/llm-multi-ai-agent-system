"""
Create AWS Glue Database for LangGraph Preprocessing Node
"""

import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def create_glue_database():
    """Create Glue database programmatically following copilot instructions"""
    
    glue_client = boto3.client(
        'glue',
        region_name=os.getenv('AWS_REGION'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    
    database_name = os.getenv('GLUE_DATABASE_NAME', 'stock_prediction_db')
    
    try:
        # Try to create the database
        response = glue_client.create_database(
            DatabaseInput={
                'Name': database_name,
                'Description': 'Database for LangGraph multi-agent stock prediction preprocessing workflow'
            }
        )
        
        print(f"✅ Successfully created Glue database: {database_name}")
        print(f"🔧 Database ready for LangGraph preprocessing node tools")
        return True
        
    except glue_client.exceptions.AlreadyExistsException:
        print(f"ℹ️  Database '{database_name}' already exists")
        
        # Try to get database details
        try:
            db_detail = glue_client.get_database(Name=database_name)
            print(f"📋 Database details: {db_detail['Database']['Name']}")
            print(f"✅ Database is accessible and ready")
            return True
        except Exception as e:
            print(f"⚠️  Database exists but access error: {e}")
            return False
        
    except Exception as e:
        print(f"💥 Error creating database: {e}")
        print("🔧 Check your AWS Glue permissions")
        return False

if __name__ == "__main__":
    create_glue_database()
