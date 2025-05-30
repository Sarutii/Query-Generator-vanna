#!/usr/bin/env python3
"""
Vanna Vector Store Training and Diagnostic Script
This script helps you populate and verify your ChromaDB vector store for RAG functionality.
"""

import os
import json
import logging
import oracledb
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        config = config or {}
        config['model'] = 'mistral'
        config['persist_directory'] = './vanna-data'
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

def connect_to_oracle():
    """Connect to Oracle database"""
    try:
        # Initialize Oracle client
        oracledb.init_oracle_client(lib_dir=r"C:\Users\ahmed\Downloads\instantclient-basic-windows.x64-23.8.0.25.04\instantclient_23_8")
        
        connection = oracledb.connect(
            user='IAS202538',
            password='123',
            dsn='localhost:1521/xepdb1'
        )
        return connection
    except Exception as e:
        logging.error(f"Failed to connect to Oracle: {e}")
        return None

def get_schema_information(connection):
    """Extract schema information from Oracle database"""
    schema_info = {
        'tables': {},
        'views': {},
        'columns': {}
    }
    
    try:
        with connection.cursor() as cursor:
            # Get all tables owned by the user
            cursor.execute("""
                SELECT table_name, num_rows, last_analyzed
                FROM user_tables
                ORDER BY table_name
            """)
            
            tables = cursor.fetchall()
            logging.info(f"Found {len(tables)} tables")
            
            for table_name, num_rows, last_analyzed in tables:
                schema_info['tables'][table_name] = {
                    'row_count': num_rows,
                    'last_analyzed': str(last_analyzed) if last_analyzed else None
                }
                
                # Get column information for each table
                cursor.execute("""
                    SELECT column_name, data_type, data_length, nullable, data_default
                    FROM user_tab_columns
                    WHERE table_name = :table_name
                    ORDER BY column_id
                """, {'table_name': table_name})
                
                columns = cursor.fetchall()
                schema_info['columns'][table_name] = []
                
                for col_name, data_type, data_length, nullable, data_default in columns:
                    schema_info['columns'][table_name].append({
                        'name': col_name,
                        'type': data_type,
                        'length': data_length,
                        'nullable': nullable == 'Y',
                        'default': str(data_default) if data_default else None
                    })
            
            # Get views
            cursor.execute("""
                SELECT view_name, text
                FROM user_views
                ORDER BY view_name
            """)
            
            views = cursor.fetchall()
            logging.info(f"Found {len(views)} views")
            
            for view_name, view_text in views:
                schema_info['views'][view_name] = {
                    'definition': view_text
                }
                
    except Exception as e:
        logging.error(f"Error extracting schema: {e}")
    
    return schema_info

def create_training_data(schema_info):
    """Create training data from schema information"""
    training_data = {
        #ddl statements read from sql file
        'ddl_statements': [],
        'documentation': [],
        'sample_questions': []
    }

    #read DDL statements from SQL file
    ddl_file_path = r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Query-Generator-vanna\oracle_schema_20250415_054319.sql"
    if os.path.exists(ddl_file_path):
        with open(ddl_file_path, 'r') as f:
            ddl_statements = f.read().strip().split(';')
            training_data['ddl_statements'].extend([stmt.strip() for stmt in ddl_statements if stmt.strip()])
    
    #read documentation from text file and split at line like ------------------------------------------------------------
    # This assumes the documentation is structured with sections separated by newlines
    
    doc_file_path = r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Query-Generator-vanna\Full Data Discrebtion.txt"
    if os.path.exists(doc_file_path):
        try:
            with open(doc_file_path, 'r', encoding='utf-8') as f:
                documentation = f.read().strip().split('------------------------------------------------------------')
                training_data['documentation'].extend([doc.strip() for doc in documentation if doc.strip()])

        except UnicodeDecodeError:
            # Try alternative encodings if UTF-8 fails
            try:
                with open(doc_file_path, 'r', encoding='latin1') as f:
                    documentation = f.read().strip().split('------------------------------------------------------------')
                    training_data['documentation'].extend([doc.strip() for doc in documentation if doc.strip()])
                logging.info("Documentation file read with latin1 encoding")
            except Exception as e:
                logging.error(f"Failed to read documentation file: {e}")
    
    # Generate DDL statements
    # for table_name, table_info in schema_info['tables'].items():
    #     if table_name in schema_info['columns']:
    #         columns = schema_info['columns'][table_name]
            
    #         # Create table description
    #         ddl = f"-- Table: {table_name}\n"
    #         if table_info['row_count']:
    #             ddl += f"-- Rows: {table_info['row_count']}\n"
            
    #         ddl += f"CREATE TABLE {table_name} (\n"
    #         col_definitions = []
            
    #         for col in columns:
    #             col_def = f"    {col['name']} {col['type']}"
    #             if col['length'] and col['type'] in ['VARCHAR2', 'CHAR']:
    #                 col_def += f"({col['length']})"
    #             if not col['nullable']:
    #                 col_def += " NOT NULL"
    #             col_definitions.append(col_def)
            
    #         ddl += ",\n".join(col_definitions)
    #         ddl += "\n);"
            
    #         training_data['ddl_statements'].append(ddl)
            
            # # Create documentation
            # doc = f"Table {table_name} contains the following columns:\n"
            # for col in columns:
            #     doc += f"- {col['name']}: {col['type']}"
            #     if col['length'] and col['type'] in ['VARCHAR2', 'CHAR']:
            #         doc += f"({col['length']})"
            #     doc += f" ({'nullable' if col['nullable'] else 'not null'})\n"
            
            # training_data['documentation'].append(doc)
    
    # Generate sample questions and SQL pairs
    sample_questions = [
        ("Show me all accounts", "SELECT * FROM ACCOUNT WHERE ROWNUM <= 100"),
        ("Count total number of accounts", "SELECT COUNT(*) FROM ACCOUNT"),
        ("Get account types", "SELECT * FROM ACCOUNT_TYPES"),
        ("Show account groupings", "SELECT * FROM ACCOUNT_GROUPING"),
        ("List all branches", "SELECT * FROM ALL_BRANCHES"),
        ("Show recent maintenance jobs", "SELECT * FROM AMS_MNT_JOB_ORDR WHERE ROWNUM <= 50"),
        ("Get asset information", "SELECT * FROM AMS_PLN_AST WHERE ROWNUM <= 100"),
        ("Show service notifications", "SELECT * FROM AMS_SRV_NTF WHERE ROWNUM <= 50"),
        ("List customer changes", "SELECT * FROM ARS_CSTMR_CHNG_MST WHERE ROWNUM <= 100"),
        ("Show sales orders", "SELECT * FROM ARS_AUTO_SLS_ORDR_MST WHERE ROWNUM <= 100"),
    ]
    
    training_data['sample_questions'] = sample_questions
    
    return training_data

def train_vanna_model(vn, training_data):
    """Train the Vanna model with schema and sample data"""
    logging.info("Starting Vanna model training...")
    
    # Add DDL statements
    logging.info(f"Adding {len(training_data['ddl_statements'])} DDL statements...")
    for i, ddl in enumerate(training_data['ddl_statements']):
        try:
            vn.train(ddl=ddl)
            if (i + 1) % 10 == 0:
                logging.info(f"Added {i + 1} DDL statements")
        except Exception as e:
            logging.warning(f"Failed to add DDL {i}: {e}")
    
    # Add documentation
    logging.info(f"Adding {len(training_data['documentation'])} documentation entries...")
    for i, doc in enumerate(training_data['documentation']):
        try:
            vn.train(documentation=doc)
            if (i + 1) % 10 == 0:
                logging.info(f"Added {i + 1} documentation entries")
        except Exception as e:
            logging.warning(f"Failed to add documentation {i}: {e}")
    
    # Add sample questions
    logging.info(f"Adding {len(training_data['sample_questions'])} sample Q&A pairs...")
    for i, (question, sql) in enumerate(training_data['sample_questions']):
        try:
            vn.train(question=question, sql=sql)
            if (i + 1) % 5 == 0:
                logging.info(f"Added {i + 1} Q&A pairs")
        except Exception as e:
            logging.warning(f"Failed to add Q&A pair {i}: {e}")
    
    logging.info("Training completed!")

def diagnose_vector_store(vn):
    """Diagnose the current state of the vector store"""
    logging.info("=== VECTOR STORE DIAGNOSTICS ===")
    
    try:
        # Check if data directory exists
        if os.path.exists('./vanna-data'):
            logging.info("✅ Vanna data directory exists")
            
            # List contents
            contents = os.listdir('./vanna-data')
            logging.info(f"Directory contents: {contents}")
            
            # Check ChromaDB collection
            try:
                # Try to get similar questions
                test_question = "show me accounts"
                similar = vn.get_similar_question_sql_pairs(test_question)
                logging.info(f"✅ Found {len(similar)} similar questions for test query")
                
                if similar:
                    logging.info("Sample similar questions:")
                    for i, (q, sql) in enumerate(similar[:3]):
                        logging.info(f"  {i+1}. Q: {q}")
                        logging.info(f"     SQL: {sql[:100]}...")
                
                # Try to get DDL
                ddl_info = vn.get_related_ddl(test_question)
                logging.info(f"✅ Found {len(ddl_info)} DDL entries for test query")
                
                if ddl_info:
                    logging.info("Sample DDL entries:")
                    for i, ddl in enumerate(ddl_info[:2]):
                        logging.info(f"  {i+1}. {ddl[:100]}...")
                
                # Try to get documentation
                docs = vn.get_related_documentation(test_question)
                logging.info(f"✅ Found {len(docs)} documentation entries for test query")
                
                if docs:
                    logging.info("Sample documentation:")
                    for i, doc in enumerate(docs[:2]):
                        logging.info(f"  {i+1}. {doc[:100]}...")
                
            except Exception as e:
                logging.error(f"❌ Error accessing vector store data: {e}")
                return False
                
        else:
            logging.warning("⚠️ Vanna data directory does not exist")
            return False
            
    except Exception as e:
        logging.error(f"❌ Diagnostic failed: {e}")
        return False
    
    return True

def save_training_data(training_data, filename='training_data.json'):
    """Save training data to file for backup"""
    try:
        with open(filename, 'w') as f:
            # Convert to serializable format
            serializable_data = {
                'ddl_statements': training_data['ddl_statements'],
                'documentation': training_data['documentation'],
                'sample_questions': training_data['sample_questions']
            }
            json.dump(serializable_data, f, indent=2, default=str)
        logging.info(f"Training data saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save training data: {e}")

def main():
    """Main function to run training or diagnostics"""
    print("=== Vanna Vector Store Training and Diagnostics ===")
    print("1. Diagnose current vector store")
    print("2. Extract schema and train model")
    print("3. Both (diagnose then train)")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    # Initialize Vanna
    vn = MyVanna()
    
    if choice in ['1', '3']:
        print("\n--- Running Diagnostics ---")
        has_data = diagnose_vector_store(vn)
        
        if has_data and choice == '1':
            print("✅ Vector store appears to have data!")
            return
        elif not has_data:
            print("⚠️ Vector store appears empty or has issues")
    
    if choice in ['2', '3']:
        print("\n--- Starting Training Process ---")
        
        # Connect to Oracle
        connection = connect_to_oracle()
        if not connection:
            print("❌ Failed to connect to Oracle database")
            return
        
        try:
            # Extract schema
            print("Extracting schema information...")
            schema_info = get_schema_information(connection)
            print(f"Found {len(schema_info['tables'])} tables and {len(schema_info['views'])} views")
            
            # Create training data
            print("Creating training data...")
            training_data = create_training_data(schema_info)
            
            # Save training data for backup
            save_training_data(training_data)
            
            # Connect Vanna to Oracle
            vn.connect_to_oracle(
                user='IAS202538',
                password='123',
                dsn="localhost:1521/xepdb1"
            )
            
            # Train the model
            train_vanna_model(vn, training_data)
            
            # Run final diagnostics
            print("\n--- Final Diagnostics ---")
            diagnose_vector_store(vn)
            
        finally:
            connection.close()
    
    print("\n=== Process Complete ===")

def quick_check():
    """Quick check of vector store status"""
    print("=== Quick Vector Store Check ===")
    
    if not os.path.exists('./vanna-data'):
        print("❌ No vanna-data directory found")
        print("Run this script with option 2 or 3 to create training data")
        return False
    
    vn = MyVanna()
    
    try:
        # Test queries
        test_queries = [
            "show me all accounts",
            "count customers", 
            "list all tables",
            "get maintenance jobs"
        ]
        
        total_results = 0
        for query in test_queries:
            similar = vn.get_similar_question_sql_pairs(query)
            ddl = vn.get_related_ddl(query)
            docs = vn.get_related_documentation(query)
            
            results = len(similar) + len(ddl) + len(docs)
            total_results += results
            print(f"Query '{query}': {results} total results (Q&A: {len(similar)}, DDL: {len(ddl)}, Docs: {len(docs)})")
        
        if total_results > 0:
            print(f"✅ Vector store has data! Total results across test queries: {total_results}")
            return True
        else:
            print("⚠️ Vector store exists but appears empty")
            return False
            
    except Exception as e:
        print(f"❌ Error checking vector store: {e}")
        return False

if __name__ == "__main__":
    # Quick check first
    if len(os.sys.argv) > 1 and os.sys.argv[1] == 'quick':
        quick_check()
    else:
        main()