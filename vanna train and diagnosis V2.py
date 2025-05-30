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
        'ddl_statements': [],
        'documentation': [],
        'sample_questions': []
    }
    
    # Generate DDL statements
    for table_name, table_info in schema_info['tables'].items():
        if table_name in schema_info['columns']:
            columns = schema_info['columns'][table_name]
            
            # Create table description
            ddl = f"-- Table: {table_name}\n"
            if table_info['row_count']:
                ddl += f"-- Rows: {table_info['row_count']}\n"
            
            ddl += f"CREATE TABLE {table_name} (\n"
            col_definitions = []
            
            for col in columns:
                col_def = f"    {col['name']} {col['type']}"
                if col['length'] and col['type'] in ['VARCHAR2', 'CHAR']:
                    col_def += f"({col['length']})"
                if not col['nullable']:
                    col_def += " NOT NULL"
                col_definitions.append(col_def)
            
            ddl += ",\n".join(col_definitions)
            ddl += "\n);"
            
            training_data['ddl_statements'].append(ddl)
            
            # Create documentation
            doc = f"Table {table_name} contains the following columns:\n"
            for col in columns:
                doc += f"- {col['name']}: {col['type']}"
                if col['length'] and col['type'] in ['VARCHAR2', 'CHAR']:
                    doc += f"({col['length']})"
                doc += f" ({'nullable' if col['nullable'] else 'not null'})\n"
            
            training_data['documentation'].append(doc)
    
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
                test_question = "show me accounts"
                
                # Check available methods and try different approaches
                logging.info("Checking available Vanna methods...")
                methods = [attr for attr in dir(vn) if not attr.startswith('_')]
                logging.info(f"Available methods: {[m for m in methods if 'similar' in m.lower() or 'ddl' in m.lower() or 'doc' in m.lower()]}")
                
                # Try different method names for getting training data
                similar_data = []
                ddl_data = []
                doc_data = []
                
                # Method 1: Try get_training_data
                if hasattr(vn, 'get_training_data'):
                    training_data = vn.get_training_data()
                    logging.info(f"✅ Retrieved training data: {type(training_data)}")
                    if isinstance(training_data, list):
                        logging.info(f"Training data length: {len(training_data)}")
                        for i, item in enumerate(training_data[:3]):
                            logging.info(f"  Item {i+1}: {str(item)[:100]}...")
                
                # Method 2: Try similarity search directly
                if hasattr(vn, 'get_similar_question_sql'):
                    similar_data = vn.get_similar_question_sql(test_question)
                elif hasattr(vn, 'get_similar_sql'):
                    similar_data = vn.get_similar_sql(test_question)
                elif hasattr(vn, 'similarity_search'):
                    similar_data = vn.similarity_search(test_question)
                
                if similar_data:
                    logging.info(f"✅ Found {len(similar_data)} similar items")
                    for i, item in enumerate(similar_data[:3]):
                        logging.info(f"  {i+1}. {str(item)[:100]}...")
                
                # Method 3: Try DDL methods
                if hasattr(vn, 'get_related_ddl'):
                    ddl_data = vn.get_related_ddl(test_question)
                elif hasattr(vn, 'get_ddl'):
                    ddl_data = vn.get_ddl(test_question)
                
                if ddl_data:
                    logging.info(f"✅ Found {len(ddl_data)} DDL entries")
                    for i, ddl in enumerate(ddl_data[:2]):
                        logging.info(f"  {i+1}. {str(ddl)[:100]}...")
                
                # Method 4: Try documentation methods
                if hasattr(vn, 'get_related_documentation'):
                    doc_data = vn.get_related_documentation(test_question)
                elif hasattr(vn, 'get_documentation'):
                    doc_data = vn.get_documentation(test_question)
                
                if doc_data:
                    logging.info(f"✅ Found {len(doc_data)} documentation entries")
                    for i, doc in enumerate(doc_data[:2]):
                        logging.info(f"  {i+1}. {str(doc)[:100]}...")
                
                # Method 5: Try accessing ChromaDB directly
                if hasattr(vn, 'chroma_collection') or hasattr(vn, 'collection'):
                    collection = getattr(vn, 'chroma_collection', None) or getattr(vn, 'collection', None)
                    if collection:
                        try:
                            count = collection.count()
                            logging.info(f"✅ ChromaDB collection has {count} documents")
                            
                            # Try to query the collection
                            results = collection.query(
                                query_texts=[test_question],
                                n_results=3
                            )
                            if results:
                                logging.info(f"✅ Sample query returned {len(results.get('documents', [[]]))} results")
                        except Exception as e:
                            logging.warning(f"Could not query ChromaDB collection: {e}")
                
                # Summary
                total_data = len(similar_data) + len(ddl_data) + len(doc_data)
                if total_data > 0:
                    logging.info(f"✅ Vector store contains data! Total items found: {total_data}")
                    return True
                else:
                    logging.warning("⚠️ No training data found in vector store")
                    return False
                
            except Exception as e:
                logging.error(f"❌ Error accessing vector store data: {e}")
                import traceback
                logging.error(f"Full traceback: {traceback.format_exc()}")
                return False
                
        else:
            logging.warning("⚠️ Vanna data directory does not exist")
            return False
            
    except Exception as e:
        logging.error(f"❌ Diagnostic failed: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
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
        # First, let's see what methods are available
        methods = [attr for attr in dir(vn) if not attr.startswith('_')]
        relevant_methods = [m for m in methods if any(keyword in m.lower() for keyword in ['similar', 'ddl', 'doc', 'train', 'query'])]
        print(f"Available relevant methods: {relevant_methods}")
        
        # Test different approaches
        test_queries = [
            "show me all accounts",
            "count customers", 
        ]
        
        total_results = 0
        for query in test_queries:
            query_results = 0
            
            # Try different method combinations
            try:
                if hasattr(vn, 'get_similar_question_sql'):
                    similar = vn.get_similar_question_sql(query)
                    query_results += len(similar) if similar else 0
                elif hasattr(vn, 'get_similar_sql'):
                    similar = vn.get_similar_sql(query)
                    query_results += len(similar) if similar else 0
            except:
                pass
            
            try:
                if hasattr(vn, 'get_related_ddl'):
                    ddl = vn.get_related_ddl(query)
                    query_results += len(ddl) if ddl else 0
            except:
                pass
            
            try:
                if hasattr(vn, 'get_related_documentation'):
                    docs = vn.get_related_documentation(query)
                    query_results += len(docs) if docs else 0
            except:
                pass
            
            # Try accessing training data directly
            try:
                if hasattr(vn, 'get_training_data'):
                    training_data = vn.get_training_data()
                    if training_data:
                        query_results += len(training_data)
                        break  # Only count once
            except:
                pass
            
            total_results += query_results
            print(f"Query '{query}': {query_results} results")
        
        # Try to access ChromaDB collection directly
        try:
            if hasattr(vn, 'chroma_collection'):
                collection = vn.chroma_collection
                if collection:
                    count = collection.count()
                    print(f"ChromaDB collection document count: {count}")
                    total_results += count
            elif hasattr(vn, 'collection'):
                collection = vn.collection
                if collection:
                    count = collection.count()
                    print(f"ChromaDB collection document count: {count}")
                    total_results += count
        except Exception as e:
            print(f"Could not access ChromaDB collection: {e}")
        
        if total_results > 0:
            print(f"✅ Vector store has data! Total results: {total_results}")
            return True
        else:
            print("⚠️ Vector store exists but appears empty")
            print("Run the training process (option 2 or 3) to populate it")
            return False
            
    except Exception as e:
        print(f"❌ Error checking vector store: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    # Quick check first
    if len(os.sys.argv) > 1 and os.sys.argv[1] == 'quick':
        quick_check()
    else:
        main()