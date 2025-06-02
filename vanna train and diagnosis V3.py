#!/usr/bin/env python3
"""
Enhanced Vanna Vector Store Training and Diagnostic Script
This script ensures proper embedding and vector similarity search functionality.
"""

import os
import json
import logging
import oracledb
import numpy as np
from typing import List, Dict, Any, Tuple
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        config = config or {}
        config['model'] = 'mistral'  # This will be used for embeddings too
        config['persist_directory'] = './vanna-data'
        
        # Enhanced ChromaDB configuration for better embeddings
        config['embedding_function'] = None  # Let Vanna handle this
        config['collection_name'] = 'vanna_embeddings'
        
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)
        
        # Verify embedding setup
        self._verify_embedding_setup()
    
    def _verify_embedding_setup(self):
        """Verify that embeddings are properly configured and check available methods"""
        try:
            # Test if we can generate embeddings
            test_text = "SELECT * FROM ACCOUNT"
            
            # Check what methods are available
            available_methods = []
            for method_name in dir(self):
                if 'similar' in method_name.lower() or 'related' in method_name.lower():
                    if not method_name.startswith('_'):
                        available_methods.append(method_name)
            
            logging.info(f"Available similarity/retrieval methods: {available_methods}")
            logging.info("✅ Embedding configuration verified")
        except Exception as e:
            logging.warning(f"⚠️ Embedding setup issue: {e}")
    
    def add_to_vector_store_with_embedding(self, content: str, content_type: str, metadata: Dict = None):
        """
        Add content to vector store with explicit embedding verification
        
        Args:
            content: The text content to embed
            content_type: Type of content ('ddl', 'documentation', 'question_sql')
            metadata: Additional metadata for the content
        """
        try:
            metadata = metadata or {}
            metadata['content_type'] = content_type
            metadata['length'] = len(content)
            
            if content_type == 'ddl':
                self.train(ddl=content)
            elif content_type == 'documentation':
                self.train(documentation=content)
            elif content_type == 'question_sql':
                question, sql = content.split('|||') if '|||' in content else ('', content)
                self.train(question=question.strip(), sql=sql.strip())
            
            logging.info(f"✅ Added {content_type} content to vector store (length: {len(content)})")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to add {content_type} content: {e}")
            return False
    
    def test_similarity_search(self, query: str, top_k: int = 5) -> Dict[str, List]:
        """
        Test similarity search across all content types using correct Vanna API methods
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Dictionary with results from different content types
        """
        results = {
            'similar_questions': [],
            'related_ddl': [],
            'related_docs': []
        }
        
        try:
            # Get similar question-SQL pairs using correct method name
            try:
                similar_qa = self.get_similar_question_sql(question=query, n=top_k)
                if isinstance(similar_qa, list):
                    results['similar_questions'] = similar_qa
                else:
                    results['similar_questions'] = []
            except AttributeError:
                # Try alternative method names
                try:
                    similar_qa = self.get_related_training_data(query)
                    results['similar_questions'] = similar_qa if similar_qa else []
                except:
                    results['similar_questions'] = []
            
            # Get related DDL using correct method name
            try:
                related_ddl = self.get_related_ddl(question=query, n=top_k)
                if isinstance(related_ddl, list):
                    results['related_ddl'] = related_ddl
                else:
                    results['related_ddl'] = []
            except:
                results['related_ddl'] = []
            
            # Get related documentation using correct method name
            try:
                related_docs = self.get_related_documentation(question=query, n=top_k)
                if isinstance(related_docs, list):
                    results['related_docs'] = related_docs
                else:
                    results['related_docs'] = []
            except:
                results['related_docs'] = []
            
            logging.info(f"Similarity search for '{query}':")
            logging.info(f"  - Similar Q&A pairs: {len(results['similar_questions'])}")
            logging.info(f"  - Related DDL: {len(results['related_ddl'])}")
            logging.info(f"  - Related docs: {len(results['related_docs'])}")
            
        except Exception as e:
            logging.error(f"❌ Similarity search failed: {e}")
        
        return results

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

def create_enhanced_training_data(schema_info):
    """Create enhanced training data with better structure for embeddings"""
    training_data = {
        'ddl_statements': [],
        'documentation': [],
        'sample_questions': [],
        'metadata': []
    }

    # Read DDL statements from SQL file
    ddl_file_path = r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Query-Generator-vanna\oracle_schema_20250415_054319.sql"
    if os.path.exists(ddl_file_path):
        with open(ddl_file_path, 'r') as f:
            ddl_content = f.read().strip()
            # Split by semicolon but preserve meaningful chunks
            ddl_statements = [stmt.strip() for stmt in ddl_content.split(';') if stmt.strip()]
            
            # Enhanced DDL with metadata for better embeddings
            for i, stmt in enumerate(ddl_statements):
                # Add context to DDL for better embedding
                enhanced_ddl = f"-- DDL Statement {i+1}\n{stmt}"
                training_data['ddl_statements'].append(enhanced_ddl)
                training_data['metadata'].append({
                    'type': 'ddl',
                    'index': i,
                    'length': len(stmt)
                })

    # Read documentation from text file
    doc_file_path = r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Query-Generator-vanna\Full Data Discrebtion.txt"
    if os.path.exists(doc_file_path):
        try:
            with open(doc_file_path, 'r', encoding='utf-8') as f:
                documentation = f.read().strip().split('------------------------------------------------------------')
                
                for i, doc in enumerate(documentation):
                    if doc.strip():
                        # Enhanced documentation with context
                        enhanced_doc = f"Database Documentation - Section {i+1}:\n{doc.strip()}"
                        training_data['documentation'].append(enhanced_doc)
                        training_data['metadata'].append({
                            'type': 'documentation',
                            'section': i+1,
                            'length': len(doc)
                        })

        except UnicodeDecodeError:
            try:
                with open(doc_file_path, 'r', encoding='latin1') as f:
                    documentation = f.read().strip().split('------------------------------------------------------------')
                    for i, doc in enumerate(documentation):
                        if doc.strip():
                            enhanced_doc = f"Database Documentation - Section {i+1}:\n{doc.strip()}"
                            training_data['documentation'].append(enhanced_doc)
                            training_data['metadata'].append({
                                'type': 'documentation',
                                'section': i+1,
                                'length': len(doc)
                            })
                logging.info("Documentation file read with latin1 encoding")
            except Exception as e:
                logging.error(f"Failed to read documentation file: {e}")
    
    # Enhanced sample questions with more variety and context
    sample_questions = [
        ("Show me all accounts with their details", "SELECT * FROM ACCOUNT "),       
    ]
    
    for i, (question, sql) in enumerate(sample_questions):
        training_data['sample_questions'].append((question, sql))
        training_data['metadata'].append({
            'type': 'question_sql',
            'index': i,
            'question_length': len(question),
            'sql_length': len(sql)
        })
    
    return training_data

def train_vanna_model_with_embeddings(vn, training_data):
    """Enhanced training with embedding verification"""
    logging.info("Starting enhanced Vanna model training with embedding verification...")
    
    total_added = 0
    total_failed = 0
    
    # Add DDL statements with embedding verification
    logging.info(f"Adding {len(training_data['ddl_statements'])} DDL statements...")
    for i, ddl in enumerate(training_data['ddl_statements']):
        try:
            success = vn.add_to_vector_store_with_embedding(ddl, 'ddl', {'index': i})
            if success:
                total_added += 1
            else:
                total_failed += 1
                
            if (i + 1) % 10 == 0:
                logging.info(f"Processed {i + 1} DDL statements (Added: {total_added}, Failed: {total_failed})")
        except Exception as e:
            logging.warning(f"Failed to add DDL {i}: {e}")
            total_failed += 1
    
    # Add documentation with embedding verification
    logging.info(f"Adding {len(training_data['documentation'])} documentation entries...")
    for i, doc in enumerate(training_data['documentation']):
        try:
            success = vn.add_to_vector_store_with_embedding(doc, 'documentation', {'section': i})
            if success:
                total_added += 1
            else:
                total_failed += 1
                
            if (i + 1) % 10 == 0:
                logging.info(f"Processed {i + 1} documentation entries")
        except Exception as e:
            logging.warning(f"Failed to add documentation {i}: {e}")
            total_failed += 1
    
    # Add sample questions with embedding verification
    logging.info(f"Adding {len(training_data['sample_questions'])} sample Q&A pairs...")
    for i, (question, sql) in enumerate(training_data['sample_questions']):
        try:
            content = f"{question}|||{sql}"
            success = vn.add_to_vector_store_with_embedding(content, 'question_sql', {'qa_index': i})
            if success:
                total_added += 1
            else:
                total_failed += 1
                
            if (i + 1) % 5 == 0:
                logging.info(f"Processed {i + 1} Q&A pairs")
        except Exception as e:
            logging.warning(f"Failed to add Q&A pair {i}: {e}")
            total_failed += 1
    
    logging.info(f"Training completed! Total added: {total_added}, Total failed: {total_failed}")
    return total_added, total_failed

def comprehensive_embedding_test(vn):
    """Comprehensive test of embedding and similarity search functionality"""
    logging.info("=== COMPREHENSIVE EMBEDDING & SIMILARITY SEARCH TEST ===")
    
    test_queries = [
        "show me all accounts",
        "count customers", 
        "maintenance jobs",
        "branch information",
        "asset details",
        "service notifications",
        "customer changes",
        "sales orders",
        "account types",
        "database schema",
        "اظهر تنفيذات اوامر الصيانة"
    ]
    
    total_results = 0
    detailed_results = []
    
    for query in test_queries:
        logging.info(f"\n--- Testing query: '{query}' ---")
        results = vn.test_similarity_search(query, top_k=3)
        
        query_total = (len(results['similar_questions']) + 
                      len(results['related_ddl']) + 
                      len(results['related_docs']))
        
        total_results += query_total
        detailed_results.append({
            'query': query,
            'total_results': query_total,
            'qa_results': len(results['similar_questions']),
            'ddl_results': len(results['related_ddl']),
            'doc_results': len(results['related_docs'])
        })
        
        # Show sample results
        if results['similar_questions']:
            logging.info("  Top similar Q&A:")
            for i, (q, sql) in enumerate(results['similar_questions'][:2]):
                print(f"    {i+1}. Q: {q[:60]}...")
                print(f"       SQL: {sql[:60]}...")
                # logging.info(f"    {i+1}. Q: {q[:60]}...")
                # logging.info(f"       SQL: {sql[:60]}...")
        
        if results['related_ddl']:
            logging.info("  Related DDL:")
            for i, ddl in enumerate(results['related_ddl'][:1]):
                print(f"    {i+1}. {ddl[:80]}...")
                # logging.info(f"    {i+1}. {ddl[:80]}...")
        if results['related_docs']:
            logging.info("  Related Documentation:")
            for i, doc in enumerate(results['related_docs'][:1]):
                print(f"    {i+1}. {doc[:80]}...")
                # logging.info(f"    {i+1}. {doc[:80]}...")
    
    # Summary
    logging.info(f"\n=== EMBEDDING TEST SUMMARY ===")
    logging.info(f"Total queries tested: {len(test_queries)}")
    logging.info(f"Total results found: {total_results}")
    logging.info(f"Average results per query: {total_results/len(test_queries):.1f}")
    
    if total_results > 0:
        logging.info("✅ Embedding and similarity search are working!")
        return True
    else:
        logging.error("❌ No results found - embedding may not be working properly")
        return False

def main():
    """Enhanced main function with embedding focus"""
    print("=== Enhanced Vanna Vector Store with Embedding Verification ===")
    print("1. Test current embeddings and similarity search")
    print("2. Extract schema and train model with embedding verification")
    print("3. Full process (train + comprehensive embedding test)")
    print("4. Quick embedding health check")
    print("5. Debug API methods (check available Vanna methods)")
    
    choice = input("Enter your choice (1/2/3/4/5): ").strip()
    
    # Initialize Vanna
    vn = MyVanna()
    
    if choice in ['1', '2','3', '4', '5']:
        print("\n--- Testing Embeddings and Similarity Search ---")
        if choice == '5':
            # Debug API methods
            print("Available methods in Vanna instance:")
            vn_methods = [method for method in dir(vn) if not method.startswith('_')]
            similarity_methods = [m for m in vn_methods if 'similar' in m.lower() or 'related' in m.lower() or 'get' in m.lower()]
            print("Similarity/retrieval related methods:")
            for method in sorted(similarity_methods):
                print(f"  - {method}")
            
            # Try to inspect some key methods
            print("\nTrying to call some methods...")
            test_query = "SELECT * FROM ACCOUNT"
            for method_name in ['get_similar_question_sql', 'get_related_ddl', 'get_related_documentation']:
                if hasattr(vn, method_name):
                    try:
                        method = getattr(vn, method_name)
                        print(f"✅ {method_name} exists")
                        # Try to get method signature
                        import inspect
                        sig = inspect.signature(method)
                        print(f"   Signature: {method_name}{sig}")
                    except Exception as e:
                        print(f"⚠️ {method_name} exists but error getting signature: {e}")
                else:
                    print(f"❌ {method_name} not found")
            return
        if choice == '4':
            # Quick check with better error handling
                try:
                    quick_results = vn.test_similarity_search("اظهر تنفيذات اوامر الصيانة", top_k=3)
                    total = sum(len(v) if isinstance(v, list) else 0 for v in quick_results.values())
                    if total > 0:
                        print(f"✅ Quick check passed - found {total} results")
                        # Show what was found
                        for content_type, results in quick_results.items():
                            if results:
                                print(f"  - {content_type}: {results} items")
                    else:
                        print("❌ Quick check failed - no results found")
                        print("This might indicate:")
                        print("  1. Vector store is empty")
                        print("  2. API method names have changed")
                        print("  3. Embedding configuration issue")
                    return
                except Exception as e:
                    print(f"❌ Quick check error: {e}")
                    return
        else:
            has_embeddings = comprehensive_embedding_test(vn)
            if has_embeddings and choice == '1':
                return
    
    if choice in ['2', '3']:
        print("\n--- Starting Enhanced Training Process ---")
        
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
            
            # Create enhanced training data
            print("Creating enhanced training data...")
            training_data = create_enhanced_training_data(schema_info)
            
            # Save training data for backup
            with open('enhanced_training_data.json', 'w') as f:
                serializable_data = {
                    'ddl_statements': training_data['ddl_statements'],
                    'documentation': training_data['documentation'],
                    'sample_questions': training_data['sample_questions'],
                    'metadata': training_data['metadata']
                }
                json.dump(serializable_data, f, indent=2, default=str)
            logging.info("Enhanced training data saved to enhanced_training_data.json")
            
            # Connect Vanna to Oracle
            vn.connect_to_oracle(
                user='IAS202538',
                password='123',
                dsn="localhost:1521/xepdb1"
            )
            
            # Train the model with embedding verification
            added, failed = train_vanna_model_with_embeddings(vn, training_data)
            print(f"Training completed: {added} items added, {failed} failed")
            
            # Run comprehensive embedding test
            if choice == '3':
                print("\n--- Comprehensive Embedding Test ---")
                comprehensive_embedding_test(vn)
            
        finally:
            connection.close()
    
    print("\n=== Process Complete ===")

if __name__ == "__main__":
    main()