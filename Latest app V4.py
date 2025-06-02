from flask import Flask, render_template, request, jsonify, session, send_file
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
import os
import tempfile
import json
import oracledb
import sqlparse
import logging
import uuid
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enhanced MyVanna class with proper vector similarity search
import logging
import numpy as np
from typing import List, Tuple, Dict, Any

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        config = config or {}
        config['model'] = 'mistral'
        config['persist_directory'] = './vanna-data'
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)
        self.custom_oracle_prompt = ""
        
        # Similarity search thresholds
        self.ddl_similarity_threshold = 0.3  # Adjust based on your needs
        self.doc_similarity_threshold = 0.3
        self.question_similarity_threshold = 0.4


    def set_custom_oracle_prompt(self, prompt: str):
        """Set the custom Oracle SQL prompt"""
        self.custom_oracle_prompt = prompt

    def set_similarity_thresholds(self, ddl_threshold=0.3, doc_threshold=0.3, question_threshold=0.4):
        """Set similarity thresholds for filtering results"""
        self.ddl_similarity_threshold = ddl_threshold
        self.doc_similarity_threshold = doc_threshold
        self.question_similarity_threshold = question_threshold

    def _get_chromadb_collections(self):
        """Get ChromaDB collections safely"""
        try:
            import chromadb
            chroma_client = chromadb.PersistentClient(path='./vanna-data')
            collections = chroma_client.list_collections()
            return chroma_client, collections
        except Exception as e:
            logging.error(f"Failed to access ChromaDB: {e}")
            return None, []

    def search_similar_content_with_scores(self, question: str, content_type: str = "all", 
                                         n_results: int = 3) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar content using vector similarity with scores
        
        Args:
            question: The user's question
            content_type: 'ddl', 'documentation', 'questions', or 'all'
            n_results: Maximum number of results to return
            
        Returns:
            List of tuples: (content, similarity_score, metadata)
        """
        results_with_scores = []
        
        try:
            chroma_client, collections = self._get_chromadb_collections()
            if not chroma_client:
                return []
            
            for collection in collections:
                try:
                    # Use ChromaDB's query method which returns similarity scores
                    search_results = collection.query(
                        query_texts=[question],
                        n_results=n_results * 2,  # Get more to filter by type and threshold
                        include=['documents', 'metadatas', 'distances']
                    )
                    
                    if (search_results and 
                        'documents' in search_results and 
                        'distances' in search_results and
                        search_results['documents']):
                        
                        documents = search_results['documents'][0]  # First query results
                        distances = search_results['distances'][0]
                        metadatas = search_results.get('metadatas', [[{}] * len(documents)])[0]
                        
                        for doc, distance, metadata in zip(documents, distances, metadatas):
                            # Convert distance to similarity score (ChromaDB uses distance, lower = more similar)
                            similarity_score = 1.0 / (1.0 + distance) if distance > 0 else 1.0
                            
                            # Determine content type
                            doc_upper = doc.upper()
                            detected_type = self._detect_content_type(doc, metadata)
                            
                            # Filter by content type if specified
                            if content_type != "all" and detected_type != content_type:
                                continue
                            
                            # Apply threshold filtering
                            threshold = self._get_threshold_for_type(detected_type)
                            if similarity_score >= threshold:
                                results_with_scores.append((doc, similarity_score, {
                                    'type': detected_type,
                                    'collection': collection.name,
                                    'distance': distance,
                                    'metadata': metadata
                                }))
                
                except Exception as e:
                    logging.debug(f"Error querying collection {collection.name}: {e}")
                    continue
        
        except Exception as e:
            logging.error(f"Error in similarity search: {e}")
        
        # Sort by similarity score (highest first) and limit results
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        return results_with_scores[:n_results]

    def _detect_content_type(self, content: str, metadata: Dict) -> str:
        """Detect the type of content (DDL, documentation, or question)"""
        content_upper = content.upper()
        
        # Check metadata first
        if metadata:
            if 'type' in metadata:
                return metadata['type'].lower()
            if 'category' in metadata:
                return metadata['category'].lower()
        
        # Detect DDL
        if any(keyword in content_upper for keyword in 
               ['CREATE TABLE', 'CREATE VIEW', 'ALTER TABLE', 'DROP TABLE', 'CREATE INDEX']):
            return 'ddl'
        
        # Detect documentation (descriptions, explanations)
        if any(keyword in content.lower() for keyword in 
               ['contains', 'stores', 'represents', 'description', 'field', 'column']):
            return 'documentation'
        
        # Detect SQL queries
        if any(keyword in content_upper for keyword in 
               ['SELECT', 'INSERT', 'UPDATE', 'DELETE']) and '?' not in content:
            return 'questions'
        
        return 'unknown'

    def _get_threshold_for_type(self, content_type: str) -> float:
        """Get similarity threshold for content type"""
        thresholds = {
            'ddl': self.ddl_similarity_threshold,
            'documentation': self.doc_similarity_threshold,
            'questions': self.question_similarity_threshold,
            'unknown': 0.2
        }
        return thresholds.get(content_type, 0.2)

    def get_related_ddl_with_similarity(self, question: str, n_results: int = 5) -> List[Tuple[str, float]]:
        """Get related DDL statements using vector similarity search"""
        logging.info(f"Searching for DDL similar to: {question}")
        
        results = self.search_similar_content_with_scores(
            question, 
            content_type="ddl", 
            n_results=n_results
        )
        
        ddl_results = [(content, score) for content, score, metadata in results]
        
        logging.info(f"Found {len(ddl_results)} DDL entries with similarity scores")
        for i, (ddl, score) in enumerate(ddl_results):
            logging.info(f"DDL {i+1} (score: {score:.3f}): {ddl[:100]}...")
        
        return ddl_results

    def get_related_documentation_with_similarity(self, question: str, n_results: int = 3) -> List[Tuple[str, float]]:
        """Get related documentation using vector similarity search"""
        logging.info(f"Searching for documentation similar to: {question}")
        
        results = self.search_similar_content_with_scores(
            question, 
            content_type="documentation", 
            n_results=n_results
        )
        
        doc_results = [(content, score) for content, score, metadata in results]
        
        logging.info(f"Found {len(doc_results)} documentation entries with similarity scores")
        for i, (doc, score) in enumerate(doc_results):
            logging.info(f"Doc {i+1} (score: {score:.3f}): {doc[:100]}...")
        
        return doc_results

    def get_similar_questions_with_similarity(self, question: str, n_results: int = 3) -> List[Tuple[str, str, float]]:
        """Get similar questions and their SQL using vector similarity search"""
        logging.info(f"Searching for similar questions to: {question}")
        
        results = self.search_similar_content_with_scores(
            question, 
            content_type="questions", 
            n_results=n_results
        )
        
        question_pairs = []
        for content, score, metadata in results:
            # Try to extract question-SQL pairs
            if self._looks_like_sql(content):
                question_pairs.append(("Similar query", content, score))
            else:
                # Look for associated SQL in the same context/metadata
                question_pairs.append((content, "-- No SQL found", score))
        
        logging.info(f"Found {len(question_pairs)} similar questions with similarity scores")
        for i, (q, sql, score) in enumerate(question_pairs):
            logging.info(f"Question {i+1} (score: {score:.3f}): {q[:50]}...")
        
        return question_pairs

    def _looks_like_sql(self, content: str) -> bool:
        """Check if content looks like SQL"""
        content_upper = content.upper()
        return any(keyword in content_upper for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE'])

    def preview_similar_content(self, question: str, show_all_types: bool = True) -> Dict[str, List]:
        """Preview similar content before building context - for debugging/validation"""
        print(f"\n{'='*60}")
        print(f"SIMILARITY SEARCH PREVIEW for: '{question}'")
        print(f"{'='*60}")
        
        preview_results = {
            'ddl': [],
            'documentation': [],
            'questions': [],
            'all': []
        }
        
        if show_all_types:
            # Get all types of content
            all_results = self.search_similar_content_with_scores(question, content_type="all", n_results=15)
            preview_results['all'] = all_results
            
            print(f"\n🔍 ALL SIMILAR CONTENT (Top 15):")
            print(f"{'Type':<15} {'Score':<8} {'Content Preview'}")
            print("-" * 80)
            
            for content, score, metadata in all_results:
                content_type = metadata.get('type', 'unknown')
                preview = content.replace('\n', ' ')[:60] + "..." if len(content) > 60 else content
                print(f"{content_type:<15} {score:<8.3f} {preview}")
        
        # DDL Content
        ddl_results = self.get_related_ddl_with_similarity(question, n_results=7)
        preview_results['ddl'] = ddl_results
        
        print(f"\n📋 DDL/SCHEMA CONTENT (Threshold: {self.ddl_similarity_threshold}):")
        if ddl_results:
            for i, (ddl, score) in enumerate(ddl_results, 1):
                print(f"  {i}. Score: {score:.3f}")
                print(f"     Content: {ddl[:150]}...")
                print()
        else:
            print("  No DDL content found above threshold")
        
        # Documentation Content
        doc_results = self.get_related_documentation_with_similarity(question, n_results=5)
        preview_results['documentation'] = doc_results
        
        print(f"\n📚 DOCUMENTATION CONTENT (Threshold: {self.doc_similarity_threshold}):")
        if doc_results:
            for i, (doc, score) in enumerate(doc_results, 1):
                print(f"  {i}. Score: {score:.3f}")
                print(f"     Content: {doc[:150]}...")
                print()
        else:
            print("  No documentation found above threshold")
        
        # Similar Questions
        question_results = self.get_similar_questions_with_similarity(question, n_results=5)
        preview_results['questions'] = question_results
        
        print(f"\n❓ SIMILAR QUESTIONS (Threshold: {self.question_similarity_threshold}):")
        if question_results:
            for i, (q, sql, score) in enumerate(question_results, 1):
                print(f"  {i}. Score: {score:.3f}")
                print(f"     Question: {q[:100]}...")
                print(f"     SQL: {sql[:100]}...")
                print()
        else:
            print("  No similar questions found above threshold")
        
        print(f"{'='*60}")
        
        return preview_results

    def build_context_from_rag_with_similarity(self, question: str, preview: bool = False) -> str:
        """Build comprehensive context string from RAG components using similarity search"""
        
        if preview:
            # Show preview of similar content
            self.preview_similar_content(question)
        
        context_parts = []
        
        # Get DDL/Schema information with similarity scores
        # ddl_results = self.get_related_ddl_with_similarity(question, n_results=7)
        # if ddl_results:
        #     context_parts.append("=== DATABASE SCHEMA (DDL) ===")
        #     for i, (ddl, score) in enumerate(ddl_results, 1):
        #         context_parts.append(f"Schema {i} (Similarity: {score:.3f}):")
        #         context_parts.append(ddl.strip())
        #         context_parts.append("")
        
        # Get similar questions/examples with similarity scores
        # similar_questions = self.get_similar_questions_with_similarity(question, n_results=5)
        # if similar_questions:
        #     context_parts.append("=== SIMILAR QUERY EXAMPLES ===")
        #     for i, (q, sql, score) in enumerate(similar_questions, 1):
        #         context_parts.append(f"Example {i} (Similarity: {score:.3f}):")
        #         context_parts.append(f"Question: {q}")
        #         context_parts.append(f"SQL: {sql}")
        #         context_parts.append("")
        
        # Get documentation with similarity scores
        docs = self.get_related_documentation_with_similarity(question, n_results=5)
        if docs:
            context_parts.append("=== TABLE/COLUMN DOCUMENTATION ===")
            for i, (doc, score) in enumerate(docs, 1):
                context_parts.append(f"Documentation {i} (Similarity: {score:.3f}):")
                context_parts.append(doc.strip())
                context_parts.append("")
        
        # Add a detailed summary section
        if context_parts:
            summary_parts = []
            # if ddl_results:
            #     avg_ddl_score = sum(score for _, score in ddl_results) / len(ddl_results)
            #     summary_parts.append(f"{len(ddl_results)} schema definitions (avg similarity: {avg_ddl_score:.3f})")
            # if similar_questions:
            #     avg_q_score = sum(score for _, _, score in similar_questions) / len(similar_questions)
            #     summary_parts.append(f"{len(similar_questions)} example queries (avg similarity: {avg_q_score:.3f})")
            if docs:
                avg_doc_score = sum(score for _, score in docs) / len(docs)
                summary_parts.append(f"{len(docs)} documentation entries (avg similarity: {avg_doc_score:.3f})")
            
            context_summary = f"=== CONTEXT SUMMARY ===\nRetrieved context: {', '.join(summary_parts)}\nSimilarity thresholds: DDL({self.ddl_similarity_threshold}), Docs({self.doc_similarity_threshold}), Questions({self.question_similarity_threshold})\n\n"
            return context_summary + "\n".join(context_parts)
        
        return "\n".join(context_parts)

    def generate_sql_with_preview(self, question: str, preview_similarity: bool = True, **kwargs) -> str:
        """Generate SQL with optional similarity preview"""
        logging.info(f"Generating SQL with vector similarity search for: {question}")
        
        if preview_similarity:
            print(f"\n🔍 PREVIEWING SIMILAR CONTENT FOR: '{question}'")
            self.preview_similar_content(question)
            
            # Ask user if they want to continue (for interactive mode)
            # In production, you might want to remove this or make it configurable
            try:
                user_input = input("\nContinue with SQL generation? (y/n): ").lower().strip()
                if user_input != 'y' and user_input != 'yes':
                    return "SQL generation cancelled by user"
            except:
                # In non-interactive mode, continue automatically
                pass
        
        # Build context using similarity search
        rag_context = self.build_context_from_rag_with_similarity(question, preview=False)
        
        # Generate SQL prompt with enhanced context
        if self.custom_oracle_prompt and rag_context:
            system_content = f"""{self.custom_oracle_prompt}

{rag_context}

IMPORTANT INSTRUCTIONS:
1. Use ONLY the tables and columns shown in the schema definitions above
2. Follow the patterns from the example queries provided (pay attention to similarity scores)
3. Reference the documentation to understand what each table/column contains
4. Generate clean Oracle SQL that matches the examples and schema
5. Prioritize higher similarity scored examples and schema definitions
6. If the question asks about data not covered in the schema, explain what's missing

Using the above context and following Oracle SQL rules, generate a query for the question below."""
            
        elif self.custom_oracle_prompt:
            system_content = self.custom_oracle_prompt
        elif rag_context:
            system_content = f"""You are an Oracle SQL expert. Use the following context to generate queries:

{rag_context}

Generate clean Oracle SQL queries following these rules:
- Use ROWNUM for limiting rows instead of LIMIT
- Use SYSDATE for current date
- Use || for string concatenation
- Only reference tables and columns that exist in the provided schema context
- Follow the patterns shown in the example queries (prioritize higher similarity scores)"""
        else:
            system_content = "You are an Oracle SQL expert. Generate clean, executable Oracle SQL queries using proper Oracle syntax."
        
        messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': question}
        ]
        
        # Log enhanced context information
        context_length = len(messages[0]['content'])
        logging.info(f"Enhanced system prompt length: {context_length} characters")
        
        # Generate response using Ollama
        response = self.submit_prompt(messages)
        
        # Extract clean SQL from response
        sql = self._extract_sql_from_response(response)
        logging.info(f"Generated SQL with similarity context: {sql}")
        
        return sql

    def check_vector_embeddings_status(self) -> Dict[str, Any]:
        """Check if the vector store is properly storing embedded vectors"""
        status = {
            'collections_found': 0,
            'total_embeddings': 0,
            'embedding_dimensions': [],
            'collections_info': [],
            'has_vectors': False,
            'errors': []
        }
        
        try:
            chroma_client, collections = self._get_chromadb_collections()
            status['collections_found'] = len(collections)
            
            for collection in collections:
                try:
                    count = collection.count()
                    collection_info = {
                        'name': collection.name,
                        'count': count,
                        'has_embeddings': False,
                        'embedding_dimension': None
                    }
                    
                    if count > 0:
                        # Try to peek at data to check for embeddings
                        peek_data = collection.peek(limit=1)
                        if peek_data and 'embeddings' in peek_data and peek_data['embeddings']:
                            embedding = peek_data['embeddings'][0]
                            if embedding:  # Not None or empty
                                collection_info['has_embeddings'] = True
                                collection_info['embedding_dimension'] = len(embedding)
                                status['embedding_dimensions'].append(len(embedding))
                                status['total_embeddings'] += count
                                status['has_vectors'] = True
                    
                    status['collections_info'].append(collection_info)
                    
                except Exception as e:
                    status['errors'].append(f"Error checking collection {collection.name}: {e}")
                    
        except Exception as e:
            status['errors'].append(f"Error accessing ChromaDB: {e}")
        
        return status

# Additional functions to test the enhanced functionality

def test_enhanced_similarity_search():
    """Test the enhanced similarity search functionality"""
    print("=== TESTING ENHANCED SIMILARITY SEARCH ===")
    
    # Initialize enhanced Vanna
    vn_enhanced = MyVanna()
    
    # Set similarity thresholds
    vn_enhanced.set_similarity_thresholds(
        ddl_threshold=0.2,
        doc_threshold=0.2,
        question_threshold=0.3
    )
    
    # Test vector embeddings status
    print("\n1. Checking Vector Embeddings Status...")
    embedding_status = vn_enhanced.check_vector_embeddings_status()
    print(f"Collections found: {embedding_status['collections_found']}")
    print(f"Total embeddings: {embedding_status['total_embeddings']}")
    print(f"Has vectors: {embedding_status['has_vectors']}")
    
    if embedding_status['collections_info']:
        for info in embedding_status['collections_info']:
            print(f"  - {info['name']}: {info['count']} items, "
                  f"embeddings: {info['has_embeddings']}, "
                  f"dimensions: {info['embedding_dimension']}")
    
    if embedding_status['errors']:
        print("Errors:", embedding_status['errors'])
    
    # Test similarity search
    print("\n2. Testing Similarity Search...")
    test_question = "show me account information"
    
    try:
        # Preview similar content
        preview_results = vn_enhanced.preview_similar_content(test_question)
        
        # Test SQL generation with preview
        print("\n3. Testing SQL Generation with Similarity...")
        sql = vn_enhanced.generate_sql_with_preview(test_question, preview_similarity=True)
        print(f"Generated SQL: {sql}")
        
    except Exception as e:
        print(f"Error in similarity search test: {e}")
    
    return vn_enhanced

def demonstrate_similarity_thresholds():
    """Demonstrate how different similarity thresholds affect results"""
    print("=== SIMILARITY THRESHOLD DEMONSTRATION ===")
    
    vn_demo = MyVanna()
    test_question = "show customer accounts"
    
    thresholds_to_test = [0.1, 0.3, 0.5, 0.7]
    
    for threshold in thresholds_to_test:
        print(f"\n--- Testing with threshold: {threshold} ---")
        vn_demo.set_similarity_thresholds(
            ddl_threshold=threshold,
            doc_threshold=threshold,
            question_threshold=threshold
        )
        
        # Get results with this threshold
        ddl_results = vn_demo.get_related_ddl_with_similarity(test_question, n_results=5)
        doc_results = vn_demo.get_related_documentation_with_similarity(test_question, n_results=3)
        
        print(f"DDL results: {len(ddl_results)}")
        print(f"Doc results: {len(doc_results)}")
        
        # Show score ranges
        if ddl_results:
            scores = [score for _, score in ddl_results]
            print(f"DDL score range: {min(scores):.3f} - {max(scores):.3f}")
        
        if doc_results:
            scores = [score for _, score in doc_results]
            print(f"Doc score range: {min(scores):.3f} - {max(scores):.3f}")

# Example usage in your Flask app route
def enhanced_ask_sql_route():
    """Enhanced version of your ask_sql route with similarity preview"""
    question = request.form.get("question")
    preview_mode = request.form.get("preview", "false").lower() == "true"
    
    if not question:
        return jsonify({"status": "error", "message": "No question provided"})
    
    try:
        logging.info(f"Processing question with enhanced similarity: {question}")
        
        # Check if vector store has embeddings
        embedding_status = vn.check_vector_embeddings_status()
        if not embedding_status['has_vectors']:
            return jsonify({
                "status": "warning", 
                "message": "Vector store found but no embeddings detected. The RAG system may not work properly.",
                "embedding_info": embedding_status
            })
        
        # Generate SQL with similarity search
        if preview_mode:
            # In preview mode, show similar content first
            preview_results = vn.preview_similar_content(question)
            return jsonify({
                "status": "preview",
                "message": "Similar content found. Review and confirm to generate SQL.",
                "preview_data": {
                    "ddl_count": len(preview_results['ddl']),
                    "doc_count": len(preview_results['documentation']),
                    "question_count": len(preview_results['questions'])
                },
                "question": question
            })
        else:
            # Normal mode - generate SQL directly
            sql = vn.generate_sql_with_preview(question, preview_similarity=False)
            
            return jsonify({
                "status": "success", 
                "sql": sql,
                "context_used": True,
                "embedding_status": embedding_status
            })
        
    except Exception as e:
        logging.error(f"Error in enhanced SQL generation: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

# Initialize the Flask app and rest of the code remains unchanged (omitted for brevity)
app = Flask(__name__)
app.secret_key = 'vanna-rag-sql-assistant-secret-key'

# Initialize Oracle client (use the same path as in your training script)
try:
    oracledb.init_oracle_client(lib_dir=r"C:\Users\ahmed\Downloads\instantclient-basic-windows.x64-23.8.0.25.04\instantclient_23_8")
    logging.info("Oracle client initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Oracle client: {e}")

# Initialize Vanna
vn = MyVanna()

# Set the enhanced Oracle SQL prompt that works with RAG
oracle_prompt = """You are an expert SQL assistant writing SQL queries for an Oracle Database.
You must follow Oracle SQL syntax strictly and only use tables that exist in the user's schema.

Before generating the query, use the provided context from the vector store to understand the available schema. Only reference tables and columns that are confirmed to exist in the schema based on the context provided.

Oracle SQL rules to follow:
- Use `ROWNUM` for limiting rows instead of `LIMIT` (e.g., `WHERE ROWNUM <= 10`).
- Use `SYSDATE` for the current date.
- Use `TO_DATE('YYYY-MM-DD', 'YYYY-MM-DD')` to parse dates.
- For string concatenation, use `||` operator.
- Avoid PostgreSQL/MySQL syntax like `LIMIT`, `ILIKE`, or `TRUE/FALSE` — these are not valid in Oracle.
- Use `DUAL` for selecting constants (e.g., `SELECT 1 FROM DUAL`).
- For pagination, use ROW_NUMBER() OVER() or ROWNUM with nested queries instead of OFFSET/FETCH.
- Use Oracle's hierarchical query syntax with CONNECT BY and PRIOR for tree-structured data.
- Remember that Oracle's NVL() is equivalent to COALESCE() in other dialects.
- For date arithmetic, use date + number for days (e.g., SYSDATE + 7 for a week later).

Important constraints:
1. ONLY reference tables that exist in the schema retrieved from the vector store context.
2. NEVER make up table names or columns that aren't confirmed in the retrieved schema.
3. Use proper Oracle join syntax and appropriate table aliases if needed.
4. Always use fully qualified column names in joins to avoid ambiguity.
5. When no specific limit is requested, add WHERE ROWNUM <= 100 to prevent large result sets.

Always output clean, runnable Oracle SQL with appropriate table and column references.
Do not include ANY explanatory text before or after the SQL query - just return the SQL itself.
Do not assume any table names or columns that are not explicitly mentioned in the schema retrieved from the vector store."""

vn.set_custom_oracle_prompt(oracle_prompt)

# Connect to Oracle using the same credentials as in training script
try:
    vn.connect_to_oracle(
        user='IAS202538',
        password='123',
        dsn="localhost:1521/xepdb1"
    )
    logging.info("Vanna connected to Oracle successfully")
except Exception as e:
    logging.error(f"Failed to connect Vanna to Oracle: {e}")

# Enable RAG features
vn.allow_llm_to_see_data = True

def get_valid_table_names():
    """Get valid table names from the database"""
    valid_tables = set()
    try:
        with oracledb.connect(
            user='IAS202538',
            password='123',
            dsn='localhost:1521/xepdb1'
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT table_name FROM user_tables ORDER BY table_name")
                tables = cursor.fetchall()
                valid_tables = {table[0].upper() for table in tables}
                logging.info(f"Found {len(valid_tables)} valid tables")
    except Exception as e:
        logging.error(f"Failed to get table names: {e}")
    
    return valid_tables

# Get valid table names from database
valid_table_names = get_valid_table_names()

def extract_table_names(sql_query):
    """Extract table names from an SQL query using sqlparse."""
    try:
        parsed = sqlparse.parse(sql_query)
        table_names = set()
        
        for statement in parsed:
            for token in statement.flatten():
                if token.ttype is sqlparse.tokens.Name:
                    # Check if this might be a table name by looking at surrounding context
                    token_str = token.value.upper().strip()
                    if (token_str in valid_table_names and 
                        len(token_str) > 2 and  # Avoid single characters
                        not token_str in ['AS', 'ON', 'AND', 'OR', 'IN', 'IS', 'NOT']):
                        table_names.add(token_str)
        
        return table_names
    except Exception as e:
        logging.warning(f"Error parsing SQL: {e}")
        return set()

def is_valid_sql(sql_query, valid_tables):
    """Check if all table names in the SQL query are in the valid tables set."""
    try:
        referenced_tables = extract_table_names(sql_query)
        if not referenced_tables:
            # If no tables detected, try a simple string search
            sql_upper = sql_query.upper()
            for table in valid_tables:
                if table in sql_upper:
                    referenced_tables.add(table)
        
        if not referenced_tables:
            return True  # If no tables found, allow (might be a simple query)
        
        invalid_tables = referenced_tables - valid_tables
        return len(invalid_tables) == 0
    except Exception as e:
        logging.warning(f"Error validating SQL: {e}")
        return True  # Allow execution if validation fails

def check_vector_store_status():
    """Enhanced check if the vector store has been populated with training data"""
    if not os.path.exists('./vanna-data'):
        return False, "Vanna data directory not found"
    
    try:
        # Method 1: Try to get training data directly (most reliable)
        if hasattr(vn, 'get_training_data'):
            try:
                training_data = vn.get_training_data()
                if training_data is not None:
                    # Handle DataFrame case - FIX THE AMBIGUITY WARNING
                    if hasattr(training_data, 'empty'):  # It's a DataFrame
                        if not training_data.empty:  # Use .empty instead of truth value
                            row_count = len(training_data)
                            return True, f"Vector store ready with {row_count} training entries (DataFrame)"
                    # Handle list case
                    elif isinstance(training_data, list) and len(training_data) > 0:
                        return True, f"Vector store ready with {len(training_data)} training entries (List)"
            except Exception as e:
                logging.debug(f"get_training_data failed: {e}")
        
        # Method 2: Test similarity search functionality with enhanced checking
        test_queries = ["show me accounts", "select", "table", "database"]
        total_results = 0
        
        for test_query in test_queries:
            try:
                # Try different similarity search methods
                if hasattr(vn, 'get_similar_question_sql'):
                    similar = vn.get_similar_question_sql(test_query)
                    if similar is not None:
                        # Handle DataFrame case
                        if hasattr(similar, 'empty'):  # It's a DataFrame
                            if not similar.empty:  # Use .empty instead of truth value
                                total_results += len(similar)
                                break  # Found data, no need to continue
                        # Handle list case
                        elif isinstance(similar, list) and len(similar) > 0:
                            total_results += len(similar)
                            break
                elif hasattr(vn, 'get_similar_sql'):
                    similar = vn.get_similar_sql(test_query)
                    if similar is not None:
                        if hasattr(similar, 'empty'):  # DataFrame
                            if not similar.empty:
                                total_results += len(similar)
                                break
                        elif isinstance(similar, list) and len(similar) > 0:
                            total_results += len(similar)
                            break
                elif hasattr(vn, 'similarity_search'):
                    similar = vn.similarity_search(test_query, n_results=1)
                    if similar and len(similar) > 0:
                        total_results += len(similar)
                        break
            except Exception as e:
                logging.debug(f"Similarity search failed for '{test_query}': {e}")
                continue
        
        if total_results > 0:
            return True, f"Vector store ready with similarity search results ({total_results} items found)"
        
        # Method 3: Try to get DDL information using enhanced method
        try:
            if hasattr(vn, 'get_related_ddl'):
                ddl_data = vn.get_related_ddl("tables", n_results=3)
                if ddl_data and len(ddl_data) > 0:
                    return True, f"Vector store ready with {len(ddl_data)} DDL entries"
        except Exception as e:
            logging.debug(f"get_related_ddl failed: {e}")
        
        # Method 4: Try to get documentation using enhanced method
        try:
            if hasattr(vn, 'get_related_documentation'):
                doc_data = vn.get_related_documentation("account", n_results=3)
                if doc_data and len(doc_data) > 0:
                    return True, f"Vector store ready with {len(doc_data)} documentation entries"
        except Exception as e:
            logging.debug(f"get_related_documentation failed: {e}")
        
        # Method 5: Check ChromaDB collection directly with enhanced error handling
        try:
            import chromadb
            chroma_client = chromadb.PersistentClient(path='./vanna-data')
            collections = chroma_client.list_collections()
            
            total_docs = 0
            collection_info = []
            
            for collection in collections:
                try:
                    count = collection.count()
                    total_docs += count
                    collection_info.append(f"{collection.name}: {count}")
                except Exception as e:
                    logging.debug(f"Failed to count collection {collection.name}: {e}")
                    collection_info.append(f"{collection.name}: error")
            
            if total_docs > 0:
                return True, f"Vector store ready with {total_docs} documents ({', '.join(collection_info)})"
            else:
                return False, f"ChromaDB collections exist but are empty ({len(collections)} collections found: {', '.join(collection_info)})"
                
        except Exception as e:
            logging.debug(f"Direct ChromaDB access failed: {e}")
        
        # Method 6: Test if we can generate a simple query (ultimate test)
        try:
            # Try to generate SQL for a simple question
            test_sql = vn.generate_sql_with_preview("show me accounts")
            if test_sql and len(test_sql.strip()) > 10:  # Got a reasonable response
                return True, "Vector store ready (SQL generation successful)"
        except Exception as e:
            logging.debug(f"SQL generation test failed: {e}")
        
        # Method 7: Test enhanced RAG context building
        try:
            context = vn.build_context_from_rag("show me accounts")
            if context and len(context.strip()) > 50:  # Got substantial context
                return True, "Vector store ready (RAG context building successful)"
        except Exception as e:
            logging.debug(f"RAG context test failed: {e}")
        
        # If we get here, directory exists but we couldn't find data
        # Let's check what files actually exist
        try:
            all_files = []
            for root, dirs, files in os.walk('./vanna-data'):
                all_files.extend(files)
            
            if all_files:
                return False, f"Vector store directory exists with {len(all_files)} files, but data verification failed. Files: {all_files[:5]}..."
            else:
                return False, "Vector store directory exists but is completely empty"
        except:
            return False, "Vector store directory exists but cannot read contents"
    
    except Exception as e:
        return False, f"Error checking vector store: {str(e)}"


def detailed_vector_store_diagnosis():
    """Enhanced diagnosis of vector store status"""
    print("=== ENHANCED VECTOR STORE DIAGNOSIS ===")
    
    # Check directory structure
    if os.path.exists('./vanna-data'):
        print("✅ vanna-data directory exists")
        
        # List all files
        all_files = []
        for root, dirs, files in os.walk('./vanna-data'):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    filesize = os.path.getsize(filepath)
                    all_files.append((filepath, filesize))
                except:
                    all_files.append((filepath, "unknown size"))
        
        print(f"📁 Found {len(all_files)} files:")
        for filepath, size in all_files:
            print(f"   - {filepath}: {size} bytes")
    else:
        print("❌ vanna-data directory not found")
        return
    
    # Test enhanced methods
    print("\n=== TESTING ENHANCED VANNA METHODS ===")
    
    # Test 1: Enhanced get_training_data
    try:
        if hasattr(vn, 'get_training_data'):
            training_data = vn.get_training_data()
            if training_data is not None:
                # Handle DataFrame case properly
                if hasattr(training_data, 'empty'):  # It's a DataFrame
                    if not training_data.empty:
                        print(f"✅ get_training_data: Found DataFrame with {len(training_data)} rows")
                        print(f"   Columns: {list(training_data.columns) if hasattr(training_data, 'columns') else 'N/A'}")
                        # Show first few items safely
                        for i in range(min(3, len(training_data))):
                            row = training_data.iloc[i]
                            preview = str(row.to_dict())[:100] + "..." if len(str(row.to_dict())) > 100 else str(row.to_dict())
                            print(f"   Row {i}: {preview}")
                    else:
                        print("❌ get_training_data: DataFrame is empty")
                # Handle list case
                elif isinstance(training_data, list):
                    if len(training_data) > 0:
                        print(f"✅ get_training_data: Found list with {len(training_data)} items")
                        # Show first few items safely
                        for i, item in enumerate(training_data[:3]):
                            preview = str(item)[:100] + "..." if len(str(item)) > 100 else str(item)
                            print(f"   Item {i}: {preview}")
                    else:
                        print("❌ get_training_data: List is empty")
                else:
                    print(f"❓ get_training_data: Found {type(training_data)} with content: {str(training_data)[:100]}...")
            else:
                print("❌ get_training_data: Returned None")
        else:
            print("❌ get_training_data: Method not available")
    except Exception as e:
        print(f"❌ get_training_data failed: {e}")
    
    # Test 2: Enhanced similarity search
    print("\n--- Testing Similarity Search ---")
    test_queries = ["show me accounts", "select", "table", "database"]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        
        # Try get_similar_question_sql
        try:
            if hasattr(vn, 'get_similar_question_sql'):
                similar = vn.get_similar_question_sql(query)
                if similar is not None:
                    if hasattr(similar, 'empty'):  # DataFrame
                        if not similar.empty:
                            print(f"✅ get_similar_question_sql: Found {len(similar)} results (DataFrame)")
                            print(f"   Columns: {list(similar.columns) if hasattr(similar, 'columns') else 'N/A'}")
                        else:
                            print("❌ get_similar_question_sql: Empty DataFrame")
                    elif isinstance(similar, list) and len(similar) > 0:
                        print(f"✅ get_similar_question_sql: Found {len(similar)} results (List)")
                    else:
                        print(f"❓ get_similar_question_sql: {type(similar)} - {similar}")
                else:
                    print("❌ get_similar_question_sql: Returned None")
            else:
                print("❌ get_similar_question_sql: Method not available")
        except Exception as e:
            print(f"❌ get_similar_question_sql failed: {e}")
        
        # Try get_similar_sql
        try:
            if hasattr(vn, 'get_similar_sql'):
                similar = vn.get_similar_sql(query)
                if similar is not None:
                    if hasattr(similar, 'empty'):  # DataFrame
                        if not similar.empty:
                            print(f"✅ get_similar_sql: Found {len(similar)} results (DataFrame)")
                        else:
                            print("❌ get_similar_sql: Empty DataFrame")
                    elif isinstance(similar, list) and len(similar) > 0:
                        print(f"✅ get_similar_sql: Found {len(similar)} results (List)")
                    else:
                        print(f"❓ get_similar_sql: {type(similar)} - {similar}")
                else:
                    print("❌ get_similar_sql: Returned None")
            else:
                print("❌ get_similar_sql: Method not available")
        except Exception as e:
            print(f"❌ get_similar_sql failed: {e}")
        
        # Break after first successful query to avoid spam
        break
    
    # Test 3: Enhanced DDL and Documentation
    print("\n--- Testing DDL and Documentation ---")
    try:
        if hasattr(vn, 'get_related_ddl'):
            ddl_data = vn.get_related_ddl("tables", n_results=3)
            if ddl_data and len(ddl_data) > 0:
                print(f"✅ get_related_ddl: Found {len(ddl_data)} DDL entries")
                for i, ddl in enumerate(ddl_data[:2]):
                    preview = str(ddl)[:100] + "..." if len(str(ddl)) > 100 else str(ddl)
                    print(f"   DDL {i}: {preview}")
            else:
                print("❌ get_related_ddl: No data found")
        else:
            print("❌ get_related_ddl: Method not available")
    except Exception as e:
        print(f"❌ get_related_ddl failed: {e}")
    
    try:
        if hasattr(vn, 'get_related_documentation'):
            doc_data = vn.get_related_documentation("account", n_results=3)
            if doc_data and len(doc_data) > 0:
                print(f"✅ get_related_documentation: Found {len(doc_data)} documentation entries")
                for i, doc in enumerate(doc_data[:2]):
                    preview = str(doc)[:100] + "..." if len(str(doc)) > 100 else str(doc)
                    print(f"   Doc {i}: {preview}")
            else:
                print("❌ get_related_documentation: No data found")
        else:
            print("❌ get_related_documentation: Method not available")
    except Exception as e:
        print(f"❌ get_related_documentation failed: {e}")
    
    # Test 4: Enhanced ChromaDB direct access
    print("\n--- Testing ChromaDB Direct Access ---")
    try:
        import chromadb
        chroma_client = chromadb.PersistentClient(path='./vanna-data')
        collections = chroma_client.list_collections()
        
        print(f"✅ ChromaDB: Found {len(collections)} collections")
        total_docs = 0
        
        for collection in collections:
            try:
                count = collection.count()
                total_docs += count
                print(f"   - {collection.name}: {count} documents")
                
                # Try to peek at some data
                if count > 0:
                    try:
                        peek_data = collection.peek(limit=2)
                        if peek_data and 'documents' in peek_data:
                            for i, doc in enumerate(peek_data['documents'][:2]):
                                preview = str(doc)[:100] + "..." if len(str(doc)) > 100 else str(doc)
                                print(f"     Sample {i}: {preview}")
                    except Exception as e:
                        print(f"     Could not peek data: {e}")
            except Exception as e:
                print(f"   - {collection.name}: Error counting - {e}")
        
        print(f"📊 Total documents across all collections: {total_docs}")
        
    except Exception as e:
        print(f"❌ ChromaDB direct access failed: {e}")
    
    # Test 5: Enhanced SQL Generation
    print("\n--- Testing SQL Generation ---")
    try:
        test_sql = vn.generate_sql_with_preview("show me accounts")
        if test_sql and len(test_sql.strip()) > 10:
            print(f"✅ SQL Generation successful")
            print(f"   Generated SQL: {test_sql[:200]}...")
        else:
            print(f"❌ SQL Generation failed or returned insufficient data: '{test_sql}'")
    except Exception as e:
        print(f"❌ SQL Generation failed: {e}")
    
    # Test 6: Enhanced RAG Context Building
    print("\n--- Testing RAG Context Building ---")
    try:
        if hasattr(vn, 'build_context_from_rag'):
            context = vn.build_context_from_rag("show me accounts")
            if context and len(context.strip()) > 50:
                print(f"✅ RAG Context building successful")
                print(f"   Context length: {len(context)} characters")
                print(f"   Context preview: {context[:200]}...")
            else:
                print(f"❌ RAG Context building failed or insufficient: '{context}'")
        else:
            print("❌ build_context_from_rag: Method not available")
    except Exception as e:
        print(f"❌ RAG Context building failed: {e}")
    
    # Final summary
    print("\n=== DIAGNOSIS SUMMARY ===")
    is_ready, message = check_vector_store_status()
    if is_ready:
        print(f"✅ OVERALL STATUS: {message}")
    else:
        print(f"❌ OVERALL STATUS: {message}")
        print("💡 RECOMMENDATION: Try re-running the training process or check your data sources")
    
    print("=== END DIAGNOSIS ===")


# Additional helper function for debugging
def test_vanna_methods():
    """Test all available Vanna methods to understand what's available"""
    print("=== AVAILABLE VANNA METHODS ===")
    
    vanna_methods = [method for method in dir(vn) if not method.startswith('_')]
    print(f"Found {len(vanna_methods)} public methods:")
    
    for method in sorted(vanna_methods):
        try:
            method_obj = getattr(vn, method)
            if callable(method_obj):
                print(f"✅ {method}() - callable")
            else:
                print(f"📋 {method} - attribute")
        except Exception as e:
            print(f"❌ {method} - error: {e}")
    
    print("=== END METHOD LIST ===")


# Quick status check function
def quick_vector_store_check():
    """Quick one-line status check"""
    is_ready, message = check_vector_store_status()
    status_emoji = "✅" if is_ready else "❌"
    print(f"{status_emoji} Vector Store Status: {message}")
    return is_ready


# Test function you can run directly
def test_vector_store_from_flask():
    """Test function to run the same checks as your training script"""
    print("=== TESTING VECTOR STORE FROM FLASK APP ===")
    
    # Initialize the same way as in Flask app
    vn_test = MyVanna()
    
    # Try the same methods that work in your training script
    print("\n1. Testing get_training_data...")
    try:
        training_data = vn_test.get_training_data()
        if training_data:
            print(f"✅ Found {len(training_data)} training items")
            return True
        else:
            print("⚠️ get_training_data returned None/empty")
    except Exception as e:
        print(f"❌ get_training_data failed: {e}")
    
    print("\n2. Testing similarity search...")
    try:
        similar = vn_test.get_similar_question_sql("show accounts")
        if similar:
            print(f"✅ Found {len(similar)} similar items")
            return True
        else:
            print("⚠️ No similar items found")
    except Exception as e:
        print(f"❌ Similarity search failed: {e}")
    
    print("\n3. Testing SQL generation...")
    try:
        sql = vn_test.generate_sql_with_preview("show me accounts")
        if sql and len(sql.strip()) > 5:
            print(f"✅ Generated SQL: {sql[:50]}...")
            return True
        else:
            print("⚠️ No SQL generated")
    except Exception as e:
        print(f"❌ SQL generation failed: {e}")
    
    return False

@app.route("/", methods=["GET"])
def index():
    # Check vector store status
    has_data, status_message = check_vector_store_status()
    
    return render_template("index.html", 
                         vector_store_status=status_message,
                         vector_store_ready=has_data)

@app.route("/ask", methods=["POST"])
def ask_sql():
    question = request.form.get("question")
    if not question:
        return jsonify({"status": "error", "message": "No question provided"})
    
    try:
        logging.info(f"Processing question: {question}")
        
        # Check if vector store is ready
        has_data, status_message = check_vector_store_status()
        if not has_data:
            return jsonify({
                "status": "warning", 
                "message": f"Vector store not ready: {status_message}. Run the training script first.",
                "sql": ""
            })
        
        # Generate SQL using RAG + custom prompt
        sql = vn.generate_sql_with_preview(question=question)
        
        if not sql:
            return jsonify({"status": "error", "message": "No SQL generated"})
        
        # Clean up the SQL
        sql = sql.strip()
        if sql.endswith(';'):
            sql = sql[:-1]  # Remove trailing semicolon for Oracle
        
        # Validate the generated SQL
        if is_valid_sql(sql, valid_table_names):
            logging.info(f"Valid SQL generated: {sql}")
            return jsonify({"status": "success", "sql": sql})
        else:
            referenced_tables = extract_table_names(sql)
            invalid_tables = referenced_tables - valid_table_names
            logging.warning(f"Invalid tables referenced: {invalid_tables}")
            return jsonify({
                "status": "warning", 
                "message": f"Query may reference non-existent tables: {list(invalid_tables)}. Proceed with caution.",
                "sql": sql
            })
        
    except Exception as e:
        logging.error(f"Error generating SQL: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route("/execute", methods=["POST"])
def execute_sql():
    sql_code = request.form.get("sql")
    if not sql_code:
        return jsonify({"status": "error", "message": "No SQL provided"})
    
    try:
        with oracledb.connect(
            user='IAS202538',
            password='123',
            dsn='localhost:1521/xepdb1'
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql_code)
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    rows = cursor.fetchall()
                    
                    # Convert Oracle-specific data types to JSON-serializable formats
                    serializable_rows = []
                    for row in rows:
                        serializable_row = []
                        for item in row:
                            if hasattr(item, 'read'):  # LOB objects
                                serializable_row.append(str(item.read()))
                            elif isinstance(item, datetime):
                                serializable_row.append(item.isoformat())
                            else:
                                serializable_row.append(str(item) if item is not None else None)
                        serializable_rows.append(serializable_row)
                    
                    data = [dict(zip(columns, row)) for row in serializable_rows]
                else:
                    connection.commit()
                    data = f"{cursor.rowcount} row(s) affected."
        
        return jsonify({"status": "success", "data": data})
        
    except Exception as e:
        logging.error(f"Error executing SQL: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route("/download", methods=["POST"])
def download():
    sql_code = request.form.get("sql")
    file_type = request.form.get("file_type", "txt")
    suffix = f".{file_type}"
    
    with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix=suffix) as tmp:
        tmp.write(sql_code)
        tmp_path = tmp.name
    
    return send_file(tmp_path, as_attachment=True, download_name=f"query{suffix}")

@app.route("/vector-store-status", methods=["GET"])
def vector_store_status():
    """API endpoint to check vector store status"""
    has_data, status_message = check_vector_store_status()
    return jsonify({
        "ready": has_data,
        "message": status_message,
        "table_count": len(valid_table_names)
    })

@app.route("/toggle-theme", methods=["POST"])
def toggle_theme():
    current = session.get('theme', 'light')
    session['theme'] = 'dark' if current == 'light' else 'light'
    session.modified = True
    return jsonify({"theme": session['theme']})

@app.context_processor
def inject_theme():
    return dict(theme=session.get('theme', 'light'))



if __name__ == "__main__":
    print("=== Vanna RAG SQL Assistant ===")
    has_data, status_message = check_vector_store_status()
    if has_data:
        print(f"âœ… {status_message}")
        print(f"âœ… Found {len(valid_table_names)} database tables")
        print("âœ… RAG functionality enabled")
    else:
        print(f"âš ï¸� Warning: {status_message}")
        print("Please run the training script first:")
        print("  python train_vanna.py")
        print("Choose option 2 or 3 to populate the vector store with training data")
    print("\nStarting Flask application...")
    app.run(debug=True, host='localhost', port=8080)