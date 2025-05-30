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

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        config = config or {}
        config['model'] = 'mistral'
        config['persist_directory'] = './vanna-data'
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)
        self.custom_oracle_prompt = ""

    def set_custom_oracle_prompt(self, prompt: str):
        """Set the custom Oracle SQL prompt"""
        self.custom_oracle_prompt = prompt

    def get_related_ddl(self, question: str, n_results: int = 5) -> list:
        """Get related DDL statements using RAG similarity search"""
        try:
            ddl_results = []
            
            # Method 1: Try get_similar_question_sql for DDL
            if hasattr(self, 'get_similar_question_sql'):
                try:
                    similar_data = self.get_similar_question_sql(question)
                    if similar_data is not None:
                        # Handle DataFrame case
                        if hasattr(similar_data, 'empty'):  # It's a DataFrame
                            if not similar_data.empty:
                                for _, row in similar_data.iterrows():
                                    # Look for DDL in different possible columns
                                    for col in ['content', 'ddl', 'sql', 'question', 'training_data']:
                                        if col in row and row[col] is not None:
                                            content = str(row[col])
                                            if any(keyword in content.upper() for keyword in ['CREATE TABLE', 'CREATE VIEW', 'ALTER TABLE']):
                                                ddl_results.append(content)
                        # Handle list/tuple case
                        elif isinstance(similar_data, (list, tuple)):
                            for item in similar_data:
                                if isinstance(item, (tuple, list)) and len(item) >= 2:
                                    content = str(item[1]) if len(item) > 1 else str(item[0])
                                elif isinstance(item, str):
                                    content = item
                                else:
                                    content = str(item)
                                
                                if any(keyword in content.upper() for keyword in ['CREATE TABLE', 'CREATE VIEW', 'ALTER TABLE']):
                                    ddl_results.append(content)
                except Exception as e:
                    logging.debug(f"get_similar_question_sql failed: {e}")
            
            # Method 2: Try direct training data access for DDL
            if hasattr(self, 'get_training_data') and len(ddl_results) < n_results:
                try:
                    training_data = self.get_training_data()
                    if training_data is not None:
                        # Handle DataFrame case
                        if hasattr(training_data, 'empty'):  # It's a DataFrame
                            if not training_data.empty:
                                for _, row in training_data.iterrows():
                                    for col in ['content', 'ddl', 'sql', 'question', 'training_data']:
                                        if col in row and row[col] is not None:
                                            content = str(row[col])
                                            if any(keyword in content.upper() for keyword in ['CREATE TABLE', 'CREATE VIEW', 'ALTER TABLE']):
                                                ddl_results.append(content)
                                                if len(ddl_results) >= n_results:
                                                    break
                        # Handle list case
                        elif isinstance(training_data, list):
                            for item in training_data:
                                content = str(item)
                                if any(keyword in content.upper() for keyword in ['CREATE TABLE', 'CREATE VIEW', 'ALTER TABLE']):
                                    ddl_results.append(content)
                                    if len(ddl_results) >= n_results:
                                        break
                except Exception as e:
                    logging.debug(f"get_training_data for DDL failed: {e}")
            
            # Method 3: Try ChromaDB direct query for DDL
            if len(ddl_results) < n_results:
                try:
                    import chromadb
                    chroma_client = chromadb.PersistentClient(path='./vanna-data')
                    collections = chroma_client.list_collections()
                    
                    for collection in collections:
                        try:
                            # Query for DDL-related content
                            results = collection.query(
                                query_texts=[question],
                                n_results=min(10, n_results * 2),
                                where={"$or": [
                                    {"content": {"$contains": "CREATE TABLE"}},
                                    {"content": {"$contains": "CREATE VIEW"}},
                                    {"content": {"$contains": "ALTER TABLE"}}
                                ]} if hasattr(collection, 'query') else None
                            )
                            
                            if results and 'documents' in results:
                                for doc_list in results['documents']:
                                    for doc in doc_list:
                                        if any(keyword in doc.upper() for keyword in ['CREATE TABLE', 'CREATE VIEW', 'ALTER TABLE']):
                                            ddl_results.append(doc)
                                            if len(ddl_results) >= n_results:
                                                break
                        except Exception as e:
                            logging.debug(f"ChromaDB DDL query failed for collection {collection.name}: {e}")
                            continue
                except Exception as e:
                    logging.debug(f"ChromaDB DDL access failed: {e}")
            
            return ddl_results[:n_results]
            
        except Exception as e:
            logging.warning(f"Could not retrieve DDL context: {e}")
            return []

    def get_related_documentation(self, question: str, n_results: int = 3) -> list:
        """Get related documentation using RAG similarity search"""
        try:
            doc_results = []
            
            # Method 1: Try get_similar_question_sql for documentation
            if hasattr(self, 'get_similar_question_sql'):
                try:
                    similar_data = self.get_similar_question_sql(question)
                    if similar_data is not None:
                        # Handle DataFrame case - FIX THE AMBIGUITY WARNING
                        if hasattr(similar_data, 'empty'):  # It's a DataFrame
                            if not similar_data.empty:  # Use .empty instead of truth value
                                for _, row in similar_data.iterrows():
                                    for col in ['content', 'documentation', 'description', 'question']:
                                        if col in row and row[col] is not None:
                                            content = str(row[col])
                                            # Look for documentation-like content
                                            if any(keyword in content.lower() for keyword in 
                                                 ['contains', 'table', 'column', 'field', 'stores', 'represents', 'description']):
                                                doc_results.append(content)
                        # Handle list/tuple case
                        elif isinstance(similar_data, (list, tuple)):
                            for item in similar_data:
                                if isinstance(item, (tuple, list)) and len(item) >= 2:
                                    content = str(item[0]) if len(item) > 1 else str(item)
                                elif isinstance(item, str):
                                    content = item
                                else:
                                    content = str(item)
                                
                                if any(keyword in content.lower() for keyword in 
                                     ['contains', 'table', 'column', 'field', 'stores', 'represents']):
                                    doc_results.append(content)
                except Exception as e:
                    logging.debug(f"get_similar_question_sql for docs failed: {e}")
            
            # Method 2: Try direct training data access for documentation
            if hasattr(self, 'get_training_data') and len(doc_results) < n_results:
                try:
                    training_data = self.get_training_data()
                    if training_data is not None:
                        # Handle DataFrame case - FIX THE AMBIGUITY WARNING
                        if hasattr(training_data, 'empty'):  # It's a DataFrame
                            if not training_data.empty:  # Use .empty instead of truth value
                                for _, row in training_data.iterrows():
                                    for col in ['content', 'documentation', 'description', 'question']:
                                        if col in row and row[col] is not None:
                                            content = str(row[col])
                                            if (any(keyword in content.lower() for keyword in 
                                                  ['contains', 'table', 'column', 'field', 'stores', 'represents']) and
                                                not any(keyword in content.upper() for keyword in ['SELECT', 'CREATE TABLE', 'INSERT'])):
                                                doc_results.append(content)
                                                if len(doc_results) >= n_results:
                                                    break
                        # Handle list case
                        elif isinstance(training_data, list):
                            for item in training_data:
                                content = str(item)
                                if (any(keyword in content.lower() for keyword in 
                                      ['contains', 'table', 'column', 'field', 'stores', 'represents']) and
                                    not any(keyword in content.upper() for keyword in ['SELECT', 'CREATE TABLE', 'INSERT'])):
                                    doc_results.append(content)
                                    if len(doc_results) >= n_results:
                                        break
                except Exception as e:
                    logging.debug(f"get_training_data for docs failed: {e}")
            
            return doc_results[:n_results]
            
        except Exception as e:
            logging.warning(f"Could not retrieve documentation: {e}")
            return []

    def get_similar_questions(self, question: str, n_results: int = 3) -> list:
        """Get similar questions and their SQL using RAG"""
        try:
            question_pairs = []
            
            # Method 1: Try get_similar_question_sql
            if hasattr(self, 'get_similar_question_sql'):
                try:
                    similar = self.get_similar_question_sql(question)
                    if similar is not None:
                        # Handle DataFrame case - FIX THE AMBIGUITY WARNING
                        if hasattr(similar, 'empty'):  # It's a DataFrame
                            if not similar.empty:  # Use .empty instead of truth value
                                for _, row in similar.iterrows():
                                    # Try to extract question-SQL pairs from DataFrame
                                    q_text = None
                                    sql_text = None
                                    
                                    # Look for question in different columns
                                    for q_col in ['question', 'query', 'q', 'user_question']:
                                        if q_col in row and row[q_col] is not None:
                                            q_text = str(row[q_col])
                                            break
                                    
                                    # Look for SQL in different columns
                                    for sql_col in ['sql', 'query', 'answer', 'response']:
                                        if sql_col in row and row[sql_col] is not None:
                                            content = str(row[sql_col])
                                            if any(keyword in content.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                                                sql_text = content
                                                break
                                    
                                    if q_text and sql_text:
                                        question_pairs.append((q_text, sql_text))
                                    elif sql_text:  # Just SQL without question
                                        question_pairs.append(("Example query", sql_text))
                        
                        # Handle list/tuple case
                        elif isinstance(similar, (list, tuple)):
                            for item in similar:
                                if isinstance(item, (tuple, list)) and len(item) >= 2:
                                    q_text = str(item[0])
                                    sql_text = str(item[1])
                                    question_pairs.append((q_text, sql_text))
                                elif isinstance(item, str) and any(keyword in item.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                                    question_pairs.append(("Example query", item))
                except Exception as e:
                    logging.debug(f"get_similar_question_sql failed: {e}")
            
            # Method 2: Try direct training data access for Q&A pairs
            if hasattr(self, 'get_training_data') and len(question_pairs) < n_results:
                try:
                    training_data = self.get_training_data()
                    if training_data is not None:
                        # Handle DataFrame case
                        if hasattr(training_data, 'empty'):  # It's a DataFrame
                            if not training_data.empty:  # Use .empty instead of truth value
                                for _, row in training_data.iterrows():
                                    q_text = None
                                    sql_text = None
                                    
                                    # Look for question-SQL pairs
                                    for q_col in ['question', 'query', 'q', 'user_question']:
                                        if q_col in row and row[q_col] is not None:
                                            q_text = str(row[q_col])
                                            break
                                    
                                    for sql_col in ['sql', 'query', 'answer', 'response']:
                                        if sql_col in row and row[sql_col] is not None:
                                            content = str(row[sql_col])
                                            if any(keyword in content.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                                                sql_text = content
                                                break
                                    
                                    if q_text and sql_text:
                                        question_pairs.append((q_text, sql_text))
                                        if len(question_pairs) >= n_results:
                                            break
                        # Handle list case
                        elif isinstance(training_data, list):
                            for item in training_data:
                                if isinstance(item, (tuple, list)) and len(item) >= 2:
                                    q_text = str(item[0])
                                    sql_text = str(item[1])
                                    if any(keyword in sql_text.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                                        question_pairs.append((q_text, sql_text))
                                        if len(question_pairs) >= n_results:
                                            break
                except Exception as e:
                    logging.debug(f"get_training_data for Q&A failed: {e}")
            
            return question_pairs[:n_results]
            
        except Exception as e:
            logging.warning(f"Could not retrieve similar questions: {e}")
            return []

    def build_context_from_rag(self, question: str) -> str:
        """Build comprehensive context string from RAG components"""
        context_parts = []
        
        # Get DDL/Schema information (more comprehensive)
        ddl_info = self.get_related_ddl(question, n_results=7)
        if ddl_info:
            context_parts.append("=== DATABASE SCHEMA (DDL) ===")
            for i, ddl in enumerate(ddl_info, 1):
                context_parts.append(f"Schema {i}:")
                context_parts.append(ddl.strip())
                context_parts.append("")
        
        # Get similar questions/examples (more examples)
        similar_questions = self.get_similar_questions(question, n_results=5)
        if similar_questions:
            context_parts.append("=== SIMILAR QUERY EXAMPLES ===")
            for i, (q, sql) in enumerate(similar_questions, 1):
                context_parts.append(f"Example {i}:")
                context_parts.append(f"Question: {q}")
                context_parts.append(f"SQL: {sql}")
                context_parts.append("")
        
        # Get documentation (more comprehensive)
        docs = self.get_related_documentation(question, n_results=5)
        if docs:
            context_parts.append("=== TABLE/COLUMN DOCUMENTATION ===")
            for i, doc in enumerate(docs, 1):
                context_parts.append(f"Documentation {i}:")
                context_parts.append(doc.strip())
                context_parts.append("")
        
        # Add a summary section for context awareness
        if context_parts:
            summary_parts = []
            if ddl_info:
                summary_parts.append(f"{len(ddl_info)} schema definitions")
            if similar_questions:
                summary_parts.append(f"{len(similar_questions)} example queries")
            if docs:
                summary_parts.append(f"{len(docs)} documentation entries")
            
            context_summary = f"=== CONTEXT SUMMARY ===\nAvailable context: {', '.join(summary_parts)}\n\n"
            return context_summary + "\n".join(context_parts)
        
        return "\n".join(context_parts)

    def get_sql_prompt(self, question: str, **kwargs) -> list:
        """Generate SQL prompt combining custom Oracle rules with comprehensive RAG context"""
        
        # Build comprehensive RAG context
        rag_context = self.build_context_from_rag(question)
        
        # Enhanced system prompt with better context integration
        if self.custom_oracle_prompt and rag_context:
            system_content = f"""{self.custom_oracle_prompt}

{rag_context}

IMPORTANT INSTRUCTIONS:
1. Use ONLY the tables and columns shown in the schema definitions above
2. Follow the patterns from the example queries provided
3. Reference the documentation to understand what each table/column contains
4. Generate clean Oracle SQL that matches the examples and schema
5. If the question asks about data not covered in the schema, explain what's missing

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
- Follow the patterns shown in the example queries"""
        else:
            system_content = "You are an Oracle SQL expert. Generate clean, executable Oracle SQL queries using proper Oracle syntax."
        
        return [
            {
                'role': 'system', 
                'content': system_content
            },
            {
                'role': 'user', 
                'content': question
            }
        ]

    def generate_sql(self, question: str, **kwargs) -> str:
        """Generate SQL using comprehensive RAG context + custom Oracle rules"""
        logging.info(f"Generating SQL with enhanced RAG context for: {question}")
        
        # Get the prompt with comprehensive RAG context
        messages = self.get_sql_prompt(question, **kwargs)
        
        # Log context information for debugging
        context_length = len(messages[0]['content'])
        logging.info(f"System prompt length: {context_length} characters")
        
        # Count context components for debugging
        system_content = messages[0]['content']
        schema_count = system_content.count('=== DATABASE SCHEMA (DDL) ===')
        example_count = system_content.count('=== SIMILAR QUERY EXAMPLES ===')
        doc_count = system_content.count('=== TABLE/COLUMN DOCUMENTATION ===')
        
        logging.info(f"RAG Context: {schema_count} schema sections, {example_count} example sections, {doc_count} doc sections")
        
        # Generate response using Ollama
        response = self.submit_prompt(messages)
        
        # Extract clean SQL from response
        sql = self._extract_sql_from_response(response)
        logging.info(f"Generated SQL: {sql}")
        
        return sql

    def _extract_sql_from_response(self, response: str) -> str:
        """Extract clean SQL from the LLM response"""
        # Remove common prefixes/suffixes that LLMs add
        lines = response.strip().split('\n')
        sql_lines = []
        
        in_sql_block = False
        for line in lines:
            line = line.strip()
            
            # Check for SQL code blocks
            if line.startswith('```sql') or line.startswith('```SQL'):
                in_sql_block = True
                continue
            elif line.startswith('```') and in_sql_block:
                break
            elif in_sql_block:
                sql_lines.append(line)
                continue
            
            # Skip empty lines and common explanatory text
            if (line and 
                not line.lower().startswith('here') and 
                not line.lower().startswith('the sql') and
                not line.lower().startswith('this query') and
                not line.lower().startswith('note:') and
                not line.lower().startswith('explanation:') and
                not line.startswith('--')):  # Skip SQL comments at the start
                
                # Check if line looks like SQL (contains SQL keywords)
                sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH', 'FROM', 'WHERE']
                if any(keyword in line.upper() for keyword in sql_keywords):
                    sql_lines.append(line)
        
        result = ' '.join(sql_lines).strip()
        
        # If no SQL found in structured way, try to extract from full response
        if not result:
            # Look for lines that contain SQL keywords
            for line in response.split('\n'):
                line = line.strip()
                if any(keyword in line.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                    result = line
                    break
        
        return result if result else response.strip()

# Initialize the Flask app
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
            test_sql = vn.generate_sql("show me accounts")
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
        test_sql = vn.generate_sql("show me accounts")
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
        sql = vn_test.generate_sql("show me accounts")
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
        sql = vn.generate_sql(question=question)
        
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
    
    # Check vector store status on startup
    has_data, status_message = check_vector_store_status()
    
    if has_data:
        print(f"✅ {status_message}")
        print(f"✅ Found {len(valid_table_names)} database tables")
        print("✅ RAG functionality enabled")
    else:
        print(f"⚠️ Warning: {status_message}")
        print("Please run the training script first:")
        print("  python train_vanna.py")
        print("Choose option 2 or 3 to populate the vector store with training data")
    
    print("\nStarting Flask application...")
    app.run(debug=True, host='localhost', port=5000)