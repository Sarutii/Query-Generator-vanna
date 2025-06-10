from flask import Flask, render_template, request, jsonify, session
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
import os
import oracledb
import sqlparse
import logging
import chromadb
import shutil
import glob
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Multilingual Embedding Function (same as training script)
class MultilingualEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        logging.info("✅ Loaded multilingual embedding model: paraphrase-multilingual-MiniLM-L12-v2")

    def __call__(self, input: list[str]) -> list[list[float]]:
        try:
            if isinstance(input, str):
                input = [input]
            embeddings = self.model.encode(input, convert_to_numpy=True)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            return embeddings.tolist()
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            return [[0.0] * 384 for _ in input]

# FIXED: Use the same FixedMyVanna class from training script
class FixedMyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None, reset_data=False):
        """
        Initialize Vanna with GUARANTEED vector store population
        """
        
        # Set up the embedding function
        self.multilingual_embedding = MultilingualEmbeddingFunction()
        
        # Use a FIXED persist directory (SAME as training script)
        persist_dir = './vanna-multilingual-persistent'
        
        # Clean slate approach - ONLY if explicitly requested
        if reset_data:
            logging.info("🗑️ RESET REQUESTED - Cleaning ChromaDB to start fresh...")
            self._clean_chromadb_completely(persist_dir)
        else:
            logging.info("📂 Using existing vector store data (if available)")

        # Create persist directory if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)
        
        # CRITICAL FIX: Initialize ChromaDB client BEFORE Vanna
        try:
            self.chroma_client = chromadb.PersistentClient(path=persist_dir)
            logging.info(f"✅ ChromaDB client initialized at {persist_dir}")
        except Exception as e:
            logging.error(f"❌ Failed to initialize ChromaDB client: {e}")
            raise
        
        # Separate configs for each parent class
        chroma_config = {
            'persist_directory': persist_dir,
            'embedding_function': self.multilingual_embedding,
            'client': self.chroma_client  # Pass the client explicitly
        }

        ollama_config = {
            'model': 'mistral'
        }

        # Store config for operations
        self.chroma_config = chroma_config
        self.persist_dir = persist_dir
        
        # Initialize parent classes WITH EXPLICIT ERROR HANDLING
        try:
            logging.info("🔧 Initializing ChromaDB_VectorStore...")
            ChromaDB_VectorStore.__init__(self, config=chroma_config)
            logging.info("✅ ChromaDB_VectorStore initialized")
            
            logging.info("🔧 Initializing Ollama...")
            Ollama.__init__(self, config=ollama_config)
            logging.info("✅ Ollama initialized")
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize Vanna: {e}")
            raise
        
        # Custom Oracle prompt
        self.custom_oracle_prompt = ""
        
        # CRITICAL: Verify vector store is actually accessible
        self._verify_vector_store_accessibility()
        self._check_existing_data()

    def _clean_chromadb_completely(self, target_dir: str):
        """Completely remove specific ChromaDB directory"""
        try:
            if os.path.exists(target_dir) and os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
                logging.info(f"✅ Removed directory {target_dir}")
            
            # Also clean any database files in current directory
            file_patterns = ['./chroma.sqlite3*', './*.db']
            for pattern in file_patterns:
                for path in glob.glob(pattern):
                    if os.path.exists(path) and os.path.isfile(path):
                        try:
                            os.remove(path)
                            logging.info(f"✅ Removed file {path}")
                        except Exception as e:
                            logging.warning(f"⚠️ Error removing file {path}: {e}")
                        
        except Exception as e:
            logging.warning(f"⚠️ Error cleaning ChromaDB: {e}")

    def _verify_vector_store_accessibility(self):
        """CRITICAL: Verify that the vector store is actually accessible"""
        try:
            logging.info("🔍 Verifying vector store accessibility...")
            
            # Check ChromaDB client
            collections = self.chroma_client.list_collections()
            logging.info(f"   ChromaDB collections: {[c.name for c in collections]}")
            
            logging.info("✅ Vector store accessibility verified")
            return True
            
        except Exception as e:
            logging.error(f"❌ Vector store accessibility check failed: {e}")
            return False

    def _check_existing_data(self):
        """Check if vector store already contains training data"""
        try:
            collections = self.chroma_client.list_collections()
            total_docs = 0
            
            collection_info = {}
            for collection in collections:
                count = collection.count()
                total_docs += count
                collection_info[collection.name] = count
            
            if total_docs > 0:
                logging.info(f"📊 Existing data found:")
                for name, count in collection_info.items():
                    logging.info(f"   Collection '{name}': {count} documents")
                logging.info(f"   Total documents: {total_docs}")
                return True
            else:
                logging.info("📭 Vector store appears to be empty - needs training data")
                return False
                
        except Exception as e:
            logging.warning(f"⚠️ Could not check existing data: {e}")
            return False

    def set_custom_oracle_prompt(self, prompt: str):
        self.custom_oracle_prompt = prompt
    def generate_sql_with_preview(self, question: str, similarity_threshold: float = 0.7, force_simple_limit: bool = True, **kwargs) -> str:
        logging.info(f"Generating SQL for: {question}")
        rag_context = self.build_context_from_rag(question, similarity_threshold=similarity_threshold, force_simple_limit=force_simple_limit)
        system_content = f"{self.custom_oracle_prompt}\n{rag_context}\nTask: Generate the SQL query that answers the user's question using the provided database schema and descriptions. Ensure the query is correct and adheres to Oracle SQL syntax. Provide only the SQL query as your response."
        messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': question}
        ]
        response = self.submit_prompt(messages)
        sql = self._extract_sql_from_response(response)
        logging.info(f"Generated SQL: {sql}")
        return sql

    def build_context_from_rag(self, question: str, similarity_threshold: float = 0.7, force_simple_limit: bool = False) -> str:
        """
        Build RAG context with explicit similarity filtering and result limits
        """
        context_parts = []
        
        if force_simple_limit:
            # Simple approach: just limit the results from original methods
            logging.info("Using simple limit approach (no similarity filtering)")
            
            ddl_results = self.get_related_ddl(question, n_results=10)  # Get more initially
            ddl_results = ddl_results[:3] if ddl_results else []  # Force limit to 3
            
            docs = self.get_related_documentation(question, n_results=10)  # Get more initially  
            docs = docs[:3] if docs else []  # Force limit to 3
            
            logging.info(f"Simple limit: {len(ddl_results)} DDL, {len(docs)} docs")
            
        else:
            # Get DDL results with similarity filtering
            ddl_results = self._get_filtered_ddl(question, max_results=3, similarity_threshold=similarity_threshold)
            
            # Get documentation results with similarity filtering
            docs = self._get_filtered_documentation(question, max_results=3, similarity_threshold=similarity_threshold)
        
        # Build context from DDL results
        if ddl_results:
            context_parts.append("=== DATABASE SCHEMA (DDL) ===")
            for i, ddl in enumerate(ddl_results, 1):
                context_parts.append(f"Schema {i}:")
                context_parts.append(ddl.strip())
                context_parts.append("")
        
        # Build context from documentation results
        if docs:
            context_parts.append("=== TABLE/COLUMN DOCUMENTATION ===")
            for i, doc in enumerate(docs, 1):
                context_parts.append(f"Documentation {i}:")
                context_parts.append(doc.strip())
                context_parts.append("")
        
        return "\n".join(context_parts)

    def _get_filtered_ddl(self, question: str, max_results: int = 3, similarity_threshold: float = 0.7):
        """Get DDL results with explicit similarity filtering"""
        try:
            # First try the simple approach - just limit the original method results
            logging.info(f"Getting DDL results for: {question}")
            all_results = self.get_related_ddl(question, n_results=10)  # Get more to have options
            
            if not all_results:
                logging.warning("No DDL results from get_related_ddl")
                return []
            
            logging.info(f"Got {len(all_results)} DDL results from get_related_ddl")
            
            # Try direct ChromaDB access for similarity scores
            try:
                collections = self.chroma_client.list_collections()
                ddl_collection = None
                
                # Find the DDL collection
                for collection in collections:
                    if 'ddl' in collection.name.lower():
                        ddl_collection = collection
                        break
                
                if ddl_collection:
                    # Query with similarity scoring
                    query_results = ddl_collection.query(
                        query_texts=[question],
                        n_results=min(len(all_results), 3),
                        include=['documents', 'distances', 'metadatas']
                    )
                    
                    if query_results and query_results.get('documents') and query_results.get('distances'):
                        documents = query_results['documents'][0]
                        distances = query_results['distances'][0]
                        
                        # Debug: Log actual distance values
                        logging.info(f"Distance values: {distances[:5]}")  # Show first 5
                        logging.info(f"Min distance: {min(distances)}, Max distance: {max(distances)}")
                        
                        # Filter by distance threshold (lower distance = higher similarity)
                        # For cosine distance: 0 = identical, 2 = opposite
                        # For euclidean distance: 0 = identical, higher = more different
                        
                        filtered_results = []
                        for doc, distance in zip(documents, distances):
                            # Use distance directly - lower is better
                            # Adjust threshold: 0.3 means very similar, 1.0 means moderately similar
                            distance_threshold = 1.0 - similarity_threshold  # Convert similarity to distance
                            
                            logging.info(f"Distance: {distance:.3f}, Threshold: {distance_threshold:.3f}")
                            
                            if distance <= distance_threshold and len(filtered_results) < max_results:
                                filtered_results.append(doc)
                                logging.info(f"DDL result added with distance: {distance:.3f}")
                        
                        if filtered_results:
                            logging.info(f"DDL: {len(filtered_results)} results after distance filtering")
                            return filtered_results
            
            except Exception as e:
                logging.warning(f"ChromaDB direct access failed: {e}")
            
            # Fallback: Just return the first N results from the original method
            limited_results = all_results[:max_results]
            logging.info(f"DDL: Using fallback - returning first {len(limited_results)} results")
            return limited_results
            
        except Exception as e:
            logging.error(f"Error in filtered DDL retrieval: {e}")
            return []

    def _get_filtered_documentation(self, question: str, max_results: int = 3, similarity_threshold: float = 0.7):
        """Get documentation results with explicit similarity filtering"""
        try:
            # First try the simple approach - just limit the original method results
            logging.info(f"Getting documentation results for: {question}")
            all_results = self.get_related_documentation(question, n_results=10)  # Get more to have options
            
            if not all_results:
                logging.warning("No documentation results from get_related_documentation")
                return []
            
            logging.info(f"Got {len(all_results)} documentation results from get_related_documentation")
            
            # Try direct ChromaDB access for similarity scores
            try:
                collections = self.chroma_client.list_collections()
                doc_collection = None
                
                # Find the documentation collection
                for collection in collections:
                    if 'doc' in collection.name.lower() and 'ddl' not in collection.name.lower():
                        doc_collection = collection
                        break
                
                if doc_collection:
                    # Query with similarity scoring
                    query_results = doc_collection.query(
                        query_texts=[question],
                        n_results=min(len(all_results), 10),
                        include=['documents', 'distances', 'metadatas']
                    )
                    
                    if query_results and query_results.get('documents') and query_results.get('distances'):
                        documents = query_results['documents'][0]
                        distances = query_results['distances'][0]
                        
                        # Debug: Log actual distance values
                        logging.info(f"Doc distance values: {distances[:5]}")  # Show first 5
                        
                        # Filter by distance threshold (lower distance = higher similarity)
                        filtered_results = []
                        for doc, distance in zip(documents, distances):
                            # Use distance directly - lower is better
                            distance_threshold = 1.0 - similarity_threshold  # Convert similarity to distance
                            
                            logging.info(f"Doc distance: {distance:.3f}, Threshold: {distance_threshold:.3f}")
                            
                            if distance <= distance_threshold and len(filtered_results) < max_results:
                                filtered_results.append(doc)
                                logging.info(f"Doc result added with distance: {distance:.3f}")
                        
                        if filtered_results:
                            logging.info(f"Documentation: {len(filtered_results)} results after distance filtering")
                            return filtered_results
            
            except Exception as e:
                logging.warning(f"ChromaDB direct access failed for docs: {e}")
            
            # Fallback: Just return the first N results from the original method
            limited_results = all_results[:max_results]
            logging.info(f"Documentation: Using fallback - returning first {len(limited_results)} results")
            return limited_results
            
        except Exception as e:
            logging.error(f"Error in filtered documentation retrieval: {e}")
            return []    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL from LLM response, handling various response formats"""
        try:
            if not response:
                return ""
            
            # Remove common prefixes and suffixes
            response = response.strip()
            
            # Look for SQL code blocks
            import re
            
            # Check for SQL code blocks (```sql ... ```)
            sql_block_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
            if sql_block_match:
                return sql_block_match.group(1).strip()
            
            # Check for generic code blocks (``` ... ```)
            code_block_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if code_block_match:
                potential_sql = code_block_match.group(1).strip()
                # Check if it looks like SQL
                if any(keyword in potential_sql.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']):
                    return potential_sql
            
            # Look for SQL keywords at the start of lines
            lines = response.split('\n')
            sql_lines = []
            capturing_sql = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    if capturing_sql:
                        sql_lines.append(line)
                    continue
                
                # Start capturing if line begins with SQL keyword
                if any(line.upper().startswith(keyword) for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'WITH']):
                    capturing_sql = True
                    sql_lines = [line]
                elif capturing_sql:
                    # Continue capturing if it looks like part of SQL
                    if (line.upper().startswith(('FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'ON', 'GROUP', 'ORDER', 'HAVING', 'UNION', 'AND', 'OR')) or
                        line.endswith(',') or line.endswith('(') or sql_lines[-1].endswith(',') or sql_lines[-1].endswith('(')):
                        sql_lines.append(line)
                    else:
                        # Stop capturing if we hit non-SQL content
                        break
            
            if sql_lines:
                return '\n'.join(sql_lines).strip()
            
            # If no structured SQL found, return the whole response if it contains SQL keywords
            if any(keyword in response.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']):
                return response.strip()
            
            # Last resort: return as-is
            return response.strip()
            
        except Exception as e:
            logging.error(f"Error extracting SQL from response: {e}")
            return response.strip() if response else ""

    def comprehensive_vector_store_test(self):
        """Comprehensive test of the vector store"""
        try:
            logging.info("🧪 Running comprehensive vector store test...")
            
            # Direct ChromaDB access
            collections = self.chroma_client.list_collections()
            total_docs = 0
            
            logging.info(f"📊 Found {len(collections)} collections:")
            for collection in collections:
                count = collection.count()
                total_docs += count
                logging.info(f"   '{collection.name}': {count} documents")
                
                # Get sample data
                if count > 0:
                    try:
                        sample = collection.get(limit=2, include=['documents', 'metadatas'])
                        for i, doc in enumerate(sample.get('documents', [])):
                            logging.info(f"      Sample {i+1}: {doc[:100]}...")
                    except Exception as e:
                        logging.warning(f"      Error getting samples: {e}")
            
            logging.info(f"📈 Total documents in vector store: {total_docs}")
            
            # Test Vanna methods
            if total_docs > 0:
                logging.info("🔍 Testing Vanna retrieval methods...")
                
                try:
                    ddl_results = self.get_related_ddl("SELECT", n_results=2)
                    logging.info(f"   get_related_ddl: {len(ddl_results) if ddl_results else 0} results")
                except Exception as e:
                    logging.warning(f"   get_related_ddl failed: {e}")
                
                try:
                    doc_results = self.get_related_documentation("database", n_results=2)
                    logging.info(f"   get_related_documentation: {len(doc_results) if doc_results else 0} results")
                except Exception as e:
                    logging.warning(f"   get_related_documentation failed: {e}")
                
                try:
                    qa_results = self.get_similar_question_sql("show tables", n_results=2)
                    logging.info(f"   get_similar_question_sql: {len(qa_results) if qa_results else 0} results")
                except Exception as e:
                    logging.warning(f"   get_similar_question_sql failed: {e}")
            
            return total_docs > 0
            
        except Exception as e:
            logging.error(f"❌ Comprehensive test failed: {e}")
            return False

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'vanna-rag-sql-assistant-secret-key'

# Initialize Oracle Client
try:
    oracledb.init_oracle_client(lib_dir=r"C:\Users\ahmed\Downloads\instantclient-basic-windows.x64-23.8.0.25.04\instantclient_23_8")
    logging.info("Oracle client initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Oracle client: {e}")

# FIXED: Initialize Vanna with FixedMyVanna (same as training script)
vn = FixedMyVanna()

# Set Custom Oracle Prompt
oracle_prompt = """Instruction: You are an SQL expert specializing in Oracle databases. Based on the provided database schema and descriptions, generate an SQL query that directly answers the user's question. The query must be executable on an Oracle database and should be provided without any additional text or explanations.

Database Schema and Descriptions:

Below are the relevant table definitions and descriptions retrieved from the database schema:"""
vn.set_custom_oracle_prompt(oracle_prompt)

# Connect to Oracle
try:
    vn.connect_to_oracle(user='IAS202538', password='123', dsn="localhost:1521/xepdb1")
    logging.info("Vanna connected to Oracle successfully")
except Exception as e:
    logging.error(f"Failed to connect Vanna to Oracle: {e}")

# Enable RAG Features
vn.allow_llm_to_see_data = True

# Utility Functions
def get_valid_table_names():
    """Get valid table names from the database"""
    valid_tables = set()
    try:
        with oracledb.connect(user='IAS202538', password='123', dsn='localhost:1521/xepdb1') as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT table_name FROM user_tables ORDER BY table_name")
                tables = cursor.fetchall()
                valid_tables = {table[0].upper() for table in tables}
                logging.info(f"Found {len(valid_tables)} valid tables")
    except Exception as e:
        logging.error(f"Failed to get table names: {e}")
    return valid_tables

valid_table_names = get_valid_table_names()

def extract_table_names(sql_query):
    """Extract table names from an SQL query using sqlparse"""
    try:
        parsed = sqlparse.parse(sql_query)
        table_names = set()
        for statement in parsed:
            for token in statement.flatten():
                if token.ttype is sqlparse.tokens.Name:
                    token_str = token.value.upper().strip()
                    if token_str in valid_table_names and len(token_str) > 2 and token_str not in ['AS', 'ON', 'AND', 'OR', 'IN', 'IS', 'NOT']:
                        table_names.add(token_str)
        return table_names
    except Exception as e:
        logging.warning(f"Error parsing SQL: {e}")
        return set()

def is_valid_sql(sql_query, valid_tables):
    """Check if all table names in the SQL query are valid"""
    try:
        referenced_tables = extract_table_names(sql_query)
        return len(referenced_tables - valid_tables) == 0 if referenced_tables else True
    except Exception as e:
        logging.warning(f"Error validating SQL: {e}")
        return True

def check_vector_store_status():
    """IMPROVED: Check if the vector store has been populated with training data"""
    try:
        # Use the comprehensive test from FixedMyVanna
        has_data = vn.comprehensive_vector_store_test()
        
        if has_data:
            # Additional checks
            collections = vn.chroma_client.list_collections()
            total_docs = sum(c.count() for c in collections)
            
            # Test actual retrieval
            test_queries = ["SELECT", "database", "table"]
            working_retrievals = 0
            
            for query in test_queries:
                try:
                    ddl_results = vn.get_related_ddl(query, n_results=1)
                    if ddl_results and len(ddl_results) > 0:
                        working_retrievals += 1
                except:
                    pass
            
            if working_retrievals > 0:
                return True, f"Vector store is ready with {total_docs} documents across {len(collections)} collections"
            else:
                return False, f"Vector store has {total_docs} documents but retrieval is not working"
        else:
            return False, "Vector store is empty or not properly initialized"
            
    except Exception as e:
        logging.error(f"Error checking vector store: {e}")
        return False, f"Error checking vector store: {str(e)}"

# Flask Routes
@app.route("/", methods=["GET"])
def index():
    has_data, status_message = check_vector_store_status()
    return render_template("index.html", vector_store_status=status_message, vector_store_ready=has_data)

@app.route("/ask", methods=["POST"])
def ask_sql():
    question = request.form.get("question")
    if not question:
        return jsonify({"status": "error", "message": "No question provided"})
    
    has_data, status_message = check_vector_store_status()
    if not has_data:
        return jsonify({"status": "error", "message": f"Vector store not ready: {status_message}. Please train the model first."})
    
    try:
        logging.info(f"Processing question: {question}")
        sql = vn.generate_sql_with_preview(question=question)
        if not sql:
            return jsonify({"status": "error", "message": "No SQL generated"})
        sql = sql.strip().rstrip(';')  # Clean up SQL for Oracle
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
        with oracledb.connect(user='IAS202538', password='123', dsn='localhost:1521/xepdb1') as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql_code)
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    rows = cursor.fetchall()
                    data = [dict(zip(columns, row)) for row in rows]
                else:
                    connection.commit()
                    data = f"{cursor.rowcount} row(s) affected."
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        logging.error(f"Error executing SQL: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route("/vector-store-status", methods=["GET"])
def vector_store_status():
    has_data, status_message = check_vector_store_status()
    
    # Additional debugging info
    debug_info = {}
    try:
        collections = vn.chroma_client.list_collections()
        debug_info = {
            "collections": len(collections),
            "total_docs": sum(c.count() for c in collections),
            "collection_details": {c.name: c.count() for c in collections}
        }
    except Exception as e:
        debug_info = {"error": str(e)}
    
    return jsonify({
        "ready": has_data, 
        "message": status_message, 
        "table_count": len(valid_table_names),
        "debug_info": debug_info
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
    app.run(debug=True, host='localhost', port=8080)