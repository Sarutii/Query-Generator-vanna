#!/usr/bin/env python3
"""
Enhanced Multilingual Vanna Vector Store Training and Diagnostic Script
Fixed version with PERSISTENT vector store and proper top-k similarity retrieval
"""

import os
import json
import logging
import shutil
import oracledb
import numpy as np
import uuid
import glob
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
from chromadb.api.types import EmbeddingFunction


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MultilingualEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        logging.info(f"✅ Loaded multilingual embedding model: paraphrase-multilingual-MiniLM-L12-v2")

    def __call__(self, input: List[str]) -> List[List[float]]:
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

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None, reset_data=False):
        """
        Initialize Vanna with PERSISTENT vector store
        
        Args:
            config: Configuration dictionary
            reset_data: Only set to True if you want to DELETE all existing training data
        """
        
        # Set up the embedding function
        self.multilingual_embedding = MultilingualEmbeddingFunction()

        # Use a FIXED persist directory (not random)
        persist_dir = './vanna-multilingual-persistent'
        
        # Clean slate approach - ONLY if explicitly requested
        if reset_data:
            logging.info("🗑️ RESET REQUESTED - Cleaning ChromaDB to start fresh...")
            self._clean_chromadb_completely(persist_dir)
        else:
            logging.info("📂 Using existing vector store data (if available)")

        # Create persist directory if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)
        
        # Separate configs for each parent class
        chroma_config = {
            'persist_directory': persist_dir,
            'embedding_function': self.multilingual_embedding
        }

        ollama_config = {
            'model': 'mistral'
        }

        # Store config for operations
        self.chroma_config = chroma_config
        self.persist_dir = persist_dir
        
        # Initialize parent classes
        try:
            ChromaDB_VectorStore.__init__(self, config=chroma_config)
            Ollama.__init__(self, config=ollama_config)
            logging.info(f"✅ Initialized Vanna with multilingual embeddings at {persist_dir}")
        except Exception as e:
            logging.error(f"❌ Failed to initialize Vanna: {e}")
            raise
        
        # Verify setup and check existing data
        self._verify_embedding_setup()
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

    def _check_existing_data(self):
        """Check if vector store already contains training data"""
        try:
            # Try to get some sample data to see if store is populated
            test_results = {
                'ddl_count': 0,
                'doc_count': 0,
                'qa_count': 0
            }
            
            # Test with a simple query
            try:
                ddl_results = super().get_related_ddl("SELECT")
                test_results['ddl_count'] = len(ddl_results) if ddl_results else 0
            except:
                pass
                
            try:
                doc_results = super().get_related_documentation("database")
                test_results['doc_count'] = len(doc_results) if doc_results else 0
            except:
                pass
                
            try:
                qa_results = super().get_similar_question_sql("show")
                test_results['qa_count'] = len(qa_results) if qa_results else 0
            except:
                pass
            
            total_items = sum(test_results.values())
            
            if total_items > 0:
                logging.info(f"📊 Existing data found - DDL: {test_results['ddl_count']}, Docs: {test_results['doc_count']}, Q&A: {test_results['qa_count']}")
                logging.info("✅ Vector store contains existing training data - ready to use!")
                return True
            else:
                logging.info("📭 Vector store appears to be empty - needs training data")
                return False
                
        except Exception as e:
            logging.warning(f"⚠️ Could not check existing data: {e}")
            return False

    def reset_vector_store(self):
        """Delete all previous training data and reset the vector store"""
        try:
            logging.info("🗑️ Resetting vector store - deleting all previous training data...")
            
            # Delete the current persist directory
            if os.path.exists(self.persist_dir):
                shutil.rmtree(self.persist_dir)
                logging.info(f"✅ Deleted existing vector store directory: {self.persist_dir}")
            
            # Recreate the directory
            os.makedirs(self.persist_dir, exist_ok=True)
            
            # Reinitialize ChromaDB with fresh data
            ChromaDB_VectorStore.__init__(self, config=self.chroma_config)
            
            logging.info("✅ Vector store reset complete - ready for fresh training data")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to reset vector store: {e}")
            return False
    
    def _verify_embedding_setup(self):
        """Verify that multilingual embeddings are properly configured"""
        try:
            # Test multilingual embedding
            test_texts = [
                "SELECT * FROM ACCOUNT",
                "اظهر جميع الحسابات",
                "Show all customer information"
            ]
            
            embeddings = self.multilingual_embedding(test_texts)
            logging.info(f"✅ Multilingual embedding test successful - generated {len(embeddings)} embeddings")
            logging.info(f"   Embedding dimension: {len(embeddings[0])}")
            
        except Exception as e:
            logging.warning(f"⚠️ Multilingual embedding setup issue: {e}")
    
    def add_to_vector_store_with_embedding(self, content: str, content_type: str, metadata: Dict = None):
        """Add content to vector store with multilingual embedding verification"""
        try:
            metadata = metadata or {}
            metadata['content_type'] = content_type
            metadata['length'] = len(content)
            metadata['language'] = self._detect_language(content)
            
            # For multilingual content, preprocess it
            processed_content = self._preprocess_multilingual_content(content)
            
            if content_type == 'ddl':
                self.train(ddl=processed_content)
            elif content_type == 'documentation':
                self.train(documentation=processed_content)
            elif content_type == 'question_sql':
                question, sql = processed_content.split('|||') if '|||' in processed_content else ('', processed_content)
                self.train(question=question.strip(), sql=sql.strip())
            
            logging.info(f"✅ Added {content_type} content to vector store (length: {len(content)}, lang: {metadata['language']})")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to add {content_type} content: {e}")
            return False
    
    def _preprocess_multilingual_content(self, content: str) -> str:
        """Preprocess multilingual content to make it more embedding-friendly"""
        # Add English keywords/context for Arabic content
        arabic_chars = sum(1 for char in content if '\u0600' <= char <= '\u06FF')
        if arabic_chars > len(content) * 0.1:
            content = f"Multilingual content (Arabic/English): {content}"
        
        return content
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection for logging purposes"""
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        if arabic_chars > len(text) * 0.1:
            return "ar/en"
        elif arabic_chars > 0:
            return "mixed"
        else:
            return "en"
    
    # ===== FIXED SIMILARITY SEARCH METHODS =====
    
    def get_similar_question_sql_fixed(self, question: str, n: int = 5, similarity_threshold: float = 0.1) -> List[Tuple[str, str]]:
        """Fixed version that properly limits results to top-n"""
        try:
            all_results = super().get_similar_question_sql(question)
            
            if not all_results or not isinstance(all_results, list):
                return []
            
            if hasattr(self, 'multilingual_embedding'):
                query_embedding = self.multilingual_embedding([question])[0]
                scored_results = []
                
                for item in all_results:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        q_text, sql_text = item[0], item[1]
                        item_embedding = self.multilingual_embedding([q_text])[0]
                        similarity = self._cosine_similarity(query_embedding, item_embedding)
                        
                        if similarity >= similarity_threshold:
                            scored_results.append((similarity, item))
                
                scored_results.sort(key=lambda x: x[0], reverse=True)
                return [item for _, item in scored_results[:n]]
            else:
                return all_results[:n]
                
        except Exception as e:
            logging.error(f"Error in get_similar_question_sql_fixed: {e}")
            return []
    
    def get_related_ddl_fixed(self, question: str, n: int = 5, similarity_threshold: float = 0.1) -> List[str]:
        """Fixed version that properly limits results to top-n"""
        try:
            all_results = super().get_related_ddl(question)
            
            if not all_results or not isinstance(all_results, list):
                return []
            
            if hasattr(self, 'multilingual_embedding'):
                query_embedding = self.multilingual_embedding([question])[0]
                scored_results = []
                
                for ddl_text in all_results:
                    if isinstance(ddl_text, str) and ddl_text.strip():
                        ddl_embedding = self.multilingual_embedding([ddl_text])[0]
                        similarity = self._cosine_similarity(query_embedding, ddl_embedding)
                        
                        if similarity >= similarity_threshold:
                            scored_results.append((similarity, ddl_text))
                
                scored_results.sort(key=lambda x: x[0], reverse=True)
                return [ddl for _, ddl in scored_results[:n]]
            else:
                return all_results[:n]
                
        except Exception as e:
            logging.error(f"Error in get_related_ddl_fixed: {e}")
            return []
    
    def get_related_documentation_fixed(self, question: str, n: int = 5, similarity_threshold: float = 0.1) -> List[str]:
        """Fixed version that properly limits results to top-n"""
        try:
            all_results = super().get_related_documentation(question)
            
            if not all_results or not isinstance(all_results, list):
                return []
            
            if hasattr(self, 'multilingual_embedding'):
                query_embedding = self.multilingual_embedding([question])[0]
                scored_results = []
                
                for doc_text in all_results:
                    if isinstance(doc_text, str) and doc_text.strip():
                        doc_embedding = self.multilingual_embedding([doc_text])[0]
                        similarity = self._cosine_similarity(query_embedding, doc_embedding)
                        
                        if similarity >= similarity_threshold:
                            scored_results.append((similarity, doc_text))
                
                scored_results.sort(key=lambda x: x[0], reverse=True)
                return [doc for _, doc in scored_results[:n]]
            else:
                return all_results[:n]
                
        except Exception as e:
            logging.error(f"Error in get_related_documentation_fixed: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import numpy as np
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            return dot_product / (norm_v1 * norm_v2)
        except Exception:
            return 0.0
    
    def test_multilingual_similarity_search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.1) -> Dict[str, List]:
        """Test multilingual similarity search across all content types"""
        results = {
            'similar_questions': [],
            'related_ddl': [],
            'related_docs': []
        }
        
        try:
            logging.info(f"🔍 Testing multilingual similarity search for: '{query}' (detected lang: {self._detect_language(query)})")
            
            results['similar_questions'] = self.get_similar_question_sql_fixed(
                question=query, n=top_k, similarity_threshold=similarity_threshold
            )
            
            results['related_ddl'] = self.get_related_ddl_fixed(
                question=query, n=top_k, similarity_threshold=similarity_threshold
            )
            
            results['related_docs'] = self.get_related_documentation_fixed(
                question=query, n=top_k, similarity_threshold=similarity_threshold
            )
            
            total_results = len(results['similar_questions']) + len(results['related_ddl']) + len(results['related_docs'])
            logging.info(f"  📊 Results: Q&A={len(results['similar_questions'])}, DDL={len(results['related_ddl'])}, Docs={len(results['related_docs'])}, Total={total_results}")
                        
        except Exception as e:
            logging.error(f"❌ Multilingual similarity search failed: {e}")
        
        return results

def connect_to_oracle():
    """Connect to Oracle database"""
    try:
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

def load_multilingual_training_data():
    """Load training data optimized for multilingual embeddings"""
    training_data = {
        'ddl_statements': [],
        'documentation': [],
        'sample_questions': [],
        'metadata': []
    }

    # Read DDL statements from SQL file
    ddl_file_path = r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Query-Generator-vanna\DDL_latest.sql"
    if os.path.exists(ddl_file_path):
        with open(ddl_file_path, 'r') as f:
            ddl_content = f.read().strip()
            ddl_statements = [stmt.strip() for stmt in ddl_content.split(';') if stmt.strip()]
            
            for i, stmt in enumerate(ddl_statements):
                enhanced_ddl = f"-- DDL Statement {i+1}\n{stmt}"
                training_data['ddl_statements'].append(enhanced_ddl)
                training_data['metadata'].append({
                    'type': 'ddl',
                    'index': i,
                    'length': len(stmt)
                })
            logging.info(f"✅ Loaded {len(training_data['ddl_statements'])} DDL statements")
    else:
        logging.error(f"❌ DDL file not found: {ddl_file_path}")

    # Load multilingual documentation
    doc_file_path = r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Query-Generator-vanna\Test_sample.txt"
    if os.path.exists(doc_file_path):
        try:
            with open(doc_file_path, 'r', encoding='utf-8') as f:
                documentation = f.read().strip().split('=' * 80)
                
                for i, doc in enumerate(documentation):
                    if doc.strip():
                        enhanced_doc = f"Database Documentation Section {i+1} (Multilingual):\n{doc.strip()}"
                        training_data['documentation'].append(enhanced_doc)
                        
                        arabic_chars = sum(1 for char in doc if '\u0600' <= char <= '\u06FF')
                        lang = "ar/en" if arabic_chars > 0 else "en"
                        
                        training_data['metadata'].append({
                            'type': 'documentation',
                            'section': i+1,
                            'length': len(doc),
                            'language': lang
                        })
                        
            logging.info(f"✅ Loaded {len(training_data['documentation'])} multilingual documentation sections")

        except Exception as e:
            logging.error(f"Failed to read multilingual documentation: {e}")
    
    return training_data

def train_multilingual_vanna_model(vn, training_data):
    """Train Vanna model with multilingual data"""
    logging.info("🚀 Starting multilingual Vanna model training...")
    
    total_added = 0
    total_failed = 0
    
    # Add DDL statements
    logging.info(f"📊 Adding {len(training_data['ddl_statements'])} DDL statements...")
    for i, ddl in enumerate(training_data['ddl_statements']):
        try:
            success = vn.add_to_vector_store_with_embedding(ddl, 'ddl', {'index': i})
            if success:
                total_added += 1
            else:
                total_failed += 1
                
            if (i + 1) % 5 == 0:
                logging.info(f"  Processed {i + 1}/{len(training_data['ddl_statements'])} DDL statements")
        except Exception as e:
            logging.warning(f"Failed to add DDL {i}: {e}")
            total_failed += 1

    # Add multilingual documentation
    logging.info(f"📚 Adding {len(training_data['documentation'])} multilingual documentation entries...")
    for i, doc in enumerate(training_data['documentation']):
        try:
            success = vn.add_to_vector_store_with_embedding(doc, 'documentation', {'section': i})
            if success:
                total_added += 1
            else:
                total_failed += 1
                
            if (i + 1) % 5 == 0:
                logging.info(f"  Processed {i + 1}/{len(training_data['documentation'])} documentation entries")
        except Exception as e:
            logging.warning(f"Failed to add documentation {i}: {e}")
            total_failed += 1

    return total_added, total_failed

def main():
    """Main function with persistent vector store"""
    print("=== Multilingual Vanna Vector Store (PERSISTENT) ===")
    print("1. Train with multilingual embeddings (preserves existing data)")
    print("2. Test current multilingual embeddings")
    print("3. RESET + Train + Test (DELETES all existing data)")
    print("4. Quick multilingual check")
    print("5. Check existing data status")
    
    choice = input("Enter your choice (1/2/3/4/5): ").strip()
    
    # Initialize Vanna WITHOUT forcing reset
    reset_requested = (choice == '3')  # Only reset if explicitly chosen
    
    print(f"🔧 Initializing multilingual Vanna (reset={reset_requested})...")
    try:
        vn = MyVanna(reset_data=reset_requested)
        print("✅ Vanna initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Vanna: {e}")
        return
    
    if choice == '5':
        print("📊 Checking existing data status...")
        vn._check_existing_data()
        return
    
    if choice in ['1', '3']:
        # Load and train (reset already handled in init if choice == '3')
        print("📚 Loading multilingual training data...")
        training_data = load_multilingual_training_data()
        
        # Connect to Oracle
        try:
            vn.connect_to_oracle(
                user='IAS202538',
                password='123',
                dsn="localhost:1521/xepdb1"
            )
            print("✅ Connected to Oracle database")
        except Exception as e:
            print(f"⚠️ Oracle connection failed: {e}")
        
        # Train the model
        added, failed = train_multilingual_vanna_model(vn, training_data)
        print(f"🎓 Training completed: {added} items added, {failed} failed")
    
    if choice in ['2', '4']:
        # Test embeddings
        if choice == '4':
            print("🔍 Quick multilingual check...")
            results = vn.test_multilingual_similarity_search("اظهر جميع الحسابات", top_k=2)
            total = sum(len(v) if isinstance(v, list) else 0 for v in results.values())
            print(f"✅ Found {total} results for Arabic query")
        else:
            print("🧪 Running comprehensive test...")
            test_queries = ["show all accounts", "اظهر جميع الحسابات", "customer information"]
            for query in test_queries:
                results = vn.test_multilingual_similarity_search(query, top_k=3)
                total = sum(len(v) if isinstance(v, list) else 0 for v in results.values())
                print(f"Query '{query}': {total} results")
    
    print(f"\n🎉 Process complete! Vector store persisted at: {vn.persist_dir}")
    print("💡 TIP: In your other applications, initialize with MyVanna(reset_data=False) to use existing data")

if __name__ == "__main__":
    main()