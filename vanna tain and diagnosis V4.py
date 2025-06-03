#!/usr/bin/env python3
"""
Enhanced Multilingual Vanna Vector Store Training and Diagnostic Script
Fixed version with proper top-k similarity retrieval
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
    def __init__(self, config=None, force_reset=False):
        # Clean slate approach - always start fresh to avoid embedding function conflicts
        if force_reset:
            logging.info("🗑️ Cleaning ChromaDB to avoid embedding function conflicts...")
            self._clean_chromadb_completely()

        # Set up the embedding function
        self.multilingual_embedding = MultilingualEmbeddingFunction()

        # Use a completely fresh directory and unique collection names
        import time
        unique_id = str(uuid.uuid4())[:8]
        timestamp = str(int(time.time()))
        
        # Create unique persist directory
        persist_dir = f'./vanna-multilingual-data'
        
        # Separate configs for each parent class
        chroma_config = {
            'persist_directory': persist_dir,
            'embedding_function': self.multilingual_embedding  # Pass the instance directly
        }

        ollama_config = {
            'model': 'mistral'
        }

        # Store config for reset operations
        self.chroma_config = chroma_config
        self.persist_dir = persist_dir
        
        # Initialize parent classes
        try:
            ChromaDB_VectorStore.__init__(self, config=chroma_config)
            Ollama.__init__(self, config=ollama_config)
            logging.info(f"✅ Initialized Vanna with multilingual embeddings at {persist_dir}")
        except Exception as e:
            logging.error(f"❌ Failed to initialize Vanna: {e}")
            # Try one more time with complete cleanup
            self._clean_chromadb_completely()
            ChromaDB_VectorStore.__init__(self, config=chroma_config)
            Ollama.__init__(self, config=ollama_config)
        
        # Verify setup
        self._verify_embedding_setup()

    def _clean_chromadb_completely(self):
        """Completely remove all ChromaDB data to start fresh"""
        try:
            import shutil
            import glob
            
            # Remove all vanna-data and vanna-multilingual directories
            directory_patterns = ['./vanna-data*', './vanna-multilingual*']
            file_patterns = ['./chroma.sqlite3*', './*.db', './*.sqlite*']
            
            # Clean directories
            for pattern in directory_patterns:
                for path in glob.glob(pattern):
                    if os.path.exists(path) and os.path.isdir(path):
                        try:
                            shutil.rmtree(path)
                            logging.info(f"✅ Removed directory {path}")
                        except PermissionError:
                            logging.warning(f"⚠️ Permission denied removing directory {path}")
                        except Exception as e:
                            logging.warning(f"⚠️ Error removing directory {path}: {e}")
            
            # Clean files
            for pattern in file_patterns:
                for path in glob.glob(pattern):
                    if os.path.exists(path) and os.path.isfile(path):
                        try:
                            os.remove(path)
                            logging.info(f"✅ Removed file {path}")
                        except PermissionError:
                            logging.warning(f"⚠️ Permission denied removing file {path}")
                        except Exception as e:
                            logging.warning(f"⚠️ Error removing file {path}: {e}")
                        
        except Exception as e:
            logging.warning(f"⚠️ Error cleaning ChromaDB: {e}")

    def reset_vector_store(self):
        """Delete all previous training data and reset the vector store"""
        try:
            logging.info("🗑️ Resetting vector store - deleting all previous training data...")
            
            # Delete the current persist directory
            persist_dir = self.persist_dir
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
                logging.info(f"✅ Deleted existing vector store directory: {persist_dir}")
            
            # Recreate the directory
            os.makedirs(persist_dir, exist_ok=True)
            
            # Reinitialize ChromaDB with fresh data using the same config
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
            
            # Check what methods are available
            available_methods = []
            for method_name in dir(self):
                if 'similar' in method_name.lower() or 'related' in method_name.lower():
                    if not method_name.startswith('_'):
                        available_methods.append(method_name)
            
            logging.info(f"Available similarity/retrieval methods: {available_methods}")
            
        except Exception as e:
            logging.warning(f"⚠️ Multilingual embedding setup issue: {e}")
    
    def add_to_vector_store_with_embedding(self, content: str, content_type: str, metadata: Dict = None):
        """
        Add content to vector store with multilingual embedding verification
        """
        try:
            metadata = metadata or {}
            metadata['content_type'] = content_type
            metadata['length'] = len(content)
            metadata['language'] = self._detect_language(content)
            
            # For multilingual content, we can preprocess it to make it more embedding-friendly
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
        """
        Preprocess multilingual content to make it more embedding-friendly
        """
        # Add English keywords/context for Arabic content to improve searchability
        arabic_chars = sum(1 for char in content if '\u0600' <= char <= '\u06FF')
        if arabic_chars > len(content) * 0.1:  # If significant Arabic content
            # Add some English context
            content = f"Multilingual content (Arabic/English): {content}"
        
        return content
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection for logging purposes"""
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        if arabic_chars > len(text) * 0.1:  # More than 10% Arabic characters
            return "ar/en"
        elif arabic_chars > 0:
            return "mixed"
        else:
            return "en"
    
    # ===== FIXED SIMILARITY SEARCH METHODS =====
    
    def get_similar_question_sql_fixed(self, question: str, n: int = 5, similarity_threshold: float = 0.1) -> List[Tuple[str, str]]:
        """
        Fixed version of get_similar_question_sql that properly limits results to top-n
        """
        try:
            # First get all similar results using parent method
            all_results = super().get_similar_question_sql(question)
            
            if not all_results or not isinstance(all_results, list):
                return []
            
            # If we have embeddings available, compute similarity scores
            if hasattr(self, 'multilingual_embedding'):
                query_embedding = self.multilingual_embedding([question])[0]
                scored_results = []
                
                for item in all_results:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        q_text, sql_text = item[0], item[1]
                        # Compute similarity with the question
                        item_embedding = self.multilingual_embedding([q_text])[0]
                        similarity = self._cosine_similarity(query_embedding, item_embedding)
                        
                        if similarity >= similarity_threshold:
                            scored_results.append((similarity, item))
                
                # Sort by similarity (descending) and take top-n
                scored_results.sort(key=lambda x: x[0], reverse=True)
                return [item for _, item in scored_results[:n]]
            else:
                # Fallback: just take first n results
                return all_results[:n]
                
        except Exception as e:
            logging.error(f"Error in get_similar_question_sql_fixed: {e}")
            return []
    
    def get_related_ddl_fixed(self, question: str, n: int = 5, similarity_threshold: float = 0.1) -> List[str]:
        """
        Fixed version of get_related_ddl that properly limits results to top-n
        """
        try:
            # Get all related DDL using parent method
            all_results = super().get_related_ddl(question)
            
            if not all_results or not isinstance(all_results, list):
                return []
            
            # If we have embeddings available, compute similarity scores
            if hasattr(self, 'multilingual_embedding'):
                query_embedding = self.multilingual_embedding([question])[0]
                scored_results = []
                
                for ddl_text in all_results:
                    if isinstance(ddl_text, str) and ddl_text.strip():
                        # Compute similarity
                        ddl_embedding = self.multilingual_embedding([ddl_text])[0]
                        similarity = self._cosine_similarity(query_embedding, ddl_embedding)
                        
                        if similarity >= similarity_threshold:
                            scored_results.append((similarity, ddl_text))
                
                # Sort by similarity and take top-n
                scored_results.sort(key=lambda x: x[0], reverse=True)
                return [ddl for _, ddl in scored_results[:n]]
            else:
                # Fallback: just take first n results
                return all_results[:n]
                
        except Exception as e:
            logging.error(f"Error in get_related_ddl_fixed: {e}")
            return []
    
    def get_related_documentation_fixed(self, question: str, n: int = 5, similarity_threshold: float = 0.1) -> List[str]:
        """
        Fixed version of get_related_documentation that properly limits results to top-n
        """
        try:
            # Get all related documentation using parent method
            all_results = super().get_related_documentation(question)
            
            if not all_results or not isinstance(all_results, list):
                return []
            
            # If we have embeddings available, compute similarity scores
            if hasattr(self, 'multilingual_embedding'):
                query_embedding = self.multilingual_embedding([question])[0]
                scored_results = []
                
                for doc_text in all_results:
                    if isinstance(doc_text, str) and doc_text.strip():
                        # Compute similarity
                        doc_embedding = self.multilingual_embedding([doc_text])[0]
                        similarity = self._cosine_similarity(query_embedding, doc_embedding)
                        
                        if similarity >= similarity_threshold:
                            scored_results.append((similarity, doc_text))
                
                # Sort by similarity and take top-n
                scored_results.sort(key=lambda x: x[0], reverse=True)
                return [doc for _, doc in scored_results[:n]]
            else:
                # Fallback: just take first n results
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
        except Exception as e:
            logging.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def test_multilingual_similarity_search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.1) -> Dict[str, List]:
        """
        Test multilingual similarity search across all content types - FIXED VERSION
        """
        results = {
            'similar_questions': [],
            'related_ddl': [],
            'related_docs': []
        }
        
        try:
            logging.info(f"🔍 Testing multilingual similarity search for: '{query}' (detected lang: {self._detect_language(query)})")
            logging.info(f"   Parameters: top_k={top_k}, similarity_threshold={similarity_threshold}")
            
            # Use FIXED methods that properly limit results
            results['similar_questions'] = self.get_similar_question_sql_fixed(
                question=query, n=top_k, similarity_threshold=similarity_threshold
            )
            
            results['related_ddl'] = self.get_related_ddl_fixed(
                question=query, n=top_k, similarity_threshold=similarity_threshold
            )
            
            results['related_docs'] = self.get_related_documentation_fixed(
                question=query, n=top_k, similarity_threshold=similarity_threshold
            )
            
            # Log the results with similarity scores if available
            total_results = len(results['similar_questions']) + len(results['related_ddl']) + len(results['related_docs'])
            logging.info(f"  📊 Results: Q&A={len(results['similar_questions'])}, DDL={len(results['related_ddl'])}, Docs={results['related_docs']}, Total={total_results}")
            
            # Show top results with similarity info
            if results['similar_questions']:
                logging.info("  🎯 Top similar Q&A found:")
                for i, item in enumerate(results['similar_questions'][:2]):
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        q, sql = item[0], item[1]
                        logging.info(f"    {i+1}. Q: {q[:80]}...")
                        logging.info(f"       SQL: {sql[:80]}...")
            
            if results['related_docs']:
                logging.info("  📄 Top related docs found:")
                for i, doc in enumerate(results['related_docs'][:2]):
                    logging.info(f"    {i+1}. Doc: {doc[:80]}...")
                        
        except Exception as e:
            logging.error(f"❌ Multilingual similarity search failed: {e}")
        
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

def load_multilingual_training_data():
    """Load training data optimized for multilingual embeddings"""
    training_data = {
        'ddl_statements': [],
        'documentation': [],
        'sample_questions': [],
        'metadata': []
    }

    # Load multilingual documentation
    doc_file_path = r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Query-Generator-vanna\Test_sample.txt"
    if os.path.exists(doc_file_path):
        try:
            with open(doc_file_path, 'r', encoding='utf-8') as f:
                documentation = f.read().strip().split('=' * 80)
                
                for i, doc in enumerate(documentation):
                    if doc.strip():
                        # Enhanced multilingual documentation
                        enhanced_doc = f"Database Documentation Section {i+1} (Multilingual):\n{doc.strip()}"
                        training_data['documentation'].append(enhanced_doc)
                        
                        # Detect if contains Arabic
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
    
    logging.info(f"✅ Multilingual training completed! Added: {total_added}, Failed: {total_failed}")
    return total_added, total_failed

def comprehensive_multilingual_test(vn):
    """Comprehensive test of multilingual embedding and similarity search with proper top-k limiting"""
    logging.info("=== COMPREHENSIVE MULTILINGUAL EMBEDDING TEST (WITH TOP-K LIMITING) ===")
    
    test_queries = [
        # English queries
        "show me all accounts",
        "customer information", 
        "account balance",
        "database schema",
        
        # Arabic queries
        "اظهر جميع الحسابات",
        "معلومات العملاء",
        "رصيد الحساب", 
        "هيكل قاعدة البيانات",
        
        # Mixed queries
        "show accounts - اظهر الحسابات",
        "customer data - بيانات العملاء"
    ]
    
    total_results = 0
    detailed_results = []
    
    # Test with different top_k values and similarity thresholds
    test_params = [
        {'top_k': 2, 'similarity_threshold': 0.1},
        {'top_k': 3, 'similarity_threshold': 0.2},
    ]
    
    for params in test_params:
        logging.info(f"\n--- Testing with params: {params} ---")
        
        for query in test_queries[:4]:  # Test first 4 queries with each param set
            logging.info(f"\n--- Testing query: '{query}' ---")
            results = vn.test_multilingual_similarity_search(
                query, 
                top_k=params['top_k'], 
                similarity_threshold=params['similarity_threshold']
            )
            
            query_total = (len(results['similar_questions']) + 
                          len(results['related_ddl']) + 
                          len(results['related_docs']))
            
            total_results += query_total
            detailed_results.append({
                'query': query,
                'params': params,
                'total_results': query_total,
                'qa_results': len(results['similar_questions']),
                'ddl_results': len(results['related_ddl']),
                'doc_results': len(results['related_docs'])
            })
            
            # Verify that we're not getting more results than requested
            max_expected = params['top_k']
            if results['similar_questions'] and len(results['similar_questions']) > max_expected:
                logging.warning(f"⚠️ Q&A results ({len(results['similar_questions'])}) exceed top_k ({max_expected})")
            if results['related_ddl'] and len(results['related_ddl']) > max_expected:
                logging.warning(f"⚠️ DDL results ({len(results['related_ddl'])}) exceed top_k ({max_expected})")
            if results['related_docs'] and len(results['related_docs']) > max_expected:
                logging.warning(f"⚠️ Doc results ({len(results['related_docs'])}) exceed top_k ({max_expected})")
    
    # Summary
    logging.info(f"\n=== MULTILINGUAL EMBEDDING TEST SUMMARY ===")
    logging.info(f"Total queries tested: {len([r for r in detailed_results])}")
    logging.info(f"Total results found: {total_results}")
    
    # Check if top-k limiting is working
    top_k_working = True
    for result in detailed_results:
        max_allowed = result['params']['top_k']
        if (result['qa_results'] > max_allowed or 
            result['ddl_results'] > max_allowed or 
            result['doc_results'] > max_allowed):
            top_k_working = False
            logging.warning(f"❌ Top-k limiting failed for query: {result['query']}")
    
    if top_k_working:
        logging.info("✅ Top-k limiting is working correctly!")
    else:
        logging.error("❌ Top-k limiting is NOT working properly")
    
    if total_results > 0:
        logging.info("✅ Multilingual embedding and similarity search are working!")
        return True
    else:
        logging.error("❌ No results found - multilingual embedding may not be working properly")
        return False

def main():
    """Main function with multilingual support and data reset"""
    print("=== Multilingual Vanna Vector Store with FIXED Top-K Retrieval ===")
    print("1. Reset data and train with multilingual embeddings")
    print("2. Test current multilingual embeddings (FIXED)")
    print("3. Reset + Train + Test (Full Process)")
    print("4. Quick multilingual embedding check (FIXED)")
    print("5. Debug embedding model")
    print("6. Test top-k limiting specifically")
    
    choice = input("Enter your choice (1/2/3/4/5/6): ").strip()
    
    # Initialize Multilingual Vanna with force reset
    print("🔧 Initializing multilingual Vanna...")
    try:
        vn = MyVanna(force_reset=True)  # Force clean start
        print("✅ Vanna initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Vanna: {e}")
        return
    
    if choice == '5':
        # Debug embedding model
        print("🔍 Testing multilingual embedding model...")
        test_texts = [
            "SELECT * FROM ACCOUNT",
            "اظهر جميع الحسابات",
            "Show customer information",
            "معلومات العملاء"
        ]
        
        embeddings = vn.multilingual_embedding(test_texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   Embedding dimension: {len(embeddings[0])}")
        print(f"   Sample embedding (first 5 values): {embeddings[0][:5]}")
        return
    
    if choice in ['1', '3']:
        # Reset vector store
        print("🗑️ Resetting vector store...")
        if not vn.reset_vector_store():
            print("❌ Failed to reset vector store")
            return
        
        # Load multilingual training data
        print("📚 Loading multilingual training data...")
        training_data = load_multilingual_training_data()
        print(f"Loaded: {len(training_data['ddl_statements'])} DDL, {len(training_data['documentation'])} docs, {len(training_data['sample_questions'])} Q&A")
        
        # Connect Vanna to Oracle
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
    
    if choice in ['2', '3', '4', '6']:
        # Test multilingual embeddings with FIXED top-k limiting
        if choice == '4':
            print("🔍 Quick multilingual embedding check (FIXED)...")
            quick_results = vn.test_multilingual_similarity_search("اظهر جميع الحسابات", top_k=2, similarity_threshold=0.1)
            total = sum(len(v) if isinstance(v, list) else 0 for v in quick_results.values())
            print(f"✅ Quick check: found {total} results for Arabic query (requested top_k=2)")
            
            # Test English query
            quick_results_en = vn.test_multilingual_similarity_search("show all accounts", top_k=2, similarity_threshold=0.1)
            total_en = sum(len(v) if isinstance(v, list) else 0 for v in quick_results_en.values())
            print(f"✅ Quick check: found {total_en} results for English query (requested top_k=2)")
            
        elif choice == '6':
            print("🎯 Testing top-k limiting specifically...")
            test_query = "show all accounts"
            for k in [1, 2, 3]:
                print(f"\n--- Testing with top_k={k} ---")
                results = vn.test_multilingual_similarity_search(test_query, top_k=k, similarity_threshold=0.0)
                for result_type, items in results.items():
                    if items:
                        print(f"  {result_type}: {len(items)} results (expected max: {k})")
                        if len(items) > k:
                            print(f"  ❌ PROBLEM: Got {len(items)} results but requested only {k}")
                        else:
                            print(f"  ✅ OK: Got {len(items)} results as expected")
        else:
            print("🧪 Running comprehensive multilingual test (FIXED)...")
            comprehensive_multilingual_test(vn)
    
    print("\n🎉 Multilingual Vanna process complete!")

if __name__ == "__main__":
    main()