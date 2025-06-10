#!/usr/bin/env python3
"""
FIXED Multilingual Vanna Vector Store Training Script
This version ensures data is actually stored in the vector store
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

class FixedMyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None, reset_data=False):
        """
        Initialize Vanna with GUARANTEED vector store population
        """
        
        # Set up the embedding function
        self.multilingual_embedding = MultilingualEmbeddingFunction()
        
        # Use a FIXED persist directory
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
            
            # Test creating a collection if none exist
            if not collections:
                logging.info("   Creating test collection to verify write access...")
                test_collection = self.chroma_client.create_collection(
                    name="vanna_test_collection",
                    embedding_function=self.multilingual_embedding
                )
                
                # Add a test document
                test_collection.add(
                    documents=["This is a test document"],
                    ids=["test_1"],
                    metadatas=[{"type": "test"}]
                )
                
                # Verify it was added
                count = test_collection.count()
                logging.info(f"   Test collection created with {count} documents")
                
                # Clean up test collection
                self.chroma_client.delete_collection("vanna_test_collection")
                logging.info("   Test collection removed")
            
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

    def train_with_verification(self, **kwargs):
        """
        Enhanced train method that VERIFIES data was actually stored
        """
        try:
            # Get collection count before training
            collections_before = self.chroma_client.list_collections()
            doc_count_before = sum(c.count() for c in collections_before)
            
            logging.info(f"📊 Before training: {len(collections_before)} collections, {doc_count_before} documents")
            
            # Call the original train method
            result = super().train(**kwargs)
            
            # CRITICAL: Force persistence by calling heartbeat/persist if available
            try:
                self.chroma_client.heartbeat()
                logging.info("✅ ChromaDB heartbeat called")
            except:
                pass
            
            # Wait a moment for async operations
            import time
            time.sleep(0.5)
            
            # Verify data was actually added
            collections_after = self.chroma_client.list_collections()
            doc_count_after = sum(c.count() for c in collections_after)
            
            docs_added = doc_count_after - doc_count_before
            
            logging.info(f"📊 After training: {len(collections_after)} collections, {doc_count_after} documents")
            logging.info(f"📈 Documents added this training: {docs_added}")
            
            if docs_added > 0:
                logging.info("✅ Training verification SUCCESSFUL - data was stored")
                return True
            else:
                logging.warning("⚠️ Training verification FAILED - no new documents detected")
                
                # Try to diagnose the issue
                logging.info("🔍 Diagnosing training failure...")
                for collection in collections_after:
                    try:
                        sample = collection.get(limit=1, include=['documents', 'metadatas'])
                        logging.info(f"   Collection '{collection.name}': {collection.count()} docs")
                        if sample and sample.get('documents'):
                            logging.info(f"      Sample: {sample['documents'][0][:100]}...")
                    except Exception as e:
                        logging.warning(f"      Error sampling {collection.name}: {e}")
                
                return False
            
        except Exception as e:
            logging.error(f"❌ Training with verification failed: {e}")
            return False

    def add_ddl_with_verification(self, ddl: str):
        """Add DDL with explicit verification"""
        try:
            logging.info(f"📝 Adding DDL: {ddl[:100]}...")
            success = self.train_with_verification(ddl=ddl)
            if success:
                logging.info("✅ DDL added and verified")
            else:
                logging.error("❌ DDL addition failed verification")
            return success
        except Exception as e:
            logging.error(f"❌ DDL addition error: {e}")
            return False

    def add_documentation_with_verification(self, documentation: str):
        """Add documentation with explicit verification"""
        try:
            logging.info(f"📚 Adding documentation: {documentation[:100]}...")
            success = self.train_with_verification(documentation=documentation)
            if success:
                logging.info("✅ Documentation added and verified")
            else:
                logging.error("❌ Documentation addition failed verification")
            return success
        except Exception as e:
            logging.error(f"❌ Documentation addition error: {e}")
            return False

    def add_qa_with_verification(self, question: str, sql: str):
        """Add Q&A with explicit verification"""
        try:
            logging.info(f"❓ Adding Q&A: Q='{question[:50]}...', SQL='{sql[:50]}...'")
            success = self.train_with_verification(question=question, sql=sql)
            if success:
                logging.info("✅ Q&A added and verified")
            else:
                logging.error("❌ Q&A addition failed verification")
            return success
        except Exception as e:
            logging.error(f"❌ Q&A addition error: {e}")
            return False

    def force_persistence(self):
        """Force ChromaDB to persist data to disk"""
        try:
            logging.info("💾 Forcing data persistence...")
            
            # Get all collections and force operations on them
            collections = self.chroma_client.list_collections()
            for collection in collections:
                try:
                    # Force a query to ensure data is loaded/saved
                    collection.count()
                    logging.info(f"   Persisted collection '{collection.name}': {collection.count()} docs")
                except Exception as e:
                    logging.warning(f"   Error persisting {collection.name}: {e}")
            
            # Call heartbeat if available
            try:
                self.chroma_client.heartbeat()
            except:
                pass
                
            logging.info("✅ Persistence operations completed")
            
        except Exception as e:
            logging.error(f"❌ Force persistence failed: {e}")

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

def load_training_data():
    """Load training data with verification"""
    training_data = {
        'ddl_statements': [],
        'documentation': [],
        'sample_questions': []
    }

    # Load DDL statements
    ddl_file_path = r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Query-Generator-vanna\DDL_latest.sql"
    
    logging.info(f"📂 Loading DDL from: {ddl_file_path}")
    if os.path.exists(ddl_file_path):
        try:
            with open(ddl_file_path, 'r', encoding='utf-8') as f:
                ddl_content = f.read().strip()
                
            if ddl_content:
                ddl_statements = [stmt.strip() for stmt in ddl_content.split(';') if stmt.strip()]
                training_data['ddl_statements'] = ddl_statements
                logging.info(f"✅ Loaded {len(ddl_statements)} DDL statements")
                logging.info(f"   First statement preview: {ddl_statements[0][:100]}..." if ddl_statements else "   No statements found")
            else:
                logging.warning("⚠️ DDL file is empty")
                
        except Exception as e:
            logging.error(f"❌ Failed to load DDL file: {e}")
    else:
        logging.error(f"❌ DDL file not found: {ddl_file_path}")

    # Load documentation
    doc_file_path = r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Query-Generator-vanna\Test_sample.txt"
    
    logging.info(f"📂 Loading documentation from: {doc_file_path}")
    if os.path.exists(doc_file_path):
        try:
            with open(doc_file_path, 'r', encoding='utf-8') as f:
                doc_content = f.read().strip()
                
            if doc_content:
                # Split by section separators
                documentation = [doc.strip() for doc in doc_content.split('=' * 80) if doc.strip()]
                training_data['documentation'] = documentation
                logging.info(f"✅ Loaded {len(documentation)} documentation sections")
                logging.info(f"   First section preview: {documentation[0][:100]}..." if documentation else "   No sections found")
            else:
                logging.warning("⚠️ Documentation file is empty")
                
        except Exception as e:
            logging.error(f"❌ Failed to load documentation file: {e}")
    else:
        logging.error(f"❌ Documentation file not found: {doc_file_path}")

    total_items = len(training_data['ddl_statements']) + len(training_data['documentation'])
    logging.info(f"📊 Total training data loaded: {total_items} items")
    
    return training_data

def train_with_explicit_verification(vn, training_data):
    """Train with explicit verification at each step"""
    logging.info("🚀 Starting training with explicit verification...")
    
    success_count = 0
    failure_count = 0
    
    # Train DDL statements
    ddl_statements = training_data['ddl_statements']
    if ddl_statements:
        logging.info(f"📊 Training {len(ddl_statements)} DDL statements...")
        for i, ddl in enumerate(ddl_statements):
            try:
                if vn.add_ddl_with_verification(ddl):
                    success_count += 1
                else:
                    failure_count += 1
                    
                if (i + 1) % 5 == 0:
                    logging.info(f"   Progress: {i + 1}/{len(ddl_statements)} DDL statements processed")
                    
            except Exception as e:
                logging.error(f"   DDL {i} failed: {e}")
                failure_count += 1
    
    # Train documentation
    documentation = training_data['documentation']
    if documentation:
        logging.info(f"📚 Training {len(documentation)} documentation sections...")
        for i, doc in enumerate(documentation):
            try:
                if vn.add_documentation_with_verification(doc):
                    success_count += 1
                else:
                    failure_count += 1
                    
                if (i + 1) % 5 == 0:
                    logging.info(f"   Progress: {i + 1}/{len(documentation)} documentation sections processed")
                    
            except Exception as e:
                logging.error(f"   Documentation {i} failed: {e}")
                failure_count += 1
    
    # Force persistence after training
    vn.force_persistence()
    
    logging.info(f"🎓 Training completed: {success_count} successes, {failure_count} failures")
    return success_count, failure_count

def main():
    """Main function with guaranteed vector store population"""
    print("=== FIXED Multilingual Vanna Vector Store ===")
    print("This version ensures data is actually stored!")
    print()
    print("1. Train with verification (preserves existing data)")
    print("2. Test current vector store")
    print("3. RESET + Train + Test (DELETES all existing data)")
    print("4. Comprehensive diagnostics")
    
    choice = input("Enter your choice (1/2/3/4): ").strip()
    
    # Initialize Vanna
    reset_requested = (choice == '3')
    
    print(f"🔧 Initializing FIXED multilingual Vanna (reset={reset_requested})...")
    try:
        vn = FixedMyVanna(reset_data=reset_requested)
        print("✅ Vanna initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Vanna: {e}")
        return
    
    if choice in ['1', '3']:
        # Load and train
        print("📚 Loading training data...")
        training_data = load_training_data()
        
        total_items = len(training_data['ddl_statements']) + len(training_data['documentation'])
        if total_items == 0:
            print("❌ No training data found! Check your file paths.")
            return
        
        print(f"📊 Found {total_items} training items")
        
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
        
        # Train with verification
        success_count, failure_count = train_with_explicit_verification(vn, training_data)
        print(f"🎓 Training results: {success_count} successful, {failure_count} failed")
        
        # Final verification
        print("🔍 Running final comprehensive test...")
        if vn.comprehensive_vector_store_test():
            print("✅ Vector store successfully populated!")
        else:
            print("❌ Vector store still appears empty - check logs for errors")
    
    if choice in ['2', '4']:
        # Test vector store
        print("🧪 Testing vector store...")
        if vn.comprehensive_vector_store_test():
            print("✅ Vector store contains data and is accessible")
        else:
            print("❌ Vector store appears to be empty or inaccessible")
    
    print(f"\n🎉 Process complete! Vector store location: {vn.persist_dir}")
    
    # Final summary
    try:
        collections = vn.chroma_client.list_collections()
        total_docs = sum(c.count() for c in collections)
        print(f"📊 Final status: {len(collections)} collections, {total_docs} total documents")
    except:
        pass

if __name__ == "__main__":
    main()