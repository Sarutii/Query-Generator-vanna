import os
import sys
import time
import shutil
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore

class MyVanna(ChromaDB_VectorStore, Ollama):
    """
    A simplified Vanna implementation that works with latest Vanna versions
    """
    
    def __init__(self, config=None):
        # Initialize with minimal configuration
        config = config or {}
        config['model'] = 'mistral'
        config['persist_directory'] = './vanna-data'
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)
        # Removed: self.vector_store = ChromaDB_VectorStore(config=config)
    
    # Required abstract method implementations
    def system_message(self, message):
        return Ollama.system_message(self, message) if hasattr(Ollama, 'system_message') else None
    
    def user_message(self, message):
        return Ollama.user_message(self, message) if hasattr(Ollama, 'user_message') else None
    
    def assistant_message(self, message):
        return Ollama.assistant_message(self, message) if hasattr(Ollama, 'assistant_message') else None
    
    def submit_prompt(self, prompt):
        return Ollama.generate(self, prompt) if hasattr(Ollama, 'generate') else None
    
    def train(self, documentation=None, sql=None):
        """Train on documentation or SQL"""
        if documentation:
            print(f"Training on documentation ({len(documentation)} chars)")
            self.add_documentation(documentation)  # Use inherited method from ChromaDB_VectorStore
        if sql:
            print(f"Training on SQL ({len(sql)} chars)")
            self.add_sql(sql)  # Use inherited method from ChromaDB_VectorStore
    
    def generate_sql(self, question):
        """Generate SQL for the given question"""
        # Get relevant documentation context
        context = self.get_context(question)  # Use self instead of self.vector_store
        context_str = "\n\n".join(context)
        
        # Construct prompt
        prompt = f"{self.prompt_prefix}\n\nSchema information:\n{context_str}\n\nQuestion: {question}\n\nSQL:"
        
        # Generate SQL
        return self.submit_prompt(prompt)

# Removed global instantiation here to avoid issues during import
# vn = MyVanna(config={'model': 'mistral'})

def train_vanna_on_text_file(training_text_file):
    """
    Train Vanna on the generated training text file,
    deleting any previous training data first
    
    Args:
        training_text_file (str): Path to the training text file
    """
    print("Starting fresh Vanna training on text file...")
    start_time = time.time()
    
    # Delete previous training data if it exists
    persist_directory = './vanna-data'
    if os.path.exists(persist_directory):
        print(f"Deleting previous training data at {persist_directory}...")
        shutil.rmtree(persist_directory)
        print("Previous training data deleted successfully.")
    
    # Create a fresh directory
    os.makedirs(persist_directory, exist_ok=True)
    
    # Initialize Vanna
    try:
        vn = MyVanna()
        print("Successfully created MyVanna instance")
    except Exception as e:
        print(f"Failed to create MyVanna instance: {e}")
        return False
    
    # Read the training text file
    try:
        with open(training_text_file, 'r', encoding='utf-8') as f:
            training_text = f.read()
            
        print(f"Successfully read training text file: {training_text_file}")
        print(f"Training text size: {len(training_text)} characters")
        
        # Split the training text into individual table documentation chunks
        table_docs = training_text.split("-" * 60)
        table_docs = [doc.strip() for doc in table_docs if doc.strip()]
        
        print(f"Found {len(table_docs)} table documentation chunks")
        
        # Train Vanna on each table documentation chunk
        for i, doc in enumerate(table_docs):
            try:
                vn.train(documentation=doc)
                if i % 10 == 0 or i == len(table_docs) - 1:
                    print(f"Trained on {i+1}/{len(table_docs)} table documentation chunks")
            except Exception as e:
                print(f"Error training on chunk {i+1}: {e}")
                print(f"First 100 chars of problematic chunk: {doc[:100]}")
                continue
        
        print("Successfully trained Vanna on all table documentation")
        
    except Exception as e:
        print(f"Error reading or training on text file: {e}")
        return False
    
    end_time = time.time()
    print(f"Training completed successfully in {end_time - start_time:.2f} seconds!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to the generated training text file if no argument is provided
        training_text_file = r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Query-Generator-vanna\Full Data Discrebtion.txt"
        print(f"No file specified, using default: {training_text_file}")
    else:
        training_text_file = sys.argv[1]
    
    if train_vanna_on_text_file(training_text_file):
        print("Training complete! You can now run the main application.")
    else:
        print("Training failed. Please check the error messages above.")