"""
Script to train Vanna on Oracle schema files and save the trained model.
Run this script separately before starting the main application.
"""

from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
import json
import sys
import time

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        config = config or {}
        config['model'] = 'mistral'
        config['persist_directory'] = './vanna-data'
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)
        self.prompt_prefix = ""

    def set_prompt_prefix(self, prompt:str):
        self.prompt_prefix = prompt

    def get_prompt_prefix(self)-> str:
        return self.prompt_prefix
    
    def generate_sql(self, question: str) -> str:
        full_prompt = f"{self.prompt_prefix}\nQuestion: {question}"
        return super().generate_sql(full_prompt)

def train_vanna_on_schema_files(sql_schema_file):
                                # , json_schema_file):
    """
    Train Vanna on SQL schema file and JSON schema file
    
    Args:
        sql_schema_file (str): Path to SQL schema file
        json_schema_file (str): Path to JSON schema file with relationships
    """
    print("Starting Vanna training on schema files...")
    start_time = time.time()
    
    # Initialize Vanna
    vn = MyVanna()
    
    # Step 1: Read the SQL schema file
    try:
        with open(sql_schema_file, 'r') as f:
            sql_schema = f.read()
            
        print(f"Successfully read SQL schema file: {sql_schema_file}")
        print(f"SQL schema size: {len(sql_schema)} characters")
        
        # Train Vanna on SQL schema
        vn.train(sql=sql_schema)
        print("Successfully trained Vanna on SQL schema")
        
    except Exception as e:
        print(f"Error reading or training on SQL schema file: {e}")
        return False
    
    # Step 2: Read and process the JSON schema file with relationships
    # try:
    #     with open(json_schema_file, 'r') as f:
    #         json_schema = json.load(f)
            
    #     print(f"Successfully read JSON schema file: {json_schema_file}")
        
    #     # Extract and format table documentation from JSON
    #     table_docs = []
    #     table_names = []
        
    #     # Process tables and their columns
    #     if 'tables' in json_schema:
    #         for table in json_schema['tables']:
    #             table_name = table.get('table_name', '')
    #             table_names.append(table_name)
    #             table_comment = table.get('comment', 'No description available')
                
    #             doc = f"Table: {table_name}\nDescription: {table_comment}\n"
                
    #             # Add columns information
    #             if 'columns' in table and table['columns']:
    #                 doc += "Columns:\n"
    #                 for column in table['columns']:
    #                     col_name = column.get('column_name', '')
    #                     col_type = column.get('data_type', '')
    #                     col_comment = column.get('comment', 'No description')
    #                     nullable = "NOT NULL" if not column.get('nullable', True) else "NULL"
                        
    #                     doc += f"- {col_name} ({col_type}, {nullable}): {col_comment}\n"
                
    #             table_docs.append(doc)
        
        # # Process relationships if available
        # if 'relationships' in json_schema:
        #     relationships_doc = "Table Relationships:\n"
        #     for rel in json_schema['relationships']:
        #         from_table = rel.get('from_table', '')
        #         from_column = rel.get('from_column', '')
        #         to_table = rel.get('to_table', '')
        #         to_column = rel.get('to_column', '')
        #         rel_type = rel.get('type', 'FK')
                
        #         relationships_doc += f"- {from_table}.{from_column} -> {to_table}.{to_column} ({rel_type})\n"
            
        #     table_docs.append(relationships_doc)
        
        # # Train Vanna on each table documentation
        # for doc in table_docs:
        #     vn.train(documentation=doc)
        #     print(f"Trained on table documentation (first 100 chars):\n{doc[:100]}...")
        
        # print("Successfully trained Vanna on JSON schema relationships")
        
        # # Save the list of table names for the main application to use
        # with open('schema_tables.json', 'w') as f:
        #     json.dump({"tables": table_names}, f)
        # print(f"Saved table names to schema_tables.json")
        
    # except Exception as e:
    #     print(f"Error reading or training on JSON schema file: {e}")
    #     return False
    
    end_time = time.time()
    print(f"Schema training completed successfully in {end_time - start_time:.2f} seconds!")
    return True

if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print("Usage: python train_vanna.py <path_to_schema.sql> <path_to_schema_relationships.json>")
    #     sys.exit(1)
    
    sql_schema_file = r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Table_Column_Comment.sql"
    # json_schema_file = r"C:\Users\ahmed\Desktop\Projects\Query Generator\oracle_schema_20250415_054319.json"
    
    if train_vanna_on_schema_files(sql_schema_file):
                                #    , json_schema_file):
        print("Training complete! You can now run the main application.")
    else:
        print("Training failed. Please check the error messages above.")