#!/usr/bin/env python3
"""
Enhanced Vanna Vector Store with Description-Focused Similarity Search
This script focuses on table/column descriptions for better similarity matching.
"""

import os
import json
import logging
import re
import oracledb
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MyVannaWithDescriptions(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        config = config or {}
        config['model'] = 'mistral'
        config['persist_directory'] = './vanna-data-descriptions'
        
        # Enhanced ChromaDB configuration for better embeddings
        config['embedding_function'] = None
        config['collection_name'] = 'vanna_descriptions'
        
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)
        
        # Store parsed descriptions separately
        self.parsed_descriptions = {
            'table_descriptions': [],
            'column_descriptions': [],
            'general_descriptions': []
        }
        
        self._verify_embedding_setup()
    
    def _verify_embedding_setup(self):
        """Verify that embeddings are properly configured"""
        try:
            test_text = "SELECT * FROM ACCOUNT"
            available_methods = []
            for method_name in dir(self):
                if 'similar' in method_name.lower() or 'related' in method_name.lower():
                    if not method_name.startswith('_'):
                        available_methods.append(method_name)
            
            logging.info(f"Available similarity/retrieval methods: {available_methods}")
            logging.info("✅ Embedding configuration verified")
        except Exception as e:
            logging.warning(f"⚠️ Embedding setup issue: {e}")
    
    def parse_documentation_descriptions(self, doc_content: str) -> Dict[str, List[Dict]]:
        """
        Parse documentation to extract table and column descriptions
        
        Args:
            doc_content: Raw documentation content
            
        Returns:
            Dictionary with parsed descriptions
        """
        descriptions = {
            'table_descriptions': [],
            'column_descriptions': [],
            'general_descriptions': []
        }
        
        try:
            # Split content into sections
            sections = doc_content.split('------------------------------------------------------------')
            
            for section_idx, section in enumerate(sections):
                if not section.strip():
                    continue
                
                lines = section.strip().split('\n')
                current_table = None
                
                for line_idx, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Detect table names (usually uppercase or have specific patterns)
                    if self._is_table_line(line):
                        current_table = self._extract_table_name(line)
                        # Store table description
                        table_desc = {
                            'table_name': current_table,
                            'description': line,
                            'section_idx': section_idx,
                            'line_idx': line_idx,
                            'type': 'table'
                        }
                        descriptions['table_descriptions'].append(table_desc)
                        
                    # Detect column descriptions
                    elif self._is_column_description(line):
                        col_info = self._parse_column_description(line)
                        if col_info:
                            col_desc = {
                                'table_name': current_table,
                                'column_name': col_info.get('column'),
                                'description': line,
                                'parsed_info': col_info,
                                'section_idx': section_idx,
                                'line_idx': line_idx,
                                'type': 'column'
                            }
                            descriptions['column_descriptions'].append(col_desc)
                    
                    # General descriptions (non-table/column specific)
                    else:
                        if len(line) > 20:  # Only meaningful descriptions
                            general_desc = {
                                'description': line,
                                'table_context': current_table,
                                'section_idx': section_idx,
                                'line_idx': line_idx,
                                'type': 'general'
                            }
                            descriptions['general_descriptions'].append(general_desc)
            
            logging.info(f"Parsed descriptions:")
            logging.info(f"  - Table descriptions: {len(descriptions['table_descriptions'])}")
            logging.info(f"  - Column descriptions: {len(descriptions['column_descriptions'])}")
            logging.info(f"  - General descriptions: {len(descriptions['general_descriptions'])}")
            
        except Exception as e:
            logging.error(f"Error parsing documentation: {e}")
        
        return descriptions
    
    def _is_table_line(self, line: str) -> bool:
        """Detect if a line contains a table definition/description"""
        # Common patterns for table descriptions
        table_indicators = [
            'TABLE', 'table', 'جدول', 'جداول',
            'CREATE TABLE', 'create table',
            # Add more patterns based on your documentation format
        ]
        
        # Check if line has table indicators or is all uppercase (common for table names)
        line_upper = line.upper()
        return (any(indicator.upper() in line_upper for indicator in table_indicators) or
                (len(line.split()) <= 3 and line.isupper() and len(line) > 2))
    
    def _extract_table_name(self, line: str) -> str:
        """Extract table name from a table description line"""
        # Remove common prefixes/suffixes
        cleaned = re.sub(r'(CREATE TABLE|TABLE|table|جدول)', '', line, flags=re.IGNORECASE)
        cleaned = cleaned.strip('():;')
        
        # Extract the first word that looks like a table name
        words = cleaned.split()
        for word in words:
            if word.isalnum() or '_' in word:
                return word.strip('(),;')
        
        return cleaned.strip() if cleaned else 'UNKNOWN'
    
    def _is_column_description(self, line: str) -> bool:
        """Detect if a line contains a column description"""
        # Common patterns for column descriptions
        column_indicators = [
            ':', '-', 'عمود', 'حقل', 'field', 'column',
            'VARCHAR', 'NUMBER', 'DATE', 'INTEGER',
            # Add more data type indicators
        ]
        
        return any(indicator in line for indicator in column_indicators)
    
    def _parse_column_description(self, line: str) -> Optional[Dict]:
        """Parse column description to extract column name and details"""
        try:
            # Try to extract column name (usually before : or -)
            if ':' in line:
                parts = line.split(':', 1)
                column_name = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ''
            elif '-' in line:
                parts = line.split('-', 1)
                column_name = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ''
            else:
                # Try to extract from other patterns
                words = line.split()
                column_name = words[0] if words else ''
                description = ' '.join(words[1:]) if len(words) > 1 else ''
            
            return {
                'column': column_name,
                'description': description,
                'full_line': line
            }
        except:
            return None
    
    def add_description_to_vector_store(self, desc_data: Dict, content_type: str):
        """
        Add parsed description to vector store with enhanced metadata
        
        Args:
            desc_data: Parsed description data
            content_type: Type of description ('table_desc', 'column_desc', 'general_desc')
        """
        try:
            # Create enhanced content for better embeddings
            if content_type == 'table_desc':
                enhanced_content = f"Table: {desc_data['table_name']} - {desc_data['description']}"
                metadata = {
                    'type': 'table_description',
                    'table_name': desc_data['table_name'],
                    'original_description': desc_data['description']
                }
                
            elif content_type == 'column_desc':
                table_context = f" in table {desc_data['table_name']}" if desc_data['table_name'] else ""
                enhanced_content = f"Column: {desc_data['column_name']}{table_context} - {desc_data['description']}"
                metadata = {
                    'type': 'column_description',
                    'table_name': desc_data['table_name'],
                    'column_name': desc_data['column_name'],
                    'original_description': desc_data['description']
                }
                
            else:  # general_desc
                table_context = f" (Context: {desc_data['table_context']})" if desc_data['table_context'] else ""
                enhanced_content = f"Description{table_context}: {desc_data['description']}"
                metadata = {
                    'type': 'general_description',
                    'table_context': desc_data.get('table_context'),
                    'original_description': desc_data['description']
                }
            
            # Add to vector store as documentation
            self.train(documentation=enhanced_content)
            logging.debug(f"✅ Added {content_type}: {enhanced_content[:60]}...")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to add {content_type}: {e}")
            return False
    
    def enhanced_similarity_search(self, query: str, top_k: int = 5) -> Dict[str, List]:
        """
        Enhanced similarity search focused on descriptions
        
        Args:
            query: Search query (Arabic or English)
            top_k: Number of results to return
            
        Returns:
            Dictionary with categorized results
        """
        results = {
            'table_descriptions': [],
            'column_descriptions': [],
            'general_descriptions': [],
            'related_ddl': [],
            'similar_questions': []
        }
        
        try:
            # Enhance query for better matching
            enhanced_queries = self._generate_enhanced_queries(query)
            
            all_related_docs = []
            
            # Search with multiple query variations
            for enhanced_query in enhanced_queries:
                try:
                    related_docs = self.get_related_documentation(question=enhanced_query, n=top_k*2)
                    if isinstance(related_docs, list):
                        all_related_docs.extend(related_docs)
                except Exception as e:
                    logging.debug(f"Search failed for query '{enhanced_query}': {e}")
            
            # Remove duplicates and categorize results
            seen_docs = set()
            for doc in all_related_docs:
                if doc not in seen_docs:
                    seen_docs.add(doc)
                    categorized = self._categorize_search_result(doc)
                    if categorized:
                        result_type, result_data = categorized
                        if result_type in results:
                            results[result_type].append(result_data)
            
            # Limit results per category
            for category in results:
                results[category] = results[category][:top_k]
            
            # Also search for DDL and Q&A
            try:
                related_ddl = self.get_related_ddl(question=query, n=top_k)
                if isinstance(related_ddl, list):
                    results['related_ddl'] = related_ddl
            except:
                pass
            
            try:
                similar_qa = self.get_similar_question_sql(question=query, n=top_k)
                if isinstance(similar_qa, list):
                    results['similar_questions'] = similar_qa
            except:
                pass
            
            # Log results
            total_desc_results = (len(results['table_descriptions']) + 
                                len(results['column_descriptions']) + 
                                len(results['general_descriptions']))
            
            logging.info(f"Enhanced similarity search for '{query}':")
            logging.info(f"  - Table descriptions: {len(results['table_descriptions'])}")
            logging.info(f"  - Column descriptions: {len(results['column_descriptions'])}")
            logging.info(f"  - General descriptions: {len(results['general_descriptions'])}")
            logging.info(f"  - Related DDL: {len(results['related_ddl'])}")
            logging.info(f"  - Similar Q&A: {len(results['similar_questions'])}")
            
        except Exception as e:
            logging.error(f"❌ Enhanced similarity search failed: {e}")
        
        return results
    
    def _generate_enhanced_queries(self, query: str) -> List[str]:
        """Generate multiple query variations for better matching"""
        queries = [query]
        
        # Add variations
        queries.append(f"table {query}")
        queries.append(f"column {query}")
        queries.append(f"جدول {query}")
        queries.append(f"عمود {query}")
        
        # Add keyword variations
        common_terms = {
            'show': ['display', 'get', 'اظهر', 'عرض'],
            'all': ['*', 'جميع', 'كل'],
            'account': ['accounts', 'حساب', 'حسابات'],
            'customer': ['customers', 'عميل', 'عملاء'],
            'order': ['orders', 'طلب', 'طلبات'],
            'maintenance': ['صيانة', 'maintenance'],
            'branch': ['فرع', 'فروع', 'branches']
        }
        
        original_query = query.lower()
        for eng_term, variations in common_terms.items():
            if eng_term in original_query:
                for variation in variations:
                    new_query = original_query.replace(eng_term, variation)
                    if new_query != original_query:
                        queries.append(new_query)
        
        return list(set(queries))  # Remove duplicates
    
    def _categorize_search_result(self, doc: str) -> Optional[Tuple[str, Dict]]:
        """Categorize a search result based on its content"""
        try:
            doc_lower = doc.lower()
            
            # Check if it's a table description
            if any(indicator in doc_lower for indicator in ['table:', 'جدول:', 'table ', 'جدول ']):
                return ('table_descriptions', {
                    'content': doc,
                    'relevance_score': self._calculate_relevance_score(doc, 'table')
                })
            
            # Check if it's a column description
            elif any(indicator in doc_lower for indicator in ['column:', 'عمود:', 'field:', 'حقل:']):
                return ('column_descriptions', {
                    'content': doc,
                    'relevance_score': self._calculate_relevance_score(doc, 'column')
                })
            
            # General description
            else:
                return ('general_descriptions', {
                    'content': doc,
                    'relevance_score': self._calculate_relevance_score(doc, 'general')
                })
                
        except Exception as e:
            logging.debug(f"Error categorizing result: {e}")
            return None
    
    def _calculate_relevance_score(self, doc: str, category: str) -> float:
        """Calculate a simple relevance score for ranking"""
        # Simple scoring based on content length and category
        base_score = min(len(doc) / 100, 1.0)  # Normalize by length
        
        # Bonus for category-specific terms
        category_bonus = {
            'table': 0.1 if 'table' in doc.lower() or 'جدول' in doc.lower() else 0,
            'column': 0.1 if 'column' in doc.lower() or 'عمود' in doc.lower() else 0,
            'general': 0.05
        }
        
        return base_score + category_bonus.get(category, 0)

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

def train_with_parsed_descriptions(vn, doc_file_path):
    """Train the model with parsed descriptions from documentation"""
    if not os.path.exists(doc_file_path):
        logging.error(f"Documentation file not found: {doc_file_path}")
        return 0, 0
    
    try:
        # Read documentation with proper encoding
        try:
            with open(doc_file_path, 'r', encoding='utf-8') as f:
                doc_content = f.read()
        except UnicodeDecodeError:
            with open(doc_file_path, 'r', encoding='latin1') as f:
                doc_content = f.read()
        
        # Parse descriptions
        logging.info("Parsing documentation for descriptions...")
        parsed_descriptions = vn.parse_documentation_descriptions(doc_content)
        
        # Store parsed descriptions
        vn.parsed_descriptions = parsed_descriptions
        
        total_added = 0
        total_failed = 0
        
        # Add table descriptions
        logging.info(f"Adding {len(parsed_descriptions['table_descriptions'])} table descriptions...")
        for desc in parsed_descriptions['table_descriptions']:
            success = vn.add_description_to_vector_store(desc, 'table_desc')
            if success:
                total_added += 1
            else:
                total_failed += 1
        
        # Add column descriptions
        logging.info(f"Adding {len(parsed_descriptions['column_descriptions'])} column descriptions...")
        for desc in parsed_descriptions['column_descriptions']:
            success = vn.add_description_to_vector_store(desc, 'column_desc')
            if success:
                total_added += 1
            else:
                total_failed += 1
        
        # Add general descriptions
        logging.info(f"Adding {len(parsed_descriptions['general_descriptions'])} general descriptions...")
        for desc in parsed_descriptions['general_descriptions']:
            success = vn.add_description_to_vector_store(desc, 'general_desc')
            if success:
                total_added += 1
            else:
                total_failed += 1
        
        logging.info(f"Description training completed: {total_added} added, {total_failed} failed")
        return total_added, total_failed
        
    except Exception as e:
        logging.error(f"Error training with descriptions: {e}")
        return 0, 0

def test_description_focused_search(vn):
    """Test the enhanced description-focused similarity search"""
    logging.info("=== DESCRIPTION-FOCUSED SIMILARITY SEARCH TEST ===")
    
    test_queries = [
        "account details",
        "customer information", 
        "maintenance orders",
        "branch data",
        "asset information",
        "حسابات العملاء",
        "طلبات الصيانة",
        "معلومات الفروع",
        "بيانات الأصول",
        "show me accounts",
        "اظهر تنفيذات اوامر الصيانة"
    ]
    
    total_results = 0
    
    for query in test_queries:
        logging.info(f"\n--- Testing description search for: '{query}' ---")
        results = vn.enhanced_similarity_search(query, top_k=3)
        
        query_total = (len(results['table_descriptions']) + 
                      len(results['column_descriptions']) + 
                      len(results['general_descriptions']))
        
        total_results += query_total
        
        # Show results by category
        if results['table_descriptions']:
            print("  📊 Table Descriptions:")
            for i, result in enumerate(results['table_descriptions'][:2]):
                content = result['content'] if isinstance(result, dict) else result
                print(f"    {i+1}. {content[:80]}...")
        
        if results['column_descriptions']:
            print("  📋 Column Descriptions:")
            for i, result in enumerate(results['column_descriptions'][:2]):
                content = result['content'] if isinstance(result, dict) else result
                print(f"    {i+1}. {content[:80]}...")
        
        if results['general_descriptions']:
            print("  📝 General Descriptions:")
            for i, result in enumerate(results['general_descriptions'][:1]):
                content = result['content'] if isinstance(result, dict) else result
                print(f"    {i+1}. {content[:80]}...")
    
    logging.info(f"\n=== DESCRIPTION SEARCH TEST SUMMARY ===")
    logging.info(f"Total queries tested: {len(test_queries)}")
    logging.info(f"Total description results found: {total_results}")
    logging.info(f"Average description results per query: {total_results/len(test_queries):.1f}")
    
    return total_results > 0

def main():
    """Main function with description-focused options"""
    print("=== Enhanced Vanna with Description-Focused Similarity Search ===")
    print("1. Test description-focused similarity search")
    print("2. Train with parsed descriptions")
    print("3. Full process (parse + train + test descriptions)")
    print("4. Debug parsed descriptions")
    
    choice = input("Enter your choice (1/2/3/4): ").strip()
    
    # Initialize enhanced Vanna
    vn = MyVannaWithDescriptions()
    
    doc_file_path = r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Query-Generator-vanna\Full Data Discrebtion.txt"
    
    if choice == '1':
        print("\n--- Testing Description-Focused Search ---")
        test_description_focused_search(vn)
        
    elif choice == '2':
        print("\n--- Training with Parsed Descriptions ---")
        added, failed = train_with_parsed_descriptions(vn, doc_file_path)
        print(f"Training completed: {added} descriptions added, {failed} failed")
        
    elif choice == '3':
        print("\n--- Full Process: Parse, Train, and Test ---")
        
        # Connect to Oracle for DDL
        connection = connect_to_oracle()
        if connection:
            vn.connect_to_oracle(
                user='IAS202538',
                password='123',
                dsn="localhost:1521/xepdb1"
            )
            connection.close()
        
        # Train with DDL first
        ddl_file_path = r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Query-Generator-vanna\oracle_schema_20250415_054319.sql"
        if os.path.exists(ddl_file_path):
            with open(ddl_file_path, 'r') as f:
                ddl_content = f.read().strip()
                ddl_statements = [stmt.strip() for stmt in ddl_content.split(';') if stmt.strip()]
                
                for i, ddl in enumerate(ddl_statements[:10]):  # Limit for testing
                    try:
                        vn.train(ddl=ddl)
                        if (i + 1) % 5 == 0:
                            print(f"Added {i + 1} DDL statements")
                    except Exception as e:
                        logging.warning(f"Failed to add DDL {i}: {e}")
        
        # Train with parsed descriptions
        added, failed = train_with_parsed_descriptions(vn, doc_file_path)
        print(f"Description training: {added} added, {failed} failed")
        
        # Test the enhanced search
        print("\n--- Testing Enhanced Description Search ---")
        test_description_focused_search(vn)
        
    elif choice == '4':
        print("\n--- Debugging Parsed Descriptions ---")
        try:
            with open(doc_file_path, 'r', encoding='utf-8') as f:
                doc_content = f.read()
        except UnicodeDecodeError:
            with open(doc_file_path, 'r', encoding='latin1') as f:
                doc_content = f.read()
        
        parsed = vn.parse_documentation_descriptions(doc_content)
        
        print(f"\nParsed {len(parsed['table_descriptions'])} table descriptions")
        print("Sample table descriptions:")
        for i, desc in enumerate(parsed['table_descriptions'][:3]):
            print(f"  {i+1}. Table: {desc['table_name']}")
            print(f"     Description: {desc['description'][:60]}...")
        
        print(f"\nParsed {len(parsed['column_descriptions'])} column descriptions")
        print("Sample column descriptions:")
        for i, desc in enumerate(parsed['column_descriptions'][:3]):
            print(f"  {i+1}. Column: {desc['column_name']} in {desc['table_name']}")
            print(f"     Description: {desc['description'][:60]}...")
    
    print("\n=== Process Complete ===")

if __name__ == "__main__":
    main()