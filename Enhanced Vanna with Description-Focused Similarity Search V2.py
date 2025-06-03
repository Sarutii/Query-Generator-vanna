#!/usr/bin/env python3
"""
Enhanced Vanna Vector Store - Document Section Retrieval Based on Description Matching
This script finds similar descriptions and returns the complete document sections.
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
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MyVannaDocumentRetrieval(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        config = config or {}
        config['model'] = 'mistral'
        config['persist_directory'] = './vanna-data-documents'
        
        config['embedding_function'] = None
        config['collection_name'] = 'vanna_document_sections'
        
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)
        
        # Store document sections with their descriptions
        self.document_sections = []
        self.description_to_section_map = {}
        
        self._verify_embedding_setup()
    
    def _verify_embedding_setup(self):
        """Verify that embeddings are properly configured"""
        try:
            available_methods = []
            for method_name in dir(self):
                if 'similar' in method_name.lower() or 'related' in method_name.lower():
                    if not method_name.startswith('_'):
                        available_methods.append(method_name)
            
            logging.info(f"Available similarity methods: {available_methods}")
            logging.info("✅ Embedding configuration verified")
        except Exception as e:
            logging.warning(f"⚠️ Embedding setup issue: {e}")
    
    def parse_document_sections_with_descriptions(self, doc_content: str) -> List[Dict]:
        """
        Parse documentation into sections and extract all descriptions from each section
        
        Args:
            doc_content: Raw documentation content
            
        Returns:
            List of document sections with their descriptions
        """
        document_sections = []
        
        try:
            # Split content into sections
            sections = doc_content.split('------------------------------------------------------------')
            
            for section_idx, section in enumerate(sections):
                if not section.strip():
                    continue
                
                section_data = {
                    'section_idx': section_idx,
                    'full_content': section.strip(),
                    'descriptions': [],
                    'length': len(section.strip())
                }
                
                # Extract all descriptions from this section
                lines = section.strip().split('\n')
                current_table = None
                
                for line_idx, line in enumerate(lines):
                    line = line.strip()
                    if not line or len(line) < 10:  # Skip very short lines
                        continue
                    
                    # Every meaningful line is a potential description
                    description_data = {
                        'text': line,
                        'line_idx': line_idx,
                        'section_idx': section_idx,
                        'type': self._classify_description_type(line)
                    }
                    
                    # Try to extract table context
                    if self._is_table_line(line):
                        current_table = self._extract_table_name(line)
                        description_data['table_context'] = current_table
                        description_data['is_table_header'] = True
                    else:
                        description_data['table_context'] = current_table
                        description_data['is_table_header'] = False
                    
                    # Extract column info if it's a column description
                    if self._is_column_description(line):
                        col_info = self._parse_column_description(line)
                        if col_info:
                            description_data['column_info'] = col_info
                    
                    section_data['descriptions'].append(description_data)
                
                # Only add sections that have meaningful descriptions
                if section_data['descriptions']:
                    document_sections.append(section_data)
                    logging.debug(f"Section {section_idx}: {len(section_data['descriptions'])} descriptions")
            
            logging.info(f"Parsed {len(document_sections)} document sections")
            logging.info(f"Total descriptions: {sum(len(sec['descriptions']) for sec in document_sections)}")
            
        except Exception as e:
            logging.error(f"Error parsing document sections: {e}")
        
        return document_sections
    
    def _classify_description_type(self, line: str) -> str:
        """Classify the type of description"""
        line_lower = line.lower()
        
        if self._is_table_line(line):
            return 'table'
        elif self._is_column_description(line):
            return 'column'
        elif any(keyword in line_lower for keyword in ['description', 'وصف', 'معلومات', 'بيانات']):
            return 'description'
        elif any(keyword in line_lower for keyword in ['note', 'ملاحظة', 'تنبيه']):
            return 'note'
        else:
            return 'general'
    
    def _is_table_line(self, line: str) -> bool:
        """Detect if a line contains a table definition/description"""
        table_indicators = [
            'TABLE', 'table', 'جدول', 'جداول',
            'CREATE TABLE', 'create table',
        ]
        
        line_upper = line.upper()
        return (any(indicator.upper() in line_upper for indicator in table_indicators) or
                (len(line.split()) <= 3 and line.isupper() and len(line) > 2))
    
    def _extract_table_name(self, line: str) -> str:
        """Extract table name from a table description line"""
        cleaned = re.sub(r'(CREATE TABLE|TABLE|table|جدول)', '', line, flags=re.IGNORECASE)
        cleaned = cleaned.strip('():;')
        
        words = cleaned.split()
        for word in words:
            if word.isalnum() or '_' in word:
                return word.strip('(),;')
        
        return cleaned.strip() if cleaned else 'UNKNOWN'
    
    def _is_column_description(self, line: str) -> bool:
        """Detect if a line contains a column description"""
        column_indicators = [
            ':', '-', 'عمود', 'حقل', 'field', 'column',
            'VARCHAR', 'NUMBER', 'DATE', 'INTEGER',
        ]
        
        return any(indicator in line for indicator in column_indicators)
    
    def _parse_column_description(self, line: str) -> Optional[Dict]:
        """Parse column description to extract column name and details"""
        try:
            if ':' in line:
                parts = line.split(':', 1)
                column_name = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ''
            elif '-' in line and not line.startswith('--'):
                parts = line.split('-', 1)
                column_name = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ''
            else:
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
    
    def add_document_section_to_vector_store(self, section_data: Dict):
        """
        Add each description from a document section to vector store
        Each description points back to its full document section
        
        Args:
            section_data: Document section with descriptions
        """
        added = 0
        failed = 0
        
        try:
            section_idx = section_data['section_idx']
            
            # Add each description as a separate embedding that points to the full section
            for desc in section_data['descriptions']:
                try:
                    # Create enhanced description for better embedding
                    enhanced_desc = self._create_enhanced_description(desc, section_data)
                    
                    # Add to vector store as documentation
                    self.train(documentation=enhanced_desc)
                    
                    # Map this description to its section
                    desc_id = f"section_{section_idx}_line_{desc['line_idx']}"
                    self.description_to_section_map[desc_id] = section_data
                    
                    added += 1
                    logging.debug(f"✅ Added description: {enhanced_desc[:60]}...")
                    
                except Exception as e:
                    logging.warning(f"Failed to add description: {e}")
                    failed += 1
            
            logging.info(f"Section {section_idx}: {added} descriptions added, {failed} failed")
            return added, failed
            
        except Exception as e:
            logging.error(f"❌ Failed to add section {section_data.get('section_idx', 'unknown')}: {e}")
            return 0, 1
    
    def _create_enhanced_description(self, desc_data: Dict, section_data: Dict) -> str:
        """Create enhanced description text for better embedding"""
        base_text = desc_data['text']
        desc_type = desc_data['type']
        table_context = desc_data.get('table_context', '')
        
        # Add context based on type
        if desc_type == 'table':
            enhanced = f"Table Definition: {base_text}"
        elif desc_type == 'column':
            if table_context:
                enhanced = f"Column in {table_context}: {base_text}"
            else:
                enhanced = f"Column Description: {base_text}"
        else:
            if table_context:
                enhanced = f"Information about {table_context}: {base_text}"
            else:
                enhanced = f"Database Information: {base_text}"
        
        return enhanced
    
    def find_similar_descriptions_and_return_documents(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Find descriptions similar to the question and return their complete document sections
        
        Args:
            question: User's question
            top_k: Maximum number of document sections to return
            
        Returns:
            List of complete document sections containing similar descriptions
        """
        try:
            logging.info(f"Searching for descriptions similar to: '{question}'")
            
            # Generate enhanced queries for better matching
            enhanced_queries = self._generate_enhanced_queries(question)
            
            all_related_docs = []
            
            # Search with multiple query variations
            for enhanced_query in enhanced_queries:
                try:
                    related_docs = self.get_related_documentation(question=enhanced_query, n=top_k*3)
                    if isinstance(related_docs, list):
                        all_related_docs.extend(related_docs)
                        logging.debug(f"Found {len(related_docs)} results for query: '{enhanced_query}'")
                except Exception as e:
                    logging.debug(f"Search failed for query '{enhanced_query}': {e}")
            
            # Remove duplicates
            unique_docs = list(set(all_related_docs))
            logging.info(f"Found {len(unique_docs)} unique matching descriptions")
            
            # For each matching description, find its document section
            matching_sections = {}
            section_scores = {}
            
            for doc in unique_docs:
                # Calculate similarity score
                similarity_score = self._calculate_similarity_score(question, doc)
                
                # Find which section this description belongs to
                section_data = self._find_section_for_description(doc)
                
                if section_data:
                    section_idx = section_data['section_idx']
                    
                    # Keep track of the best score for each section
                    if section_idx not in section_scores or similarity_score > section_scores[section_idx]:
                        section_scores[section_idx] = similarity_score
                        matching_sections[section_idx] = {
                            'section_data': section_data,
                            'matching_description': doc,
                            'similarity_score': similarity_score
                        }
            
            # Sort sections by similarity score and return top results
            sorted_sections = sorted(matching_sections.values(), 
                                   key=lambda x: x['similarity_score'], 
                                   reverse=True)
            
            result_sections = []
            for i, section_info in enumerate(sorted_sections[:top_k]):
                result_sections.append({
                    'rank': i + 1,
                    'section_idx': section_info['section_data']['section_idx'],
                    'full_document': section_info['section_data']['full_content'],
                    'matching_description': section_info['matching_description'],
                    'similarity_score': section_info['similarity_score'],
                    'total_descriptions': len(section_info['section_data']['descriptions'])
                })
            
            logging.info(f"Returning {len(result_sections)} complete document sections")
            
            return result_sections
            
        except Exception as e:
            logging.error(f"❌ Error finding similar descriptions: {e}")
            return []
    
    def _generate_enhanced_queries(self, query: str) -> List[str]:
        """Generate multiple query variations for better matching"""
        queries = [query]
        
        # Add variations with common database terms
        queries.append(f"information about {query}")
        queries.append(f"data about {query}")
        queries.append(f"معلومات عن {query}")
        queries.append(f"بيانات عن {query}")
        
        # Add keyword variations
        common_terms = {
            'show': ['display', 'get', 'اظهر', 'عرض'],
            'all': ['*', 'جميع', 'كل'],
            'account': ['accounts', 'حساب', 'حسابات'],
            'customer': ['customers', 'عميل', 'عملاء', 'زبون', 'زبائن'],
            'order': ['orders', 'طلب', 'طلبات'],
            'maintenance': ['صيانة', 'صيانات', 'إصلاح'],
            'branch': ['فرع', 'فروع', 'branches'],
            'service': ['خدمة', 'خدمات', 'services'],
            'job': ['وظيفة', 'مهمة', 'عمل', 'jobs'],
            'asset': ['أصل', 'أصول', 'assets'],
            'notification': ['إشعار', 'إشعارات', 'تنبيه', 'notifications']
        }
        
        query_lower = query.lower()
        for eng_term, variations in common_terms.items():
            if eng_term in query_lower:
                for variation in variations:
                    new_query = query_lower.replace(eng_term, variation)
                    if new_query != query_lower:
                        queries.append(new_query)
            
            # Also check if any variation is in the query
            for variation in variations:
                if variation in query_lower:
                    new_query = query_lower.replace(variation, eng_term)
                    if new_query != query_lower:
                        queries.append(new_query)
        
        return list(set(queries))  # Remove duplicates
    
    def _calculate_similarity_score(self, question: str, description: str) -> float:
        """Calculate a simple similarity score between question and description"""
        try:
            # Simple word overlap scoring
            question_words = set(question.lower().split())
            desc_words = set(description.lower().split())
            
            if not question_words or not desc_words:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(question_words.intersection(desc_words))
            union = len(question_words.union(desc_words))
            
            jaccard_score = intersection / union if union > 0 else 0.0
            
            # Bonus for exact phrase matches
            phrase_bonus = 0.1 if any(word in description.lower() for word in question.lower().split() if len(word) > 3) else 0.0
            
            # Length normalization bonus
            length_bonus = min(len(description) / 200, 0.1)
            
            return jaccard_score + phrase_bonus + length_bonus
            
        except Exception as e:
            logging.debug(f"Error calculating similarity score: {e}")
            return 0.0
    
    def _find_section_for_description(self, description: str) -> Optional[Dict]:
        """Find which document section contains this description"""
        try:
            # Simple approach: check if description text appears in any section
            for section in self.document_sections:
                if description.strip() in section['full_content'] or \
                   any(desc['text'] in description for desc in section['descriptions']):
                    return section
            
            # If not found directly, try fuzzy matching
            best_match = None
            best_score = 0.0
            
            for section in self.document_sections:
                for desc in section['descriptions']:
                    score = self._calculate_similarity_score(description, desc['text'])
                    if score > best_score:
                        best_score = score
                        best_match = section
            
            return best_match if best_score > 0.3 else None
            
        except Exception as e:
            logging.debug(f"Error finding section for description: {e}")
            return None

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

def train_with_document_sections(vn, doc_file_path):
    """Train the model with document sections and their descriptions"""
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
        
        # Parse document sections with descriptions
        logging.info("Parsing document sections with descriptions...")
        document_sections = vn.parse_document_sections_with_descriptions(doc_content)
        
        # Store sections in the Vanna instance
        vn.document_sections = document_sections
        
        total_added = 0
        total_failed = 0
        
        # Add each section to vector store
        logging.info(f"Adding {len(document_sections)} document sections to vector store...")
        for section in document_sections:
            added, failed = vn.add_document_section_to_vector_store(section)
            total_added += added
            total_failed += failed
            
            if (section['section_idx'] + 1) % 5 == 0:
                logging.info(f"Processed {section['section_idx'] + 1} sections")
        
        logging.info(f"Document section training completed: {total_added} descriptions added, {total_failed} failed")
        return total_added, total_failed
        
    except Exception as e:
        logging.error(f"Error training with document sections: {e}")
        return 0, 0

def test_document_retrieval(vn):
    """Test the document section retrieval based on description matching"""
    logging.info("=== DOCUMENT SECTION RETRIEVAL TEST ===")
    
    test_questions = [
        "show me account information",
        "customer details", 
        "maintenance job data",
        "branch information",
        "service notifications",
        "معلومات الحسابات",
        "بيانات العملاء",
        "طلبات الصيانة",
        "إشعارات الخدمة",
        "اظهر تنفيذات اوامر الصيانة",
        "asset management details"
    ]
    
    total_sections_found = 0
    
    for question in test_questions:
        print(f"\n--- Question: '{question}' ---")
        
        # Find similar descriptions and return complete document sections
        matching_sections = vn.find_similar_descriptions_and_return_documents(question, top_k=3)
        
        total_sections_found += len(matching_sections)
        
        if matching_sections:
            print(f"  ✅ Found {len(matching_sections)} relevant document sections:")
            
            for i, section in enumerate(matching_sections):
                print(f"\n  📄 Document Section {section['rank']} (Score: {section['similarity_score']:.3f}):")
                print(f"     Section Index: {section['section_idx']}")
                print(f"     Matching Description: {section['matching_description'][:80]}...")
                print(f"     Total Descriptions in Section: {section['total_descriptions']}")
                print(f"     Document Preview: {section['full_document'][:150]}...")
                print("     " + "="*50)
        else:
            print("  ❌ No matching document sections found")
    
    print(f"\n=== DOCUMENT RETRIEVAL TEST SUMMARY ===")
    print(f"Total questions tested: {len(test_questions)}")
    print(f"Total document sections retrieved: {total_sections_found}")
    print(f"Average sections per question: {total_sections_found/len(test_questions):.1f}")
    
    return total_sections_found > 0

def main():
    """Main function for document section retrieval"""
    print("=== Vanna Document Section Retrieval Based on Description Matching ===")
    print("1. Test document section retrieval")
    print("2. Train with document sections")
    print("3. Full process (parse + train + test)")
    print("4. Debug document sections parsing")
    print("5. Interactive question testing")
    
    choice = input("Enter your choice (1/2/3/4/5): ").strip()
    
    # Initialize Vanna
    vn = MyVannaDocumentRetrieval()
    
    doc_file_path = r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Query-Generator-vanna\Full Data Discrebtion.txt"
    
    if choice == '1':
        print("\n--- Testing Document Section Retrieval ---")
        test_document_retrieval(vn)
        
    elif choice == '2':
        print("\n--- Training with Document Sections ---")
        added, failed = train_with_document_sections(vn, doc_file_path)
        print(f"Training completed: {added} descriptions added, {failed} failed")
        
    elif choice == '3':
        print("\n--- Full Process: Parse, Train, and Test ---")
        
        # Train with document sections
        added, failed = train_with_document_sections(vn, doc_file_path)
        print(f"Document training: {added} descriptions added, {failed} failed")
        
        # Test the document retrieval
        print("\n--- Testing Document Section Retrieval ---")
        test_document_retrieval(vn)
        
    elif choice == '4':
        print("\n--- Debugging Document Sections Parsing ---")
        try:
            with open(doc_file_path, 'r', encoding='utf-8') as f:
                doc_content = f.read()
        except UnicodeDecodeError:
            with open(doc_file_path, 'r', encoding='latin1') as f:
                doc_content = f.read()
        
        sections = vn.parse_document_sections_with_descriptions(doc_content)
        
        print(f"\nParsed {len(sections)} document sections")
        
        for i, section in enumerate(sections[:3]):  # Show first 3 sections
            print(f"\n--- Section {section['section_idx']} ---")
            print(f"Total descriptions: {len(section['descriptions'])}")
            print(f"Content length: {section['length']} characters")
            print("Sample descriptions:")
            for j, desc in enumerate(section['descriptions'][:5]):
                print(f"  {j+1}. [{desc['type']}] {desc['text'][:60]}...")
            print(f"Full content preview: {section['full_content'][:200]}...")
    
    elif choice == '5':
        print("\n--- Interactive Question Testing ---")
        
        # First ensure training is done
        if not vn.document_sections:
            print("Training with document sections first...")
            train_with_document_sections(vn, doc_file_path)
        
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print(f"\nSearching for: '{question}'...")
            matching_sections = vn.find_similar_descriptions_and_return_documents(question, top_k=2)
            
            if matching_sections:
                print(f"\n✅ Found {len(matching_sections)} relevant document sections:")
                
                for section in matching_sections:
                    print(f"\n📄 Document Section {section['rank']} (Similarity: {section['similarity_score']:.3f}):")
                    print(f"Matching Description: {section['matching_description']}")
                    print(f"\n--- COMPLETE DOCUMENT SECTION ---")
                    print(section['full_document'])
                    print("--- END OF DOCUMENT SECTION ---\n")
            else:
                print("❌ No matching document sections found")
    
    print("\n=== Process Complete ===")

if __name__ == "__main__":
    main()