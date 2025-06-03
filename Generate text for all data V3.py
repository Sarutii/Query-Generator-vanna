import json
import pandas as pd

# Load the JSON schema
with open(r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Query-Generator-vanna\oracle_schema_20250415_054319.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

# Load table and column comments
table_comments = pd.read_csv(r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Query-Generator-vanna\Table_comments_final.csv")
column_comments = pd.read_csv(r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Query-Generator-vanna\Column_Comments_final.csv")

# Normalize column names
table_comments.columns = [col.strip().lower() for col in table_comments.columns]
column_comments.columns = [col.strip().lower() for col in column_comments.columns]

# Create lookup dictionaries for table comments (both Arabic and English)
table_arabic_map = dict(zip(table_comments['table_name'], table_comments['arabic comment']))
table_english_map = dict(zip(table_comments['table_name'], table_comments['english comment']))

# Create lookup dictionaries for column comments, handling TABLE.COLUMN format
column_arabic_map = {}
column_english_map = {}
for _, row in column_comments.iterrows():
    if '.' in row['column_name']:
        table_name, col_name = row['column_name'].split('.', 1)
        column_arabic_map[(table_name, col_name)] = row['arabic comment']
        column_english_map[(table_name, col_name)] = row['english comment']

# Function to create bilingual description optimized for similarity search
def create_bilingual_description(arabic_comment, english_comment):
    """Create description optimized for multilingual similarity search"""
    if arabic_comment and english_comment:
        # Clean and normalize comments
        arabic_clean = str(arabic_comment).strip()
        english_clean = str(english_comment).strip()
        return f"{arabic_clean} - {english_clean}"
    elif arabic_comment:
        return str(arabic_comment).strip()
    elif english_comment:
        return str(english_comment).strip()
    else:
        return ""

# Generate structured training chunks optimized for similarity search
training_chunks = []

for table_name, table_data in schema.items():
    # Get table descriptions in both languages
    table_arabic = table_arabic_map.get(table_name, "")
    table_english = table_english_map.get(table_name, "")
    table_desc = create_bilingual_description(table_arabic, table_english)
    
    # 1. TABLE OVERVIEW CHUNK - Comprehensive table information
    table_chunk = f"""TABLE: {table_name}
DESCRIPTION: {table_desc}
TYPE: Table Overview
KEYWORDS: {table_name.lower()}, table, جدول, database schema, قاعدة بيانات

COLUMNS SUMMARY:"""
    
    column_names = []
    pk_columns = []
    for col in table_data.get("columns", []):
        column_names.append(col["name"])
        if col.get("is_primary_key"):
            pk_columns.append(col["name"])
    
    table_chunk += f"\nColumn Names: {', '.join(column_names)}"
    if pk_columns:
        table_chunk += f"\nPrimary Keys: {', '.join(pk_columns)}"
    
    # Add foreign key relationships
    if "foreign_keys" in table_data and table_data["foreign_keys"]:
        fk_info = []
        for fk in table_data["foreign_keys"]:
            ref_table = fk["reference_table"]
            fk_info.append(f"{table_name} → {ref_table}")
        table_chunk += f"\nRelationships: {', '.join(fk_info)}"
    
    training_chunks.append(table_chunk)
    
    # 2. INDIVIDUAL COLUMN CHUNKS - Detailed column information
    for col in table_data.get("columns", []):
        col_name = col["name"]
        col_type = col["data_type"]
        nullable = "Nullable" if col["nullable"] == "Y" else "Not Nullable"
        is_pk = col.get("is_primary_key", False)
        
        # Get column descriptions in both languages
        col_arabic = column_arabic_map.get((table_name, col_name), "")
        col_english = column_english_map.get((table_name, col_name), "")
        col_desc = create_bilingual_description(col_arabic, col_english)
        
        # Create rich column chunk
        column_chunk = f"""COLUMN: {table_name}.{col_name}
TABLE: {table_name}
COLUMN_NAME: {col_name}
DATA_TYPE: {col_type}
NULLABLE: {nullable}
PRIMARY_KEY: {is_pk}
DESCRIPTION: {col_desc}
TYPE: Column Detail
KEYWORDS: {col_name.lower()}, {table_name.lower()}, column, عمود, field, حقل, {col_type.lower()}"""

        if col["default"]:
            column_chunk += f"\nDEFAULT_VALUE: {col['default']}"
        
        training_chunks.append(column_chunk)
    
    # 3. RELATIONSHIP CHUNKS - Foreign key relationships
    if "foreign_keys" in table_data and table_data["foreign_keys"]:
        for fk in table_data["foreign_keys"]:
            cols = ", ".join(fk["columns"])
            ref_table = fk["reference_table"]
            ref_cols = ", ".join(fk["reference_columns"])
            
            relationship_chunk = f"""RELATIONSHIP: {table_name} → {ref_table}
SOURCE_TABLE: {table_name}
SOURCE_COLUMNS: {cols}
TARGET_TABLE: {ref_table}
TARGET_COLUMNS: {ref_cols}
TYPE: Foreign Key Relationship
KEYWORDS: {table_name.lower()}, {ref_table.lower()}, relationship, علاقة, foreign key, مفتاح خارجي, reference, مرجع"""
            
            training_chunks.append(relationship_chunk)

# 4. GENERATE DOMAIN-SPECIFIC CHUNKS for better contextual understanding
table_groups = {}
for table_name in schema.keys():
    # Group tables by common prefixes or patterns
    prefix = table_name.split('_')[0] if '_' in table_name else table_name[:3]
    if prefix not in table_groups:
        table_groups[prefix] = []
    table_groups[prefix].append(table_name)

for prefix, tables in table_groups.items():
    if len(tables) > 1:  # Only create groups with multiple tables
        domain_chunk = f"""DOMAIN_GROUP: {prefix.upper()}
RELATED_TABLES: {', '.join(tables)}
TABLE_COUNT: {len(tables)}
TYPE: Domain Group
KEYWORDS: {prefix.lower()}, domain, مجال, related tables, جداول مترابطة, module, وحدة"""
        
        training_chunks.append(domain_chunk)

# Save chunks to file with clear separators
with open("vanna_training_chunks.txt", "w", encoding="utf-8") as f:
    f.write("\n\n" + "="*80 + "\n\n".join(training_chunks))

# Also save as JSON for programmatic access
chunks_data = []
for i, chunk in enumerate(training_chunks):
    chunks_data.append({
        "id": f"chunk_{i+1}",
        "content": chunk,
        "type": chunk.split('\n')[0].split(':')[0] if ':' in chunk.split('\n')[0] else "UNKNOWN"
    })

with open("vanna_training_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks_data, f, ensure_ascii=False, indent=2)

print(f"✅ Generated {len(training_chunks)} optimized training chunks:")
print(f"   - Text file: vanna_training_chunks.txt")
print(f"   - JSON file: vanna_training_chunks.json")
print(f"   - Chunks include: Table overviews, Column details, Relationships, and Domain groups")
print(f"   - Optimized for multilingual similarity search with paraphrase-multilingual-MiniLM-L12-v2")