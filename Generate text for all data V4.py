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

# Function to create bilingual description
def create_bilingual_description(arabic_comment, english_comment):
    """Create concise bilingual description"""
    if arabic_comment and english_comment:
        return f"{str(arabic_comment).strip()} - {str(english_comment).strip()}"
    elif arabic_comment:
        return str(arabic_comment).strip()
    elif english_comment:
        return str(english_comment).strip()
    else:
        return ""

# Generate COMPACT training chunks - ONE chunk per table with ALL information
training_chunks = []

for table_name, table_data in schema.items():
    # Get table descriptions
    table_arabic = table_arabic_map.get(table_name, "")
    table_english = table_english_map.get(table_name, "")
    table_desc = create_bilingual_description(table_arabic, table_english)
    
    # Start building the comprehensive table chunk
    chunk = f"TABLE: {table_name}"
    
    if table_desc:
        chunk += f"\nDESCRIPTION: {table_desc}"
    
    # Add all columns in a compact format
    chunk += f"\nCOLUMNS:"
    
    columns_info = []
    pk_columns = []
    
    for col in table_data.get("columns", []):
        col_name = col["name"]
        col_type = col["data_type"]
        nullable = "NULL" if col["nullable"] == "Y" else "NOT NULL"
        is_pk = col.get("is_primary_key", False)
        
        if is_pk:
            pk_columns.append(col_name)
        
        # Get column description
        col_arabic = column_arabic_map.get((table_name, col_name), "")
        col_english = column_english_map.get((table_name, col_name), "")
        col_desc = create_bilingual_description(col_arabic, col_english)
        
        # Compact column format
        col_info = f"{col_name} ({col_type}, {nullable})"
        if is_pk:
            col_info += " [PK]"
        if col["default"]:
            col_info += f" DEFAULT:{col['default']}"
        if col_desc:
            col_info += f" | {col_desc}"
        
        columns_info.append(col_info)
    
    # Add columns to chunk
    for col_info in columns_info:
        chunk += f"\n  - {col_info}"
    
    # Add primary keys summary
    if pk_columns:
        chunk += f"\nPRIMARY_KEYS: {', '.join(pk_columns)}"
    
    # Add foreign keys in compact format
    if "foreign_keys" in table_data and table_data["foreign_keys"]:
        chunk += f"\nFOREIGN_KEYS:"
        for fk in table_data["foreign_keys"]:
            cols = ", ".join(fk["columns"])
            ref_table = fk["reference_table"]
            ref_cols = ", ".join(fk["reference_columns"])
            chunk += f"\n  - {cols} → {ref_table}({ref_cols})"
    
    # Add multilingual keywords for better search
    keywords = [table_name.lower(), "جدول", "table"]
    keywords.extend([col["name"].lower() for col in table_data.get("columns", [])])
    if "foreign_keys" in table_data:
        keywords.extend([fk["reference_table"].lower() for fk in table_data["foreign_keys"]])
    
    chunk += f"\nKEYWORDS: {', '.join(set(keywords))}"
    
    training_chunks.append(chunk)

# Save compact chunks to file
with open("vanna_training_compact.txt", "w", encoding="utf-8") as f:
    separator = "\n" + "="*80 + "\n"
    f.write(separator.join(training_chunks))

# Also save as JSON with metadata
chunks_data = []
for i, chunk in enumerate(training_chunks):
    table_name = chunk.split('\n')[0].replace('TABLE: ', '')
    chunks_data.append({
        "id": f"table_{i+1}",
        "table_name": table_name,
        "content": chunk,
        "type": "table_complete"
    })

with open("vanna_training_compact.json", "w", encoding="utf-8") as f:
    json.dump(chunks_data, f, ensure_ascii=False, indent=2)

print(f"✅ Generated {len(training_chunks)} COMPACT training chunks (one per table)")
print(f"   - Text file: vanna_training_compact.txt")
print(f"   - JSON file: vanna_training_compact.json")
print(f"   - Each chunk contains complete table information")
print(f"   - Optimized size for efficient ChromaDB embedding and search")
print(f"   - Multilingual keywords included for better similarity matching")