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

# Create lookup dictionaries for table comments
table_comment_map = dict(zip(table_comments['table_name'], table_comments['arabic comment']))

# Create lookup dictionary for column comments, handling TABLE.COLUMN format
column_comment_map = {}
for _, row in column_comments.iterrows():
    # Check if column name contains table name (format: TABLE.COLUMN)
    if '.' in row['column_name']:
        table_name, col_name = row['column_name'].split('.', 1)
        column_comment_map[(table_name, col_name)] = row['arabic comment']


# Generate training text
training_texts = []
for table_name, table_data in schema.items():
    text = f"Table: {table_name}"
    table_desc = table_comment_map.get(table_name, "")
    if table_desc:
        text += f"\nDescription: {table_desc}"
    
    text += "\nColumns:"
    for col in table_data.get("columns", []):
        col_name = col["name"]
        col_type = col["data_type"]
        nullable = "Nullable" if col["nullable"] == "Y" else "Not Nullable"
        default = f", Default: {col['default']}" if col["default"] else ""
        is_pk = "Primary Key" if col.get("is_primary_key") else ""
        comment = column_comment_map.get((table_name, col_name), "")
        
        line = f" - {col_name} ({col_type}) [{nullable}{default}] {is_pk}"
        if comment:
            line += f"\n   Description: {comment}"
        text += f"\n{line}"
    
    if "foreign_keys" in table_data and table_data["foreign_keys"]:
        text += "\nForeign Keys:"
        for fk in table_data["foreign_keys"]:
            cols = ", ".join(fk["columns"])
            ref_table = fk["reference_table"]
            ref_cols = ", ".join(fk["reference_columns"])
            text += f"\n - ({cols}) → {ref_table}({ref_cols})"
    
    training_texts.append(text + "\n" + "-"*60)

# Save to file
with open("vanna_training_text.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(training_texts))

print("✅ Training text generated: vanna_training_text.txt")