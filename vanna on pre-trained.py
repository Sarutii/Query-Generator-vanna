from flask import Flask, render_template, request, jsonify, session, send_file
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
import os
import tempfile
import json
import oracledb

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

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Needed for session

# Use thin mode for Oracle client
oracledb.init_oracle_client(lib_dir=r'C:\Users\ahmed\Downloads\instantclient-basiclite-windows.x64-23.7.0.25.01\instantclient_23_7')

# Initialize Vanna - this will load the pre-trained data
vn = MyVanna()

# Load table names from the saved file (created during training)
table_names = []
try:
    with open('schema_tables.json', 'r') as f:
        schema_data = json.load(f)
        table_names = schema_data.get('tables', [])
    print(f"Loaded {len(table_names)} table names from schema_tables.json")
except Exception as e:
    print(f"Warning: Could not load schema tables: {e}")

# Add table names to the prompt prefix
tables_str = ", ".join(table_names) if table_names else "the tables you were trained on"

# Set the enhanced Oracle SQL prompt with table names
vn.set_prompt_prefix(f"""You are an expert SQL assistant writing SQL queries for an Oracle Database.
You must follow Oracle SQL syntax strictly and only use tables that exist in the user's schema.

Available tables: {tables_str}

Oracle SQL rules to follow:
- Use `ROWNUM` for limiting rows instead of `LIMIT` (e.g., `WHERE ROWNUM <= 10`).
- Use `SYSDATE` for the current date.
- Use `TO_DATE('YYYY-MM-DD', 'YYYY-MM-DD')` to parse dates.
- For string concatenation, use `||` operator.
- Avoid PostgreSQL/MySQL syntax like `LIMIT`, `ILIKE`, or `TRUE/FALSE` — these are not valid in Oracle.
- Use `DUAL` for selecting constants (e.g., `SELECT 1 FROM DUAL`).
- For pagination, use ROW_NUMBER() OVER() or ROWNUM with nested queries instead of OFFSET/FETCH.
- Use Oracle's hierarchical query syntax with CONNECT BY and PRIOR for tree-structured data.
- Remember that Oracle's NVL() is equivalent to COALESCE() in other dialects.
- For date arithmetic, use date + number for days (e.g., SYSDATE + 7 for a week later).

Important constraints:
1. ONLY reference tables that actually exist in the list of available tables provided above.
2. NEVER make up table names or columns that aren't in the tables you were trained on.
3. Use proper Oracle join syntax and appropriate table aliases.
4. Always use fully qualified column names in joins to avoid ambiguity.

Always output clean, runnable Oracle SQL with appropriate table and column references.
If the user asks a question, respond with the correct Oracle SQL query that will work specifically in their Oracle database.
Do not include ANY explanatory text before or after the SQL query - just return the SQL itself.""")

# Connect to Oracle
vn.connect_to_oracle(
    user='IAS202538',
    password='123',
    dsn="localhost:1521/xepdb1"  # Easy connect string: host:port/service_name
)

vn.allow_llm_to_see_data = True
vn.allow_llm_to_see_sql = True
vn.allow_sql_to_see_data = True
vn.allow_sql_to_see_llm = True
vn.allow_sql_to_see_sql = True

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_sql():
    question = request.form.get("question")
    try:
        sql = vn.generate_sql(question=question)
        return jsonify({"status": "success", "sql": sql})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/execute", methods=["POST"])
def execute_sql():
    sql_code = request.form.get("sql")
    try:
        # Establish Oracle DB connection
        with oracledb.connect(
            user='IAS202538',
            password='123',
            dsn='localhost:1521/xepdb1'
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql_code)

                # Fetch data only if the statement returns rows (e.g., SELECT)
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    rows = cursor.fetchall()
                    data = [dict(zip(columns, row)) for row in rows]
                else:
                    connection.commit()
                    data = f"{cursor.rowcount} row(s) affected."

        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/download", methods=["POST"])
def download():
    sql_code = request.form.get("sql")
    file_type = request.form.get("file_type", "txt")
    suffix = f".{file_type}"
    with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix=suffix) as tmp:
        tmp.write(sql_code)
        tmp_path = tmp.name

    return send_file(tmp_path, as_attachment=True, download_name=f"query{suffix}")

@app.route("/toggle-theme", methods=["POST"])
def toggle_theme():
    current = session.get('theme', 'light')
    session['theme'] = 'dark' if current == 'light' else 'light'
    session.modified = True
    return jsonify({"theme": session['theme']})

@app.context_processor
def inject_theme():
    return dict(theme=session.get('theme', 'light'))

if __name__ == "__main__":
    if not os.path.exists('./vanna-data'):
        print("\n⚠️ Warning: Training data directory not found.")
        print("Please run the training script first: python train_vanna.py <schema.sql> <schema.json>\n")
    
    app.run(debug=True)