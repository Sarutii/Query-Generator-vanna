# Query SQL Generator

This repository contains a web application that utilizes the [Vanna AI library](https://vanna.ai/docs/) to generate SQL queries from natural language input, specifically tailored for Oracle databases. The application offers a user-friendly interface where users can input questions in plain English, receive Oracle-compliant SQL queries, execute them against a database, and export the results. It is designed to simplify SQL query creation for both technical and non-technical users working with Oracle databases.

---

## Features

- **Natural Language to SQL Conversion**: Converts plain English questions into Oracle-compliant SQL queries using the Vanna AI library.
- **SQL Execution**: Allows users to execute generated SQL queries directly against an Oracle database and view the results.
- **SQL Export**: Enables exporting SQL queries as `.txt` or `.sql` files.
- **Theme Toggle**: Supports switching between light and dark themes for an enhanced user experience.
- **Schema Training**: Trains the Vanna model on database schema files for improved query accuracy.
- **Comment Extraction and Processing**: Includes utilities to extract and process table and column comments from SQL files, aiding in schema documentation.

---

## Installation

Follow these steps to set up the Yehia SQL Generator on your local machine:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Sarutii/Query-Generator-vanna.git
   cd Query-Generator-vanna
   ```

2. **Install Dependencies**:
   Ensure Python 3.x is installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   *Note*: You may need to create a `requirements.txt` file with dependencies like `flask`, `vanna`, `pandas`, `oracledb`, and `googletrans`.

3. **Set Up Oracle Client**:
   - Download and install the Oracle Instant Client from [Oracle's website](https://www.oracle.com/database/technologies/instant-client.html).
   - Update the `lib_dir` path in `app.py` and `vanna_on_pre-trained.py` to point to your Oracle Instant Client directory, e.g.:
     ```python
     oracledb.init_oracle_client(lib_dir="C:/path/to/instantclient")
     ```

4. **Train the Vanna Model** (Optional but recommended):
   - Prepare your database schema in a SQL file (e.g., `schema.sql`).
   - Run the training script:
     ```bash
     python vanna_training_on_data.py path/to/schema.sql
     ```
   - This trains the Vanna model on your schema and saves the trained data to `./vanna-data`.

---

## Configuration

1. **Database Connection**:
   - Modify the Oracle connection details in `app.py` and `vanna_on_pre-trained.py`:
     ```python
     vn.connect_to_oracle(
         user='YOUR_USERNAME',
         password='YOUR_PASSWORD',
         dsn="localhost:1521/YOUR_SERVICE_NAME"
     )
     ```

2. **Oracle Client Path**:
   - Ensure the `lib_dir` in `oracledb.init_oracle_client()` matches your Oracle Instant Client installation path.

3. **Theme Preference**:
   - The application defaults to a light theme, toggleable to dark via the web interface.

---

## Usage

1. **Run the Application**:
   - Start the Flask app with the untrained version:
     ```bash
     python app.py
     ```
   - Or use the pre-trained version if trained:
     ```bash
     python vanna_on_pre-trained.py
     ```

2. **Access the Web Interface**:
   - Open a browser and go to `http://localhost:5000`.

3. **Generate SQL**:
   - Enter a question (e.g., "List all employees hired this year").
   - Click "Generate SQL" to see the Oracle SQL query.

4. **Edit and Execute SQL**:
   - Edit the generated SQL in the textarea if needed.
   - Click "Execute" to run it against the Oracle database and view results.

5. **Export SQL**:
   - Click "Export" and choose `.txt` or `.sql` to download the query.

---

## Examples

- **Input**: "Show employees hired after January 1, 2023"  
  **Output**: 
  ```sql
  SELECT * FROM employees WHERE hire_date > TO_DATE('2023-01-01', 'YYYY-MM-DD');
  ```

- **Input**: "Count the number of transactions per customer"  
  **Output**: 
  ```sql
  SELECT customer_id, COUNT(*) as transaction_count FROM transactions GROUP BY customer_id;
  ```

---

## Scripts and Utilities

The repository includes several utility scripts for schema processing and training:

- **`Column_comment_extractor.py`**:
  - Extracts column names and comments from SQL files (e.g., `COMMENT ON COLUMN ...`).
  - Saves results to a CSV file (e.g., `COLUMN_Comment.csv`).

- **`Table_comment_extractor.py`**:
  - Extracts table names and comments from SQL files (e.g., `COMMENT ON TABLE ...`).
  - Translates comments to English using Google Translate and saves to a CSV file (e.g., `Table_Comment.csv`).

- **`conversion.py`**:
  - Converts column comments to Arabic and saves them to a new CSV file (e.g., `Column_Comments_arabic.csv`).

- **`compining.py`**:
  - Combines table names with Arabic and English comments into a single text file (e.g., `combined_comments.txt`).

- **`vanna_training_on_data.py`**:
  - Trains the Vanna model on a SQL schema file and saves the trained data.

---

## Limitations

- **Query Accuracy**: Depends on the quality of schema training and complexity of input questions.
- **Oracle-Specific**: Designed for Oracle databases; adapting to other systems requires modification.
- **Manual Review**: Always verify generated SQL before production use.

---

## Contributing

Contributions are encouraged! Submit pull requests or report issues on the [GitHub repository](https://github.com/Sarutii/Query-Generator-vanna).

---

## Additional Resources

- [Vanna AI Documentation](https://vanna.ai/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Oracle Instant Client](https://www.oracle.com/database/technologies/instant-client.html)

---

**Note**: This tool uses AI to generate SQL and may not always produce perfect queries. Always validate the output before execution.
