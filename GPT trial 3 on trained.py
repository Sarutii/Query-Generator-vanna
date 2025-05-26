from flask import Flask, render_template, request, jsonify, session, send_file
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
import os
import tempfile
import json
import oracledb
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

import re

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        config = config or {}
        config['model'] = 'mistral'
        config['persist_directory'] = './vanna-data'
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)
        self.prompt_prefix = ""

        self.embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(persist_directory="./vanna-data", embedding_function=self.embedding_model)
        
        self.table_names = ['ACCOUNT', 'ACCOUNT_CURR', 'ACCOUNT_GROUPING', 'ACCOUNT_REPORT_TYPE', 'ACCOUNT_TYPES',
    'ADDR_LST_DTL', 'ADJST_INOUT_QTY_TMP', 'ALL_BRANCHES', 'AMS_ADD_MTR_TRNS', 'AMS_AST_ADD_MTR',
    'AMS_COD_DTL', 'AMS_COD_MST', 'AMS_COD_PRV', 'AMS_FUL_AST_CNSMPTN', 'AMS_FUL_COD_DTL',
    'AMS_FUL_MOVMNT', 'AMS_FUL_REQ', 'AMS_FUL_TNK', 'AMS_FUL_TRNS', 'AMS_MNT_AST_RCV',
    'AMS_MNT_CHK_DTL', 'AMS_MNT_CHK_MST', 'AMS_MNT_CHK_PRT', 'AMS_MNT_CNTRCT', 'AMS_MNT_CNTRCT_PRC',
    'AMS_MNT_CNTRCT_SRV', 'AMS_MNT_CNTRCT_TAX_MOVMNT', 'AMS_MNT_EXEC_DTL', 'AMS_MNT_EXEC_MST',
    'AMS_MNT_JOB_CHK', 'AMS_MNT_JOB_DLVR', 'AMS_MNT_JOB_EQPMNT', 'AMS_MNT_JOB_LBR', 'AMS_MNT_JOB_ORDR',
    'AMS_MNT_JOB_PRBLM', 'AMS_MNT_JOB_PRT', 'AMS_MNT_JOB_SRV', 'AMS_MNT_REQ_DTL', 'AMS_MNT_REQ_MST',
    'AMS_MNT_TAX_MOVMNT', 'AMS_MSUR_CNVRT', 'AMS_MSUR_UNT', 'AMS_ODMTR_HST', 'AMS_ODMTR_READ_DTL',
    'AMS_ODMTR_READ_MST', 'AMS_PARA', 'AMS_PLN_AST', 'AMS_PLN_DTL', 'AMS_PLN_LBR', 'AMS_PLN_MST',
    'AMS_PLN_PRT', 'AMS_PLN_SRV', 'AMS_PLN_UPD', 'AMS_REQ_PRT_DTL', 'AMS_REQ_PRT_MST', 'AMS_SPR_PRT_CHNG',
    'AMS_SPR_PRT_RCV', 'AMS_SPR_PRT_RNW', 'AMS_SRV', 'AMS_SRV_CNTR', 'AMS_SRV_CNTR_CONN_SRV',
    'AMS_SRV_CNTR_LBR', 'AMS_SRV_CNTR_MTRL', 'AMS_SRV_CONN_AST', 'AMS_SRV_CONN_AST_GRP', 'AMS_SRV_CONN_SPR',
    'AMS_SRV_NTF', 'AMS_SRV_NTF_AST', 'AMS_SRV_NTF_PRT', 'AMS_SRV_NTF_SRV', 'AMS_TAX_SRV', 'AMS_TNK_RFLNG',
    'APEX_IR_PRMTR', 'APEX_IR_SLCT', 'APEX_PARA', 'APEX_TOKEN', 'API_TESTER', 'APS_BILL_FLLWUP_MOVMNT',
    'APS_FLLW_TRNS_MST', 'APS_FRGHT_ITMS_EXCL', 'APS_NET_SALES_ITM_CMPNNT_TMP', 'APS_PO_ITMS_EXCL',
    'APS_PO_ITMS_EXCL_TMP', 'APS_RQST_AUTO_TMP', 'APS_RQST_ITM_TMP', 'APS_RT_PRCHS_RSN', 'APX_FVRT_SCR',
    'APX_SCR', 'APX_SCR_PRV', 'AQ$_DOC_SYS_ID_Q_TBL_G', 'AQ$_DOC_SYS_ID_Q_TBL_H', 'AQ$_DOC_SYS_ID_Q_TBL_I',
    'AQ$_DOC_SYS_ID_Q_TBL_L', 'AQ$_DOC_SYS_ID_Q_TBL_S', 'AQ$_DOC_SYS_ID_Q_TBL_T', 'ARCHV_PARA',
    'ARS_ANSWR_QUESTNNR_DTL', 'ARS_ANSWR_QUESTNNR_MST', 'ARS_API_EXTRNL_TMP', 'ARS_AUTO_SLS_ORDR_DTL',
    'ARS_AUTO_SLS_ORDR_MST', 'ARS_BILL_CRDT_CRD', 'ARS_BILL_CRDT_CRD_BR', 'ARS_BILL_DOC_REF',
    'ARS_BILL_FLLWUP_MOVMNT', 'ARS_CALC_CMPNS_PERCNT_QTY_TMP', 'ARS_CALC_CMPNS_QTY_TMP', 'ARS_CODE_DTL',
    'ARS_CODE_MST', 'ARS_CONN_CST_DRVR', 'ARS_CSTMR_BANK', 'ARS_CSTMR_CHNG_DTL', 'ARS_CSTMR_CHNG_MST',
    'ARS_CSTMR_ITM', 'ARS_CST_LGN_HSTRY', 'ARS_CST_TRNS_GPS', 'ARS_DLVRY_GDS_DTL', 'ARS_DLVRY_GDS_MST',
    'ARS_EXP_BTCH_TMP', 'ARS_FDA_ITM_LST_DTL_TMP', 'ARS_FDA_ITM_LST_TMP', 'ARS_INSTLMNT_TYP_DTL',
    'ARS_INSTLMNT_TYP_MST', 'ARS_INTRMDT_CMPNY', 'ARS_ITM_CPN_TMP', 'ARS_KPI_NET_SALES_VW_YR',
    'ARS_LOAD_AMT_ITM_DTL', 'ARS_LOCTN_GEO_SMAN', 'ARS_MRKTING_AGNCY', 'ARS_MSG_CSS_OLD', 'ARS_NEWS_CSS',
    'ARS_NEWS_CSS11', 'ARS_NEWS_CSS22', 'ARS_PARA_CSS', 'ARS_PARA_CSS_OLD', 'ARS_PRM_FREE_QTY_TMP',
    'ARS_QT_PRM_EXPDT_PYMNT_TMP', 'ARS_QUESTNNR_DTL', 'ARS_QUESTNNR_MST', 'ARS_QUESTNNR_SUB_DTL',
    'ARS_QUESTN_QUESTNNR', 'ARS_RET_BILL_AUTO_TMP', 'ARS_RQ_EXD_LMT', 'ARS_SHW_INFRMTN_TRNS_OLD',
    'ARS_SHW_ITM_INFO', 'ARS_SLS_DLVRY_PRMT_DTL', 'ARS_SLS_DLVRY_PRMT_MST', 'AR_MSG_CSS', 'ASD', 'ASDASD',
    'ASSEMBLE_KIT_ITEMS', 'ASSEMBLE_KIT_ITEMS_DET', 'ASSEMBLE_KIT_ITEMS_TST', 'ATTCH_WHATSAPP_BOT', 'AUD',
    'AUTO_ORDER_DETAIL', 'AUTO_SALES_ORDER', 'BACKUP', 'BARCODE_LABELS_COUNTER', 'BAS', 'BAS2', 'BASE64',
    'BGT_APRV_RQ_FNC_DTL', 'BGT_APRV_RQ_FNC_MST', 'BIN_DETAILS', 'BIN_DETAILS_CPY', 'BOM_INSRT_ERR',
    'BRN_MANUAL_ATT_CONTROL', 'CASH_AT_BANK', 'CASH_INCOME', 'CASH_IN_HAND', 'CC_CODE_ACTIONS',
    'CC_GROUPING', 'CHCK_CP_QTY', 'CHK_GNR_DDC_TBL_20220818', 'CITIES', 'CLOB_TST', 'CMPNS_QTY',
    'CMPT_CMPNS_QTY_TMP', 'CMS_CASE', 'CMS_CASE_HNLING_HSTRY', 'CMS_CASE_REQ', 'CMS_CMPLXITY',
    'CMS_CORR_ACTION', 'CMS_ESCALATION', 'CMS_ESCL_SETUP', 'CMS_FAQS', 'CMS_FRWD_TYPE', 'CMS_HNDLING',
    'CMS_ITM_AGNT_SETUP', 'CMS_PRIORITY', 'CMS_ROOTCAUSE', 'CMS_SEVERITY', 'CMS_STATUS', 'CMS_SUBCASE',
    'CMS_TEMPLATING', 'CMS_TICKET_TYPE', 'CMS_TRACK', 'CNTRY', 'CODE_V', 'COLLERCTOR', 'COLLERCTOR_TMP_K',
    'COLUMN_LOG_']

    def set_prompt_prefix(self, prompt: str):
        self.prompt_prefix = prompt

    def get_prompt_prefix(self) -> str:
        return self.prompt_prefix
    
    def get_context_from_vectorstore(self, query: str, top_k: int = 4, threshold: float = 0.2):
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        filtered = [doc for doc, score in results if score >= threshold]
        return "\n".join([doc.page_content for doc in filtered])


    def generate_sql(self, question: str, max_attempts: int = 3) -> str:
        context = self.get_context_from_vectorstore(question)
        if not context:
            raise ValueError("No relevant information found in vector store for your question.")

        prompt = f"""{self.prompt_prefix}
    Context:
    {context}

    Question: {question}
    """

        for _ in range(max_attempts):
            sql = super().generate_sql(prompt)
            referenced = self.extract_referenced_tables(sql)
            if all(tbl in self.table_names for tbl in referenced):
                return sql

        raise ValueError("Model generated SQL using invalid table names.")

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key'

# Initialize Oracle client
oracledb.init_oracle_client(lib_dir=r"C:\\Users\\ahmed\\Downloads\\instantclient-basic-windows.x64-23.8.0.25.04\\instantclient_23_8")

# Load table names from uploaded CSV



vn = MyVanna()

tables_str = ", ".join(vn.table_names) if vn.table_names else "the tables you were trained on"


# Minimal prompt prefix with instructions only
vn.set_prompt_prefix(f"""
You are an expert SQL assistant writing SQL queries for an Oracle Database.
Use only these valid table names: {tables_str}.
Do NOT fabricate table names or columns.
Follow strict Oracle syntax: ROWNUM, SYSDATE, TO_DATE('YYYY-MM-DD', 'YYYY-MM-DD'), DUAL, etc.
Return only executable Oracle SQL, no explanations.
""")

# Connect to Oracle
vn.connect_to_oracle(
    user='IAS202538',
    password='123',
    dsn="localhost:1521/xepdb1"
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
        for _ in range(3):  # Retry if invalid table name appears
            sql = vn.generate_sql(question=question)
            if any(tbl in sql for tbl in vn.table_names):
                break
        else:
            return jsonify({"status": "error", "message": "No valid table name found in SQL."})
        return jsonify({"status": "success", "sql": sql})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/execute", methods=["POST"])
def execute_sql():
    sql_code = request.form.get("sql")
    try:
        with oracledb.connect(user='IAS202538', password='123', dsn='localhost:1521/xepdb1') as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql_code)
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
