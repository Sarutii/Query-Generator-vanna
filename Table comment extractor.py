import re
import pandas as pd
from googletrans import Translator


def extract_table_info(sql_file_path=None, text = None):
    """
    Extract table names and their comments from text following the pattern:
    COMMENT ON TABLE table_name IS 'comment'
    
    Args:
        text (str): Input text containing table definitions
        
    Returns:
        pd.DataFrame: DataFrame with columns 'table_name' and 'comment'
    """

    if sql_file_path:
        with open(sql_file_path, 'r') as f:
            text = f.read()
    # Split by newlines to handle multiple entries
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    table_data = []
    translator = Translator()

    for line in lines:
        if "TABLE" in line:
            try:
                # Split by "TABLE" to get parts after each occurrence
                parts = re.split(r'TABLE\s+', line)
                
                # Skip the first part (before the first "TABLE")
                for part in parts[1:]:
                    # Extract table name (first word after "TABLE")
                    table_name = part.split()[0]
                    
                    # Find comment between single quotes after "IS"
                    comment_match = re.search(r"IS\s+'([^']*)'", part) or re.search(r'IS\s+"([^"]*)"', part)
                    comment = comment_match.group(1) if comment_match else ''
                    comment = comment.replace(";", "").strip()
                    #encode into 'windows-1256'
                    text_bytes = comment.encode('latin-1')
                    # Decode the bytes to text using the target encoding
                    commentArabic = text_bytes.decode('windows-1256', errors='replace')
                    #translate the arabic comment to english using google translate api
                    #if the comment is empty, skip the translation
                    if not commentArabic:
                        commentEnglish = ''
                    else:
                        # Translate Arabic comment to English
                        # Use the Google Translate API to translate the Arabic comment to English
                        # Note: You may need to handle API limits and errors in a production environment
                        commentEnglish = translator.translate(commentArabic, src='ar', dest='en').text
                    


                    table_data.append({
                        'table_name': table_name,
                        'comment': comment,
                        'Arabic comment': commentArabic,
                        'English comment': commentEnglish
                    })
            except Exception as e:
                print(f"Error processing line: {line}")
                print(f"Error details: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    return df

# Example usage
sample_text = """COMMENT ON TABLE  ACCOUNT_GROUPING IS ' ãÌãæÚÉ ÇáÍÓÇÈÇÊ  ; '"""

# Process the text and get the DataFrame
result_df = extract_table_info(sql_file_path= r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Table_Comment.sql")

# Display the result
print(result_df)
#save to csv
#but encode every column on its own encoding
result_df.to_csv(r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Table_Comment.csv", index=False)

# If you have multiple table definitions, you can add them like this:
"""
multiple_tables = \"""COMMENT ON TABLE  ACCOUNT_GROUPING IS ' ãÌãæÚÉ ÇáÍÓÇÈÇÊ  ; '
COMMENT ON TABLE  USER_ACCOUNTS IS ' ÍÓÇÈÇÊ ÇáãÓÊÎÏãíä ; '
COMMENT ON TABLE  TRANSACTION_LOG IS ' ÓÌá ÇáãÚÇãáÇÊ ; '\"""

result_df = extract_table_info(multiple_tables)
print(result_df)
"""