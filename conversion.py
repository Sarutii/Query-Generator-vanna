import pandas as pd

# Read the CSV file with UTF-8 encoding and error handling
df = pd.read_csv(r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\COLUMN_Comment.csv", encoding='utf-8')

# Instead of trying to encode/decode which is causing problems, try this safer approach:
def convert_text(text):
    if not isinstance(text, str):
        return text
    try:
        # Try a direct approach without intermediate encoding
        return text
    except Exception as e:
        # If any error occurs, return the original text
        print(f"Error converting text: {e}")
        return text

#if comment is not empty, convert it to arabic
def convert_to_arabic(text):
    if not isinstance(text, str) or not text.strip():
        return text
    try:
        # Encode to bytes and decode to Arabic using windows-1256
        return text.encode('latin-1', errors='replace').decode('windows-1256', errors='replace')
    except Exception as e:
        print(f"Error converting to Arabic: {e}")
        return text

df['Arabic comment'] = df['comment'].apply(convert_text).apply(convert_to_arabic)

# Print sample of the data to understand what we're working with
print("First few rows of Arabic comment column:")
print(df['Arabic comment'].head())

# Uncomment to save the file
df.to_csv(r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Column_Comments_arabic.csv", index=False , encoding='utf-8-sig')

# Diagnostic information
print("\nSample data overview:")
for i, row in df[['comment', 'Arabic comment']].head().iterrows():
    print(f"Row {i}:")
    print(f"Original: {row['comment']}")
    print(f"Converted: {row['Arabic comment']}")
    print("-" * 40)

