#read the csv file and combine the table names with the arabic and english comments into a single string and save it into txt file
import pandas as pd

df = pd.read_csv(r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Table_comments_final.csv")
df['combined'] = df.apply(lambda x: f"Table Name: {x['table_name']}\nArabic Comment: {x['Arabic comment']}\nEnglish Comment: {x['English comment']}\n\n", axis=1)
combined_text = ''.join(df['combined'].tolist())
with open(r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\combined_comments.txt", 'w', encoding='utf-8') as f:
    f.write(combined_text)
