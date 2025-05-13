import pandas as pd
from googletrans import Translator
import time

# i want to read csv file and get a column named arabic comment and translate it to english and save it in a new column named english comment
def translate_arabic_to_english(input_csv_path, output_csv_path):
    # Read the CSV file with UTF-8 encoding and error handling
    df = pd.read_csv(input_csv_path, encoding='utf-8')
    
    # Initialize the Google Translator
    translator = Translator()

    # Function to translate Arabic text to English
    def translate_text(text):
        if not isinstance(text, str) or not text.strip():
            return text
        
        # Maximum number of retry attempts
        # max_retries = 3
        # retry_count = 0
        
        # while retry_count < max_retries:
        try:
            # Don't translate if empty
            if not text.strip():
                return text
            
            translated = translator.translate(text, src='ar', dest='en')
            return translated.text
            
        except AttributeError as e:
            # Specific handling for the 'raise_Exception' issue
            if "'Translator' object has no attribute 'raise_Exception'" in str(e):
                print(f"Caught known translator error, returning original text: {e}")
                return text  # Return original text if this specific error occurs
            else:
                print(f"Attribute error: {e}")
                # retry_count += 1
                # time.sleep(1)  # Wait before retrying
                
        except Exception as e:
            print(f"Error translating text: {e}")
            # retry_count += 1
            # time.sleep(1)  # Wait before retrying
        
        # # If all retries fail, return original text
        # print(f"Failed to translate after {max_retries} attempts. Returning original text.")
        # return text
    
    # Apply translation to the 'Arabic comment' column and save it in a new column 'English comment'
    df['English comment'] = df['Arabic comment'].apply(translate_text)
    print("Translation completed. Sample of translated comments:")
    print(df[['Arabic comment', 'English comment']].head())
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    print(f"Translation completed. Translated comments saved to {output_csv_path}.")

# Example usage
input_csv_path = r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Column_Comments_arabic.csv"
output_csv_path = r"C:\Users\ahmed\Desktop\Projects\Query Generator\Vanna_app\Column_Comments_final.csv"

translate_arabic_to_english(input_csv_path, output_csv_path)
