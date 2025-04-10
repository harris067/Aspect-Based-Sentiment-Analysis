import pandas as pd
import ast

df = pd.read_csv("fulloutput.csv")

expanded_rows = []

for _, row in df.iterrows():
    id_value = row["id"]
    sentence = row["Sentence"]
    aspect_sentiment_pairs = ast.literal_eval(row["aspect_sentiment_pairs"])  
    
    for aspect, sentiment in aspect_sentiment_pairs:
        expanded_rows.append([id_value, sentence, aspect, sentiment])


expanded_df = pd.DataFrame(expanded_rows, columns=["id", "text", "span", "label"])


expanded_df.to_csv("existingworktrainingdata.csv", index=False)

print("Expanded CSV file saved as 'finaloutput.csv'")
