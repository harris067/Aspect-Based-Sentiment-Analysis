import pandas as pd
import re
def clean_sentence(sentence: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9.!? ]+', '', sentence)
    cleaned = re.sub(r'([.!?])\1+', r'\1', cleaned)
    cleaned = cleaned.lower()
    return cleaned
df = pd.read_csv("Restaurants_Train.csv", encoding="utf-8")
df["Sentence"] = df["Sentence"].apply(clean_sentence)
df.to_csv("processed_sentences.csv", encoding="utf-8", index=False)
print("Processing complete! Check 'processed_sentences.csv'.")
