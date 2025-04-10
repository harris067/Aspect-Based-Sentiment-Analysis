import pandas as pd
import openai

openai.api_key = ""
def process_sentence(sentence):
    prompt = f"""
Objective: Extract all correct aspect-sentiment pairs from the sentence provided below. 
Output the results as a JSON list of pairs in the format: [["aspect1", "sentiment1"], ["aspect2", "sentiment2"]].

Sentence: "{sentence}"

Rules:
- Assign "positive" sentiment for positive opinions.
- Assign "negative" sentiment for negative opinions.
- For sentences  where the sentiment is absent for an aspect (e.g., "It took half an hour to get our check, which was perfect since we could sit, have drinks and talk!", here drinks' sentiment is neutral"),they are neutral sentiments.
- Format the output as JSON list like in the following example: 

Example:
Input: "I have to say they have one of the fastest delivery times in the city, but their customer service could use some improvement."
Expected Output: [["delivery times", "positive"], ["customer service", "negative"]]
- Do not include any explanations, commentary,what u did, or extra text but only the outputs for the given sentence.
"""
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content

df = pd.read_csv("merged.csv", encoding="utf-8")

df["aspect_sentiment_pairs"] = df["Sentence"].apply(process_sentence)

df.to_csv("fulloutput.csv", encoding="utf-8", index=False)

print("Data augmentation complete! Check 'sample_output_data.csv'.")
