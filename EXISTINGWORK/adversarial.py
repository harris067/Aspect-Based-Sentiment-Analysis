import ollama
import pandas as pd
import json

file_path = "aug_data.csv" 
df = pd.read_csv(file_path)

required_columns = ['id', 'Sentence', 'aspect_sentiment_pairs']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Dataset must have '{col}' column.")

def generate_prompt(original_text, aspects_sentiments):
    aspects_text = ", ".join([f"\"{aspect}\" (sentiment: \"{sentiment}\")" for aspect, sentiment in aspects_sentiments])
    prompt = (
        f"Your task is to rephrase the given sentence while unaltering the words given in '{aspects_text}' "
        f"IMPORTANT: Each aspect MUST appear exactly as given (do not modify, replace, or remove it). "
        f"Original Sentence: \"{original_text}\"\n"
        f"The following words should not be altered in any way while rephrasing the sentence: {aspects_text}\n\n"
        f"Rephrase the sentence so that each aspect in '{aspects_text}' present in the sentence is unaltered and unmodified and rephrase the rest of the sentence.\n"
        f"Change only the descriptive or opinion parts of the sentence.\n"
        f"Do not include any greetings, commentary, or extra text. Output only the rephrased sentence.\n"
    )
    return prompt

def generate_adversarial_text(row):
    original_text = row["Sentence"]
    try:
        aspects_sentiments = json.loads(row["aspect_sentiment_pairs"])
    except Exception as e:
        print(f"Error parsing aspect_sentiment_pairs for id {row['id']}: {e}")
        return original_text 

    prompt = generate_prompt(original_text, aspects_sentiments)
    response = ollama.generate(model="llama2:7b", prompt=prompt)
    adv_text = str(response.get("response", "")).strip()


    if ":" in adv_text:
        adv_text = adv_text.split(":", 1)[-1].strip()


    tokens_to_remove = ["aspect", "opinion"]
    for token in tokens_to_remove:
        adv_text = adv_text.replace(token, "")
    
    
    adv_text = " ".join(adv_text.split())

    
    if not adv_text or adv_text.lower() == original_text.lower():
        return original_text
    return adv_text

df["Sentence"] = df.apply(generate_adversarial_text, axis=1)
output_file = "adversarial_data.csv"
df.to_csv(output_file, index=False)
print(f"Adversarial dataset saved: {output_file}")
