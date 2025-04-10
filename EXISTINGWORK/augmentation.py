import pandas as pd
import ollama

df = pd.read_csv("processed_sentences.csv", encoding="utf-8")

def process_sentence(sentence):
    prompt = f"""You are given a CSV dataset with two columns: id, sentences. The output dataset must be the modified dataset of the input dataset 
    with the sentences modified to their respective augmented sentences. DO NOT include any explanations, commentary, 
    or extra text but only the augmented sentence output.
Follow these rules to augment the sentence for aspect-based sentiment analysis:
- Consider commas as separators for sub-sentences. 
- Every sub-sentence must have an explicit aspect (subject) and corresponding opinion. If missing, add it based on context.
- Each sub-sentence should follow the pattern: [Subject/Aspect] [connecting verb] [rest of the sentence]. Paraphrase if needed.
- Make sure all the sentences are in passive voice
- When encountering cut-short slang terms, expand them to follow the structure.
Important Note: Do not include any explanations, commentary, what you did, or extra text but only the augmented sentence output.
Now, process the following sentence:
Sentence: {sentence}
Output:"""

    response = ollama.chat(model="llama2:7b", messages=[{"role": "user", "content": prompt}])
    result = response["message"]["content"]
    
    
    if ":" in result:
        result = result.split(":", 1)[-1].strip()  
    
    return result

df["Sentence"] = df["Sentence"].apply(process_sentence)
df.to_csv("aug_data.csv", encoding="utf-8", index=False)
print("Data augmentation complete! Check 'aug_data.csv'.")


