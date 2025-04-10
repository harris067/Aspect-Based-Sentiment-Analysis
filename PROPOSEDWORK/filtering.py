import pandas as pd
import ollama
import ast 


df = pd.read_csv("dependancy_output.csv", encoding="utf-8")

def process_aspect_opinion(sentence, aspect_opinion_pairs):
    prompt = f"""You will receive a sentence and a list of aspect-opinion pairs.
Your task:
1. Extract aspect-opinion pairs from the given sentence for restaurant domain. Validate this list by making it more meaningful. Remove pairs with unnecessary aspects(not present in your extracted pairs) from the list.Make the opinions meaningful.Replace pronoun aspects with corresponding nouns.

2. Aspects and opinions must strictly be present in the sentence and aspect belong to the class of aspects of restaurant domain. For example: waiter,staff,food,ambiance,menu,seating,experince and like that and its related attributes.   

3. Sentiment values are one of these: "positive", "negative", or "neutral". 
   Opinions are words, phrases, or expressions that convey a subjective evaluation, judgment, or feeling towards an aspect or entity. The sentiment must reflect the emotional tone of its corresponding opinion.

4. Neutral sentiment is strictly for opinions that do not have any emotional tone.
   - Words like "nice", "pleasant", and "decent" should be classified as positive.
   - Words like "average", "okay" should be neutral.

5. **Output only a list of aspect-opinion-sentiment triples** in this format:
   [["aspect", "opinion", "sentiment"]] where aspect and opinion are the extracted pairs from the sentence, and sentiment is determined.

6. **No explanations, no extra text—just the structured list.**
Now, process this:
Sentence: "{sentence}"
Output:

"""


    
    try:
        response = ollama.chat(model="llama2:7b", messages=[{"role": "user", "content": prompt}])
        raw_output = response["message"]["content"].strip() 

        
        start = raw_output.find("[[")
        end = raw_output.rfind("]]") + 2
        cleaned_output = raw_output[start:end]

        
        aspect_opinion_triples = ast.literal_eval(cleaned_output)

        
        if isinstance(aspect_opinion_triples, list) and all(
            isinstance(triple, list) and len(triple) == 3 for triple in aspect_opinion_triples
        ):
            return str(aspect_opinion_triples) 
        else:
            return "[]"  
    except (SyntaxError, ValueError, KeyError):
        return "[]"  

def process_with_retry(sentence, aspect_opinion_pairs, max_retries=1000):
    for _ in range(max_retries):
        result = process_aspect_opinion(sentence,aspect_opinion_pairs)
        if result != "[]": 
            return result
    return "[]"  

def clean_faulty_outputs(result):
    if "aspect" in result and "opinion" in result and "sentiment" in result:
        return "[]"  
    return result


df["aspect_opinion_sentiment_triples"] = df.apply(
    lambda row: clean_faulty_outputs(process_with_retry(row["sentence"],row["aspect_opinion_pairs"])), axis=1
)


df.to_csv("filtering_output.csv", encoding="utf-8", index=False)

print("Aspect-based sentiment analysis enhancement complete! Check 'filtering_output.csv'.")
