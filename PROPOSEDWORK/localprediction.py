import os
import gc
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

save_dir = "models" 
tokenizer = AutoTokenizer.from_pretrained(save_dir)
inference_model = AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=torch.float16)
inference_model.to("cuda:0")  
inference_model.eval()


def process_prompt(user_prompt, model):
    print('Prediction running')
    text_input = f"### Human: {user_prompt} ###"
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=int(len(tokenizer.encode(user_prompt)) * 3.5),
        device=0  
    )
    return pipe(text_input)


input_file = "50sentences.csv"  
output_file = "predicted_annotations.csv" 


df = pd.read_csv(input_file)
df["text"] = df["text"].str.lower() 


predictions = []


for idx, row in df.iterrows():
    sentence = row["text"]

    result = process_prompt(sentence, inference_model)
  
    generated_text = result[0]["generated_text"]
    predictions.append(generated_text)


df["prediction"] = predictions


df.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")
