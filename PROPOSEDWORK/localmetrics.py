import pandas as pd
import numpy as np
import re
from sklearn.metrics import hamming_loss


def extract_aos_from_actual(row):
    """
    Convert actual row into a set of A-O-S triples.
    """
    aspects = [x.strip() for x in str(row['span']).split(",") if x.strip()]
    opinions = [x.strip() for x in str(row['opinion']).split(",") if x.strip()]
    sentiments = [x.strip() for x in str(row['sentiment']).split(",") if x.strip()]
    aos_set = set(sorted(zip(aspects, opinions, sentiments)))
    return aos_set

def extract_aos_from_pred(pred_str):
    """
    Extract A-O-S triples from the predicted output.
    """
    a_match = re.search(r"aspect detected:\s*(.*?)\s*##", pred_str)
    o_match = re.search(r"opinion detected:\s*(.*?)\s*##", pred_str)
    s_match = re.search(r"sentiment detected:\s*(.*)", pred_str)
    
    if a_match and o_match and s_match:
        aspects = [x.strip() for x in a_match.group(1).split(",") if x.strip()]
        opinions = [x.strip() for x in o_match.group(1).split(",") if x.strip()]
        sentiments = [x.strip() for x in s_match.group(1).split(",") if x.strip()]
        aos_set = set(sorted(zip(aspects, opinions, sentiments)))
        return aos_set
    else:
        return set()

df_actual = pd.read_csv("actual.csv")
df_actual = df_actual.apply(lambda col: col.map(lambda x: x.lower().strip() if isinstance(x, str) else x))
df_actual['aos'] = df_actual.apply(extract_aos_from_actual, axis=1)


df_pred = pd.read_csv('predicted_annotations.csv')
df_pred = df_pred.apply(lambda col: col.map(lambda x: x.lower().strip() if isinstance(x, str) else x))
df_pred['aos'] = df_pred['prediction'].apply(extract_aos_from_pred)


df_merged = df_actual[['text', 'aos']].merge(df_pred[['text', 'aos']], on='text', suffixes=('_actual', '_pred'))

global_triples = sorted(set().union(*df_merged['aos_actual']).union(*df_merged['aos_pred']))
triple_to_idx = {triple: i for i, triple in enumerate(global_triples)}

def aos_to_vector(aos_set):
    vec = [0] * len(global_triples)
    for triple in aos_set:
        if triple in triple_to_idx:
            vec[triple_to_idx[triple]] = 1
    return np.array(vec)

df_merged['vector_actual'] = df_merged['aos_actual'].apply(aos_to_vector)
df_merged['vector_pred'] = df_merged['aos_pred'].apply(aos_to_vector)

actual_vectors = np.stack(df_merged['vector_actual'].values)
pred_vectors = np.stack(df_merged['vector_pred'].values)


TP = np.sum(np.logical_and(actual_vectors == 1, pred_vectors == 1))
TN = np.sum(np.logical_and(actual_vectors == 0, pred_vectors == 0))
FP = np.sum(np.logical_and(actual_vectors == 0, pred_vectors == 1))
FN = np.sum(np.logical_and(actual_vectors == 1, pred_vectors == 0))

precision = TP / (TP + FP) if (TP + FP) > 0 else 0  
recall = TP / (TP + FN) if (TP + FN) > 0 else 0     
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
mcc = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))) \
      if ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) > 0 else 0
hamming = hamming_loss(actual_vectors, pred_vectors)
fdr = FP / (FP + TP) if (FP + TP) > 0 else 0

print("\n=== Evaluation Metrics ===")
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"Precision: {precision:.4f} (TP / (TP + FP))")
print(f"Recall: {recall:.4f} (TP / (TP + FN))")
print(f"F1 Score: {f1:.4f} (2 * (Precision * Recall) / (Precision + Recall))")
print(f"MCC: {mcc:.4f} (((TP * TN) - (FP * FN)) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)))")
print(f"Hamming Loss: {hamming:.4f} ((FP + FN) / total samples)")
print(f"False Discovery Rate: {fdr:.4f} (FP / (FP + TP))")

empty_preds = df_merged[df_merged['aos_pred'].apply(lambda x: len(x) == 0)]
if not empty_preds.empty:
    print("\n⚠️ Warning: Some predictions are empty!")
    print(empty_preds[['text', 'aos_actual', 'aos_pred']].head())


df_actual[['text', 'aos']].to_csv("actual_aos.csv", index=False)

df_pred[['text', 'aos']].to_csv("predicted_aos.csv", index=False)

print("\n✅ A-O-S files saved: 'actual_aos.csv' and 'predicted_aos.csv'")
