import pandas as pd
from setfit import AbsaModel

model = AbsaModel.from_pretrained(
        r"D:\ASPECT BASED SENTIMENT ANALYSIS\EXISTINGWORK\models\models\setfit-absa-model-aspect",
        r"D:\ASPECT BASED SENTIMENT ANALYSIS\EXISTINGWORK\models\models\setfit-absa-model-polarity",
    )

file_path = "data.csv" 
df = pd.read_csv(file_path)
predictions = model.predict(df["text"])
df["predicted_label"] = predictions 

df_eval = pd.read_csv("data.csv")

df_eval["predicted_label"] = predictions  

df_grouped = df_eval.groupby(["id", "text"], sort=False).apply(lambda x: x[["span", "predicted_label"]].values.tolist()).reset_index()

df_grouped.columns = ["id", "text", "predicted_aspect_sentiment_pairs"]
df_grouped.to_csv("data2.csv", index=False)
