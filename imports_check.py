import pandas as pd

df = pd.read_pickle("Dataset/affectnet/df_with_image_vad_embedding.pkl")
print(df.columns)
print(df.head())
