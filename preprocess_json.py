import os
import json
import joblib
import numpy as np
import pandas as pd
from fastembed import TextEmbedding

# Load a lightweight ONNX-based embedding model (downloads once, ~50MB, no torch needed)
print("Loading embedding model...")
model = TextEmbedding("BAAI/bge-small-en-v1.5")

jsons = os.listdir("jsons")  # List all the jsons
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating embeddings for {json_file}")

    texts = [c["text"] for c in content["chunks"]]
    embeddings = list(model.embed(texts))  # returns a generator, convert to list

    for i, chunk in enumerate(content["chunks"]):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = np.array(embeddings[i])
        chunk_id += 1
        my_dicts.append(chunk)

df = pd.DataFrame.from_records(my_dicts)
joblib.dump(df, "embeddings.joblib")
print(f"✅ Saved embeddings.joblib with {len(df)} chunks.")

