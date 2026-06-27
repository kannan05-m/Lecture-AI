import os
import joblib
import numpy as np
import pandas as pd
from groq import Groq
from fastembed import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv #loads the api key every time we run program from .env 

load_dotenv()

# --- Config ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_RESULTS = 5

# --- Load resources ---
print("Loading embedding model...")
embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")

print("Loading embeddings database...")
df = joblib.load("embeddings.joblib")

client = Groq(api_key=GROQ_API_KEY)


def create_embedding(text: str):
    return np.array(list(embed_model.embed([text]))[0])


def inference(prompt: str) -> str:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# --- Main query loop ---
incoming_query = input("Ask a Question: ")

question_embedding = create_embedding(incoming_query)

similarities = cosine_similarity(
    np.vstack(df["embedding"]), [question_embedding]
).flatten()

max_indx = similarities.argsort()[::-1][:TOP_RESULTS]
top_df = df.loc[max_indx]

prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{top_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''

with open("prompt.txt", "w") as f:
    f.write(prompt)

print("\n🤖 Answer:\n")
response = inference(prompt)
print(response)

with open("response.txt", "w") as f:
    f.write(response)