import os
import joblib
import numpy as np
import streamlit as st
from groq import Groq
from fastembed import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_RESULTS = 5

# --- Page config ---
st.set_page_config(
    page_title="Lecture AI",
    page_icon="🎓",
    layout="centered"
)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Dark background */
.stApp {
    background-color: #0f1117;
    color: #e8e8e8;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Hero header */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.03em;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #6b7280;
    font-size: 1rem;
    margin: 0;
}
.hero .accent {
    color: #6ee7b7;
}

/* Input area */
.stTextInput > div > div > input {
    background-color: #1a1d27 !important;
    border: 1px solid #2d3142 !important;
    border-radius: 10px !important;
    color: #e8e8e8 !important;
    font-size: 1rem !important;
    padding: 0.8rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6ee7b7 !important;
    box-shadow: 0 0 0 2px rgba(110, 231, 183, 0.15) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #6ee7b7, #3b82f6) !important;
    color: #0f1117 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Answer card */
.answer-card {
    background: #1a1d27;
    border: 1px solid #2d3142;
    border-left: 3px solid #6ee7b7;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.5rem;
    color: #e8e8e8;
    font-size: 0.97rem;
    line-height: 1.7;
}

/* Chunk card */
.chunk-card {
    background: #13151f;
    border: 1px solid #2d3142;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
}
.chunk-meta {
    font-family: 'JetBrains Mono', monospace;
    color: #6ee7b7;
    font-size: 0.78rem;
    margin-bottom: 0.3rem;
}
.chunk-text { color: #9ca3af; }

/* Section label */
.section-label {
    color: #4b5563;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.5rem 0 0.6rem 0;
}

/* Error */
.error-box {
    background: #1f1315;
    border: 1px solid #7f1d1d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #fca5a5;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# --- Load models (cached) ---
@st.cache_resource(show_spinner=False)
def load_resources():
    embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
    df = joblib.load("embeddings.joblib")
    client = Groq(api_key=GROQ_API_KEY)
    return embed_model, df, client


def get_answer(query, embed_model, df, client):
    # Embed the query
    question_embedding = np.array(list(embed_model.embed([query]))[0])

    # Cosine similarity search
    similarities = cosine_similarity(
        np.vstack(df["embedding"]), [question_embedding]
    ).flatten()
    top_idx = similarities.argsort()[::-1][:TOP_RESULTS]
    top_df = df.loc[top_idx]

    # Build prompt
    prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{top_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
    )

    return response.choices[0].message.content, top_df


# --- UI ---
st.markdown("""
<div class="hero">
    <h1>Lecture <span class="accent">AI</span></h1>
    <p>Ask anything about your course — get answers with exact timestamps</p>
</div>
""", unsafe_allow_html=True)

# Load resources
with st.spinner("Loading models..."):
    try:
        embed_model, df, client = load_resources()
    except FileNotFoundError:
        st.markdown('<div class="error-box">⚠️ <strong>embeddings.joblib not found.</strong> Run <code>python preprocess_json.py</code> first.</div>', unsafe_allow_html=True)
        st.stop()

# Query input
query = st.text_input("", placeholder="e.g. Where is Flexbox taught in this course?", label_visibility="collapsed")
ask = st.button("Ask →")

if ask and query.strip():
    with st.spinner("Searching through lectures..."):
        try:
            answer, top_chunks = get_answer(query, embed_model, df, client)

            # Answer
            st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

            # Retrieved chunks
            st.markdown('<div class="section-label">Retrieved Context</div>', unsafe_allow_html=True)
            for _, row in top_chunks.iterrows():
                st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-meta">Video #{row['number']} · {row['title']} · {row['start']:.1f}s – {row['end']:.1f}s</div>
                    <div class="chunk-text">{row['text'].strip()}</div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)

elif ask and not query.strip():
    st.markdown('<div class="error-box">Please type a question first.</div>', unsafe_allow_html=True)