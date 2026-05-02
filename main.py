from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load data at startup
df = pd.read_csv("embeddings_clean.csv")
embeddings = np.load("embeddings.npy").astype("float32")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

class Query(BaseModel):
    text: str
    top_n: int = 10

@app.get("/biographies")
def biographies():
    return df[["name", "x", "y", "area", "profession", "nationality"]].to_dict(orient="records")

@app.post("/search")
def search(query: Query):
    q_emb = model.encode(query.text, normalize_embeddings=True).astype("float32")
    scores = embeddings @ q_emb
    top_idx = np.argsort(scores)[::-1][:query.top_n]
    results = df.iloc[top_idx][["name", "summary", "area", "profession"]].copy()
    results["similarity"] = scores[top_idx].round(4)
    return results.to_dict(orient="records")


from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory=".", html=True), name="static")

