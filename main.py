import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import ClapModel, ClapProcessor, AutoTokenizer
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict

# Initialize CLAP model, processor, and tokenizer
model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
tokenizer = AutoTokenizer.from_pretrained("laion/larger_clap_music_and_speech")

app = FastAPI()

# Mount static directories
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/audio", StaticFiles(directory="freesound_library"), name="audio")

class TextQuery(BaseModel):
    query: str

def get_text_embedding(text_query):
    text_data = tokenizer([text_query], padding=True, return_tensors="pt")
    text_embed = model.get_text_features(**text_data)
    return text_embed.detach().numpy()[0]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_nearest_audio(text_query, output_file, top_k=5) -> List[Dict]:
    data = np.load(output_file)
    embeddings = data['embeddings']
    file_paths = data['file_paths']
    timestamps = data['timestamps']

    text_embedding = get_text_embedding(text_query)

    similarities = np.array([cosine_similarity(text_embedding, emb) for emb in embeddings])

    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    nearest_files = [file_paths[i] for i in top_k_indices]
    nearest_timestamps = [timestamps[i] for i in top_k_indices]
    nearest_similarities = [similarities[i] for i in top_k_indices]

    results = []
    for i in range(top_k):
        results.append({
            "file": nearest_files[i],
            "timestamp": float(nearest_timestamps[i]),  # Ensure timestamp is a float
            "similarity": float(nearest_similarities[i])  # Ensure similarity is a float
        })
    return results

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("static/index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/search")
def search_audio(text_query: TextQuery):
    try:
        results = search_nearest_audio(text_query.query, 'precomputed_embeddings_segments.npz', top_k=5)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
