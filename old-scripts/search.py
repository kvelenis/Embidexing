import numpy as np
from transformers import ClapModel, ClapProcessor, AutoTokenizer

# Initialize CLAP model, processor, and tokenizer
model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
tokenizer = AutoTokenizer.from_pretrained("laion/larger_clap_music_and_speech")

# Function to get text embedding
def get_text_embedding(text_query):
    text_data = tokenizer([text_query], padding=True, return_tensors="pt")
    text_embed = model.get_text_features(**text_data)
    return text_embed.detach().numpy()[0]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_nearest_audio(text_query, output_file, top_k=5):
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
            "timestamp": nearest_timestamps[i],
            "similarity": nearest_similarities[i]
        })
        print(f"Nearest audio segment {i+1}:")
        print(f"File: {nearest_files[i]}, Timestamp: {nearest_timestamps[i]}, Similarity: {nearest_similarities[i]}")

    return results

# Example usage
if __name__ == "__main__":
    text_query = "dog"
    output_file = 'precomputed_embeddings_segments.npz'
    search_nearest_audio(text_query, output_file, top_k=5)
