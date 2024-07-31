import os
import numpy as np
import librosa
from transformers import ClapModel, ClapProcessor

# Initialize CLAP model and processor
model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")

# Function to create audio embedding
def embed_audio(segment, sr):
    inputs = processor(audios=segment, sampling_rate=sr, return_tensors="pt")
    audio_embed = model.get_audio_features(**inputs)
    arr = audio_embed.detach().numpy()
    return arr

def precompute_embeddings(audio_dir, output_file, segment_duration=1):
    all_embeddings = []
    all_timestamps = []
    file_paths = []

    # Walk through the directory
    for root, _, files in os.walk(audio_dir):
        for file_name in files:
            if file_name.endswith(('.wav', '.mp3')):
                file_path = os.path.join(root, file_name)
                print(f"Processing file: {file_path}")
                
                y, sr = librosa.load(file_path, sr=48000)
                segment_samples = int(segment_duration * sr)
                total_segments = len(y) // segment_samples

                for i in range(total_segments):
                    start_sample = i * segment_samples
                    end_sample = start_sample + segment_samples
                    segment = y[start_sample:end_sample]
                    segment_embed = embed_audio(segment, sr)
                    all_embeddings.append(segment_embed)
                    relative_path = os.path.relpath(file_path, audio_dir)
                    file_paths.append(relative_path)
                    timestamps = start_sample / sr
                    all_timestamps.append(timestamps)

    np.savez(output_file, embeddings=np.vstack(all_embeddings), timestamps=all_timestamps, file_paths=file_paths)

# Example usage
if __name__ == "__main__":
    audio_dir = 'freesound_library'
    output_file = 'precomputed_embeddings_segments.npz'
    precompute_embeddings(audio_dir, output_file)
