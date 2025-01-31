import os
import argparse
import torch
#import librosa
import soxr
import numpy as np
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import soundfile as sf


def process_audio(file_path, sampling_rate=16000):
    """Load and preprocess audio to match Wav2Vec 2.0 input requirements."""
    audio, sr = sf.read(file_path)
    if sr != sampling_rate:
        audio = soxr.resample(audio, sr, sampling_rate)
    return audio  # normalization not required, happens later in w2v2 pipeline


def mean_pooling(hidden_states, attention_mask=None):
    """Perform mean pooling over the hidden states."""
    mean_pooled = torch.mean(hidden_states, dim=1)
    return mean_pooled

def _get_index(second, num_frames):
    index = int(second / 2 * 16000) // 320
    if index >= num_frames:
        index = num_frames - 1
    return index

def extract_embeddings(args):
    """Extract embeddings from Wav2Vec 2.0 model with metadata."""
    print(f'extracting embeddings ...')
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load the dataset dataframe
    df = pd.read_pickle(args.df_path)  
    print(f'df loaded...')
    df["w2v2_embeddings"] = None  

    # Create output directory
    output_dir = os.path.join(args.output_path, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f'output_dir created...')

    # Iterate over the dataframe rows
    for _, row in tqdm(df.iterrows(), desc="Processing Audio Files"):
        #file_path = row["file_path"]  # Assuming the dataframe has a column 'file_path'
        file_path = row['path']
        start = row.get("start", None)  
        #end = row.get("end", None)  
        end = row.get('finish', None)

        if not file_path.endswith((".wav", ".flac")):
            continue

        audio = process_audio(file_path)

        if args.slice and start is not None and end is not None:
            audio = audio[int(start * 16000): int(end * 16000)]  # Slice based on start and end times
        
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=False)

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[1:]  # Tuple of all layer outputs (excluding projection layer)

        embeddings_metadata = {
            "audio_file": file_path,
            "layer_embeddings": {}
        }

        for i, layer_embeddings in enumerate(hidden_states):
            if not args.slice:
                start_index = _get_index(start, layer_embeddings.shape[1])
                end_index = _get_index(end, layer_embeddings.shape[1])
                layer_embeddings = layer_embeddings[:, start_index:end_index, :]
            pooled_embeddings = mean_pooling(layer_embeddings)
            embeddings_metadata["layer_embeddings"][f"layer_{i}"] = pooled_embeddings.squeeze(0).cpu().numpy()

        df.at[_, "w2v2_embeddings"] = embeddings_metadata["layer_embeddings"]

    print(f'embeddings extracted...')
    df.to_pickle(os.path.join(output_dir, f"w2v2_{args.experiment_name}.pkl"))
    print(f"Embeddings saved in {output_dir}")



if __name__ == "__main__":
    print(f'started running...')
    parser = argparse.ArgumentParser(description="Extract embeddings from Wav2Vec 2.0 model")
    parser.add_argument("--df_path", type=str, required=True, help="Path to the dataset dataframe.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment for organizing outputs.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the extracted embeddings.")
    parser.add_argument("--slice", type=bool, default=False, help="Slice input by word boundary.")
    parser.add_argument("--save_as_tensor", type=bool, default=False, help="Save embeddings as tensor.")

    args = parser.parse_args()
    print(f'args loaded: {args}')
    extract_embeddings(args)
    print(f'##############')
    print(f'CODE COMPLETED')
    print(f'##############')