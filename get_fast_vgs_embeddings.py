import sys
import os
import pickle
import argparse
import torch
from fast_vgs_models import fast_vgs, w2v2_model
import soxr
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_model(model_path):
    with open(f"{model_path}/args.pkl", "rb") as f:
        model_args = pickle.load(f)

    if not hasattr(model_args, "trim_mask"):
        model_args.trim_mask = True
        print(f'args.trim_mask set to TRUE')
    
    weights = torch.load(os.path.join(model_path, "best_bundle.pth"))

    model = w2v2_model.Wav2Vec2Model_cls(model_args)

    model.carefully_load_state_dict(weights['dual_encoder']) # will filter out weights that don't belong to w2v2

    print(f'##########################')
    print(f'Model loaded successfully!')
    print(f'##########################')

    return model


def process_audio(file_path, sampling_rate=16000):
    """Load and preprocess audio to match Wav2Vec 2.0 input requirements."""
    audio, sr = sf.read(file_path)
    if sr != sampling_rate:
        audio = soxr.resample(audio, sr, sampling_rate)
    return (audio - np.mean(audio)) / np.std(audio) # Normalize audio

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
    model = load_model(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load the dataset dataframe
    df = pd.read_pickle(args.df_path)
    print(f'df loaded...')
    df[f"{args.model_name}_embeddings"] = None

    # Create output directory
    output_dir = os.path.join(args.output_path, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f'output_dir created...')

    # Iterate over the dataframe rows
    for _, row in tqdm(df.iterrows(), desc="Processing Audio Files"):
        #file_path = row["file_path"]
        file_path = row['path']
        start = row.get("start", None)  
        #end = row.get("end", None)  
        end = row.get('finish', None)

        if not file_path.endswith((".wav", ".flac")):
            continue

        audio = process_audio(file_path)

        if args.slice and start is not None and end is not None:  # Audio slicing
            audio = audio[int(start * 16000): int(end * 16000)]

        audio = torch.FloatTensor(audio).unsqueeze(0).to(device)

        if args.model_name == 'fast_vgs_plus':
            num_layers = 12
        elif args.model_name == 'fast_vgs':
            num_layers = 8
        else:
            raise ValueError(f"Unsupported model name: {args.model_name}")
        
        embeddings_metadata = {
            "audio_file": file_path,
            "layer_embeddings": {}
        }
        
        for layer in range(num_layers):
            with torch.no_grad():
                layer_embeddings = model(source=audio, padding_mask=None, mask=False, features_only=True, superb=False, tgt_layer=layer)['layer_feats']
                if not args.slice:
                    start_index = _get_index(start, layer_embeddings.shape[1])
                    end_index = _get_index(end, layer_embeddings.shape[1])
                    layer_embeddings = layer_embeddings[:, start_index:end_index, :]
            pooled_embeddings = mean_pooling(layer_embeddings)
            embeddings_metadata["layer_embeddings"][f"layer_{layer}"] = pooled_embeddings.squeeze(0).cpu().numpy()

        # Save embeddings to the dataframe
        df.at[_, f"{args.model_name}_embeddings"] = embeddings_metadata["layer_embeddings"]

    print(f'embeddings extracted...')
    # Save the dataframe with embeddings
    df.to_pickle(os.path.join(output_dir, f"{args.model_name}_{args.experiment_name}.pkl"))
    print(f"Embeddings saved in {output_dir}")


if __name__ == "__main__":
    print(f'############')
    print(f'CODE STARTED')
    print(f'############')
    # Argument parser
    parser = argparse.ArgumentParser(description="Extract embeddings from model")
    parser.add_argument("--model_path", type=str, default=False, help="Path to model weights and args.")
    parser.add_argument("--model_name", type=str, default=False, help="Name of the model used to extract the embeddings.")
    parser.add_argument("--df_path", type=str, required=True, help="Path to the dataset dataframe.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment for organizing outputs.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the extracted embeddings.")
    parser.add_argument("--slice", type=bool, default=False, help="Slice input by word boundary.")

    args = parser.parse_args()
    print(f'args loaded: {args}')
    extract_embeddings(args)
    print(f'##############')
    print(f'CODE COMPLETED')
    print(f'##############')