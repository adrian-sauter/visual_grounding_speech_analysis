import Levenshtein
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
from jiwer import wer
from tqdm import tqdm
from functools import partial
from tqdm.contrib.concurrent import process_map
import scipy
from collections import defaultdict

# Code partially taken from Choi et al. (2024): https://github.com/juice500ml/phonetic_semantic_probing/tree/24b85b648c6512d9fe4df4139c546482080fef4c
# Copyright (c) 2022, Puyuan Peng All rights reserved.


def cos_sim(f1, f2):
    f1 = np.array(f1)
    f2 = np.array(f2)
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))


def mean_confidence_interval(data, confidence=0.95):
    # Obtained from https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def get_embedding_for_word(word, df, embd_type, layer=None):
    if layer:
        return df[df['text']==word][embd_type].values[0][layer]
    else:
        return df[df['text']==word][embd_type].values[0]
    
def get_semantic_category_map(df_w2v2, df_fast_vgs_plus, sem_categories):
    cos_sims_w2v2 = []
    cos_sims_fast_vgs_plus = []
    words = []
    for layer_idx in range(12):
        word_pairs = set()
        print(f'########### Layer {layer_idx} ###########')
        layer = f'layer_{layer_idx}'
        layer_dists_w2v2 = []
        layer_dists_fast_vgs_plus = []
        for cat, data in sem_categories.items():
            speakers_counter = defaultdict(int)
            print(f'########### Category {cat} ###########')
            for word_i in data['words']:
                for word_j in data['words']:
                    if word_i == word_j or tuple(sorted((word_i, word_j))) in word_pairs:
                        continue
                    word_pairs.add(tuple(sorted((word_i, word_j))))
                    sub_df_w2v2_i = df_w2v2[df_w2v2['text'] == word_i]
                    sub_df_fast_vgs_plus_i = df_fast_vgs_plus[df_fast_vgs_plus['text'] == word_i]

                    sub_df_w2v2_j = df_w2v2[df_w2v2['text'] == word_j]
                    sub_df_fast_vgs_plus_j = df_fast_vgs_plus[df_fast_vgs_plus['text'] == word_j]

                    if len(sub_df_w2v2_i) == 0 or len(sub_df_w2v2_j) == 0:
                        continue
                    if word_i not in words:
                        words.append(word_i)
                    for idx_i, row_i in sub_df_w2v2_i.iterrows():
                        for idx_j, row_j in sub_df_w2v2_j.iterrows():
                            layer_dists_w2v2.append(cos_sim(row_i['w2v2_embeddings'][layer], row_j['w2v2_embeddings'][layer]))

                    for idx_i, row_i in sub_df_fast_vgs_plus_i.iterrows():
                        for idx_j, row_j in sub_df_fast_vgs_plus_j.iterrows():
                            layer_dists_fast_vgs_plus.append(cos_sim(row_i['fast_vgs_plus_embeddings'][layer], row_j['fast_vgs_plus_embeddings'][layer]))
        
        cos_sims_w2v2.append(np.mean(layer_dists_w2v2))
        cos_sims_fast_vgs_plus.append(np.mean(layer_dists_fast_vgs_plus))
    


def get_random_baselines(df_subset, embd_type):
    baselines = []
    for layer_idx in range(12):
        layer = f'layer_{layer_idx}'
        embeddings = [get_embedding_for_word(word, df_subset, embd_type, layer) for word in df_subset['text']]
        shuffled_embeddings = np.random.permutation(embeddings)
        sim = cosine_similarity(embeddings, shuffled_embeddings)
        baselines.append(sim.mean())
    return np.array(baselines)


def phonetic_dist(x: list[str], y: list[str]):
    ref, hyp = (x, y) if len(x) > len(y) else (y, x)
    return wer(reference=" ".join(ref), hypothesis=" ".join(hyp))

def get_homophone(df, indices, text2phones, synonym_map, word, threshold = 0.4):
    homophones = []
    for index in indices:
        row = df.loc[index]
        if (row['text'] != word) and \
            (row['text'] not in synonym_map.get(word, set())) and \
            (word not in synonym_map.get(row.text, set())):
            if 0.0 < phonetic_dist(text2phones[word], row.phones) <= threshold:
                homophones.append(row.text)
    return set(homophones)


def get_homophone_map(words, synonym_map, df, text2phones):
    """
    Create a mapping of words to their homophones.

    Args:
        words (list): List of words to find homophones for.
        synonym_map (dict): Dictionary mapping words to their synonyms.
        df (pd.DataFrame): DataFrame with 'text' and 'phones' columns.
        text2phones (dict): Dictionary mapping words to their phonetic transcriptions.

    Returns:
        dict: A dictionary where keys are words and values are sets of homophones.
    """
    indices = df.reset_index().groupby(["text"])["index"].min()
    homophone_map = {}

    for word in tqdm(words, desc="Finding homophones"):
        homophones = get_homophone(df, indices, text2phones, synonym_map, word)
        if homophones:
            homophone_map[word] = homophones

    return homophone_map


def get_synonym_map(words, df, text2phones, threshold=0.4):
    synonym_map = {}
    for index in tqdm(df.reset_index().groupby(["text"])["index"].min()):
        row = df.loc[index]
        synonyms = []
        for s in row.synonyms:
            if threshold < 0:
                synonyms.append(s)
            else:
                if (s in words) and (phonetic_dist(row.phones, text2phones[s]) > threshold):
                    synonyms.append(s)
        synonyms = set(synonyms).intersection(words)
        if len(synonyms) > 0:
            synonym_map[row.text] = synonyms
    return synonym_map


def compute_similarity(indices, df, embd_type):
    similarities = []
    for layer_idx in range(12):
        layer = f'layer_{layer_idx}'
        embeddings = [get_embedding_for_word(word, df, embd_type, layer) for word in df['text'].iloc[indices]]
        sim = cosine_similarity(embeddings)
        similarities.append(sim)
    return similarities


def _random_sampler(df, wordmap):
    l_indices = df.index.to_numpy()
    r_indices = l_indices.copy()
    np.random.default_rng(seed=42).shuffle(r_indices)
    for l, r in zip(l_indices, r_indices):
        if l != r:
            yield l, r

def _synonym_sampler(df, wordmap):
    l_indices = df.index.to_numpy()
    for l in l_indices:
        syn = set(wordmap["synonym_map"].get(df.loc[l].text, set()))
        for r in df[df.text.isin(syn)].index:
            if l != r:
                yield l, r

def _homophone_sampler(df, wordmap):
    l_indices = df.index.to_numpy()
    for l in l_indices:
        hom = wordmap["homophone_map"].get(df.loc[l].text, set())
        for r in df[df.text.isin(hom)].index:
            if l != r:
                yield l, r

def _speaker_sampler(df, wordmap):
    for spk in df.speaker.unique():
        l_indices = df[df.speaker == spk].index.to_numpy()
        r_indices = l_indices.copy()
        np.random.default_rng(seed=42).shuffle(r_indices)
        for l, r in zip(l_indices, r_indices):
            if l != r:
                yield l, r

def _same_word_sampler(df, wordmap):
    for word in df.text.unique():
        l_indices = df[df.text == word].index.to_numpy()
        r_indices = l_indices.copy()
        np.random.default_rng(seed=42).shuffle(r_indices)
        for l, r in zip(l_indices, r_indices):
            if l != r:
                yield l, r


samplers = {
    "random": _random_sampler,
    "synonym": _synonym_sampler,
    "homophone": _homophone_sampler,
    "speaker": _speaker_sampler,
    "same_word": _same_word_sampler,
}