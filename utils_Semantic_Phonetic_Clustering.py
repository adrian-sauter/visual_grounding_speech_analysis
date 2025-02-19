import numpy as np
import torch
import torch.nn.functional as F
import Levenshtein
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
import random
from sklearn.cluster import DBSCAN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from tqdm import tqdm
import scipy

def convert_embeddings_to_arrays(df):
    for idx, row in df.iterrows():
        w2v2_embeddings = row['w2v2_embeddings']
        fast_vgs_plus_embeddings = row['fast_vgs_plus_embeddings']

        for layer in w2v2_embeddings:
            w2v2_embeddings[layer] = np.array(w2v2_embeddings[layer])
        
        for layer in fast_vgs_plus_embeddings:
            fast_vgs_plus_embeddings[layer] = np.array(fast_vgs_plus_embeddings[layer])
        
        df.at[idx, 'w2v2_embeddings'] = w2v2_embeddings
        df.at[idx, 'fast_vgs_plus_embeddings'] = fast_vgs_plus_embeddings

    return df

def load_glove_embeddings(glove_file_path):
    """
    Loads GloVe embeddings from a file into a dictionary.

    Args:
        glove_file_path (str): Path to the GloVe embeddings file.

    Returns:
        dict: A dictionary where the key is the word and the value is its embedding.
    """
    glove_embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.strip().split()
            word = split_line[0]
            embedding = np.array(list(map(float, split_line[1:])))
            glove_embeddings[word] = embedding
    return glove_embeddings


def convert_np_to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_np_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_np_to_native(item) for item in obj]
        else:
            return obj

#######################
###### BASELINES ######
#######################
def compute_baseline(embeddings, num_shuffles=10):
    """
    Computes a robust baseline similarity by averaging over multiple shuffles of the embeddings.

    Args:
        embeddings (list): List of embeddings.
        num_shuffles (int): Number of times to shuffle the embeddings.

    Returns:
        float: Average baseline similarity across multiple shuffles.
    """
    similarities = []

    for _ in range(num_shuffles):
        shuffled_embeds = random.sample(embeddings, len(embeddings))
        similarity = np.mean(cosine_similarity(embeddings, shuffled_embeds))
        similarities.append(similarity)

    return np.mean(similarities)

def compute_phonetic_baseline(words, pronunc_df, num_runs=10):
    """
    Computes the phonetic baseline by comparing the pairwise phonetic distance between
    the original list of words and a shuffled list of words, only including words with
    available ARPAbet transcriptions. The computation is repeated multiple times to 
    increase the robustness of the baseline.

    Args:
        words (list): List of words to compute similarity for.
        pronunc_df (pd.DataFrame): DataFrame containing two columns: 'Item' (word) and 'Pronunciation' (ARPAbet transcription).
        num_runs (int): Number of runs to repeat the computation.

    Returns:
        float: Average phonetic baseline similarity score across multiple runs.
    """
    valid_words = [word for word in words if word in pronunc_df['Item'].values]

    if len(valid_words) < 2:
        raise ValueError("Not enough valid words with pronunciations for baseline computation.")

    word_pronunciations = pronunc_df[pronunc_df['Item'].isin(valid_words)].set_index('Item')['Pronunciation']

    all_phonetic_distances = []

    for _ in range(num_runs):
        words_shuffled = random.sample(valid_words, len(valid_words))

        phonetic_distances = []
        for word1, word2 in zip(valid_words, words_shuffled):
            pron1 = word_pronunciations[word1]
            pron2 = word_pronunciations[word2]
            distance = Levenshtein.distance(pron1, pron2)
            phonetic_distances.append(distance)

        all_phonetic_distances.append(np.mean(phonetic_distances))

    phonetic_baseline = np.mean(all_phonetic_distances)

    return phonetic_baseline

def get_embedding_for_word(word, df, embd_type, layer=None):
    if layer:
        return df[df['words']==word][embd_type].values[0][layer]
    else:
        return df[df['words']==word][embd_type].values[0]

def get_matches(word_list, wav_list):
    word_list = [item.lower() for item in word_list]
    matches = [item for item in word_list if item in wav_list]
    return matches

def get_representative_words(words, df, percentile=90):
    """
    Find the words that most strongly represent a category based on average similarity.

    Args:
        words (list): A list of words in a specific category.
        glove_embedding_dict (dict): A dictionary of GloVe embeddings.
        percentile (float): Percentile threshold (0-100) for selecting representative words.
    
    Returns:
        list: A list of the most representative words for the category.
    """
    embeddings = []
    for word in words:
        embeddings.append(get_embedding_for_word(word, df, 'glove_embeddings'))
    sim_matrix = cosine_similarity(embeddings)
    avg_similarities = np.mean(sim_matrix, axis=1)
    
    similarity_threshold = np.percentile(avg_similarities, percentile)
    
    representative_words = [
        word for similarity, word in zip(avg_similarities.tolist(), words) 
        if similarity >= similarity_threshold
    ]
    
    return representative_words

def get_wordlist_meta_data(word_list, conc_abs_df, pronunc_df):
    concreteness_scores = conc_abs_df[
        conc_abs_df['Word'].str.lower().isin(word_list)
    ]['Conc.M']
    avg_score = np.mean(concreteness_scores)
    std_dev = np.std(concreteness_scores)

    print(f'Average concreteness score: {avg_score:.2f}')
    print(f'Standard deviation (concreteness score): {std_dev:.2f}')

    avg_sim, std_stim = compute_avg_phonetic_similarity(word_list, pronunc_df)

    print(f'Average levenshtein similarity score: {avg_sim:.2f}')
    print(f'Standard deviation (levensthein similarity score): {std_stim:.2f}')

    return {"avg_concreteness_score": avg_score, "std_concreteness_score": std_dev, 'avg_phon_similiarity_score': avg_sim, 'std_phon_similarity_score': std_stim}


def compute_avg_phonetic_similarity(words, df, baseline=None):
    """
    Computes the average pairwise phonetic similarity for a given list of words.

    Args:
        words (list): List of words to compute similarity for.
        df (pd.DataFrame): DataFrame containing two columns: 'Item' (word) and 'Pronunciation' (ARPAbet transcription).
    
    Returns:
        float: Average pairwise phonetic similarity (normalized Levenshtein similarity).
    """
    word_pronunciations = df[df['Item'].isin(words)].set_index('Item')['Pronunciation']

    if len(word_pronunciations) < len(words):
        missing_words = set(words) - set(word_pronunciations.index)
        raise ValueError(f"The following words are missing pronunciations: {missing_words}")

    word_pairs = combinations(word_pronunciations.items(), 2)

    similarities = []
    for (word1, pron1), (word2, pron2) in word_pairs:
        distance = Levenshtein.distance(pron1, pron2)
        if baseline:
            distance = distance - baseline
        max_len = max(len(pron1), len(pron2))
        similarity = 1 - (distance / max_len)
        similarities.append(similarity)
    
    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)

    return avg_similarity, std_similarity

def compute_phonetic_similarity_matrix(words, pronunc_df):
    """
    Computes the pairwise phonetic similarity matrix for a given list of words using a DataFrame.

    Args:
        words (list): List of words to compute similarity for.
        df (pd.DataFrame): DataFrame containing two columns: 'Item' (word) and 'Pronunciation' (ARPAbet transcription).
    
    Returns:
        np.ndarray: 2D matrix where the (i, j) element contains the phonetic similarity
                    between words[i] and words[j].
    """
    word_pronunciations = pronunc_df[pronunc_df['Item'].isin(words)].set_index('Item')['Pronunciation']

    missing_words = set(words) - set(word_pronunciations.index)
    if missing_words:
        raise ValueError(f"The following words are missing pronunciations: {missing_words}")

    num_words = len(words)
    similarity_matrix = np.zeros((num_words, num_words))

    # Compute pairwise phonetic similarities
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i <= j: # Only compute the upper triangle (given symmetry)
                pron1 = word_pronunciations[word1]
                pron2 = word_pronunciations[word2]
                
                # Compute Levenshtein distance
                distance = Levenshtein.distance(pron1, pron2)
                max_len = max(len(pron1), len(pron2))
                similarity = 1 - (distance / max_len)
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Ensure symmetry

    return similarity_matrix

def compute_silhouette_coefficient(embeddings, labels):
    """
    Compute the silhouette coefficient for a set of embeddings and their corresponding labels.

    Args:
        embeddings (list or np.array): A list or array of word embeddings.
        labels (list or np.array): A list or array of labels corresponding to the embeddings.
    
    Returns:
        float: The silhouette score for the embeddings and labels.
    """
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    distances = cosine_distances(embeddings)
    
    score = silhouette_score(distances, labels, metric="precomputed")
    
    return score

def compute_layerwise_similarities_for_list(df, word_list, col1='w2v2_embeddings', col2='fast_vgs_plus_embeddings', baseline=None):
    """
    Compute the cosine similarities between the 768-dimensional embeddings of W2V2 and Fast-VGS-Plus 
    for a list of words in the dataframe, separated by layers, and return average similarity per layer.

    Args:
        df (pd.DataFrame): A dataframe with columns 'words', and two other columns that hold the embeddings for different layers.
                           Each row corresponds to a word and contains dictionaries with embeddings for different layers.
        word_list (list): A list of words for which the cosine similarities will be computed.

    Returns:
        dict: A dictionary with average similarity scores for each layer between two sets of embeddings.
    """
    if baseline is None:
        baseline = 0

    similarity_dict = {}

    is_glove = False

    if col2 == 'glove_embeddings':
        is_glove = True
        col2 = col1
        col1 = 'glove_embeddings'


    filtered_df = df[df['words'].isin(word_list)]

    for _, row in filtered_df.iterrows():
        word_similarities = {}

        for layer_idx in range(12):  # Assuming 12 layers for col1 and col2
            layer_name = f'layer_{layer_idx}'
            if is_glove:
                col1_embd = row[col1]
            else:
                col1_embd = row[col1][layer_name]
            col2_embd = row[col2][layer_name]

            similarity = cosine_similarity(col1_embd.reshape(1, -1), col2_embd.reshape(1, -1)) - baseline

            word_similarities[layer_name] = {
                'similarity': similarity
            }

        similarity_dict[row['words']] = word_similarities

    avg_similarity_per_layer = {}

    for layer in similarity_dict[next(iter(similarity_dict))]:  
        similarities_for_layer = []

        for word_similarities in similarity_dict.values():
            similarities_for_layer.append(word_similarities[layer]['similarity'])

        avg_similarity_per_layer[layer] = np.mean(similarities_for_layer)

    return avg_similarity_per_layer

def compute_semantic_similarity_baseline(df, num_shuffles=10):
    """
    Computes a robust baseline semantic similarity for the dataset by averaging 
    over several shuffles of the word list.

    Args:
        df (pd.DataFrame): DataFrame with word embeddings.
        num_shuffles (int): Number of times to shuffle the word list.

    Returns:
        float: Average pairwise semantic similarity across multiple shuffles.
    """
    words = df['words'].tolist()
    embeddings = [get_embedding_for_word(word, df, 'glove_embeddings') 
            for word in words if get_embedding_for_word(word, df, 'glove_embeddings') is not None]
    all_similarities = []
    
    for _ in range(num_shuffles):
        np.random.shuffle(embeddings)  # Shuffle the embeddings list
        similarities = []
        
        for i, embd1 in enumerate(embeddings):
            for j, embd2 in enumerate(embeddings):
                if i < j:
                    similarity = cosine_similarity(embd1.reshape(1, -1), embd2.reshape(1, -1))
                    similarities.append(similarity)
        
        all_similarities.extend(similarities)  # Collect similarities from each shuffle

    return np.mean(all_similarities)



###############################
####### PHONETIC GROUPS #######
###############################
def get_top_n_similar_words(word, word_list, sim_matrix, n=20, thresh=0.1, above=True):
    """
    Retrieve the top N most similar words for a given word based on the similarity matrix.
    Only keeps words with a similarity above a certain threshold, and includes the word itself.
    """
    if word not in word_list.values:
        raise ValueError(f"'{word}' not found in the word list.")

    word_idx = word_list[word_list == word].index[0]
    similarities = sim_matrix[word_idx]
    
    # Include the word itself, and filter by similarity threshold
    if above:
        top_indices = np.argsort(similarities)[::-1]
        top_indices = top_indices[top_indices != word_idx][:n]
        top_indices = [i for i in top_indices if similarities[i] >= thresh]
    else:
        top_indices = np.argsort(similarities)
        top_indices = top_indices[top_indices != word_idx][:n]
        top_indices = [i for i in top_indices if similarities[i] <= thresh]

    # Include the word itself (it should be the most similar, so we add it back manually)
    top_words = [word] + [(word_list.iloc[i]) for i in top_indices]
    
    return top_words


def get_semantically_dissimilar_words(similar_words, df, n=30, similarity_threshold=0.7, above=False):
    """
    Filters phonetically similar words based on semantic dissimilarity using GloVe embeddings.
    Ensures the selected words are not semantically similar to each other and returns the
    average cosine dissimilarity (1 - cosine similarity) of the final list.
    """
    dissimilar_words = []
    selected_embeddings = []
    for similar_word in similar_words:
        if similar_word in df['words'].values:
            similar_word_embd = df[df['words'] == similar_word]['glove_embeddings'].values[0]
            if similar_word_embd is None:
                continue

            # Calculate similarity with previously selected embeddings
            similarities = [cosine_similarity([similar_word_embd], [embd])[0][0] for embd in selected_embeddings]

            # Check if the word is sufficiently dissimilar from the already selected words
            if above:
                if all(similarity > similarity_threshold for similarity in similarities):
                    dissimilar_words.append(similar_word)
                    selected_embeddings.append(similar_word_embd)
            else:
                if all(similarity < similarity_threshold for similarity in similarities):
                    dissimilar_words.append(similar_word)
                    selected_embeddings.append(similar_word_embd)

            # Stop if enough dissimilar words have been selected
            if len(dissimilar_words) >= n:
                break
    
    avg_cosine_sim = 0
    num_combinations = 0
    for i in range(len(selected_embeddings)):
        for j in range(i + 1, len(selected_embeddings)):
            sim_score = cosine_similarity([selected_embeddings[i]], [selected_embeddings[j]])[0][0]
            avg_cosine_sim += sim_score
            num_combinations += 1
    avg_cosine_sim /= num_combinations if num_combinations > 0 else 1
    return dissimilar_words, avg_cosine_sim


def filter_by_concreteness(word_list, conc_abs_df, pronunc_df, filter_type="concrete", low_threshold_percentile=25, high_threshold_percentile=75):
    """
    Filters the word list to keep either very concrete or very abstract words based on filter_type.
    """
    low_threshold = np.percentile(conc_abs_df['Conc.M'].values, low_threshold_percentile)
    high_threshold = np.percentile(conc_abs_df['Conc.M'].values, high_threshold_percentile)
    word_list = [word for word in word_list if word in conc_abs_df['Word'].values]
    abstract_words = [word for word in word_list if conc_abs_df[conc_abs_df['Word'].str.lower() == word]['Conc.M'].values[0] <= low_threshold]
    concrete_words = [word for word in word_list if conc_abs_df[conc_abs_df['Word'].str.lower() == word]['Conc.M'].values[0] >= high_threshold]

    if not abstract_words and filter_type == "abstract":
        return [], {}
    if not concrete_words and filter_type == "concrete":
        return [], {}

    if filter_type == "concrete":
        selected_words = concrete_words
    elif filter_type == "abstract":
        selected_words = abstract_words
    else:
        raise ValueError("filter_type must be either 'concrete' or 'abstract'.")

    scores = get_wordlist_meta_data(selected_words, conc_abs_df, pronunc_df)
    return selected_words, scores


def is_phonetically_different(df, candidate_words, existing_clusters, phon_sim, threshold=0.4):
    """
    Check if the candidate words are phonetically different from the words in existing clusters.
    """
    for cluster in existing_clusters.values():
        cluster_words = cluster['words']
        for word in candidate_words:
            for cluster_word in cluster_words:
                word_idx = df[df['words'] == word].index[0]
                cluster_word_idx = df[df['words'] == cluster_word].index[0]
                similarity = phon_sim[word_idx, cluster_word_idx]
                if similarity >= threshold:
                    return False
    return True

def create_phonetic_groups(word_pool, phon_sim, df, conc_abs_df, pronunc_df, phonetic_similarity_threshold=0.47, n=100, semantic_similarity_threshold=0.1):
    phon_cat = {}
    num_concrete = 0
    num_abstract = 0
    num_words_chosen = 0
    while num_concrete < 10 or num_abstract < 10:
        if len(word_pool) == 0:
            break  
        word = random.choice(word_pool)
        num_words_chosen += 1
        # 1. Get the top N similar words with a threshold
        similar_words = get_top_n_similar_words(word, df['words'], phon_sim, n=n, thresh=phonetic_similarity_threshold)

        # 2. Filter the similar words by semantic dissimilarity
        dissimilar_words, avg_sim = get_semantically_dissimilar_words(similar_words, df, n=n, similarity_threshold=semantic_similarity_threshold)

        # 3. Apply the concreteness or abstractness filter for both categories
        abstract_words, abstract_scores = filter_by_concreteness([word for word in dissimilar_words], conc_abs_df, pronunc_df, filter_type="abstract")
        concrete_words, concrete_scores = filter_by_concreteness([word for word in dissimilar_words], conc_abs_df, pronunc_df, filter_type="concrete")
        # If the abstract words category has at least 10 words and is unique, add it
        if len(abstract_words) >= 5 and all(w in word_pool for w in abstract_words) and abstract_scores['avg_similiarity_score'] >= 0.75:
            cluster_name = f'abstract_{word}'
            print(f'abstract cluster found: {cluster_name}, {abstract_words}')
            phon_cat[cluster_name] = {
                'words': abstract_words,
                'scores': abstract_scores
            }
            phon_cat[cluster_name]['scores']['average_sem_similarity'] = avg_sim
            # Remove the used words from word_pool
            word_pool = [w for w in word_pool if w not in abstract_words]
            print(f"Abstract cluster added: {cluster_name} with average distance {avg_sim}")
            num_abstract += 1

        # If the concrete words category has at least 10 words and is unique, add it
        if len(concrete_words) >= 5 and all(w in word_pool for w in concrete_words) and concrete_scores['avg_similiarity_score'] >= 0.75:
            cluster_name = f'concrete_{word}'
            print(f'concrete cluster found: {cluster_name}: {concrete_words}')
            phon_cat[cluster_name] = {
                'words': concrete_words,
                'scores': concrete_scores
            }
            phon_cat[cluster_name]['scores']['average_sem_similarity'] = avg_sim
            # Remove the used words from word_pool
            word_pool = [w for w in word_pool if w not in concrete_words]
            print(f"Concrete cluster added: {cluster_name} with average distance {avg_sim}")

            num_concrete += 1
        # Stop if both abstract and concrete clusters have 10 or more categories
        if num_concrete >= 10 and num_abstract >= 10:
            break

    return phon_cat, num_words_chosen

# def filter_clusters_by_semantic_similarity(clusters, df, baseline):
#     """
#     Filters clusters by removing word pairs with semantic similarity above the baseline.
    
#     Args:
#         clusters (dict): Dictionary of clusters (key: cluster ID, value: list of words).
#         df (pd.DataFrame): DataFrame with word embeddings.
#         baseline (float): Baseline semantic similarity.

#     Returns:
#         dict: Filtered clusters with no word pairs above the baseline semantic similarity.
#     """
#     filtered_clusters = {}
    
#     for cluster_id, words in clusters.items():
#         # Create a copy of the cluster to allow modification
#         words = list(words)
#         to_remove = set()

#         for i, word1 in enumerate(words):
#             for j, word2 in enumerate(words):
#                 if i < j:
#                     embd1 = get_embedding_for_word(word1, df, 'glove_embeddings')
#                     embd2 = get_embedding_for_word(word2, df, 'glove_embeddings')
#                     similarity = cosine_similarity([embd1], [embd2])
                    
#                     # If similarity exceeds the baseline, mark one word for removal
#                     if similarity > baseline:
#                         to_remove.add(word2)  # Arbitrarily remove word2; this can be adjusted

#         # Remove marked words from the cluster
#         filtered_words = [word for word in words if word not in to_remove]

#         # Only add non-empty clusters
#         if len(filtered_words) > 1:  # Clusters should have at least two words
#             filtered_clusters[cluster_id] = filtered_words

#     return filtered_clusters

# def filter_by_concreteness(word_list, conc_abs_df, pronunc_df, low_threshold_percentile=25, high_threshold_percentile=75):
#     """
#     Filters the word list to keep both very concrete and very abstract words.
#     Returns the longer list (either concrete or abstract), along with its average and standard deviation of concreteness scores.

#     Args:
#         word_list (list): List of words to filter.
#         conc_abs_df (pd.DataFrame): DataFrame with concreteness scores.
#         high_threshold_percentile (int): The percentile threshold to determine very concrete words.
#         low_threshold_percentile (int): The percentile threshold to determine very abstract words.

#     Returns:
#         list: The longer list of filtered words (either very concrete or abstract).
#         float: Average concreteness score for the selected list.
#         float: Standard deviation of the concreteness score for the selected list.
#     """
        
#     # Calculate the high and low threshold based on the given percentiles
#     low_threshold = np.percentile(conc_abs_df['Conc.M'].values, low_threshold_percentile)
#     high_threshold = np.percentile(conc_abs_df['Conc.M'].values, high_threshold_percentile)
    
#     # Filter for very abstract words (below low threshold)
#     abstract_words = [word for word in word_list if conc_abs_df[conc_abs_df['Word'].str.lower() == word]['Conc.M'].values[0] <= low_threshold]

#     # Filter for very concrete words (above high threshold)
#     concrete_words = [word for word in word_list if conc_abs_df[conc_abs_df['Word'].str.lower() == word]['Conc.M'].values[0] >= high_threshold]
    
#     # Choose the longer list
#     if len(concrete_words) > len(abstract_words):
#         selected_words = concrete_words
#     else:
#         selected_words = abstract_words
    
#     # Calculate average and standard deviation for the selected list
#     scores = get_wordlist_meta_data(selected_words, conc_abs_df, pronunc_df)

#     return selected_words, scores


######################################
####### Silhouette Coefficient #######
#####################################
def get_model_silhouettes(df, category_dict, res_types = 'sem'):

    results = {}

    embds = {}
    
    word_to_label = {word: category for category, data in category_dict.items() for word in data["words"]}

    if res_types == 'sem':
        models = ["w2v2_embeddings", "fast_vgs_plus_embeddings", "reg_bert_embeddings", "vg_bert_embeddings"]
    else:
        models = ["w2v2_embeddings", "fast_vgs_plus_embeddings"]
    
    for model in models:
        results[model] = {'all': {}, "lda": {}, "pca": {}}
        embds[model] = {'all': {}, "lda": {}, "pca": {}}

        for layer in [f"layer_{i}" for i in range(12)]:
            embeddings, labels, words = [], [], []
            for _, row in df.iterrows():
                word = row["words"]
                if word in word_to_label and layer in row[model]:
                    embeddings.append(row[model][layer])
                    labels.append(word_to_label[word])
                    words.append(word)

            embeddings = np.array(embeddings)

            all_silhouette = silhouette_score(embeddings, labels)
            results[model]["all"][layer] = all_silhouette
            embds[model]['all']['layer'] = {word: emb for word, emb in zip(words, embeddings)}
            
            # **LDA** Transformation
            lda = LinearDiscriminantAnalysis()
            lda_projected_embeddings = lda.fit_transform(embeddings, labels)

            # Store LDA projected embeddings per category and per word
            lda_projected_dict = {cat: {} for cat in category_dict}
            for proj_emb, word, category in zip(lda_projected_embeddings, words, labels):
                lda_projected_dict[category][word] = proj_emb  
            
            lda_silhouette = silhouette_score(lda_projected_embeddings, labels)

            results[model]["lda"][layer] = lda_silhouette
            embds[model]['lda'][layer] = {word: emb for word, emb in zip(words, lda_projected_embeddings)}

            # **PCA** Transformation (with 8 principal components, same as LDA)
            pca = PCA(n_components=8)
            pca_projected_embeddings = pca.fit_transform(embeddings)

            # Store PCA projected embeddings per category and per word
            pca_projected_dict = {cat: {} for cat in category_dict}
            for proj_emb, word, category in zip(pca_projected_embeddings, words, labels):
                pca_projected_dict[category][word] = proj_emb  
            
            pca_silhouette = silhouette_score(pca_projected_embeddings, labels)

            results[model]["pca"][layer] = pca_silhouette
            embds[model]['pca'][layer] = {word: emb for word, emb in zip(words, pca_projected_embeddings)}

            print(f'Finished layer {layer}')
        print(f'Finished model {model}')
        
    return results, embds

def get_glove_silhouettes(df, category_dict):

    results = {
        'glove': {'all': {}, "lda": {}, "pca": {}}
    }
    
    embds =  {
        'glove': {'all': {}, "lda": {}, "pca": {}}
    }
    
    word_to_label = {word: category for category, data in category_dict.items() for word in data["words"]}

    embeddings, labels, words = [], [], []
    for _, row in df.iterrows():
        word = row["words"]
        if word in word_to_label:
            embeddings.append(row['glove_embeddings'])
            labels.append(word_to_label[word])
            words.append(word)
    
    embeddings = np.array(embeddings)
    all_silhouette = silhouette_score(embeddings, labels)
    results['glove']["all"] = all_silhouette
    embds['glove']['all'] = {word: emb for word, emb in zip(words, embeddings)}

    # **LDA** Transformation
    lda = LinearDiscriminantAnalysis()
    lda_projected_embeddings = lda.fit_transform(embeddings, labels)

    # Store LDA projected embeddings per category and per word
    lda_projected_dict = {cat: {} for cat in category_dict}
    for proj_emb, word, category in zip(lda_projected_embeddings, words, labels):
        lda_projected_dict[category][word] = proj_emb  
    
    lda_silhouette = silhouette_score(lda_projected_embeddings, labels)

    results['glove']["lda"] = lda_silhouette
    embds['glove']['lda'] = {word: emb for word, emb in zip(words, lda_projected_embeddings)}

    # **PCA** Transformation (with 8 principal components, same as LDA)
    pca = PCA(n_components=8)
    pca_projected_embeddings = pca.fit_transform(embeddings)

    # Store PCA projected embeddings per category and per word
    pca_projected_dict = {cat: {} for cat in category_dict}
    for proj_emb, word, category in zip(pca_projected_embeddings, words, labels):
        pca_projected_dict[category][word] = proj_emb  
    
    pca_silhouette = silhouette_score(pca_projected_embeddings, labels)

    results['glove']["pca"] = pca_silhouette
    embds['glove']['pca'] = {word: emb for word, emb in zip(words, pca_projected_embeddings)}

    print(f'Finished model glove')
    return results, embds


def get_robust_clustering_results(df, categories, analysis_type = 'sem'):
    if analysis_type == 'sem':
        results = {
            'w2v2_embeddings': {
                'all_dims': {f'layer_{i}': [] for i in range(12)},
                'lda': {f'layer_{i}': [] for i in range(12)},
                'pca': {f'layer_{i}': [] for i in range(12)}
            },
            'fast_vgs_plus_embeddings': {
                'all_dims': {f'layer_{i}': [] for i in range(12)},
                'lda': {f'layer_{i}': [] for i in range(12)},
                'pca': {f'layer_{i}': [] for i in range(12)}
            },
            'reg_bert_embeddings': {
                'all_dims': {f'layer_{i}': [] for i in range(12)},
                'lda': {f'layer_{i}': [] for i in range(12)},
                'pca': {f'layer_{i}': [] for i in range(12)}
            },
            'vg_bert_embeddings': {
                'all_dims': {f'layer_{i}': [] for i in range(12)},
                'lda': {f'layer_{i}': [] for i in range(12)},
                'pca': {f'layer_{i}': [] for i in range(12)}
            },
            'glove_embeddings': {
                'all_dims': [],
                'lda': [],
                'pca': []
            }
        }
    else:
        results = {
            'w2v2_embeddings': {
                'all_dims': {f'layer_{i}': [] for i in range(12)},
                'lda': {f'layer_{i}': [] for i in range(12)},
                'pca': {f'layer_{i}': [] for i in range(12)}
            },
            'fast_vgs_plus_embeddings': {
                'all_dims': {f'layer_{i}': [] for i in range(12)},
                'lda': {f'layer_{i}': [] for i in range(12)},
                'pca': {f'layer_{i}': [] for i in range(12)}
            }
        }

    keys_list = list(categories.keys())
    if analysis_type == 'sem':
        models = ['w2v2_embeddings', 'fast_vgs_plus_embeddings', 'reg_bert_embeddings', 'vg_bert_embeddings']
    else:
        models = ['w2v2_embeddings', 'fast_vgs_plus_embeddings']

    for i in tqdm(range(len(keys_list))):  
        keys = [k for j, k in enumerate(keys_list) if j != i]  # exclude one key at a time
        print(f"Iteration {i}, removed key: {keys_list[i]}, remaining keys: {keys}")
        random.shuffle(keys)
        temp_dict = {
            key: categories[key] for key in keys
        }
        if analysis_type == 'sem':
            glove_results = get_glove_silhouettes(df, temp_dict)
            results['glove_embeddings']['all_dims'].append(glove_results[0]['glove']['all'])
            results['glove_embeddings']['lda'].append(glove_results[0]['glove']['lda'])
            results['glove_embeddings']['pca'].append(glove_results[0]['glove']['pca'])
        temp_results = get_model_silhouettes(df, temp_dict, analysis_type)
        for model in models:
            for layer in [f'layer_{i}' for i in range(12)]:
                results[model]['all_dims'][layer].append(temp_results[0][model]['all'][layer])
                results[model]['lda'][layer].append(temp_results[0][model]['lda'][layer])
                results[model]['pca'][layer].append(temp_results[0][model]['pca'][layer])
    
    return results


def mean_confidence_interval(data, confidence=0.95):
    # Obtained from https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h




########################
####### PLOTTING #######
########################
def plot_rsms(words, rsms, titles, title='RSM Comparison', savefig=True, filename="exp1/rsm_comparison.pdf"):
    """
    Plot multiple RSM heatmaps with individual colorbars.

    Args:
        words (list): List of words corresponding to the RSMs.
        rsms (list of np.array): List of RSM matrices to plot.
        titles (list of str): List of titles for each RSM.
        title (str): Title for the overall plot.
        savefig (bool): Whether to save the plot as a file.
        filename (str): File name for saving the plot.
    """
    if len(rsms) != len(titles):
        raise ValueError("The number of RSMs and titles must be the same.")
    
    num_rsms = len(rsms)
    
    plt.figure(figsize=(6 * num_rsms, 6))  
    
    for idx, (rsm, rsm_title) in enumerate(zip(rsms, titles), start=1):
        ax = plt.subplot(1, num_rsms, idx)
        sns.heatmap(
            rsm,
            xticklabels=words,
            yticklabels=words,
            cmap='coolwarm',
            ax=ax,
            cbar=True
        )
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, rotation=45)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
        ax.set_title(rsm_title, fontsize=14)

    plt.tight_layout()
    plt.suptitle(title, fontsize=18, y=1.05)

    if savefig:
        plt.savefig(filename, format="pdf")
        print(f"Plot saved to {filename}")

    plt.show()

def plot_silhouette_coefficients(glove_value=None, wav2vec2_base_values=None, fast_vgs_values=None, reg_bert_values=None, vg_bert_values=None, savefig=True, filename="exp1/silhouette_scores.pdf"):
    plt.figure(figsize=(4, 3), dpi=150) 

    # Plotting the data
    if glove_value:
        plt.plot(np.arange(12), [glove_value] * 12, label="GloVe", color="blue", linestyle=':', linewidth=2)
    if vg_bert_values:
        plt.plot(np.arange(12), vg_bert_values, label="VG-BERT", color="red", marker='D', linestyle='-', markersize=8, linewidth=2)
    if reg_bert_values:
        plt.plot(np.arange(12), reg_bert_values, label="BERT", color="orange", marker='o', linestyle='-', markersize=8, linewidth=2)
    if wav2vec2_base_values:
        plt.plot(np.arange(12), wav2vec2_base_values, label="Wav2Vec2", color="green", marker='s', linestyle='-', markersize=8, linewidth=2)
    if fast_vgs_values:
        plt.plot(np.arange(12), fast_vgs_values, label="FaST-VGS+", color="purple", marker='^', linestyle='-', markersize=8, linewidth=2)
    
    plt.xlabel('Layer Index', fontsize=16, labelpad=10)
    plt.ylabel('Silhouette Coefficient', fontsize=16, labelpad=10)
    plt.title('Silhouette Coefficients Across Layers', fontsize=18)

    plt.xticks([0, 2, 4, 6, 8, 10], fontsize=14)
    plt.yticks(np.linspace(-0.3, 0.45, 7), fontsize=14)


    plt.legend(fontsize=14)

    plt.tight_layout()

    if savefig:
        plt.savefig(filename, format="pdf")

    plt.show()


def plot_reduced_results(results, categories, category_mapping, model_mapping, reduced_type = 'lda', res_type = 'sem', title=False, savefig=False, filename=''):
    if res_type == 'sem':
        num_rows = 2
        figsize = (8, 6)
        plot_title = 'Semantic Categories'
    else:
        num_rows = 1
        figsize = (8, 3)
        plot_title = 'Phonetic Groups'
    if not title:
        plot_title = ''
    fig, axes = plt.subplots(num_rows, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Define a color palette for categories
    category_names = list(categories.keys())
    colors = sns.color_palette("husl", len(category_names))
    category_colors = {category: colors[i] for i, category in enumerate(category_names)}

    best_layers = {}
    handles, labels = [], []

    for i, (model, model_data) in enumerate(results.items()):
        # Find the best-performing layer based on silhouette score
        best_layer_idx = int(np.argmax([model_data[reduced_type][f"layer_{i}"]["silhouette_score"] for i in range(12)]))
        best_layer = f"layer_{best_layer_idx}"
        best_layers[model] = best_layer_idx
        projected_embeddings = model_data[reduced_type][best_layer]["projected_embeddings"]
        
        ax = axes[i]

        for category, words_dict in projected_embeddings.items():
            embeddings = np.array(list(words_dict.values()))  
            if embeddings.shape[0] > 0:
                if category == 'musical_instruments' or category == 'clothing' or category == 'vegetables' or category == 'vehicles' or category == 'building_materials' or category == 'organs' or 'conc' in category:
                    marker = 'o'
                else:
                    marker = '^'
                ax.scatter(embeddings[:, 0], embeddings[:, 1], label=category_mapping[category], 
                           color=category_colors[category], alpha=0.7, marker=marker)
            if i == 0:
                handles.append(plt.Line2D([0], [0], marker=marker, color='w', label=category_mapping[category], 
                                           markerfacecolor=category_colors[category], markersize=8, alpha=0.7))
                labels.append(category_mapping[category])

        # Formatting
        ax.set_title(f'{plot_title}{model_mapping[model]} (Layer {best_layers[model]})', fontsize=12)
        ax.set_xlabel("LDA Dimension 1", fontsize=11)
        ax.set_ylabel("LDA Dimension 2", fontsize=11)
    if res_type == 'phon':
        bbox_y = 1.5
        n_cols = 2
    else:
        bbox_y = 1.125
        n_cols = 3
    fig.legend(handles=handles, labels=labels, fontsize=10, ncols = n_cols, loc='upper center', bbox_to_anchor=(0.55, bbox_y))  
    plt.tight_layout()
    if savefig:
        plt.savefig(f"{filename}.pdf", bbox_inches='tight')
    plt.tight_layout()
    plt.show()

