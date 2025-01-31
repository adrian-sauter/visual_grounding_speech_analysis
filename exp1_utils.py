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


def convert_embeddings_to_arrays(df):
    for idx, row in df.iterrows():
        w2v2_embeddings = row['w2v2_embeddings']
        fast_vgs_plus_embeddings = row['fast_vgs_plus_embeddings']

        for layer in w2v2_embeddings:
            w2v2_embeddings[layer] = np.array(w2v2_embeddings[layer])
        
        for layer in fast_vgs_plus_embeddings:
            fast_vgs_plus_embeddings[layer] = np.array(fast_vgs_plus_embeddings[layer])
        
        # Update the dataframe with the converted embeddings
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
    # Filter words that have corresponding ARPAbet transcriptions in the pronunc_df
    valid_words = [word for word in words if word in pronunc_df['Item'].values]

    if len(valid_words) < 2:
        raise ValueError("Not enough valid words with pronunciations for baseline computation.")

    # Get the pronunciation of the valid words
    word_pronunciations = pronunc_df[pronunc_df['Item'].isin(valid_words)].set_index('Item')['Pronunciation']

    # List to hold phonetic distances from each run
    all_phonetic_distances = []

    # Repeat the computation for the specified number of runs
    for _ in range(num_runs):
        # Shuffle the valid list of words
        words_shuffled = random.sample(valid_words, len(valid_words))

        # Compute pairwise phonetic distances between original and shuffled list
        phonetic_distances = []
        for word1, word2 in zip(valid_words, words_shuffled):
            pron1 = word_pronunciations[word1]
            pron2 = word_pronunciations[word2]
            distance = Levenshtein.distance(pron1, pron2)
            phonetic_distances.append(distance)

        # Append the average phonetic distance for this run
        all_phonetic_distances.append(np.mean(phonetic_distances))

    # Compute the overall average phonetic baseline
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
    # Compute average pairwise similarities for each word
    sim_matrix = cosine_similarity(embeddings)
    avg_similarities = np.mean(sim_matrix, axis=1)
    #avg_similarities = compute_avg_similarity(embeddings)
    
    # Determine the similarity threshold based on the given percentile
    similarity_threshold = np.percentile(avg_similarities, percentile)
    
    # Select words with similarity above the threshold
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
    # Filter dataframe to get pronunciations for the given words
    word_pronunciations = df[df['Item'].isin(words)].set_index('Item')['Pronunciation']
    #print(f'Found pronounciation for {len(word_pronunciations)} out of {len(words)} words.')

    # Ensure that pronunciations for all words are found
    if len(word_pronunciations) < len(words):
        missing_words = set(words) - set(word_pronunciations.index)
        raise ValueError(f"The following words are missing pronunciations: {missing_words}")

    # Generate all unique pairs of pronunciations
    word_pairs = combinations(word_pronunciations.items(), 2)

    # Compute similarities for all pairs
    similarities = []
    for (word1, pron1), (word2, pron2) in word_pairs:
        # Compute Levenshtein distance
        distance = Levenshtein.distance(pron1, pron2)
        if baseline:
            distance = distance - baseline
        # Normalize distance by the length of the longer transcription
        max_len = max(len(pron1), len(pron2))
        similarity = 1 - (distance / max_len)
        similarities.append(similarity)
    
    # Compute average similarity
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
    # Filter dataframe to get pronunciations for the given words
    word_pronunciations = pronunc_df[pronunc_df['Item'].isin(words)].set_index('Item')['Pronunciation']

    # Ensure that pronunciations for all words are found
    missing_words = set(words) - set(word_pronunciations.index)
    if missing_words:
        raise ValueError(f"The following words are missing pronunciations: {missing_words}")

    # Initialize an empty similarity matrix
    num_words = len(words)
    similarity_matrix = np.zeros((num_words, num_words))

    # Compute pairwise phonetic similarities
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i <= j:  # Avoid computing twice for symmetric entries
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
    # Ensure embeddings are in a numpy array
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Compute pairwise cosine distances between all embeddings
    distances = cosine_distances(embeddings)
    
    # Compute silhouette score using the precomputed distance matrix
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


    # Filter the dataframe to only include rows for the words in word_list
    filtered_df = df[df['words'].isin(word_list)]

    for _, row in filtered_df.iterrows():
        word_similarities = {}

        # Get the embeddings for each layer from col1 and col2
        for layer_idx in range(12):  # Assuming 12 layers for col1 and col2
            layer_name = f'layer_{layer_idx}'
            if is_glove:
                col1_embd = row[col1]
            else:
                col1_embd = row[col1][layer_name]
            col2_embd = row[col2][layer_name]

            # Compute the cosine similarity between col1 (768D) and col2 (768D) for the given layer
            similarity = cosine_similarity(col1_embd.reshape(1, -1), col2_embd.reshape(1, -1)) - baseline

            # Store the similarity for the current layer
            word_similarities[layer_name] = {
                'similarity': similarity
            }

        # Store similarities for this word in the result dictionary
        similarity_dict[row['words']] = word_similarities

    # Compute the average similarity per layer
    avg_similarity_per_layer = {}

    for layer in similarity_dict[next(iter(similarity_dict))]:  # Take the first word's layers to get all layers
        similarities_for_layer = []

        # Iterate through all words
        for word_similarities in similarity_dict.values():
            # Extract the similarity value for this layer
            similarities_for_layer.append(word_similarities[layer]['similarity'])

        # Compute the average similarity for this layer
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
    # Initialize the list for dissimilar words and embeddings
    dissimilar_words = []
    selected_embeddings = []
    for similar_word in similar_words:
        if similar_word in df['words'].values:
            similar_word_embd = df[df['words'] == similar_word]['glove_embeddings'].values[0]
            if similar_word_embd is None:
                continue

            # Calculate similarity with previously selected embeddings
            similarities = [cosine_similarity([similar_word_embd], [embd])[0][0] for embd in selected_embeddings]
            #print(similar_word)
            #print(similarities)
            # Check if the word is sufficiently dissimilar from the already selected words
            if above:
                if all(similarity > similarity_threshold for similarity in similarities):
                    #print(f'word: {similar_word} is kept')
                    dissimilar_words.append(similar_word)
                    selected_embeddings.append(similar_word_embd)
            else:
                if all(similarity < similarity_threshold for similarity in similarities):
                    #print(f'word: {similar_word} is kept')
                    dissimilar_words.append(similar_word)
                    selected_embeddings.append(similar_word_embd)

            # Stop if enough dissimilar words have been selected
            if len(dissimilar_words) >= n:
                break
    
    # Calculate average cosine dissimilarity of the selected embeddings
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
        # Calculate phonetic similarity between candidate words and existing cluster words
        for word in candidate_words:
            for cluster_word in cluster_words:
                word_idx = df[df['words'] == word].index[0]
                cluster_word_idx = df[df['words'] == cluster_word].index[0]
                similarity = phon_sim[word_idx, cluster_word_idx]
                if similarity >= threshold:
                    return False  # Candidate words are not sufficiently dissimilar
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
    
    # Create the figure and set up subplots
    plt.figure(figsize=(6 * num_rsms, 6))  # Adjust figure size based on number of RSMs
    
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

    # Adjust layout to ensure everything fits
    plt.tight_layout()
    plt.suptitle(title, fontsize=18, y=1.05)

    if savefig:
        plt.savefig(filename, format="pdf")
        print(f"Plot saved to {filename}")

    # Show the plot
    plt.show()

def plot_silhouette_coefficients(glove_value=None, wav2vec2_base_values=None, fast_vgs_values=None, reg_bert_values=None, vg_bert_values=None, savefig=True, filename="exp1/silhouette_scores.pdf"):
    # Create the plot with a better aspect ratio and high resolution
    plt.figure(figsize=(4, 3), dpi=150)  # Adjusted size for better clarity

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
    
    # Adding labels and title
    plt.xlabel('Layer Index', fontsize=16, labelpad=10)
    plt.ylabel('Silhouette Coefficient', fontsize=16, labelpad=10)
    plt.title('Silhouette Coefficients Across Layers', fontsize=18)

    # Customizing x-ticks to show only 0, 5, 10
    plt.xticks([0, 2, 4, 6, 8, 10], fontsize=14)
    plt.yticks(np.linspace(-0.3, 0.45, 7), fontsize=14)

    # Adding a grid and customizing appearance
    #plt.grid(True, linestyle='--', alpha=0.7)

    # Adding legend with better placement outside the plot area
    plt.legend(fontsize=14)

    # Tight layout for better spacing
    plt.tight_layout()

    if savefig:
        plt.savefig(filename, format="pdf")

    # Display the plot
    plt.show()

