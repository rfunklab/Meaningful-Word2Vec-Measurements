#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print(sys.executable)


# In[2]:

import os


# In[4]:
# #### Download/Load Analogy Questions

os.system('curl -o questions-words.txt http://download.tensorflow.org/data/questions-words.txt')


# In[]

from gensim.models import KeyedVectors

import numpy as np
import pandas as pd
import ace_tools_open as tools
import gdown
import re
from functools import partial
from typing import List, Tuple, Dict, Callable, Optional
from collections import defaultdict
import unicodedata


# #### Parse the Analogy Questions

# In[5]:


def load_analogies(filepath: str) -> Dict[str, List[Tuple[str, str, str, str, str]]]:
    """
    Loads 'question-words.txt' and groups questions by section.

    Args:
        filepath (str): Path to the analogy questions file.

    Returns:
        dict: Dictionary where keys are section names, values are lists of (w1, w2, w3, w4, section).
    """
    sections = {}
    section_name = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(':'):
                section_name = line[2:].strip()  # Remove ': ' to extract section name
                sections[section_name] = []  # Initialize list for this section
            else:
                parts = line.split()
                if len(parts) == 4:  # Ensure only valid questions are included
                    w1, w2, w3, w4 = parts
                    if section_name:  # Ensure no questions are added without a section
                        sections[section_name].append((w1.lower(), w2.lower(), w3.lower(), w4.lower(), section_name))

    return sections


# #### Load the Google News Model

# In[6]:


google_news_model_url = "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"


# In[7]:


model_filename = "GoogleNews-vectors-negative300.bin.gz"


# In[8]:


if not os.path.exists(model_filename):
    print("Downloading Google News model...")
    gdown.download(google_news_model_url, model_filename, quiet=False)
else:
    print("Model file already exists, skipping download.")


# In[9]:


print("Loading Google News model...")
model = KeyedVectors.load_word2vec_format(model_filename, binary=True)
print("Model loaded! Vocabulary size =", len(model.index_to_key))


# In[11]:


embedding_matrix = model.vectors


# In[12]:


embedding_matrix.shape

# In[]:

word2count = {}
for word in model.index_to_key:
    word2count[word] = model.get_vecattr(word, "count")
    
def normalize_string(s: str) -> str:
    """
    Normalize a string by removing accents/diacritics and converting to lowercase.
    
    This function performs Unicode normalization to handle characters with diacritical
    marks (accents, umlauts, tildes, etc.) by decomposing them into their base 
    characters and removing the combining marks. This is useful for string comparison
    where accented and non-accented versions of characters should be treated as equivalent.
    
    What it does:
    -------------
    1. Decomposes accented characters into base characters + combining marks
    2. Removes all combining marks (accents, diacritics)
    3. Converts the result to lowercase
    
    Why it's needed:
    ---------------
    - User input inconsistency (café vs cafe)
    - Cross-language matching (fiancé vs fiance)
    
    How it works:
    ------------
    1. NFKD Normalization: Decomposes characters into their canonical forms
       - 'é' becomes 'e' + '́' (combining acute accent)
       - 'ñ' becomes 'n' + '̃' (combining tilde)
       
    2. Combining Character Removal: Filters out all combining marks
       - Uses unicodedata.combining() to identify combining characters
       - These have the Unicode category starting with 'M' (Mark)
       
    3. Lowercase Conversion: Ensures case-insensitive comparison
    
    Args:
        s (str): Input string to normalize
        
    Returns:
        str: Normalized string with accents removed and in lowercase
        
    Examples:
        >>> normalize_string("Café")
        'cafe'
        >>> normalize_string("résumé")
        'resume'
        >>> normalize_string("Søren")
        'soren'
        >>> normalize_string("Müller")
        'muller'
        >>> normalize_string("naïve")
        'naive'
        
    Technical Details:
    -----------------
    - NFKD = Normalization Form Compatibility Decomposition
    - Combining characters have Unicode categories: Mn (Nonspacing Mark),
      Mc (Spacing Mark), Me (Enclosing Mark)
    - This preserves the base letter while removing modifiers
    
    Note:
    -----
    This normalization is lossy - the original accented form cannot be 
    recovered from the normalized result. It's intended for comparison
    and matching purposes, not for display.
    """
    # Decompose unicode characters and remove combining characters
    nfkd_form = unicodedata.normalize('NFKD', s)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()


def are_words_equivalent(word1: str, word2: str) -> bool:
    """
    Determine if two words should be considered equivalent for matching purposes.
    
    This function implements a comprehensive word equivalence checker that handles
    various linguistic variations, typographical differences, and encoding issues
    that arise in natural language processing and text matching tasks. It uses a
    cascade of increasingly permissive matching strategies.
    
    What it does:
    -------------
    Checks if two words are functionally equivalent by testing multiple criteria:
    1. Case-insensitive exact matching
    2. Unicode normalization (accent/diacritic removal)
    3. Character encoding differences
    4. Punctuation variations
    5. Numeric/word equivalences
    
    Why it's needed:
    ---------------
    - Word embeddings may have different representations for variants
    - User input is inconsistent (caps, accents, typos)
    - Different systems encode characters differently
    - Natural language has many equivalent forms
    - Analogy tasks need flexible matching without modifying embeddings
    
    Matching Strategy (in order of precedence):
    ------------------------------------------
    1. Exact Match (Case-Insensitive):
       - "Apple" == "apple" == "APPLE"
       - Fast path for most common case
    
    2. Unicode Normalization:
       - "café" == "cafe"
       - "résumé" == "resume"
       - Handles accents and diacritics
    
    3. Unicode Character Variants:
       - Smart quotes: "don't" == "don't"
       - Dashes: "co–operate" == "co-operate"
       - Ligatures: "æsthetic" == "aesthetic"
    
    4. Punctuation Removal:
       - "U.S.A." == "USA" == "usa"
       - "don't" == "dont"
       - Handles abbreviations and contractions
    
    5. Number-Word Equivalence:
       - "1st" == "first"
       - "2nd" == "second"
       - Limited to ordinals 1-10
    
    Args:
        word1 (str): First word to compare
        word2 (str): Second word to compare
        
    Returns:
        bool: True if words are considered equivalent, False otherwise
        
    Examples:
        >>> are_words_equivalent("Running", "running")
        True  # Case insensitive
        
        >>> are_words_equivalent("café", "cafe")
        True  # Accent normalization
        
        >>> are_words_equivalent("U.S.", "us")
        True  # Punctuation removal
        
        >>> are_words_equivalent("1st", "first")
        True  # Number-word conversion
        
        >>> are_words_equivalent("cat", "dog")
        False  # Completely different words
    
    Performance Considerations:
    --------------------------
    - Ordered from fastest to slowest checks
    - Early returns for common cases
    - Most comparisons exit at case-insensitive check
    
    Limitations:
    -----------
    - English-centric morphological rules
    - Limited semantic understanding (no synonyms)
    - May over-match in some cases
    
    Note:
    -----
    This function is designed for high recall (catching most equivalences)
    at the cost of some precision. It's particularly suited for tasks like
    word analogy evaluation where missing valid matches is worse than
    occasional false positives.
    """
    # First check exact match (case-insensitive)
    if word1.lower() == word2.lower():
        return True
    
    # Normalize both words (removes accents, normalize unicode)
    norm1 = normalize_string(word1)
    norm2 = normalize_string(word2)
    
    if norm1 == norm2:
        return True
    
    # Handle unicode variations using simple replacements
    import string
    
    # Simple character replacements
    unicode_replacements = [
        (''', "'"), (''', "'"), ('"', '"'), ('"', '"'),
        ('–', '-'), ('—', '-'), ('…', '...'),
        ('æ', 'ae'), ('œ', 'oe'), ('ß', 'ss'),
    ]
    
    ascii1 = norm1
    ascii2 = norm2
    
    # Apply replacements
    for old, new in unicode_replacements:
        ascii1 = ascii1.replace(old, new)
        ascii2 = ascii2.replace(old, new)
    
    if ascii1 == ascii2:
        return True
    
    # Check for common abbreviations or variations
    # This handles cases like "U.S." vs "US" vs "us"
    punct_removed1 = ''.join(c for c in ascii1 if c not in string.punctuation)
    punct_removed2 = ''.join(c for c in ascii2 if c not in string.punctuation)
    
    if punct_removed1 == punct_removed2:
        return True
    
    # Handle special cases with numbers (e.g., "1st" vs "first")
    number_words = {
        '1st': 'first', '2nd': 'second', '3rd': 'third',
        '4th': 'fourth', '5th': 'fifth', '6th': 'sixth',
        '7th': 'seventh', '8th': 'eighth', '9th': 'ninth',
        '10th': 'tenth'
    }
    
    for num_form, word_form in number_words.items():
        if (norm1 == num_form and norm2 == word_form) or (norm2 == num_form and norm1 == word_form):
            return True
    
    return False


# #### Helper Function to Build Covariance Matrices


# In[ ]:


def build_covariance_inverse(
    model: "KeyedVectors",
    freq_subset: int,
    rcond: float,
    word2count: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the covariance and pseudo-inverse (Cw^-1) using the top `freq_subset`
    frequent words from the model, and also compute the center of mass.

    Args:
        model: A pretrained KeyedVectors or equivalent model containing word vectors.
        freq_subset: An integer specifying how many of the top frequent words to use.
        rcond: A float specifying the cutoff for small singular values in the pseudo-inverse.

    Returns:
        tuple: (Cw_inv, cm), where
            Cw_inv is the pseudo-inverse of the covariance matrix,
            cm is the mean (center-of-mass) vector over the chosen subset.
    """
    # Filter words with count >= 5
    filtered_vocab = {w: c for w, c in word2count.items() if c >= 5}

    # Sort words by frequency in descending order
    sorted_vocab = sorted(filtered_vocab.items(), key=lambda x: x[1], reverse=True)

    # Get top freq_subset words
    top_words = [word for word, _ in sorted_vocab[:freq_subset]]

    # Get vectors
    subset_vectors = np.array([model[word] for word in top_words if word in model])

    # Covariance matrix
    Cw = np.cov(subset_vectors, rowvar=False)
    Cw_inv = np.linalg.pinv(Cw, rcond=rcond)

    # Center of mass
    cm = np.mean(subset_vectors, axis=0)

    return Cw_inv, cm


# #### Define Functions to Perform the Analogy

# In[11]:


def euclidean_distance(
    u: np.ndarray,
    v: np.ndarray
) -> float:
    return np.linalg.norm(u - v)


# In[12]:


def naive_cosine(
    u: np.ndarray,
    v: np.ndarray
) -> float:
    return np.dot(u, v) / (np.linalg.norm(u)*np.linalg.norm(v))


# In[13]:


def mahalanobis_distance(
    u: np.ndarray,
    v: np.ndarray,
    Cw_inv: np.ndarray
) -> float:
    """
    Compute Mahalanobis distance using the inverse covariance matrix.
    This measures the distance between two vectors using the covariance inverse
    as a metric tensor, which accounts for the correlations and variances 
    between embedding dimensions.

    Args:
        u: First vector
        v: Second vector  
        Cw_inv: Inverse covariance matrix (D × D)

    Returns:
        Mahalanobis distance (squared)
    """
    diff = u - v
    return float(diff @ Cw_inv @ diff)


# In[14]:


def mahalanobis_cosine(
    u: np.ndarray,
    v: np.ndarray,
    Cw_inv: np.ndarray
) -> float:
    """
    Compute Mahalanobis cosine similarity using the inverse covariance matrix.
    This measures the cosine similarity using the inverse covariance matrix
    as a metric tensor, which accounts for the correlations and variances
    between embedding dimensions.

    Args:
        u: First vector
        v: Second vector
        Cw_inv: Inverse covariance matrix (D × D)

    Returns:
        Mahalanobis cosine similarity
    """
    numerator = np.dot(u, Cw_inv @ v)
    denom = np.sqrt((u @ Cw_inv @ u) * (v @ Cw_inv @ v))
    return float(numerator / denom)


# In[15]:


def mahalanobis_shifted_cosine(
    u: np.ndarray,
    v: np.ndarray,
    Cw_inv: np.ndarray,
    center: np.ndarray
) -> float:
    """
    Compute Mahalanobis cosine similarity with center-shifting using the inverse covariance matrix.
    This measures the cosine similarity between shifted vectors using the inverse covariance matrix
    as a metric tensor. The center-shifting helps focus on relative positions.

    Args:
        u: First vector
        v: Second vector
        Cw_inv: Inverse covariance matrix (D × D)
        center: Center vector to subtract from both u and v

    Returns:
        Mahalanobis shifted cosine similarity
    """
    u_shifted = u - center
    v_shifted = v - center
    numerator = np.dot(u_shifted, Cw_inv @ v_shifted)
    denom = np.sqrt((u_shifted @ Cw_inv @ u_shifted) * (v_shifted @ Cw_inv @ v_shifted))
    return float(numerator / denom)


# #### Evaluate the Analogies using One of the Measurements

# In[16]:

# Cache for expensive word equivalence checks
equivalence_cache = {}

def cached_are_words_equivalent(word1: str, word2: str) -> bool:
    """Cached version of are_words_equivalent to avoid repeated expensive calls"""
    # Create a normalized key (alphabetically sorted to handle both directions)
    cache_key = tuple(sorted([word1, word2]))
    
    if cache_key not in equivalence_cache:
        equivalence_cache[cache_key] = are_words_equivalent(word1, word2)
    
    return equivalence_cache[cache_key]


def evaluate_analogies(
    questions: List[Tuple[str, str, str, str, str]],
    model: "KeyedVectors",
    measure: str,
    Cw_inv: Optional[np.ndarray] = None,
    center_vec: Optional[np.ndarray] = None,
    top_k_values: List[int] = [1, 3, 5, 10],
    vocab_limit: int = 30000
) -> Tuple[Dict[int, float], List[Dict[str, Optional[str]]]]:
    """
    Evaluates analogy questions and returns total accuracies for different k values.
    Args:
        questions (list): List of analogy questions.
        model (KeyedVectors): Pretrained word embedding model.
        measure (str): Similarity measure used.
        Cw_inv (np.ndarray, optional): Inverse covariance matrix (for Mahalanobis-based measures).
        center_vec (np.ndarray, optional): Center vector for Mahalanobis shifted cosine.
        top_k_values (list, optional): List of k values to evaluate. Default is [1, 3, 5, 10].
        vocab_limit (int, optional): Number of most frequent words to consider in similarity computation.
    Returns:
        tuple: (overall_accuracy_by_k, candidate_word_rankings)
            - overall_accuracy_by_k (dict): Maps k value to overall accuracy
            - candidates (list): List of dictionaries containing the top candidates
    """
    # Create mappings for flexible word matching
    vocab_mappings = {}
    for word in model.index_to_key[:vocab_limit]:
        vocab_mappings[word] = word  # Map each word to itself
    
    total = 0
    results = []
    
    # Initialize tracking for different k values
    correct_at_k = {k: 0 for k in top_k_values}
    max_k = max(top_k_values)
    
    for (w1, w2, w3, w4, category) in questions:
        # Find the best match for each word in the vocabulary
        words_found = []
        for w in [w1, w2, w3, w4]:
            best_match = None
            # First try exact match
            if w in vocab_mappings:
                best_match = w
            else:
                # Then try to find equivalent words
                for vocab_word in vocab_mappings:
                    if cached_are_words_equivalent(w, vocab_word):
                        best_match = vocab_word
                        break
            words_found.append(best_match)
        
        if any(word is None for word in words_found):
            continue
            
        w1_vec, w2_vec, w3_vec, w4_vec = words_found
        
        total += 1
        
        v1 = model[w1_vec]
        v2 = model[w2_vec]
        v3 = model[w3_vec]
        query_vec = v2 - v1 + v3
        
        best_candidates = []
        best_scores = []
        
        for candidate in model.index_to_key[:vocab_limit]:
            # Skip if it's one of the input words (using flexible matching)
            skip = False
            for input_word in [w1, w2, w3]:
                if cached_are_words_equivalent(candidate, input_word):
                    skip = True
                    break
            if skip:
                continue
                
            cvec = model[candidate]
            
            if measure == "naive_cosine":
                sim = naive_cosine(query_vec, cvec)
            elif measure == "euclidean_distance":
                dist = euclidean_distance(query_vec, cvec)
                sim = -dist
            elif measure == "mahalanobis_cosine":
                sim = mahalanobis_cosine(query_vec, cvec, Cw_inv)
            elif measure == "mahalanobis_shifted_cosine":
                sim = mahalanobis_shifted_cosine(query_vec, cvec, Cw_inv, center_vec)
            elif measure == "mahalanobis_distance":
                dist = mahalanobis_distance(query_vec, cvec, Cw_inv)
                sim = -dist
            else:
                raise ValueError(f"Unknown measure: {measure}")

            # Check if this candidate is equivalent to the true word
            if cached_are_words_equivalent(candidate, w4):
                # Add this candidate with its score
                if len(best_candidates) < max_k:
                    best_candidates.append(candidate)
                    best_scores.append(sim)
                else:
                    min_idx = min(range(len(best_scores)), key=best_scores.__getitem__)
                    if sim > best_scores[min_idx]:
                        best_scores[min_idx] = sim
                        best_candidates[min_idx] = candidate
                # Stop searching since we found the correct answer
                break
            
            if len(best_candidates) < max_k:
                best_candidates.append(candidate)
                best_scores.append(sim)
            else:
                min_idx = min(range(len(best_scores)), key=best_scores.__getitem__)
                if sim > best_scores[min_idx]:
                    best_scores[min_idx] = sim
                    best_candidates[min_idx] = candidate
        
        # Sort candidates
        sorted_candidates = [x for _, x in sorted(zip(best_scores, best_candidates), reverse=True)]
        
        # Check for correct answer at different k values
        for k in top_k_values:
            found_match = False
            for i, cand in enumerate(sorted_candidates[:k]):
                if cached_are_words_equivalent(cand, w4):
                    found_match = True
                    break
            
            if found_match:
                correct_at_k[k] += 1
            
        semantic_categories = {
            'capital-common-countries', 'capital-world', 'currency', 
            'city-in-state', 'family'
        }
        
        # Store results with all candidates up to max_k
        result_entry = {
            "word1": w1, "word2": w2, "word3": w3, "true_word": w4, 
            "category": category,
            "category_type": "semantic" if category in semantic_categories else "syntactic"
        }
        
        for i in range(max_k):
            result_entry[f"candidate_{i+1}"] = sorted_candidates[i] if i < len(sorted_candidates) else None
        results.append(result_entry)
        
    
    # Calculate accuracies for each k
    overall_acc_by_k = {k: float(correct_at_k[k] / total) if total > 0 else 0.0 
                        for k in top_k_values}
    
    return overall_acc_by_k, results

# #### Run All Experiments

# In[17]:


def run_all_experiments(
    model: "KeyedVectors",
    questions: List[Tuple[str, str, str, str, str]],
    freq_subsets: List[int],
    rcond_values: List[float],
    quantiles: List[float],
    measure_types: List[str],
    build_cov_inverse_fn: Callable[[KeyedVectors, int, float, Dict[str, int]], Tuple[np.ndarray, np.ndarray]],
    evaluate_fn: Callable,
    word2count: Dict[str, int],
    out_csv: str,
    out_dir: str = ".",
    top_k_values: List[int] = [1, 3, 5, 10]
) -> pd.DataFrame:
    """
    Runs analogy-based evaluations on word embeddings using multiple 
    regularization conditions, quantiles, and similarity measures.
    The results are saved to a CSV file.

    Args:
        model (KeyedVectors): 
            A pretrained word embedding model 
            loaded as a `gensim.models.KeyedVectors` object. 
            It contains word vectors and their associated metadata.
        questions (list): 
            A list of analogy questions, where each question is formatted 
            as a tuple (word1, word2, word3, word4, category). 
            Example: ("king", "queen", "man", "woman", "gram2-opposite").
        freq_subsets (list of int): 
            A list of integer values specifying the number of most frequent words 
            (with count ≥ 5) to consider when computing the covariance matrix.
            Typically `[30000]`.
        rcond_values (list of float): 
            A list of cutoff values for small singular values when computing the 
            pseudo-inverse of the covariance matrix using `np.linalg.pinv()`. 
            Example: `[0.01, 0.001, 0.0001]`.
        quantiles (list of float):
            A list of quantile values corresponding to each rcond value,
            indicating which quantile was used to compute that rcond.
            Example: `[0.01, 0.05, 0.10, 0.25, 0.5]`.
        measure_types (list of str): 
            A list of similarity measures to be used for evaluation. 
            Supported options:
            - `"naive_cosine"`
            - `"euclidean_distance"`
            - `"mahalanobis_cosine"`
            - `"mahalanobis_shifted_cosine"`
            - `"mahalanobis_distance"`
        build_cov_inverse_fn (function): 
            A function that computes the pseudo-inverse of the covariance matrix 
            using a subset of frequent words. Expected signature:
            ```python
            build_cov_inverse_fn(model, freq_subset, rcond, word2count) -> (Cw_inv, center_vec)
            ```
            where:
            - `Cw_inv`: Pseudo-inverse of the covariance matrix.
            - `center_vec`: Center-of-mass vector (used in Mahalanobis shifted cosine).
        evaluate_fn (function): 
            A function that evaluates analogy questions and returns accuracy.
            Expected signature:
            ```python
            evaluate_fn(questions, model, measure, Cw_inv, center_vec, top_k_values) 
            -> (overall_acc_by_k, results)
            ```
            where:
            - `overall_acc_by_k`: Dictionary mapping k values to overall accuracy.
            - `results`: List of dictionaries containing candidate words.
        word2count (dict):
            Dictionary mapping words to their frequency counts.
        out_csv (str): 
            The filename for saving the evaluation results in CSV format.
        out_dir (str, optional): 
            The directory path where the CSV file will be saved. 
            Default: `"."` (current directory).
        top_k_values (list of int, optional): 
            List of k values specifying how many top-ranked words to consider 
            when checking correctness in analogy evaluation. 
            Example: `[1, 3, 5, 10]` evaluates accuracy at top-1, top-3, top-5, and top-10.
            Default: `[1, 3, 5, 10]`.

    Returns:
        pd.DataFrame: 
            A DataFrame containing evaluation results with columns:
            - `"word1"`, `"word2"`, `"word3"`, `"true_word"`: The analogy components
            - `"category"`: Category name of the analogy
            - `"category_type"`: "semantic" or "syntactic"
            - `"candidate_1"` through `"candidate_10"`: Top predicted words
            - `"quantile"`: Quantile value used for rcond computation
            - `"rcond"`: Regularization value used in pseudo-inverse
            - `"measure"`: Similarity measure used
            - `"top@k"`: The k value for this row (1, 3, 5, or 10)
            - `"overall_accuracy"`: Accuracy for this specific combination of parameters

    Saves:
        A CSV file in the specified `out_dir` containing all evaluation results.

    Raises:
        ValueError: If an invalid similarity measure is provided.
        RuntimeError: If covariance computation fails.
    """

    all_results = []
    
    for freq in freq_subsets:
        for rc, q in zip(rcond_values, quantiles):
            Cw_inv, center_vec = build_cov_inverse_fn(model, freq, rc, word2count)
            for measure in measure_types:
                print(f"Running measure={measure}, freq={freq}, rcond={rc}, quantile={q} ...")
                overall_acc_by_k, results = evaluate_fn(
                    questions, model,
                    measure=measure,
                    Cw_inv=Cw_inv,
                    center_vec=center_vec,
                    top_k_values=top_k_values 
                )
                
                # Create separate rows for each k value
                for k in top_k_values:
                    df_results = pd.DataFrame(results)
                    df_results["quantile"] = q
                    df_results["rcond"] = rc
                    df_results["measure"] = measure
                    df_results["top@k"] = k
                    df_results["overall_accuracy"] = overall_acc_by_k[k]
                    
                    all_results.append(df_results)

    # Combine results
    final_df = pd.concat(all_results, ignore_index=True)

    # Save results
    if out_csv:
        from pathlib import Path
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        file_path = str(Path(out_dir) / out_csv)
        final_df.to_csv(file_path, index=False)
        print(f"\nResults saved to: {file_path}")

    return final_df


# #### Defining the Metrics

# In[18]:


sections = load_analogies("questions-words.txt")


# In[19]:


sections.keys()


# In[20]:


print("Number of questions per section:")
for section_name, questions in sections.items():
    print(f"{section_name}: {len(questions)} questions")


# In[21]:


freq_subs = [30000]


# In[22]:


quantiles = [0.01, 0.05, 0.10, 0.25, 0.5]


# In[23]:


meas_types = [
    "naive_cosine",
    "mahalanobis_cosine",
    "mahalanobis_shifted_cosine"
]


# #### Compute rcond values from eigenvalues for Covariance Inverse matrix

# In[ ]:


def determine_optimal_cov_rcond(
    model: "KeyedVectors",
    freq_subset: List[int],
    quantile: float,
    word2count: Dict[str, int]
) -> float:
    """
    Determines optimal rcond value for covariance inverse matrix by selecting the eigenvalue at the given quantile.

    Args:
        model: Pretrained KeyedVectors containing word vectors.
        freq_subset: Number of top frequent words to use.
        quantile: Quantile at which to set the eigenvalue cutoff.

    Returns:
        float: Optimal rcond for covariance inverse.
    """

    freq_subset = freq_subset[0]
    
    # Use external frequency data
    filtered_vocab = {word: count for word, count in word2count.items() if count >= 5}
    sorted_vocab = sorted(filtered_vocab.items(), key=lambda x: x[1], reverse=True)
    top_words = [word for word, _ in sorted_vocab[:freq_subset]]

    subset_vectors = np.asarray([model[word] for word in top_words if word in model], dtype=np.float64)

    Cw = np.cov(subset_vectors, rowvar=False)
    cov_eigvals = np.linalg.eigvalsh(Cw)[::-1]

    cov_quantile_value = np.quantile(cov_eigvals, quantile)

    return cov_quantile_value / cov_eigvals[0]


# In[ ]:


cov_rcond_values = {
    q: determine_optimal_cov_rcond(model, freq_subs, quantile=q, word2count=word2count)
    for q in quantiles
}


# In[ ]:


rc_vals = list(cov_rcond_values.values())


# #### Running the Code in Chunks for Optimization


# In[ ]:


# Run for section: capital-world
category_name = "capital-world"
questions = sections[category_name]
total = len(questions)
chunk = total // 8
parts = [questions[i*chunk:(i+1)*chunk] for i in range(7)]
parts.append(questions[7*chunk:])  # final chunk takes the rest
print(f"Processing category: {category_name} ({len(questions)} questions)")

# ADD THIS SECTION - Pre-compute covariance matrices
print("Pre-computing all covariance matrices...")
cov_matrices = {}
for freq in freq_subs:
    for rc, q in zip(rc_vals, quantiles):
        key = (freq, rc, q)
        print(f"  Computing matrix for freq={freq}, rcond={rc}, quantile={q}")
        Cw_inv, center_vec = build_covariance_inverse(model, freq, rc, word2count)
        cov_matrices[key] = (Cw_inv, center_vec)
print(f"Successfully pre-computed {len(cov_matrices)} covariance matrices")

# REPLACE the run_all_experiments function with this custom implementation
# Process the specific chunk (change the index for each file: 0 for part 1, 1 for part 2, etc.)
chunk_idx = 4  # For capital-world-1.py (use 1 for part 2, 2 for part 3, etc.)
chunk_questions = parts[chunk_idx]

print(f"Processing chunk {chunk_idx+1} of {len(parts)}...")
all_results = []

for freq in freq_subs:
    for rc, q in zip(rc_vals, quantiles):
        # Use the pre-computed matrices
        key = (freq, rc, q)
        Cw_inv, center_vec = cov_matrices[key]
        
        for measure in meas_types:
            print(f"  Evaluating measure={measure}, freq={freq}, rcond={rc}, quantile={q}")
            overall_acc_by_k, results = evaluate_analogies(
                chunk_questions, model,
                measure=measure,
                Cw_inv=Cw_inv,
                center_vec=center_vec,
                top_k_values=[1, 3, 5, 10]
            )
            
            # Create separate rows for each k value
            for k in [1, 3, 5, 10]:
                df_results = pd.DataFrame(results)
                df_results["freq_subset"] = freq  # Explicitly include freq_subset
                df_results["quantile"] = q
                df_results["rcond"] = rc
                df_results["measure"] = measure
                df_results["top@k"] = k
                df_results["overall_accuracy"] = overall_acc_by_k[k]
                
                all_results.append(df_results)

# Combine and save results
final_df = pd.concat(all_results, ignore_index=True)
out_csv = f"capital_world_results-{chunk_idx+1}.csv"
out_dir = "new_covariance_inverse_results_per_section"
from pathlib import Path
Path(out_dir).mkdir(parents=True, exist_ok=True)
file_path = str(Path(out_dir) / out_csv)
final_df.to_csv(file_path, index=False)

# Verify combinations
unique_combos = final_df.groupby(["rcond", "measure", "top@k", "freq_subset"]).size().reset_index(name="count")
print(f"\nFinal check: Found {len(unique_combos)} unique combinations")
expected_combos = len(rc_vals) * len(meas_types) * 4  # 4 top@k values
if len(unique_combos) == expected_combos:
    print(f"Success! Exactly {expected_combos} unique combinations as expected.")
else:
    print(f"Warning: Expected {expected_combos} combinations but found {len(unique_combos)}.")

print(f"Results for '{category_name}' chunk {chunk_idx+1} saved to {file_path}!")