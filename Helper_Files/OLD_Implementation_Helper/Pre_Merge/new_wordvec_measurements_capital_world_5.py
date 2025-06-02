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

# #### Parse the Analogy Questions

# In[5]:


def load_analogies(filepath: str) -> dict:
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


# #### Helper Function to Build Covariance Matrices

# In[10]:


def build_covariance_inverse(
    model: "KeyedVectors",
    freq_subset: int,
    rcond: float
) -> tuple:
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
    # Extract words and their frequencies
    vocab_counts = {word: model.get_vecattr(word, "count") for word in model.index_to_key}

    # Filter words with count >= 5
    filtered_vocab = {word: count for word, count in vocab_counts.items() if count >= 5}

    # Sort words by frequency in descending order
    sorted_vocab = sorted(filtered_vocab.items(), key=lambda x: x[1], reverse=True)

    # Get top freq_subset words
    top_words = [word for word, _ in sorted_vocab[:freq_subset]]

    # Get vectors of the selected words
    subset_vectors = np.array([model[word] for word in top_words if word in model])

    # Compute covariance matrix and pseudo-inverse
    Cw = np.cov(subset_vectors, rowvar=False)
    Cw_inv = np.linalg.pinv(Cw, rcond=rcond)

    # Compute center of mass
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
    diff = u - v
    return float(diff @ Cw_inv @ diff)


# In[14]:


def mahalanobis_cosine(
    u: np.ndarray,
    v: np.ndarray,
    Cw_inv: np.ndarray
) -> float:
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
    u_shifted = u - center
    v_shifted = v - center
    numerator = np.dot(u_shifted, Cw_inv @ v_shifted)
    denom = np.sqrt((u_shifted @ Cw_inv @ u_shifted) * (v_shifted @ Cw_inv @ v_shifted))
    return float(numerator / denom)


# #### Evaluate the Analogies using One of the Measurements

# In[16]:


def evaluate_analogies(
    questions: list,
    model: "KeyedVectors",
    measure: str,
    Cw_inv: np.ndarray = None,
    center_vec: np.ndarray = None,
    top_k: int = 10,
    vocab_limit: int = 50000
) -> tuple:
    """
    Evaluates analogy questions and returns category and total accuracies.

    Args:
        questions (list): List of analogy questions.
        model (KeyedVectors): Pretrained word embedding model.
        measure (str): Similarity measure used.
        Cw_inv (np.ndarray, optional): Inverse covariance matrix (for Mahalanobis-based measures).
        center_vec (np.ndarray, optional): Center vector for Mahalanobis shifted cosine.
        top_k (int, optional): Number of top-ranked candidates to consider. Default is 10.
        vocab_limit (int, optional): Number of most frequent words to consider in similarity computation.

    Returns:
        tuple: (overall accuracy, category-wise accuracy, candidate word rankings)
            - overall accuracy (float): Percentage of correct answers.
            - category-wise accuracy (dict): Dictionary mapping categories to accuracy values.
            - candidates (list): List of dictionaries containing the top `k` candidate words.
    """
    correct = 0
    total = 0
    category_correct = {}
    category_total = {}

    vocab_sample = model.index_to_key[:vocab_limit]

    results = []  # To store candidates per question

    for (w1, w2, w3, w4, section) in questions:
        if (w1 not in model) or (w2 not in model) or (w3 not in model) or (w4 not in model):
            continue

        total += 1
        if section not in category_total:
            category_total[section] = 0
            category_correct[section] = 0
        category_total[section] += 1

        v1 = model[w1]
        v2 = model[w2]
        v3 = model[w3]
        query_vec = v2 - v1 + v3

        best_candidates = []
        best_scores = []

        for candidate in vocab_sample:
            if candidate in (w1, w2, w3, w4):
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

            if len(best_candidates) < top_k:
                best_candidates.append(candidate)
                best_scores.append(sim)
            else:
                min_idx = min(range(len(best_scores)), key=best_scores.__getitem__)
                if sim > best_scores[min_idx]:
                    best_scores[min_idx] = sim
                    best_candidates[min_idx] = candidate

        # Sort best candidates by similarity score (descending)
        sorted_candidates = [x for _, x in sorted(zip(best_scores, best_candidates), reverse=True)]

        # Check if correct answer exists in top_k
        canonical_w4 = w4.lower()
        found_match = any(cand.lower() == canonical_w4 for cand in sorted_candidates)

        if found_match:
            correct += 1
            category_correct[section] += 1

        # Store results in dictionary format for easier DataFrame conversion
        result_entry = {
            "word1": w1, "word2": w2, "word3": w3, "true_word": w4, "section": section
        }
        for i in range(top_k):
            result_entry[f"candidate_{i+1}"] = sorted_candidates[i] if i < len(sorted_candidates) else None

        results.append(result_entry)

    overall_acc = float(correct / total) if total > 0 else 0.0
    category_acc = {cat: float(category_correct.get(cat, 0) / category_total[cat]) if category_total[cat] > 0 else 0.0 for cat in category_total}

    return overall_acc, category_acc, results


# #### Run All Experiments

# In[17]:


def run_all_experiments(
    model: "KeyedVectors",
    questions: list,
    freq_subsets: list,
    rcond_values: list,
    measure_types: list,
    build_cov_inverse_fn,
    evaluate_fn,
    out_csv: str = "final_results.csv",
    out_dir: str = ".",
    top_k: int = 10
) -> "pd.DataFrame":
    """
    Runs analogy-based evaluations on word embeddings using multiple 
    frequency subsets, regularization conditions, and similarity measures.
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
            Example: `[50000, 100000, 150000]`.

        rcond_values (list of float): 
            A list of cutoff values for small singular values when computing the 
            pseudo-inverse of the covariance matrix using `np.linalg.pinv()`. 
            Example: `[0.01, 0.001, 0.0001]`.

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
            build_cov_inverse_fn(model, freq_subset, rcond) -> (Cw_inv, center_vec)
            ```
            where:
            - `Cw_inv`: Pseudo-inverse of the covariance matrix.
            - `center_vec`: Center-of-mass vector (used in Mahalanobis shifted cosine).

        evaluate_fn (function): 
            A function that evaluates analogy questions and returns accuracy.
            Expected signature:
            ```python
            evaluate_fn(questions, model, measure, Cw_inv, center_vec, top_k) 
            -> (overall_acc, category_acc)
            ```
            where:
            - `overall_acc`: Float representing total accuracy.
            - `category_acc`: Dictionary mapping categories to their accuracy.

        out_csv (str, optional): 
            The filename for saving the evaluation results in CSV format. 
            Default: `"final_results.csv"`.

        out_dir (str, optional): 
            The directory path where the CSV file will be saved. 
            Default: `"."` (current directory).

        top_k (int, optional): 
            Specifies how many top-ranked words to consider when checking 
            correctness in analogy evaluation. 
            Example: `top_k=1` means exact match; `top_k=5` considers 
            the correct answer within the top 5 candidates.
            Default: `10`.

    Returns:
        pd.DataFrame: 
            A DataFrame containing evaluation results with columns:
            - `"freq_subset"`: Number of most frequent words used.
            - `"rcond"`: Regularization value used in pseudo-inverse.
            - `"measure"`: Similarity measure used.
            - `"overall_accuracy"`: Total accuracy across all categories.
            - `"acc_<category>"`: Accuracy for each analogy category.

    Saves:
        A CSV file in the specified `out_dir` containing all evaluation results.

    Raises:
        ValueError: If an invalid similarity measure is provided.
        RuntimeError: If covariance computation fails.
    """

    all_results = []

    for freq in freq_subsets:
        for rc in rcond_values:
            Cw_inv, center_vec = build_cov_inverse_fn(model, freq, rc)

            for measure in measure_types:
                print(f"Running measure={measure}, freq={freq}, rcond={rc} ...")

                overall_acc, cat_acc, results = evaluate_fn(
                    questions, model,
                    measure=measure,
                    Cw_inv=Cw_inv,
                    center_vec=center_vec,
                    top_k=top_k
                )

                # Convert to DataFrame
                df_results = pd.DataFrame(results)
                df_results["freq_subset"] = freq
                df_results["rcond"] = rc
                df_results["measure"] = measure
                df_results["overall_accuracy"] = overall_acc

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


freq_subs = [10000, 20000, 30000, 50000, 100000]


# In[22]:


rc_vals = [0.001, 0.01, 0.02, 0.005]


# In[23]:


meas_types = ["naive_cosine", "euclidean_distance", "mahalanobis_cosine",
             "mahalanobis_shifted_cosine", "mahalanobis_distance"]


# #### Running the Code in Chunks for Optimization


# Run for section: capital-world
section_name = "capital-world"
questions = sections[section_name]

total = len(questions)
chunk = total // 8

parts = [questions[i*chunk:(i+1)*chunk] for i in range(7)]
parts.append(questions[7*chunk:])  # final chunk takes the rest

print(f"Processing section: {section_name} ({len(questions)} questions)")

results_df = run_all_experiments(
    model=model,
    questions=parts[4],
    freq_subsets=freq_subs,
    rcond_values=rc_vals,
    measure_types=meas_types,
    build_cov_inverse_fn=build_covariance_inverse,
    evaluate_fn=evaluate_analogies,
    out_csv="capital_world_results-5.csv",
    out_dir="results_per_section",
    top_k=10
)

print(f"Results for '{section_name}' saved!")