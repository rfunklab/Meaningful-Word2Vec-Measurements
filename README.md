# New-Word2Vec-Measurements

This repository contains code for evaluating new measurement techniques in word embeddings, including:

- Mahalanobis Cosine Similarity
- Mahalanobis Shifted Cosine Similarity
- Mahalanobis Distance

The implementation is based on replicating the methodology from Mikolov's Word2Vec Paper and extends it by testing these new distance metrics on analogy-based evaluations.

## Features

- Downloads and processes the Google News Word2Vec model.
- Parses and evaluates word analogy datasets.
- Implements and compares multiple similarity measurements.
- Builds covariance matrices and applies Mahalanobis transformations.
- Saves detailed results in CSV format.

## Getting Started

### Install Dependencies

Run the following command to install the required Python libraries:

```bash
pip install -r requirements.txt
```

### Download the Word2Vec Model

The script will automatically download the Google News Word2Vec embeddings if not found.

Alternatively, you can manually download it:
```bash
wget -O GoogleNews-vectors-negative300.bin.gz https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
```

### Run the Evaluation

To test the analogy tasks with the implemented distance measures, execute:

```bash
python New_Word2Vec_Measurements.py
```

### Evaluation Metrics

The script computes analogy accuracy using:
- Naïve Cosine Similarity
- Euclidean Distance
- Mahalanobis Cosine Similarity
- Mahalanobis Shifted Cosine Similarity
- Mahalanobis Distance

It evaluates performance across multiple analogy categories and saves results to CSV files.

### Outputs

Results are saved in the ```results_per_section/``` directory:

- ```capital-common-countries```
- ```capital-world```
- ```currency```
- ```city-in-state```
- ```family```
- ```gram1-adjective-to-adverb```
- ```gram2-opposite```
- ```gram3-comparative```
- ```gram4-superlative```
- ```gram5-present-participle```
- ```gram6-nationality-adjective```
- ```gram7-past-tense```
- ```gram8-plural```
- ```gram9-plural-verbs```

Each csv file contains:

| word1  | word2  | word3  | true_word  | candidate_1  | candidate_2  | ... | overall_accuracy |
|--------|--------|--------|------------|--------------|--------------|----|------------------|


### Customizing Parameters

Modiy ```New_Word2Vec_Measurements.py``` to:

- Change the frequency subset of words used.
- Adjust Mahalanobis inverse covariance matrix computation.
- Select different similarity measures.
  
