"""
Data loading module for fairness-aware job-matching framework.
Handles loading of embeddings, oracle data, and dataset preparation.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from fairness_framework.utils.config import Config


def load_embeddings():
    """
    Load CV embeddings from the resume embeddings file.

    Returns:
        tuple: (embeddings array, candidate_indices list)
    """
    try:
        embeddings_path = Config.get_embedding_files()['resumes']
        embeddings = np.load(embeddings_path, allow_pickle=True)
        print(f"Loaded embeddings with shape: {embeddings.shape}")

        # Create candidate indices since they're not in the file
        candidate_indices = list(range(embeddings.shape[0]))

        return embeddings, candidate_indices

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Embeddings file not found at {embeddings_path}. "
            f"Please ensure the embeddings have been generated."
        )


def load_oracle_data():
    """
    Load the oracle data with demographic information.

    Returns:
        pd.DataFrame: Oracle dataframe with gender labels
    """
    try:
        oracle_df = pd.read_csv(Config.ORACLE_PATH)
        print(f"Loaded oracle data with {len(oracle_df)} entries")
        return oracle_df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Oracle data file not found at {Config.ORACLE_PATH}. "
            f"Please ensure the evaluation dataset is available."
        )


def prepare_data_with_oracle(embeddings, candidate_indices, oracle_df):
    """
    Process embeddings and align with oracle data with robust preprocessing.

    Args:
        embeddings (np.ndarray): CV embeddings array
        candidate_indices (list): List of candidate indices
        oracle_df (pd.DataFrame): Oracle dataframe with gender labels

    Returns:
        tuple: (normalized_embeddings, valid_indices, valid_genders)
    """
    print("Aligning candidate indices with oracle data...")

    # Extract gender from oracle
    oracle_genders = oracle_df['Inferred_Gender'].tolist()

    # Map gender labels to binary (0=female, 1=male)
    gender_map = {'female': 0, 'male': 1}
    binary_genders = [
        gender_map.get(gender.lower(), -1) if pd.notna(gender) else -1
        for gender in oracle_genders
    ]

    # Create dataframe with candidate data and oracle gender
    candidate_data = pd.DataFrame({
        'candidate_index': candidate_indices,
        'gender': binary_genders[:len(candidate_indices)]
    })

    # Filter out entries with invalid gender (-1)
    valid_candidates = candidate_data[candidate_data['gender'] != -1]
    print(f"Found {len(valid_candidates)} candidates with valid gender labels")

    # Get valid candidate indices and corresponding embeddings
    valid_indices = valid_candidates['candidate_index'].tolist()
    valid_embeddings = embeddings[valid_indices]
    valid_genders = valid_candidates['gender'].values

    # Apply robust normalization
    scaler = RobustScaler()
    normalized_embeddings = scaler.fit_transform(valid_embeddings)

    print(f"Valid embeddings shape: {normalized_embeddings.shape}")
    print(f"Gender distribution: Female={sum(valid_genders == 0)}, Male={sum(valid_genders == 1)}")

    return normalized_embeddings, valid_indices, valid_genders


def load_bias_analysis_embeddings():
    """
    Load embeddings for Component 2 bias analysis.
    Jobs search for CVs.

    Returns:
        tuple: (job_embeddings, cv_passage_embeddings)
    """
    embedding_files = Config.get_embedding_files()

    try:
        job_embeddings = np.load(embedding_files['jobs_as_queries'])
        cv_passage_embeddings = np.load(embedding_files['cvs_as_passages'])

        print(f"Loaded job query embeddings: {job_embeddings.shape}")
        print(f"Loaded CV passage embeddings: {cv_passage_embeddings.shape}")

        return job_embeddings, cv_passage_embeddings

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not load bias analysis embeddings: {e}. "
            f"Please ensure all embedding files are generated."
        )


def create_cv_corpus_ids(num_cvs):
    """
    Create CV corpus IDs for bias analysis.

    Args:
        num_cvs (int): Number of CV embeddings

    Returns:
        list: List of CV corpus IDs in format ['resume_0', 'resume_1', ...]
    """
    return [f"resume_{i}" for i in range(num_cvs)]


def load_jsonl_file(file_path):
    """
    Load JSONL file and return list of dictionaries.

    Args:
        file_path (str or Path): Path to JSONL file

    Returns:
        list: List of dictionaries from JSONL file
    """
    data = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line.strip()))
        print(f"Loaded {len(data)} entries from {file_path}")
        return data

    except FileNotFoundError:
        print(f"Warning: Could not find file {file_path}")
        return []


def load_test_job_queries():
    """
    Load test job queries for bias analysis.

    Returns:
        tuple: (test_job_queries, test_job_query_ids)
    """
    try:
        with open(Config.EMBEDDINGS_PATH / "sample_jobs_100.json", "r") as f:
            test_jobs_data = json.load(f)
        print(f"Loaded {len(test_jobs_data)} test jobs")

        # Convert to query format
        test_job_queries = [
            {"q_id": job_id, "query": job_text}
            for job_id, job_text in test_jobs_data.items()
        ]

        test_job_query_ids = [item['q_id'] for item in test_job_queries]

        print(f"Reconstructed {len(test_job_queries)} test job queries")

        return test_job_queries, test_job_query_ids

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Error: Could not find test job file: {e}"
        )


def load_oracle_gender_labels(oracle_path=None):
    """
    Load gender labels and create candidate index mapping.

    Args:
        oracle_path (str, optional): Path to oracle data. Uses config default if None.

    Returns:
        dict: Mapping from candidate IDs to gender labels
    """
    if oracle_path is None:
        oracle_path = Config.ORACLE_PATH

    try:
        oracle_data = pd.read_csv(oracle_path)

        # Check for gender column
        gender_column = 'Inferred_Gender'
        if gender_column not in oracle_data.columns:
            print(f"Error: Column '{gender_column}' not found in data")
            print(f"Available columns: {oracle_data.columns.tolist()}")
            return {}

        # Extract gender from oracle
        oracle_genders = oracle_data['Inferred_Gender'].tolist()

        # Map gender labels to binary (0=female, 1=male)
        gender_map = {'female': 0, 'male': 1}
        binary_genders = [
            gender_map.get(gender.lower(), -1) if pd.notna(gender) else -1
            for gender in oracle_genders
        ]

        # Create candidate indices
        candidate_indices = list(range(len(oracle_data)))

        # Create dataframe with candidate data and oracle gender
        candidate_data = pd.DataFrame({
            'candidate_index': candidate_indices,
            'gender': binary_genders[:len(candidate_indices)]
        })

        # Filter out entries with invalid gender (-1)
        valid_candidates = candidate_data[candidate_data['gender'] != -1]
        print(f"Found {len(valid_candidates)} candidates with valid gender labels")

        # Get valid candidate indices
        valid_indices = valid_candidates['candidate_index'].tolist()
        valid_genders = valid_candidates['gender'].values

        print(f"Gender distribution: Female={sum(valid_genders == 0)}, Male={sum(valid_genders == 1)}")

        # Convert binary back to string labels for bias analysis readability
        binary_to_string = {0: 'female', 1: 'male'}
        gender_mapping = {}

        for candidate_idx, gender_binary in zip(valid_indices, valid_genders):
            gender_string = binary_to_string[gender_binary]
            gender_mapping[f"resume_{candidate_idx}"] = gender_string

        print(f"Successfully created gender mapping for {len(gender_mapping)} candidates")

        # Print final gender distribution for verification
        gender_counts = pd.Series(list(gender_mapping.values())).value_counts()
        print("Final gender distribution:")
        for gender, count in gender_counts.items():
            print(f"  {gender}: {count} ({count / len(gender_mapping) * 100:.1f}%)")

        return gender_mapping

    except Exception as error:
        print(f"Error loading oracle data: {error}")
        import traceback
        traceback.print_exc()
        return {}


def load_test_job_data():
    """
    Load test job data file.

    Returns:
        tuple: (test_job_queries, test_job_query_ids)
    """
    try:
        test_jobs_file = Config.EMBEDDINGS_PATH / "sample_jobs_100.json"

        # Load test jobs data
        with open(test_jobs_file, "r") as f:
            test_jobs_data = json.load(f)
        print(f"Loaded {len(test_jobs_data)} test jobs from {test_jobs_file}")

        # Convert to query format (matching notebook logic)
        test_job_queries = [
            {"q_id": job_id, "query": job_text}
            for job_id, job_text in test_jobs_data.items()
        ]

        # Extract query IDs
        test_job_query_ids = [item['q_id'] for item in test_job_queries]

        print(f"Reconstructed {len(test_job_queries)} test job queries")

        return test_job_queries, test_job_query_ids

    except FileNotFoundError as e:
        print(f"Error: Could not find test job file: {e}")
        return [], []


def load_resume_corpus_mapping():
    """
    Load resume corpus files to create mappings.

    Returns:
        tuple: (resume_id_to_text, resume_corpus_ids)
    """
    try:
        # Load resume corpus file (jobs search for resumes)
        corpus_resumes = load_jsonl_file(Config.EMBEDDINGS_PATH / "jobs_as_queries" / "corpus.jsonl")

        # Create text mappings from corpus data
        resume_id_to_text = {item['doc_id']: item['text'] for item in corpus_resumes}

        # Create corpus ID lists from the loaded corpus
        resume_corpus_ids = [item['doc_id'] for item in corpus_resumes]

        print(f"Resume corpus size: {len(resume_corpus_ids)}")

        # Print samples to verify format (matching notebook)
        if resume_corpus_ids:
            print(f"Sample resume corpus IDs: {resume_corpus_ids[:3]}")

        return resume_id_to_text, resume_corpus_ids

    except Exception as e:
        print(f"Error loading resume corpus mappings: {e}")
        return {}, []


def extract_valid_job_test_indices(test_job_queries, job_embeddings_shape):
    """
    Extract valid test job query indices.

    Args:
        test_job_queries (list): List of test job query dictionaries
        job_embeddings_shape (tuple): Shape of job embeddings array

    Returns:
        tuple: (valid_job_indices, valid_job_query_ids)
    """
    valid_job_indices = []
    valid_job_query_ids = []

    for query_item in test_job_queries:
        job_id = query_item['q_id']
        job_index = int(job_id.split('_')[1])

        # Verify the index is within bounds
        if job_index < job_embeddings_shape[0]:
            valid_job_indices.append(job_index)
            valid_job_query_ids.append(job_id)
        else:
            print(f"Warning: Job index {job_index} out of bounds for embedding array")

    print(f"Using {len(valid_job_indices)} valid test job queries (out of {len(test_job_queries)} total)")

    return valid_job_indices, valid_job_query_ids
