"""
Configuration module for fairness-aware job-matching framework.
Handles all path configurations and system parameters.
"""

import os
from pathlib import Path


class Config:
    """Central configuration class for all paths and parameters."""

    # Base thesis directory paths
    BASE_DATA_PATH = Path("/content/drive/MyDrive/Thesis/")

    # Data paths
    EMBEDDINGS_PATH = BASE_DATA_PATH / "complete_job_matching_embeddings"
    ORACLE_PATH = BASE_DATA_PATH / "FairWS-main/eval_dataset.csv"
    DATASETS_PATH = BASE_DATA_PATH / "Datasets"

    # Model paths
    FINE_TUNED_MODEL_PATH = BASE_DATA_PATH / "Embedding/e5-large_mixed_training/"

    # Proprietary system paths
    PROPRIETARY_SRC_PATH = str(BASE_DATA_PATH / "fine_tuning/src")
    PROPRIETARY_INFERENCE_PATH = str(BASE_DATA_PATH / "fine_tuning/inference_endpoint/src")

    # Analysis parameters
    TOP_K_VALUES = [5, 10, 20, 50]
    SIMILARITY_PERCENTILES = [50, 75, 90, 95]
    FLIP_FACTORS = [0.25, 0.5, 0.75, 1.0]

    # Population proportions
    POPULATION_PROPORTIONS = {
        'male': 5583 / 10013,
        'female': 4430 / 10013
    }

    # Training parameters
    RANDOM_SEED = 42
    DEVICE = "cuda"  # Will fallback to cpu if cuda unavailable

    @classmethod
    def update_base_path(cls, new_base_path):
        """
        Update the base path and all derived paths.

        Args:
            new_base_path (str or Path): New base directory path
        """
        cls.BASE_DATA_PATH = Path(new_base_path)
        cls.EMBEDDINGS_PATH = cls.BASE_DATA_PATH / "complete_job_matching_embeddings"
        cls.ORACLE_PATH = cls.BASE_DATA_PATH / "Datasets/eval_dataset.csv"
        cls.DATASETS_PATH = cls.BASE_DATA_PATH / "Datasets"
        cls.FINE_TUNED_MODEL_PATH = cls.BASE_DATA_PATH / "Embedding/e5-large_mixed_training/"
        cls.PROPRIETARY_SRC_PATH = str(cls.BASE_DATA_PATH / "fine_tuning/src")
        cls.PROPRIETARY_INFERENCE_PATH = str(cls.BASE_DATA_PATH / "fine_tuning/inference_endpoint/src")

    @classmethod
    def get_embedding_files(cls):
        """Get dictionary of embedding file paths."""
        return {
            'jobs': cls.EMBEDDINGS_PATH / "jobs.npy",
            'resumes': cls.EMBEDDINGS_PATH / "resumes.npy",
            'jobs_as_queries': cls.EMBEDDINGS_PATH / "jobs_as_queries.npy",
            'cvs_as_queries': cls.EMBEDDINGS_PATH / "cvs_as_queries.npy",
            'cvs_as_passages': cls.EMBEDDINGS_PATH / "cvs_as_passages.npy",
            'embedding_mapping': cls.EMBEDDINGS_PATH / "embedding_mapping.csv"
        }

    @classmethod
    def get_training_data_paths(cls, query_type):
        """
        Get training data paths for specific query type.

        Args:
            query_type (str): Either 'jobs_as_queries' or 'cvs_as_queries'

        Returns:
            dict: Dictionary of file paths for training data
        """
        base_path = cls.EMBEDDINGS_PATH / query_type
        return {
            'train_queries': base_path / "train_queries.jsonl",
            'dev_queries': base_path / "dev_queries.jsonl",
            'test_queries': base_path / "test_queries.jsonl",
            'corpus': base_path / "corpus.jsonl",
            'negative_samples': base_path / f"negative_{'resumes' if 'jobs' in query_type else 'jobs'}.json"
        }


class ModelConfig:
    """Model-specific configuration parameters."""

    # Semi-supervised VAE parameters
    VAE_HIDDEN_DIM = 512
    VAE_Z_DIM = 32
    VAE_A_DIM = 2
    VAE_LEARNING_RATE = 0.0002
    VAE_WEIGHT_DECAY = 1e-6
    VAE_BATCH_SIZE = 32
    VAE_MAX_EPOCHS = 150
    VAE_PATIENCE = 25

    # Teacher classifier parameters
    TEACHER_CV_FOLDS = 5
    TEACHER_C_VALUES = [0.1, 1.0, 10.0]
    TEACHER_MAX_ITER = 1000

    # Counterfactual manipulation parameters
    GENDER_CLASSIFIER_CV_FOLDS = 5
    PERMUTATION_TEST_ITERATIONS = 50
    BOOTSTRAP_SAMPLES = 1000

    # Bias analysis parameters
    TEST_SAMPLE_SIZE = 100  # Number of test queries
    SIGNIFICANCE_LEVEL = 0.05