"""
Fairness-aware job-matching framework.
A comprehensive toolkit for detecting and mitigating bias in job-matching systems.
"""

__version__ = "1.0.0"

# Core configuration
from .utils.config import Config, ModelConfig

# Data loading
from .data.data_loader import (
    load_embeddings,
    load_oracle_data,
    load_oracle_gender_labels,
    load_test_job_data,
    load_resume_corpus_mapping,
    extract_valid_job_test_indices,
    prepare_data_with_oracle,
    load_bias_analysis_embeddings
)

# Bias analysis
from .bias_analysis.bias_detector import (
    analyze_score_distribution_by_gender,
    analyze_top_k_representation,
    analyze_similarity_percentiles,
    run_bias_analysis
)

from .bias_analysis.statistical_tests import (
    test_representation_significance,
    comprehensive_statistical_validation
)

from .bias_analysis.validation import (
    comprehensive_validation_suite
)

# Counterfactual analysis
from .counterfactual.gender_classifier import (
    train_gender_classifier_with_teacher_params,
    analyze_gender_directions,
    comprehensive_gender_classifier_analysis,
)

from .counterfactual.manipulation import (
    flip_embeddings_along_gender_direction
)

from .counterfactual.post_intervention_bias import (
    run_counterfactual_bias_analysis,
    compare_bias_results
)

# Models
from .models.vae_baseline import (
    VAE_Gender,
    train_baseline_vae,
    extract_latent_attributes
)

from .models.semi_supervised_vae import (
    ImprovedSemiSupervisedVAE,
    run_semi_supervised_gender_vae_fixed
)

__all__ = [
    # Core
    'Config',
    'ModelConfig',

    # Data
    'load_embeddings',
    'load_oracle_data',
    'load_oracle_gender_labels',
    'load_test_job_data',
    'load_resume_corpus_mapping',
    'extract_valid_job_test_indices',
    'prepare_data_with_oracle',
    'load_bias_analysis_embeddings',

    # Bias Analysis
    'analyze_score_distribution_by_gender',
    'analyze_top_k_representation',
    'analyze_similarity_percentiles',
    'run_bias_analysis',
    'test_representation_significance',
    'comprehensive_statistical_validation',
    'comprehensive_validation_suite',

    # Counterfactual
    'train_gender_classifier_with_teacher_params',
    'analyze_gender_directions',
    'comprehensive_gender_classifier_analysis',
    'flip_embeddings_along_gender_direction',
    'run_counterfactual_bias_analysis',
    'compare_bias_results',

    # Models
    'VAE_Gender',
    'train_baseline_vae',
    'extract_latent_attributes',
    'ImprovedSemiSupervisedVAE',
    'run_semi_supervised_gender_vae_fixed',
]