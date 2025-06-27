"""
Counterfactual analysis and bias mitigation modules.
"""
from .gender_classifier import (
    train_gender_classifier_with_teacher_params,
    analyze_gender_directions,
    validate_gender_classifier_robustness,
    analyze_gender_direction_interpretability,
    visualize_gender_classifier_analysis,
    comprehensive_gender_classifier_analysis
)

from .manipulation import (
    flip_embeddings_along_gender_direction,
    test_flip_factor_optimality
)

from .post_intervention_bias import (
    run_counterfactual_bias_analysis,
    compare_bias_results,
    validate_bias_reduction_stability,
    cross_validate_counterfactual_method,
    test_post_intervention_representation,
    analyze_resumes_to_jobs_representation,
    comprehensive_dosage_response_analysis,
    analyze_intervention_success_rates
)

__all__ = [
    # Gender classifier
    'train_gender_classifier_with_teacher_params',
    'analyze_gender_directions',
    'validate_gender_classifier_robustness',
    'analyze_gender_direction_interpretability',
    'visualize_gender_classifier_analysis',
    'comprehensive_gender_classifier_analysis',

    # Manipulation
    'flip_embeddings_along_gender_direction',
    'test_flip_factor_optimality',

    # Post-intervention bias
    'run_counterfactual_bias_analysis',
    'compare_bias_results',
    'validate_bias_reduction_stability',
    'cross_validate_counterfactual_method',
    'test_post_intervention_representation',
    'analyze_resumes_to_jobs_representation',
    'comprehensive_dosage_response_analysis',
    'analyze_intervention_success_rates',
]
