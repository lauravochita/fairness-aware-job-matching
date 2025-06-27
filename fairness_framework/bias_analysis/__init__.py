"""
Bias detection and analysis modules.
"""

from .bias_detector import (
    analyze_score_distribution_by_gender,
    analyze_top_k_representation,
    analyze_similarity_percentiles,
    analyze_score_distribution_shapes,
    analyze_per_query_ranking_patterns,
    analyze_score_rank_gaps,
    analyze_embedding_geometry,
    run_bias_analysis
)

from .statistical_tests import (
    test_representation_significance,
    test_representation_significance_with_correction,
    calculate_representation_effect_sizes,
    calculate_statistical_power,
    calculate_similarity_confidence_intervals,
    bootstrap_key_metrics,
    test_variance_equality,
    validate_population_distributions,
    analyze_faiss_ranking_behavior,
    comprehensive_statistical_validation
)

from .validation import (
    verify_ranking_follows_similarity,
    manual_spot_check,
    investigate_similarity_ranking_paradox,
    analyze_per_query_gender_patterns,
    comprehensive_validation_suite
)

__all__ = [
    # Bias detector
    'analyze_score_distribution_by_gender',
    'analyze_top_k_representation',
    'analyze_similarity_percentiles',
    'analyze_score_distribution_shapes',
    'analyze_per_query_ranking_patterns',
    'analyze_score_rank_gaps',
    'analyze_embedding_geometry',
    'run_bias_analysis',

    # Statistical tests
    'test_representation_significance',
    'test_representation_significance_with_correction',
    'calculate_representation_effect_sizes',
    'calculate_statistical_power',
    'calculate_similarity_confidence_intervals',
    'bootstrap_key_metrics',
    'test_variance_equality',
    'validate_population_distributions',
    'analyze_faiss_ranking_behavior',
    'comprehensive_statistical_validation',

    # Validation
    'verify_ranking_follows_similarity',
    'manual_spot_check',
    'investigate_similarity_ranking_paradox',
    'analyze_per_query_gender_patterns',
    'comprehensive_validation_suite',
]
