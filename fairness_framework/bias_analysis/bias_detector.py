"""
Core bias detection module for fairness-aware job-matching framework.
Provides comprehensive bias analysis functions for job-matching systems.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from fairness_framework.utils.config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_score_distribution_by_gender(search_results, gender_mapping, analysis_name=""):
    """
    Analyze similarity score distributions by gender.

    Args:
        search_results (dict): Dictionary of query results with document scores
        gender_mapping (dict): Mapping of document IDs to gender labels
        analysis_name (str): Name for the analysis (for logging)

    Returns:
        dict: Statistics by gender including means, std, percentiles, and statistical tests
    """
    scores_by_gender = defaultdict(list)
    total_results = 0

    # Collect scores by gender
    for query_id, document_scores in search_results.items():
        for document_id, similarity_score in document_scores.items():
            if document_id in gender_mapping:
                gender = gender_mapping[document_id]
                scores_by_gender[gender].append(similarity_score)
                total_results += 1

    if not scores_by_gender:
        logger.warning("No results found with gender labels")
        return {}

    logger.info(f"Analyzed {total_results} query-document pairs")

    # Calculate statistics by gender
    statistics = {}
    for gender in scores_by_gender:
        scores = scores_by_gender[gender]
        statistics[gender] = {
            'count': len(scores),
            'mean': np.mean(scores),
            'std': np.std(scores),
            'median': np.median(scores),
            'percentile_25': np.percentile(scores, 25),
            'percentile_75': np.percentile(scores, 75)
        }

    # Log results
    for gender, stats in statistics.items():
        logger.info(f"{gender.upper()} candidates:")
        logger.info(f"  Count: {stats['count']}")
        logger.info(f"  Mean score: {stats['mean']:.4f}")
        logger.info(f"  Std dev: {stats['std']:.4f}")
        logger.info(f"  Median: {stats['median']:.4f}")
        logger.info(f"  25th percentile: {stats['percentile_25']:.4f}")
        logger.info(f"  75th percentile: {stats['percentile_75']:.4f}")

    # Statistical significance test
    if len(statistics) == 2:
        import scipy.stats as scipy_stats

        genders = list(statistics.keys())
        scores_group_1 = scores_by_gender[genders[0]]
        scores_group_2 = scores_by_gender[genders[1]]

        t_statistic, p_value = scipy_stats.ttest_ind(scores_group_1, scores_group_2)
        effect_size = (statistics[genders[0]]['mean'] - statistics[genders[1]]['mean']) / np.sqrt(
            (statistics[genders[0]]['std'] ** 2 + statistics[genders[1]]['std'] ** 2) / 2
        )

        logger.info(f"Statistical Test ({genders[0]} vs {genders[1]}):")
        logger.info(f"  T-statistic: {t_statistic:.4f}")
        logger.info(f"  P-value: {p_value:.4f}")
        logger.info(f"  Effect size (Cohen's d): {effect_size:.4f}")
        logger.info(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

        statistics['statistical_test'] = {
            't_statistic': t_statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05
        }

    return statistics


def analyze_top_k_representation(search_results, gender_mapping, analysis_name=""):
    """
    Analyze gender representation in top-k results.

    Args:
        search_results (dict): Dictionary of query results with document scores
        gender_mapping (dict): Mapping of document IDs to gender labels
        analysis_name (str): Name for the analysis (for logging)

    Returns:
        dict: Representation statistics for each top-k value
    """
    representation_stats = {}

    for k in Config.TOP_K_VALUES:
        gender_counts = defaultdict(int)
        total_positions = 0
        query_count = 0

        for query_id, document_scores in search_results.items():
            # Get top-k results for this query
            sorted_results = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
            top_k_docs = sorted_results[:k]

            query_count += 1

            # Count gender representation in top-k
            for doc_id, score in top_k_docs:
                if doc_id in gender_mapping:
                    gender = gender_mapping[doc_id]
                    gender_counts[gender] += 1
                    total_positions += 1

        if total_positions > 0:
            # Calculate percentages
            representation = {}
            for gender in gender_counts:
                count = gender_counts[gender]
                percentage = (count / total_positions) * 100
                representation[gender] = {
                    'count': count,
                    'percentage': percentage,
                    'avg_per_query': count / query_count
                }

            representation_stats[f'top_{k}'] = representation

            logger.info(f"Top-{k} representation ({query_count} queries, {total_positions} total positions):")
            for gender, stats in representation.items():
                logger.info(f"  {gender}: {stats['count']} ({stats['percentage']:.1f}%) - "
                            f"avg {stats['avg_per_query']:.1f} per query")

    return representation_stats


def analyze_similarity_percentiles(search_results, gender_mapping, analysis_name=""):
    """
    Analyze what percentile of similarity scores each gender achieves.

    Args:
        search_results (dict): Dictionary of query results with document scores
        gender_mapping (dict): Mapping of document IDs to gender labels
        analysis_name (str): Name for the analysis (for logging)

    Returns:
        dict: Percentile analysis by gender
    """
    # Collect all scores to calculate percentiles
    all_scores = []
    scores_by_gender = defaultdict(list)

    for query_id, document_scores in search_results.items():
        for document_id, similarity_score in document_scores.items():
            all_scores.append(similarity_score)
            if document_id in gender_mapping:
                gender = gender_mapping[document_id]
                scores_by_gender[gender].append(similarity_score)

    if not all_scores:
        logger.warning("No scores found for analysis")
        return {}

    # Calculate percentile thresholds
    percentile_thresholds = {}
    for percentile in Config.SIMILARITY_PERCENTILES:
        percentile_thresholds[percentile] = np.percentile(all_scores, percentile)

    logger.info("Similarity score percentiles:")
    for percentile, threshold in percentile_thresholds.items():
        logger.info(f"  {percentile}th percentile: {threshold:.4f}")

    # Analyze how often each gender exceeds each percentile
    percentile_analysis = {}

    for gender in scores_by_gender:
        gender_scores = scores_by_gender[gender]
        gender_analysis = {}

        for percentile, threshold in percentile_thresholds.items():
            above_threshold = sum(1 for score in gender_scores if score >= threshold)
            percentage = (above_threshold / len(gender_scores)) * 100
            gender_analysis[f'above_{percentile}th'] = {
                'count': above_threshold,
                'total': len(gender_scores),
                'percentage': percentage
            }

        percentile_analysis[gender] = gender_analysis

    logger.info("Gender performance by percentile:")
    for gender, analysis in percentile_analysis.items():
        logger.info(f"{gender.upper()}:")
        for percentile_key, stats in analysis.items():
            logger.info(f"  {percentile_key}: {stats['count']}/{stats['total']} "
                        f"({stats['percentage']:.1f}%)")

    return percentile_analysis


def analyze_score_distribution_shapes(search_results, gender_mapping):
    """
    Analyze the shape characteristics of score distributions.

    Args:
        search_results (dict): Dictionary of query results with document scores
        gender_mapping (dict): Mapping of document IDs to gender labels

    Returns:
        dict: Distribution shape metrics by gender
    """
    import scipy.stats as scipy_stats

    scores_by_gender = defaultdict(list)

    # Collect scores
    for query_id, document_scores in search_results.items():
        for document_id, similarity_score in document_scores.items():
            if document_id in gender_mapping:
                gender = gender_mapping[document_id]
                scores_by_gender[gender].append(similarity_score)

    shape_analysis = {}
    for gender, scores in scores_by_gender.items():
        # Distribution shape metrics
        skewness = scipy_stats.skew(scores)
        kurtosis = scipy_stats.kurtosis(scores)

        # Tail analysis
        top_1_percent = np.percentile(scores, 99)
        top_5_percent = np.percentile(scores, 95)

        # Count extreme scores
        extreme_high = sum(1 for s in scores if s > top_1_percent)

        shape_metrics = {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'top_1_percent_threshold': top_1_percent,
            'top_5_percent_threshold': top_5_percent,
            'extreme_high_count': extreme_high
        }

        shape_analysis[gender] = shape_metrics

        logger.info(f"{gender.upper()} Distribution Shape:")
        logger.info(f"  Skewness: {skewness:.4f}")
        logger.info(f"  Kurtosis: {kurtosis:.4f}")
        logger.info(f"  99th percentile: {top_1_percent:.4f}")
        logger.info(f"  95th percentile: {top_5_percent:.4f}")
        logger.info(f"  Candidates above 99th percentile: {extreme_high}")

    return shape_analysis


def analyze_per_query_ranking_patterns(search_results, gender_mapping):
    """
    Analyze ranking patterns within individual queries.

    Args:
        search_results (dict): Dictionary of query results with document scores
        gender_mapping (dict): Mapping of document IDs to gender labels

    Returns:
        list: Per-query ranking statistics
    """
    query_ranking_stats = []

    for query_id, document_scores in search_results.items():
        # Sort by similarity score
        ranked_results = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)

        # Analyze gender positions in ranking
        gender_positions = defaultdict(list)
        for rank, (doc_id, score) in enumerate(ranked_results, 1):
            if doc_id in gender_mapping:
                gender = gender_mapping[doc_id]
                gender_positions[gender].append(rank)

        # Calculate mean rank by gender for this query
        query_stats = {'query_id': query_id}
        for gender, positions in gender_positions.items():
            if positions:
                query_stats[f'{gender}_mean_rank'] = np.mean(positions)
                query_stats[f'{gender}_min_rank'] = min(positions)
                query_stats[f'{gender}_count'] = len(positions)

        query_ranking_stats.append(query_stats)

    # Aggregate statistics
    logger.info("Per-Query Ranking Analysis:")
    female_mean_ranks = [q.get('female_mean_rank', 0) for q in query_ranking_stats if 'female_mean_rank' in q]
    male_mean_ranks = [q.get('male_mean_rank', 0) for q in query_ranking_stats if 'male_mean_rank' in q]

    if female_mean_ranks and male_mean_ranks:
        logger.info(f"  Average rank across queries:")
        logger.info(f"    Female: {np.mean(female_mean_ranks):.2f}")
        logger.info(f"    Male: {np.mean(male_mean_ranks):.2f}")

        # Test if ranking differences are consistent
        import scipy.stats as scipy_stats
        rank_diff_ttest, rank_diff_p = scipy_stats.ttest_ind(female_mean_ranks, male_mean_ranks)
        logger.info(f"  Ranking difference significance: p={rank_diff_p:.4f}")

    return query_ranking_stats


def analyze_score_rank_gaps(search_results, gender_mapping):
    """
    Analyze the relationship between score differences and ranking differences.

    Args:
        search_results (dict): Dictionary of query results with document scores
        gender_mapping (dict): Mapping of document IDs to gender labels

    Returns:
        list: Query analysis results with score and ranking patterns
    """
    query_analysis = []

    for query_id, document_scores in search_results.items():
        ranked_results = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)

        # Find gender-based score patterns within this query
        gender_scores = defaultdict(list)
        for doc_id, score in document_scores.items():
            if doc_id in gender_mapping:
                gender = gender_mapping[doc_id]
                gender_scores[gender].append(score)

        if len(gender_scores) == 2:  # Both genders present
            genders = list(gender_scores.keys())
            scores_1 = gender_scores[genders[0]]
            scores_2 = gender_scores[genders[1]]

            # Compare score distributions within this query
            mean_diff = np.mean(scores_1) - np.mean(scores_2)

            # Analyze top positions
            top_10_docs = [doc_id for doc_id, score in ranked_results[:10]]
            top_10_gender_counts = defaultdict(int)
            for doc_id in top_10_docs:
                if doc_id in gender_mapping:
                    gender = gender_mapping[doc_id]
                    top_10_gender_counts[gender] += 1

            query_analysis.append({
                'query_id': query_id,
                'score_diff': mean_diff,
                'top_10_counts': dict(top_10_gender_counts)
            })

    # Aggregate analysis
    logger.info("Score vs Rank Gap Analysis:")
    score_diffs = [q['score_diff'] for q in query_analysis]
    logger.info(f"  Average score difference across queries: {np.mean(score_diffs):.4f}")
    logger.info(f"  Std of score differences: {np.std(score_diffs):.4f}")

    # Analyze cases where scores favor one gender but rankings favor another
    conflicting_cases = 0
    for q in query_analysis:
        top_10 = q['top_10_counts']
        if 'female' in top_10 and 'male' in top_10:
            female_top10_pct = top_10['female'] / sum(top_10.values())
            if q['score_diff'] > 0 and female_top10_pct < 0.5:  # Female scores higher but less top-10 representation
                conflicting_cases += 1
            elif q['score_diff'] < 0 and female_top10_pct > 0.5:  # Male scores higher but less top-10 representation
                conflicting_cases += 1

    logger.info(f"  Queries with score-rank conflicts: {conflicting_cases}/{len(query_analysis)}")

    return query_analysis


def analyze_embedding_geometry(embeddings, gender_mapping, corpus_ids):
    """
    Analyze geometric properties of embeddings by gender.

    Args:
        embeddings (np.ndarray): Embedding matrix
        gender_mapping (dict): Mapping of document IDs to gender labels
        corpus_ids (list): List of corpus document IDs

    Returns:
        dict: Geometric analysis results by gender
    """
    # Map resume IDs to indices
    id_to_index = {resume_id: idx for idx, resume_id in enumerate(corpus_ids)}

    embeddings_by_gender = defaultdict(list)

    for resume_id, gender in gender_mapping.items():
        if resume_id in id_to_index:
            idx = id_to_index[resume_id]
            embedding = embeddings[idx]
            embeddings_by_gender[gender].append(embedding)

    geometry_analysis = {}
    logger.info("Embedding Geometry Analysis:")

    for gender, embeddings_list in embeddings_by_gender.items():
        embeddings_array = np.array(embeddings_list)

        # Calculate centroid and spread
        centroid = np.mean(embeddings_array, axis=0)
        distances_from_centroid = np.linalg.norm(embeddings_array - centroid, axis=1)

        # Analyze embedding norms
        norms = np.linalg.norm(embeddings_array, axis=1)

        geometry_metrics = {
            'mean_distance_from_centroid': np.mean(distances_from_centroid),
            'std_distance_from_centroid': np.std(distances_from_centroid),
            'mean_embedding_norm': np.mean(norms),
            'std_embedding_norm': np.std(norms)
        }

        geometry_analysis[gender] = geometry_metrics

        logger.info(f"  {gender.upper()}:")
        logger.info(f"    Mean distance from centroid: {geometry_metrics['mean_distance_from_centroid']:.4f}")
        logger.info(f"    Std distance from centroid: {geometry_metrics['std_distance_from_centroid']:.4f}")
        logger.info(f"    Mean embedding norm: {geometry_metrics['mean_embedding_norm']:.4f}")
        logger.info(f"    Std embedding norm: {geometry_metrics['std_embedding_norm']:.4f}")

    return geometry_analysis


def run_bias_analysis(job_embeddings, resume_embeddings, test_job_queries,
                      gender_mapping, resume_corpus_ids, create_faiss_index, search_faiss):
    """
    Run comprehensive bias analysis for jobs searching resumes.

    Args:
        job_embeddings (np.ndarray): Job embedding matrix
        resume_embeddings (np.ndarray): Resume embedding matrix
        test_job_queries (list): List of test job query dictionaries
        gender_mapping (dict): Mapping of resume IDs to gender labels
        resume_corpus_ids (list): List of resume corpus IDs
        create_faiss_index (function): FAISS index creation function
        search_faiss (function): FAISS search function

    Returns:
        dict: Complete bias analysis results
    """
    logger.info("Bias analysis: Jobs searching for Resumes")

    # Extract test job query indices from job IDs
    test_job_indices = []
    valid_test_job_ids = []

    for query_item in test_job_queries:
        job_id = query_item['q_id']
        job_index = int(job_id.split('_')[1])

        # Verify the index is within bounds
        if job_index < job_embeddings.shape[0]:
            test_job_indices.append(job_index)
            valid_test_job_ids.append(job_id)
        else:
            logger.warning(f"Job index {job_index} out of bounds for embedding array")

    logger.info(f"Using {len(test_job_indices)} valid test job queries (out of {len(test_job_queries)} total)")

    if not test_job_indices:
        logger.error("No valid test job queries found")
        return {}

    # Get embeddings for test job queries
    test_job_query_embeddings = job_embeddings[test_job_indices]
    logger.info(f"Extracted job query embeddings shape: {test_job_query_embeddings.shape}")

    # Create FAISS index for resume corpus
    logger.info("Creating FAISS index for resume corpus...")
    resume_faiss_index = create_faiss_index(resume_embeddings, resume_corpus_ids)

    # Run search
    logger.info("Running FAISS search: jobs → resumes...")
    search_results = search_faiss(
        q_embeddings=test_job_query_embeddings,
        q_ids=valid_test_job_ids,
        d_ids=resume_corpus_ids,
        index=resume_faiss_index,
        top_k=max(Config.TOP_K_VALUES),
        show_progress=True
    )

    logger.info(f"Search completed. Analyzing {len(search_results)} query results...")

    # Run bias analyses
    score_stats = analyze_score_distribution_by_gender(
        search_results, gender_mapping, "Jobs → Resumes"
    )

    representation_stats = analyze_top_k_representation(
        search_results, gender_mapping, "Jobs → Resumes"
    )

    percentile_stats = analyze_similarity_percentiles(
        search_results, gender_mapping, "Jobs → Resumes"
    )

    # Additional analyses
    shape_stats = analyze_score_distribution_shapes(search_results, gender_mapping)
    ranking_patterns = analyze_per_query_ranking_patterns(search_results, gender_mapping)
    score_rank_gaps = analyze_score_rank_gaps(search_results, gender_mapping)
    geometry_stats = analyze_embedding_geometry(resume_embeddings, gender_mapping, resume_corpus_ids)

    return {
        'search_results': search_results,
        'score_distribution': score_stats,
        'top_k_representation': representation_stats,
        'percentile_analysis': percentile_stats,
        'distribution_shapes': shape_stats,
        'ranking_patterns': ranking_patterns,
        'score_rank_gaps': score_rank_gaps,
        'embedding_geometry': geometry_stats
    }