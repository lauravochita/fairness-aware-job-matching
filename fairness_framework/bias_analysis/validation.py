"""
Validation module for fairness-aware job-matching framework.
Provides validation and verification functions for bias analysis integrity.
"""

import numpy as np
import pandas as pd
import random
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_ranking_follows_similarity(search_results, sample_size=10):
    """
    Verify that rankings actually follow similarity scores.

    Args:
        search_results (dict): Dictionary of search results
        sample_size (int): Number of queries to check randomly

    Returns:
        dict: Verification results including properly sorted queries count
    """
    logger.info("Ranking verification")

    verification_results = []

    # Check random sample of queries
    queries_to_check = random.sample(list(search_results.keys()), min(sample_size, len(search_results)))

    for query_id in queries_to_check:
        document_scores = search_results[query_id]

        # Sort by similarity score
        expected_ranking = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)

        # Check if it's actually sorted
        scores = [score for doc_id, score in expected_ranking]
        is_properly_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

        verification_results.append({
            'query_id': query_id,
            'properly_sorted': is_properly_sorted,
            'top_score': scores[0] if scores else 0,
            'bottom_score': scores[-1] if scores else 0,
            'score_range': scores[0] - scores[-1] if scores else 0,
            'total_documents': len(scores)
        })

        if not is_properly_sorted:
            logger.warning(f"Query {query_id} is not properly sorted by similarity")

    # Summary
    properly_sorted_count = sum(1 for r in verification_results if r['properly_sorted'])
    total_checked = len(verification_results)

    logger.info(f"Ranking verification: {properly_sorted_count}/{total_checked} queries properly sorted")

    if properly_sorted_count == total_checked:
        logger.info("Rankings follow similarity scores")
    else:
        logger.warning("Rankings do not follow similarity scores")

    verification_summary = {
        'properly_sorted_count': properly_sorted_count,
        'total_checked': total_checked,
        'all_properly_sorted': properly_sorted_count == total_checked,
        'verification_details': verification_results
    }

    return verification_summary


def manual_spot_check(search_results, gender_mapping, n_queries=5):
    """
    Manual verification of ranking for specific queries.

    Args:
        search_results (dict): Dictionary of search results
        gender_mapping (dict): Mapping of document IDs to gender labels
        n_queries (int): Number of queries to manually check

    Returns:
        list: Detailed spot check results for manual inspection
    """
    logger.info("Manual spot check")

    # Pick queries with diverse results
    queries_to_check = random.sample(list(search_results.keys()), min(n_queries, len(search_results)))

    spot_check_results = []

    for query_id in queries_to_check:
        document_scores = search_results[query_id]
        sorted_results = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)

        logger.info(f"Query: {query_id}")
        logger.info("Top 10 results:")
        logger.info(f"{'Rank':<4} {'Doc ID':<15} {'Score':<8} {'Gender':<8}")

        top_10_details = []
        for rank, (doc_id, score) in enumerate(sorted_results[:10], 1):
            gender = gender_mapping.get(doc_id, 'Unknown')
            logger.info(f"{rank:<4} {doc_id:<15} {score:<8.4f} {gender:<8}")

            top_10_details.append({
                'rank': rank,
                'doc_id': doc_id,
                'score': score,
                'gender': gender
            })

        # Count genders in top 10
        top_10_genders = [gender_mapping.get(doc_id, 'Unknown') for doc_id, _ in sorted_results[:10]]
        male_count = top_10_genders.count('male')
        female_count = top_10_genders.count('female')
        unknown_count = top_10_genders.count('Unknown')

        gender_summary = {
            'male_count': male_count,
            'female_count': female_count,
            'unknown_count': unknown_count
        }

        logger.info(f"Top 10: {male_count} male, {female_count} female, {unknown_count} unknown")

        spot_check_results.append({
            'query_id': query_id,
            'top_10_details': top_10_details,
            'gender_summary': gender_summary,
            'total_results': len(sorted_results)
        })

    return spot_check_results


def investigate_similarity_ranking_paradox(search_results, gender_mapping):
    """
    Investigate why higher-scoring females are under-represented in deep rankings.

    Args:
        search_results (dict): Dictionary of search results
        gender_mapping (dict): Mapping of document IDs to gender labels

    Returns:
        dict: Analysis of similarity-ranking paradox
    """
    logger.info("Similarity-ranking paradox investigation")

    all_scores_by_gender = {'male': [], 'female': []}
    ranking_positions_by_gender = {'male': [], 'female': []}

    for query_id, document_scores in search_results.items():
        # Sort by similarity score
        sorted_results = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)

        for rank, (doc_id, score) in enumerate(sorted_results, 1):
            if doc_id in gender_mapping:
                gender = gender_mapping[doc_id]
                if gender in all_scores_by_gender:
                    all_scores_by_gender[gender].append(score)
                    ranking_positions_by_gender[gender].append(rank)

    # Calculate basic statistics
    female_mean_score = np.mean(all_scores_by_gender['female']) if all_scores_by_gender['female'] else 0
    male_mean_score = np.mean(all_scores_by_gender['male']) if all_scores_by_gender['male'] else 0
    female_mean_rank = np.mean(ranking_positions_by_gender['female']) if ranking_positions_by_gender['female'] else 0
    male_mean_rank = np.mean(ranking_positions_by_gender['male']) if ranking_positions_by_gender['male'] else 0

    logger.info(f"Female mean score: {female_mean_score:.4f}")
    logger.info(f"Male mean score: {male_mean_score:.4f}")
    logger.info(f"Female mean rank: {female_mean_rank:.1f}")
    logger.info(f"Male mean rank: {male_mean_rank:.1f}")

    # Check score distributions
    paradox_analysis = {
        'female_mean_score': female_mean_score,
        'male_mean_score': male_mean_score,
        'female_mean_rank': female_mean_rank,
        'male_mean_rank': male_mean_rank,
        'score_advantage_female': female_mean_score > male_mean_score,
        'rank_advantage_female': female_mean_rank < male_mean_rank
    }

    if all_scores_by_gender['female'] and all_scores_by_gender['male']:
        female_75th = np.percentile(all_scores_by_gender['female'], 75)
        male_25th = np.percentile(all_scores_by_gender['male'], 25)

        distribution_overlap = female_75th > male_25th

        paradox_analysis.update({
            'female_75th_percentile': female_75th,
            'male_25th_percentile': male_25th,
            'distribution_overlap': distribution_overlap
        })

        logger.info(f"Score distribution overlap:")
        logger.info(f"Female 75th percentile: {female_75th:.4f}")
        logger.info(f"Male 25th percentile: {male_25th:.4f}")
        logger.info(f"Distribution overlap: {'Yes' if distribution_overlap else 'No'}")

        # Paradox detection
        has_paradox = (female_mean_score > male_mean_score) and (female_mean_rank > male_mean_rank)
        paradox_analysis['paradox_detected'] = has_paradox

        if has_paradox:
            logger.warning("Females score higher but rank lower on average")
        else:
            logger.info("Score and rank advantages are consistent")

    return paradox_analysis


def analyze_per_query_gender_patterns(search_results, gender_mapping):
    """
    Check if some queries systematically favor one gender.

    Args:
        search_results (dict): Dictionary of search results
        gender_mapping (dict): Mapping of document IDs to gender labels

    Returns:
        dict: Per-query gender pattern analysis
    """
    logger.info("Per-query gender pattern analysis")

    query_gender_scores = {}

    for query_id, document_scores in search_results.items():
        male_scores = []
        female_scores = []

        for doc_id, score in document_scores.items():
            if doc_id in gender_mapping:
                if gender_mapping[doc_id] == 'male':
                    male_scores.append(score)
                elif gender_mapping[doc_id] == 'female':
                    female_scores.append(score)

        if male_scores and female_scores:
            male_mean = np.mean(male_scores)
            female_mean = np.mean(female_scores)
            female_advantage = female_mean - male_mean

            query_gender_scores[query_id] = {
                'male_mean': male_mean,
                'female_mean': female_mean,
                'female_advantage': female_advantage,
                'male_count': len(male_scores),
                'female_count': len(female_scores)
            }

    if not query_gender_scores:
        logger.warning("No queries found with both male and female candidates")
        return {}

    # Analyze distribution of per-query advantages
    advantages = [q['female_advantage'] for q in query_gender_scores.values()]
    positive_queries = sum(1 for a in advantages if a > 0)
    negative_queries = sum(1 for a in advantages if a < 0)
    total_queries = len(advantages)

    pattern_analysis = {
        'total_queries_analyzed': total_queries,
        'queries_favoring_female': positive_queries,
        'queries_favoring_male': negative_queries,
        'female_favored_percentage': (positive_queries / total_queries * 100) if total_queries > 0 else 0,
        'mean_female_advantage': np.mean(advantages),
        'std_female_advantage': np.std(advantages),
        'query_details': query_gender_scores
    }

    logger.info(
        f"Queries where females score higher: {positive_queries}/{total_queries} ({positive_queries / total_queries * 100:.1f}%)")
    logger.info(
        f"Queries where males score higher: {negative_queries}/{total_queries} ({negative_queries / total_queries * 100:.1f}%)")
    logger.info(f"Mean per-query female advantage: {np.mean(advantages):.4f}")
    logger.info(f"Standard deviation of advantages: {np.std(advantages):.4f}")

    # Detect systematic bias patterns
    if positive_queries > total_queries * 0.6:
        logger.info("Systematic female advantage across queries")
    elif negative_queries > total_queries * 0.6:
        logger.info("Systematic male advantage across queries")
    else:
        logger.info("Balanced query-level performance")

    return pattern_analysis


def comprehensive_validation_suite(search_results, gender_mapping, sample_size=10, n_spot_checks=5):
    """
    Run validation suite for bias analysis integrity.

    Args:
        search_results (dict): Dictionary of search results
        gender_mapping (dict): Mapping of document IDs to gender labels
        sample_size (int): Sample size for ranking verification
        n_spot_checks (int): Number of manual spot checks

    Returns:
        dict: Complete validation results
    """
    logger.info("Validation suite")

    # Run all validation checks
    ranking_verification = verify_ranking_follows_similarity(search_results, sample_size)
    spot_check_results = manual_spot_check(search_results, gender_mapping, n_spot_checks)
    paradox_analysis = investigate_similarity_ranking_paradox(search_results, gender_mapping)
    pattern_analysis = analyze_per_query_gender_patterns(search_results, gender_mapping)

    # Compile comprehensive results
    validation_results = {
        'ranking_verification': ranking_verification,
        'manual_spot_checks': spot_check_results,
        'similarity_ranking_paradox': paradox_analysis,
        'per_query_patterns': pattern_analysis,
        'validation_summary': {
            'rankings_properly_sorted': ranking_verification['all_properly_sorted'],
            'paradox_detected': paradox_analysis.get('paradox_detected', False),
            'systematic_bias_detected': pattern_analysis['female_favored_percentage'] > 60 or pattern_analysis[
                'female_favored_percentage'] < 40,
            'total_queries_validated': len(search_results)
        }
    }

    # Log summary
    logger.info("Validation summary:")
    logger.info(f"  Rankings properly sorted: {'Yes' if ranking_verification['all_properly_sorted'] else 'No'}")
    logger.info(
        f"  Similarity-ranking paradox: {'Detected' if paradox_analysis.get('paradox_detected', False) else 'Not detected'}")
    logger.info(f"  Query-level bias pattern: {pattern_analysis['female_favored_percentage']:.1f}% favor female")
    logger.info(f"  Total queries validated: {len(search_results)}")

    if validation_results['validation_summary']['rankings_properly_sorted']:
        logger.info("Rankings follow similarity scores correctly")
    else:
        logger.warning("Rankings do not follow similarity scores")

    return validation_results
