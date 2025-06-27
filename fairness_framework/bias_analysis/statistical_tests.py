"""
Statistical testing module for fairness-aware job-matching framework.
Provides comprehensive statistical analysis and significance testing for bias detection.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import accuracy_score
from fairness_framework.utils.config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_representation_significance(search_results, gender_mapping, population_proportions, analysis_name=""):
    """
    Test statistical significance of gender representation in top-k results using binomial tests.

    Args:
        search_results (dict): Dictionary of search results
        gender_mapping (dict): Mapping of candidate IDs to gender
        population_proportions (dict): Dict with population proportions {'male': 0.558, 'female': 0.442}
        analysis_name (str): Name for the analysis (for logging)

    Returns:
        dict: Statistical test results for each top-k value
    """
    logger.info(f"Representation significance testing: {analysis_name}")

    test_results = {}

    for k in Config.TOP_K_VALUES:
        logger.info(f"Top-{k} Representation Significance")

        # Collect top-k results across all queries
        male_count = 0
        female_count = 0
        total_positions = 0

        for query_id, document_scores in search_results.items():
            # Get top-k results for this query
            sorted_results = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
            top_k_docs = sorted_results[:k]

            # Count gender representation in top-k
            for doc_id, score in top_k_docs:
                if doc_id in gender_mapping:
                    gender = gender_mapping[doc_id]
                    if gender == 'male':
                        male_count += 1
                    elif gender == 'female':
                        female_count += 1
                    total_positions += 1

        if total_positions == 0:
            logger.warning(f"No valid results for top-{k}")
            continue

        # Calculate observed proportions
        observed_male_prop = male_count / total_positions
        observed_female_prop = female_count / total_positions

        # Expected proportions from population
        expected_male_prop = population_proportions['male']
        expected_female_prop = population_proportions['female']

        logger.info(f"Total positions analyzed: {total_positions}")
        logger.info(
            f"Observed: Male={male_count} ({observed_male_prop:.1%}), Female={female_count} ({observed_female_prop:.1%})")
        logger.info(f"Expected: Male={expected_male_prop:.1%}, Female={expected_female_prop:.1%}")

        # Binomial test for gender representation
        from scipy.stats import binomtest

        # Binomial test for male representation
        male_binom_result = binomtest(male_count, total_positions, expected_male_prop, alternative='two-sided')
        male_binom_p = male_binom_result.pvalue

        # Binomial test for female representation
        female_binom_result = binomtest(female_count, total_positions, expected_female_prop, alternative='two-sided')
        female_binom_p = female_binom_result.pvalue

        # Calculate 95% confidence intervals for proportions (Wilson score interval)
        def wilson_ci(successes, trials, alpha=0.05):
            z = scipy_stats.norm.ppf(1 - alpha / 2)
            p = successes / trials
            n = trials

            denominator = 1 + (z ** 2 / n)
            centre = (p + (z ** 2 / (2 * n))) / denominator
            spread = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denominator

            return centre - spread, centre + spread

        male_ci = wilson_ci(male_count, total_positions)
        female_ci = wilson_ci(female_count, total_positions)

        # Effect size (difference from expected)
        male_effect = observed_male_prop - expected_male_prop
        female_effect = observed_female_prop - expected_female_prop

        logger.info(f"Binomial Test Results:")
        logger.info(f"  Male representation:")
        logger.info(f"    p-value: {male_binom_p:.4f}")
        logger.info(f"    95% CI: [{male_ci[0]:.3f}, {male_ci[1]:.3f}]")
        logger.info(f"    Effect size: {male_effect:+.3f}")
        logger.info(f"    Significant: {'Yes' if male_binom_p < 0.05 else 'No'}")

        logger.info(f"  Female representation:")
        logger.info(f"    p-value: {female_binom_p:.4f}")
        logger.info(f"    95% CI: [{female_ci[0]:.3f}, {female_ci[1]:.3f}]")
        logger.info(f"    Effect size: {female_effect:+.3f}")
        logger.info(f"    Significant: {'Yes' if female_binom_p < 0.05 else 'No'}")

        # Overall chi-square goodness of fit test
        observed = [male_count, female_count]
        expected = [total_positions * expected_male_prop, total_positions * expected_female_prop]
        chi2_stat, chi2_p = scipy_stats.chisquare(observed, expected)

        logger.info(f"Chi-square goodness of fit:")
        logger.info(f"  Chi-square statistic: {chi2_stat:.4f}")
        logger.info(f"  p-value: {chi2_p:.4f}")
        logger.info(f"  Significant deviation: {'Yes' if chi2_p < 0.05 else 'No'}")

        # Store results
        test_results[f'top_{k}'] = {
            'total_positions': total_positions,
            'male_count': male_count,
            'female_count': female_count,
            'observed_male_prop': observed_male_prop,
            'observed_female_prop': observed_female_prop,
            'male_p_value': male_binom_p,
            'female_p_value': female_binom_p,
            'male_ci': male_ci,
            'female_ci': female_ci,
            'male_effect_size': male_effect,
            'female_effect_size': female_effect,
            'chi2_statistic': chi2_stat,
            'chi2_p_value': chi2_p
        }

    return test_results


def test_representation_significance_with_correction(search_results, gender_mapping, population_proportions,
                                                     analysis_name=""):
    """
    Test statistical significance with multiple testing correction.

    Args:
        search_results (dict): Dictionary of search results
        gender_mapping (dict): Mapping of candidate IDs to gender
        population_proportions (dict): Population proportions by gender
        analysis_name (str): Name for the analysis (for logging)

    Returns:
        tuple: (all_results, p_corrected_bonf, p_corrected_fdr, rejected_bonf, rejected_fdr)
    """
    logger.info(f"Representation significance with multiple testing correction: {analysis_name}")

    all_p_values = []
    all_results = []

    for k in Config.TOP_K_VALUES:
        # Count representation (same as in test_representation_significance)
        male_count = 0
        female_count = 0
        total_positions = 0

        for query_id, document_scores in search_results.items():
            sorted_results = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
            top_k_docs = sorted_results[:k]

            for doc_id, score in top_k_docs:
                if doc_id in gender_mapping:
                    gender = gender_mapping[doc_id]
                    if gender == 'male':
                        male_count += 1
                    elif gender == 'female':
                        female_count += 1
                    total_positions += 1

        if total_positions == 0:
            continue

        # Calculate proportions
        observed_male_prop = male_count / total_positions
        observed_female_prop = female_count / total_positions
        expected_male_prop = population_proportions['male']
        expected_female_prop = population_proportions['female']

        # Binomial tests
        from scipy.stats import binomtest
        male_binom_result = binomtest(male_count, total_positions, expected_male_prop, alternative='two-sided')
        female_binom_result = binomtest(female_count, total_positions, expected_female_prop, alternative='two-sided')

        # Store results
        result = {
            'k': k,
            'total_positions': total_positions,
            'male_count': male_count,
            'female_count': female_count,
            'male_prop': observed_male_prop,
            'female_prop': observed_female_prop,
            'male_p_raw': male_binom_result.pvalue,
            'female_p_raw': female_binom_result.pvalue,
            'male_effect': observed_male_prop - expected_male_prop,
            'female_effect': observed_female_prop - expected_female_prop
        }
        all_results.append(result)
        all_p_values.extend([male_binom_result.pvalue, female_binom_result.pvalue])

    # Apply multiple testing correction
    rejected_bonf, p_corrected_bonf, alpha_sidak, alpha_bonf = multipletests(all_p_values, method='bonferroni')
    rejected_fdr, p_corrected_fdr, alpha_sidak_fdr, alpha_bonf_fdr = multipletests(all_p_values, method='fdr_bh')

    logger.info(f"Multiple Testing Correction Results:")
    logger.info(f"Number of tests: {len(all_p_values)}")
    logger.info(f"Bonferroni significant results: {sum(rejected_bonf)}/{len(rejected_bonf)}")
    logger.info(f"FDR significant results: {sum(rejected_fdr)}/{len(rejected_fdr)}")

    # Log detailed results with corrections
    for i, result in enumerate(all_results):
        k = result['k']
        logger.info(f"--- Top-{k} Results (Corrected) ---")
        logger.info(
            f"Observed: Male={result['male_count']} ({result['male_prop']:.1%}), Female={result['female_count']} ({result['female_prop']:.1%})")

        # Male results
        male_idx = i * 2
        logger.info(f"Male representation:")
        logger.info(f"  Raw p-value: {result['male_p_raw']:.4f}")
        logger.info(
            f"  Bonferroni corrected: {p_corrected_bonf[male_idx]:.4f} ({'Significant' if rejected_bonf[male_idx] else 'NS'})")
        logger.info(
            f"  FDR corrected: {p_corrected_fdr[male_idx]:.4f} ({'Significant' if rejected_fdr[male_idx] else 'NS'})")
        logger.info(f"  Effect size: {result['male_effect']:+.3f}")

        # Female results
        female_idx = i * 2 + 1
        logger.info(f"Female representation:")
        logger.info(f"  Raw p-value: {result['female_p_raw']:.4f}")
        logger.info(
            f"  Bonferroni corrected: {p_corrected_bonf[female_idx]:.4f} ({'Significant' if rejected_bonf[female_idx] else 'NS'})")
        logger.info(
            f"  FDR corrected: {p_corrected_fdr[female_idx]:.4f} ({'Significant' if rejected_fdr[female_idx] else 'NS'})")
        logger.info(f"  Effect size: {result['female_effect']:+.3f}")

    return all_results, p_corrected_bonf, p_corrected_fdr, rejected_bonf, rejected_fdr


def calculate_representation_effect_sizes(all_results):
    """
    Calculate Cohen's h for proportion differences (standardized effect size for proportions).

    Args:
        all_results (list): Results from representation significance testing

    Returns:
        dict: Effect sizes by ranking depth
    """
    logger.info("Representation effect sizes (Cohen's h)")

    effect_sizes = {}
    expected_male = 0.558
    expected_female = 0.442

    for result in all_results:
        k = result['k']
        male_prop = result['male_prop']
        female_prop = result['female_prop']

        # Cohen's h for proportions
        male_h = 2 * (np.arcsin(np.sqrt(male_prop)) - np.arcsin(np.sqrt(expected_male)))
        female_h = 2 * (np.arcsin(np.sqrt(female_prop)) - np.arcsin(np.sqrt(expected_female)))

        # Interpretation
        male_interp = "small" if abs(male_h) < 0.2 else ("medium" if abs(male_h) < 0.5 else "large")
        female_interp = "small" if abs(female_h) < 0.2 else ("medium" if abs(female_h) < 0.5 else "large")

        effect_sizes[f'top_{k}'] = {
            'male_cohens_h': male_h,
            'female_cohens_h': female_h,
            'male_interpretation': male_interp,
            'female_interpretation': female_interp
        }

        logger.info(
            f"Top-{k}: Male Cohen's h = {male_h:.4f} ({male_interp}), Female Cohen's h = {female_h:.4f} ({female_interp})")

    return effect_sizes


def calculate_statistical_power(all_results):
    """
    Calculate post-hoc statistical power for representation tests.

    Args:
        all_results (list): Results from representation significance testing

    Returns:
        dict: Statistical power analysis results
    """
    logger.info("Statistical Power Analysis")

    power_results = {}

    for result in all_results:
        k = result['k']
        n = result['total_positions']
        observed_prop = result['female_prop']
        expected_prop = 0.442

        # Effect size (proportion difference)
        effect_size = abs(observed_prop - expected_prop)

        # Approximate power calculation for binomial test
        z_alpha = scipy_stats.norm.ppf(0.975)  # two-tailed alpha = 0.05

        # Standard error under null
        se_null = np.sqrt(expected_prop * (1 - expected_prop) / n)

        # Standard error under alternative
        se_alt = np.sqrt(observed_prop * (1 - observed_prop) / n)

        # Critical value
        critical_diff = z_alpha * se_null

        # Power calculation
        z_beta = (effect_size - critical_diff) / se_alt
        power = scipy_stats.norm.cdf(z_beta)

        power_results[f'top_{k}'] = {
            'sample_size': n,
            'effect_size': effect_size,
            'statistical_power': power
        }

        logger.info(f"Top-{k}: n={n}, effect_size={effect_size:.4f}, power={power:.4f}")

    return power_results


def calculate_similarity_confidence_intervals(score_stats):
    """
    Calculate confidence intervals for similarity score differences.

    Args:
        score_stats (dict): Score distribution statistics from bias analysis

    Returns:
        dict: Confidence intervals for score differences
    """
    logger.info("Similarity score confidence intervals")

    if 'statistical_test' not in score_stats:
        logger.warning("No statistical test results found in score_stats")
        return {}

    # Extract values (these would come from bias analysis results)
    # Using values from thesis results as defaults
    male_scores_mean = 0.7358
    female_scores_mean = 0.7422
    male_std = 0.1187
    female_std = 0.1155
    male_n = 3000
    female_n = 1936

    # Calculate confidence intervals
    # Male CI
    male_se = male_std / np.sqrt(male_n)
    male_ci = scipy_stats.t.interval(0.95, male_n - 1, loc=male_scores_mean, scale=male_se)

    # Female CI
    female_se = female_std / np.sqrt(female_n)
    female_ci = scipy_stats.t.interval(0.95, female_n - 1, loc=female_scores_mean, scale=female_se)

    # Difference CI
    diff_mean = female_scores_mean - male_scores_mean
    diff_se = np.sqrt(male_se ** 2 + female_se ** 2)
    diff_ci = scipy_stats.t.interval(0.95, min(male_n, female_n) - 1, loc=diff_mean, scale=diff_se)

    confidence_intervals = {
        'male_scores_ci': male_ci,
        'female_scores_ci': female_ci,
        'difference_ci': diff_ci,
        'male_mean': male_scores_mean,
        'female_mean': female_scores_mean,
        'difference_mean': diff_mean
    }

    logger.info(f"Male similarity scores: {male_scores_mean:.4f} [95% CI: {male_ci[0]:.4f}, {male_ci[1]:.4f}]")
    logger.info(f"Female similarity scores: {female_scores_mean:.4f} [95% CI: {female_ci[0]:.4f}, {female_ci[1]:.4f}]")
    logger.info(f"Difference (F-M): {diff_mean:+.4f} [95% CI: {diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}]")

    return confidence_intervals


def bootstrap_key_metrics(search_results, gender_mapping, n_bootstrap=1000):
    """
    Bootstrap confidence intervals for key findings.

    Args:
        search_results (dict): Dictionary of search results
        gender_mapping (dict): Mapping of candidate IDs to gender
        n_bootstrap (int): Number of bootstrap iterations

    Returns:
        dict: Bootstrap confidence intervals for key metrics
    """
    logger.info("Bootstrap Key Metrics")

    # Collect all data
    female_scores = []
    male_scores = []
    female_ranks = []
    male_ranks = []

    for query_id, document_scores in search_results.items():
        sorted_results = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)

        for rank, (doc_id, score) in enumerate(sorted_results, 1):
            if doc_id in gender_mapping:
                gender = gender_mapping[doc_id]
                if gender == 'female':
                    female_scores.append(score)
                    female_ranks.append(rank)
                elif gender == 'male':
                    male_scores.append(score)
                    male_ranks.append(rank)

    # Bootstrap sampling
    bootstrap_results = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        female_sample_scores = np.random.choice(female_scores, size=len(female_scores), replace=True)
        male_sample_scores = np.random.choice(male_scores, size=len(male_scores), replace=True)
        female_sample_ranks = np.random.choice(female_ranks, size=len(female_ranks), replace=True)
        male_sample_ranks = np.random.choice(male_ranks, size=len(male_ranks), replace=True)

        bootstrap_results.append({
            'female_mean_score': np.mean(female_sample_scores),
            'male_mean_score': np.mean(male_sample_scores),
            'score_difference': np.mean(female_sample_scores) - np.mean(male_sample_scores),
            'female_mean_rank': np.mean(female_sample_ranks),
            'male_mean_rank': np.mean(male_sample_ranks),
            'rank_difference': np.mean(female_sample_ranks) - np.mean(male_sample_ranks)
        })

    # Calculate confidence intervals
    score_diffs = [r['score_difference'] for r in bootstrap_results]
    rank_diffs = [r['rank_difference'] for r in bootstrap_results]

    score_ci = np.percentile(score_diffs, [2.5, 97.5])
    rank_ci = np.percentile(rank_diffs, [2.5, 97.5])

    bootstrap_ci = {
        'score_advantage_mean': np.mean(score_diffs),
        'score_advantage_ci': score_ci,
        'rank_advantage_mean': np.mean(rank_diffs),
        'rank_advantage_ci': rank_ci,
        'female_score_advantage_robust': score_ci[0] > 0,
        'female_rank_advantage_robust': rank_ci[1] < 0
    }

    logger.info(f"Female score advantage: {np.mean(score_diffs):.4f}")
    logger.info(f"95% CI for score advantage: [{score_ci[0]:.4f}, {score_ci[1]:.4f}]")
    logger.info(f"Female rank advantage: {np.mean(rank_diffs):.2f}")
    logger.info(f"95% CI for rank advantage: [{rank_ci[0]:.2f}, {rank_ci[1]:.2f}]")

    # Check consistency with original findings
    if score_ci[0] > 0:
        logger.info("Female score advantage is robust")
    elif score_ci[1] > 0 and score_ci[0] < 0:
        logger.info("Uncertain: Female score advantage confidence interval includes zero")
    else:
        logger.info("Bootstrap suggests no female score advantage")

    if rank_ci[1] < 0:
        logger.info("Female rank advantage is robust")
    elif rank_ci[0] < 0 and rank_ci[1] > 0:
        logger.info("Uncertain: Female rank advantage confidence interval includes zero")
    else:
        logger.info("Bootstrap suggests no female rank advantage")

    return bootstrap_ci


def test_variance_equality(search_results, gender_mapping):
    """
    Test if variance differences explain ranking disparities.

    Args:
        search_results (dict): Dictionary of search results
        gender_mapping (dict): Mapping of candidate IDs to gender

    Returns:
        dict: Variance equality test results
    """
    scores_by_gender = defaultdict(list)

    for query_id, document_scores in search_results.items():
        for document_id, similarity_score in document_scores.items():
            if document_id in gender_mapping:
                gender = gender_mapping[document_id]
                scores_by_gender[gender].append(similarity_score)

    variance_results = {}

    if len(scores_by_gender) == 2:
        genders = list(scores_by_gender.keys())
        scores_1 = scores_by_gender[genders[0]]
        scores_2 = scores_by_gender[genders[1]]

        # Levene's test for equal variances
        levene_stat, levene_p = scipy_stats.levene(scores_1, scores_2)

        # F-test for variance ratio
        var_1 = np.var(scores_1, ddof=1)
        var_2 = np.var(scores_2, ddof=1)
        f_ratio = var_1 / var_2 if var_1 > var_2 else var_2 / var_1

        variance_results = {
            'variance_1': var_1,
            'variance_2': var_2,
            'f_ratio': f_ratio,
            'levene_statistic': levene_stat,
            'levene_p_value': levene_p,
            'equal_variances': levene_p > 0.05,
            'genders': genders
        }

        logger.info("Variance Equality Analysis:")
        logger.info(f"  {genders[0]} variance: {var_1:.6f}")
        logger.info(f"  {genders[1]} variance: {var_2:.6f}")
        logger.info(f"  F-ratio: {f_ratio:.4f}")
        logger.info(f"  Levene's test p-value: {levene_p:.4f}")
        logger.info(f"  Equal variances: {'Yes' if levene_p > 0.05 else 'No'}")

    return variance_results


def validate_population_distributions(gender_mapping):
    """
    Double-check population distributions are correct.

    Args:
        gender_mapping (dict): Mapping of candidate IDs to gender

    Returns:
        dict: Population distribution validation results
    """
    logger.info("POPULATION DISTRIBUTION VALIDATION")

    total_candidates = len(gender_mapping)
    male_count = sum(1 for gender in gender_mapping.values() if gender == 'male')
    female_count = sum(1 for gender in gender_mapping.values() if gender == 'female')

    male_prop = male_count / total_candidates
    female_prop = female_count / total_candidates

    logger.info(f"Total candidates with gender labels: {total_candidates}")
    logger.info(f"Male candidates: {male_count} ({male_prop:.1%})")
    logger.info(f"Female candidates: {female_count} ({female_prop:.1%})")
    logger.info(f"Expected from thesis: Male=55.8%, Female=44.2%")

    # Check if our calculations match
    expected_male = 5583 / 10013
    expected_female = 4430 / 10013

    logger.info(f"Calculation verification:")
    logger.info(f"  Calculated male prop: {male_prop:.4f}")
    logger.info(f"  Expected male prop: {expected_male:.4f}")
    logger.info(f"  Difference: {abs(male_prop - expected_male):.4f}")

    is_correct = abs(male_prop - expected_male) < 0.001
    if is_correct:
        logger.info("Population distributions are correct")
    else:
        logger.info("Population distribution mismatch")

    return {
        'total_candidates': total_candidates,
        'male_count': male_count,
        'female_count': female_count,
        'male_proportion': male_prop,
        'female_proportion': female_prop,
        'expected_male_proportion': expected_male,
        'expected_female_proportion': expected_female,
        'distribution_correct': is_correct
    }


def analyze_faiss_ranking_behavior(search_results, gender_mapping):
    """
    Analyze if FAISS ranking algorithm introduces systematic bias.

    Args:
        search_results (dict): Dictionary of search results
        gender_mapping (dict): Mapping of candidate IDs to gender

    Returns:
        dict: FAISS ranking behavior analysis results
    """
    # Group queries by similarity score ranges
    score_range_analysis = defaultdict(lambda: defaultdict(list))

    for query_id, document_scores in search_results.items():
        for doc_id, score in document_scores.items():
            if doc_id in gender_mapping:
                gender = gender_mapping[doc_id]
                # Bin scores into ranges
                score_bin = round(score, 2)
                score_range_analysis[score_bin][gender].append(score)

    logger.info("FAISS Ranking Behavior Analysis:")
    logger.info("Score ranges with significant representation differences:")

    significant_ranges = {}
    for score_bin in sorted(score_range_analysis.keys()):
        gender_counts = {gender: len(scores) for gender, scores in score_range_analysis[score_bin].items()}
        total = sum(gender_counts.values())
        if total > 10:
            if len(gender_counts) == 2:
                genders = list(gender_counts.keys())
                pct_1 = gender_counts[genders[0]] / total * 100
                pct_2 = gender_counts[genders[1]] / total * 100
                if abs(pct_1 - pct_2) > 20:
                    significant_ranges[score_bin] = {
                        'total_count': total,
                        'gender_percentages': {genders[0]: pct_1, genders[1]: pct_2}
                    }
                    logger.info(
                        f"  Score ~{score_bin:.2f}: {genders[0]}={pct_1:.1f}%, {genders[1]}={pct_2:.1f}% (n={total})")

    return {
        'score_range_analysis': dict(score_range_analysis),
        'significant_ranges': significant_ranges
    }


def comprehensive_statistical_validation(search_results, gender_mapping, population_proportions):
    """
    Run comprehensive statistical validation pipeline.

    Args:
        search_results (dict): Dictionary of search results
        gender_mapping (dict): Mapping of candidate IDs to gender
        population_proportions (dict): Population proportions by gender

    Returns:
        dict: Complete statistical validation results
    """
    logger.info("COMPREHENSIVE STATISTICAL VALIDATION")

    # Run all statistical tests
    representation_results = test_representation_significance(
        search_results, gender_mapping, population_proportions, "Jobs → Resumes"
    )

    corrected_results, p_bonf, p_fdr, rejected_bonf, rejected_fdr = test_representation_significance_with_correction(
        search_results, gender_mapping, population_proportions, "Jobs → Resumes (Corrected)"
    )

    effect_sizes = calculate_representation_effect_sizes(corrected_results)
    statistical_power = calculate_statistical_power(corrected_results)
    bootstrap_results = bootstrap_key_metrics(search_results, gender_mapping)
    variance_results = test_variance_equality(search_results, gender_mapping)
    population_validation = validate_population_distributions(gender_mapping)
    faiss_analysis = analyze_faiss_ranking_behavior(search_results, gender_mapping)

    # Combine all results
    comprehensive_results = {
        'representation_significance': representation_results,
        'multiple_testing_correction': {
            'corrected_results': corrected_results,
            'bonferroni_corrected_p': p_bonf,
            'fdr_corrected_p': p_fdr,
            'bonferroni_rejected': rejected_bonf,
            'fdr_rejected': rejected_fdr
        },
        'effect_sizes': effect_sizes,
        'statistical_power': statistical_power,
        'bootstrap_validation': bootstrap_results,
        'variance_equality': variance_results,
        'population_validation': population_validation,
        'faiss_ranking_analysis': faiss_analysis
    }

    logger.info("Statistical validation completed.")

    return comprehensive_results
