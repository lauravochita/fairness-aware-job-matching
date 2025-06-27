"""
Post-intervention bias analysis module for fairness-aware job-matching framework.
Provides counterfactual analysis and intervention validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from fairness_framework.counterfactual.gender_classifier import (
    train_gender_classifier_with_teacher_params,
    analyze_gender_directions
)
from .manipulation import flip_embeddings_along_gender_direction
from fairness_framework.utils.config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_counterfactual_bias_analysis(
    gender_mapping,
    run_bias_analysis_jobs_to_resumes_func,
    run_bias_analysis_resumes_to_jobs_func=None
) -> Dict:
    """
    Executes the full counterfactual gender bias analysis pipeline:
    1. Trains a gender classifier on resume embeddings
    2. Computes gender-informative directions
    3. Flips embeddings along the gender direction
    4. Reruns bias analysis on flipped embeddings

    Args:
        gender_mapping (dict): Mapping of resume IDs to gender labels
        run_bias_analysis_jobs_to_resumes_func (callable): Function to run jobs->resumes bias analysis
        run_bias_analysis_resumes_to_jobs_func (callable, optional): Function to run resumes->jobs bias analysis

    Returns:
        dict: Complete counterfactual analysis results
    """
    print("Counterfactual gender analysis results")
    from ..data.data_loader import load_bias_analysis_embeddings
    job_embeddings, resume_embeddings = load_bias_analysis_embeddings()
    # Extract labeled data
    resume_ids_with_gender = []
    resume_indices_with_gender = []
    gender_labels = []

    for resume_id, gender in gender_mapping.items():
        index = int(resume_id.split('_')[1])
        if index < resume_embeddings.shape[0]:
            resume_ids_with_gender.append(resume_id)
            resume_indices_with_gender.append(index)
            gender_labels.append(0 if gender == "female" else 1)

    gender_labels = np.array(gender_labels)
    labeled_embeddings = resume_embeddings[resume_indices_with_gender]

    print(f"  Labeled resumes found: {len(gender_labels)}")
    print(f"  Distribution: Female={np.sum(gender_labels == 0)}, Male={np.sum(gender_labels == 1)}")

    # Train gender classifier
    classifier, cv_acc, train_acc = train_gender_classifier_with_teacher_params(
        labeled_embeddings, gender_labels
    )

    # Analyze learned gender direction
    gender_weights, direction_norm = analyze_gender_directions(classifier, labeled_embeddings)

    # Evaluate classifier
    full_predictions = classifier.predict(resume_embeddings)
    pred_counts = np.bincount(full_predictions)

    print("  Prediction distribution:")
    print(f"    Female (0): {pred_counts[0]} ({pred_counts[0] / len(full_predictions) * 100:.1f}%)")
    print(f"    Male   (1): {pred_counts[1]} ({pred_counts[1] / len(full_predictions) * 100:.1f}%)")

    # Prepare result storage
    results = {
        "original_classifier": classifier,
        "gender_weights": gender_weights,
        "original_cv_acc": cv_acc,
        "original_train_acc": train_acc,
        "flip_results": {}
    }

    flip_factors = Config.FLIP_FACTORS

    for factor in flip_factors:
        print(f"Testing counterfactual embeddings (flip_factor = {factor})")

        modified_embeddings = flip_embeddings_along_gender_direction(resume_embeddings, gender_weights, factor)
        modified_labeled = modified_embeddings[resume_indices_with_gender]
        modified_preds = classifier.predict(modified_labeled)
        modified_acc = np.mean(modified_preds == gender_labels)

        print(f"  Classifier accuracy on flipped embeddings: {modified_acc:.4f}")
        print(f"  Accuracy drop from original LogReg: {train_acc - modified_acc:.4f}")

        # Run bias analysis using modified embeddings
        original_embeddings_copy = resume_embeddings.copy()
        try:
            # Modify embeddings in place for analysis
            resume_embeddings[:] = modified_embeddings

            print("  Re-run jobs to resume analysis")
            jobs_to_resumes_result = run_bias_analysis_jobs_to_resumes_func(modified_embeddings)

            if run_bias_analysis_resumes_to_jobs_func is not None:
                print("  Re-run resume to jobs analysis")
                resumes_to_jobs_result = run_bias_analysis_resumes_to_jobs_func()
            else:
                resumes_to_jobs_result = {}

            results["flip_results"][factor] = {
                "classifier_accuracy_drop": train_acc - modified_acc,
                "jobs_to_resumes": jobs_to_resumes_result,
                "resumes_to_jobs": resumes_to_jobs_result
            }

        finally:
            # Restore original embeddings
            resume_embeddings[:] = original_embeddings_copy

    return results


def compare_bias_results(original_results: Dict, counterfactual_results: Dict):
    """
    Compares the original and counterfactually modified bias results to assess reduction in gender bias.

    Args:
        original_results (dict): Original bias analysis results
        counterfactual_results (dict): Counterfactual analysis results
    """
    print("Comparative bias analysis: pre vs. post intervention")
    for factor, cf in counterfactual_results["flip_results"].items():
        print(f"Flip Factor = {factor}")
        print(f"  Classifier Accuracy Drop: {cf['classifier_accuracy_drop']:.4f}")

        orig_dist = original_results.get("score_distribution", {})
        cf_dist = cf["jobs_to_resumes"].get("score_distribution", {})

        if "statistical_test" in orig_dist and "statistical_test" in cf_dist:
            o_test = orig_dist["statistical_test"]
            c_test = cf_dist["statistical_test"]

            print("\n  Jobs → Resumes Score Distribution:")
            print(f"    Original p-value: {o_test['p_value']:.4f}")
            print(f"    Counterfactual p-value: {c_test['p_value']:.4f}")
            print(f"    Original effect size: {o_test['effect_size']:.4f}")
            print(f"    Counterfactual effect size: {c_test['effect_size']:.4f}")
            print(f"    Δ Effect size (bias reduction): {abs(o_test['effect_size']) - abs(c_test['effect_size']):.4f}")

        if ("top_k_representation" in original_results and
                "top_k_representation" in cf["jobs_to_resumes"]):
            print("\n  Top-K Representation:")
            for k in original_results["top_k_representation"]:
                if k in cf["jobs_to_resumes"]["top_k_representation"]:
                    print(f"    {k.upper()}:")
                    for g in ['female', 'male']:
                        o_pct = original_results["top_k_representation"][k][g]["percentage"]
                        c_pct = cf["jobs_to_resumes"]["top_k_representation"][k][g]["percentage"]
                        delta = c_pct - o_pct
                        print(f"      {g.capitalize():<6}: {o_pct:.1f}% → {c_pct:.1f}% (Δ {delta:+.1f}%)")


def validate_bias_reduction_stability(original_results, counterfactual_results):
    """
    Assesses whether bias reduction generalizes across flip factors.
    Tracks statistical tests and female representation shifts.

    Args:
        original_results (dict): Original bias analysis results
        counterfactual_results (dict): Counterfactual analysis results

    Returns:
        dict: Stability analysis results
    """
    print("\nBias Reduction Stability")

    if 'flip_results' not in counterfactual_results:
        print("  No flip results found")
        return

    flip_factors = list(counterfactual_results['flip_results'].keys())
    p_vals, effect_sizes, repr_deltas = [], [], []

    for ff in flip_factors:
        result = counterfactual_results['flip_results'][ff]
        stats_data = result.get('jobs_to_resumes', {}).get('score_distribution', {}).get('statistical_test', {})

        p_vals.append(stats_data.get('p_value', np.nan))
        effect_sizes.append(abs(stats_data.get('effect_size', np.nan)))

        # Female top-k representation shift
        cf_female = result.get('jobs_to_resumes', {}).get('top_k_representation', {}).get('top_5', {}).get('female',
                                                                                                           {}).get(
            'percentage', np.nan)
        orig_female = original_results.get('top_k_representation', {}).get('top_5', {}).get('female', {}).get(
            'percentage', np.nan)

        delta = cf_female - orig_female if not np.isnan(cf_female) and not np.isnan(orig_female) else np.nan
        repr_deltas.append(delta)

    # Summary Table
    print(f"{'Flip Factor':<12} {'p-value':<10} {'Effect Size':<14} {'Δ Female %':<12}")
    for i, ff in enumerate(flip_factors):
        p_str = f"{p_vals[i]:.4f}" if not np.isnan(p_vals[i]) else "N/A"
        eff_str = f"{effect_sizes[i]:.4f}" if not np.isnan(effect_sizes[i]) else "N/A"
        delta_str = f"{repr_deltas[i]:+.1f}%" if not np.isnan(repr_deltas[i]) else "N/A"
        print(f"{ff:<12} {p_str:<10} {eff_str:<14} {delta_str:<12}")

    # Plots
    try:
        plt.figure(figsize=(12, 4))

        # P-value trend
        plt.subplot(1, 2, 1)
        valid_idx = [i for i, p in enumerate(p_vals) if not np.isnan(p)]
        if valid_idx:
            plt.plot(np.array(flip_factors)[valid_idx], np.array(p_vals)[valid_idx], 'o-', label='p-value')
            plt.axhline(0.05, color='red', linestyle='--', label='Significance Threshold')
            plt.xlabel("Flip Factor")
            plt.ylabel("p-value")
            plt.title("Statistical Significance vs Flip Factor")
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Representation trend
        if not all(np.isnan(repr_deltas)):
            plt.subplot(1, 2, 2)
            valid_idx = [i for i, d in enumerate(repr_deltas) if not np.isnan(d)]
            if valid_idx:
                plt.plot(np.array(flip_factors)[valid_idx], np.array(repr_deltas)[valid_idx], 'o-', color='green')
                plt.axhline(0, color='black', linestyle='-')
                plt.xlabel("Flip Factor")
                plt.ylabel("Δ Female % in Top-5")
                plt.title("Representation Shift vs Flip Factor")
                plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.warning(f"Could not create plots: {e}")

    return {
        'flip_factors': flip_factors,
        'p_values': p_vals,
        'effect_sizes': effect_sizes,
        'representation_changes': repr_deltas
    }


def cross_validate_counterfactual_method(embeddings, labels, gender_weights, n_folds=3):
    """
    Cross-validates performance of counterfactual flip intervention.
    Measures accuracy degradation across folds.

    Args:
        embeddings (np.ndarray): Input embeddings
        labels (np.ndarray): Gender labels
        gender_weights (np.ndarray): Gender direction weights
        n_folds (int): Number of cross-validation folds

    Returns:
        list: Cross-validation results
    """
    print("\nCross-Validation of Counterfactual Method")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(embeddings)):
        print(f"\n  Fold {fold_idx + 1}/{n_folds}")

        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_train, y_train)

        orig_acc = clf.score(X_test, y_test)

        flipped_X = flip_embeddings_along_gender_direction(X_test, gender_weights, flip_factor=0.5)
        flipped_acc = clf.score(flipped_X, y_test)
        drop = orig_acc - flipped_acc

        print(f"    Original Accuracy : {orig_acc:.4f}")
        print(f"    Flipped Accuracy  : {flipped_acc:.4f}")
        print(f"    Accuracy Drop     : {drop:.4f}")

        results.append({'fold': fold_idx + 1, 'orig_acc': orig_acc, 'flipped_acc': flipped_acc, 'drop': drop})

    # Summary
    orig = [r['orig_acc'] for r in results]
    flipped = [r['flipped_acc'] for r in results]
    drops = [r['drop'] for r in results]

    print("\n  Cross-Validation Summary:")
    print(f"    Avg Original Accuracy : {np.mean(orig):.4f} ± {np.std(orig):.4f}")
    print(f"    Avg Flipped Accuracy  : {np.mean(flipped):.4f} ± {np.std(flipped):.4f}")
    print(f"    Avg Accuracy Drop     : {np.mean(drops):.4f} ± {np.std(drops):.4f}")

    return results


def test_post_intervention_representation(original_results, post_intervention_results, gender_mapping,
                                          population_proportions, flip_factor):
    """
    Test statistical significance of gender representation after counterfactual intervention.

    Args:
        original_results: Pre-intervention search results
        post_intervention_results: Post-intervention search results
        gender_mapping: Mapping of candidate IDs to gender
        population_proportions: Expected population proportions
        flip_factor: Flip factor used for intervention

    Returns:
        list: Comparison results for each top-k value
    """
    print(f"\nPost-intervention representation (α = {flip_factor})")

    from scipy.stats import binomtest

    top_k_values = [5, 10, 20, 50]
    results_comparison = []

    for k in top_k_values:
        print(f"\nTop-{k} Post-Intervention Analysis")

        # Count representation in post-intervention results
        male_count = 0
        female_count = 0
        total_positions = 0

        for query_id, document_scores in post_intervention_results.items():
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
            print(f"No valid results for top-{k}")
            continue

        # Calculate post-intervention proportions
        post_male_prop = male_count / total_positions
        post_female_prop = female_count / total_positions

        # Expected proportions
        expected_male_prop = population_proportions['male']
        expected_female_prop = population_proportions['female']

        # Get original proportions for comparison
        orig_male_count = 0
        orig_female_count = 0
        orig_total_positions = 0

        for query_id, document_scores in original_results.items():
            sorted_results = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
            top_k_docs = sorted_results[:k]

            for doc_id, score in top_k_docs:
                if doc_id in gender_mapping:
                    gender = gender_mapping[doc_id]
                    if gender == 'male':
                        orig_male_count += 1
                    elif gender == 'female':
                        orig_female_count += 1
                    orig_total_positions += 1

        orig_male_prop = orig_male_count / orig_total_positions if orig_total_positions > 0 else 0
        orig_female_prop = orig_female_count / orig_total_positions if orig_total_positions > 0 else 0

        print(f"Expected: Male={expected_male_prop:.1%}, Female={expected_female_prop:.1%}")
        print(f"Original: Male={orig_male_prop:.1%}, Female={orig_female_prop:.1%}")
        print(f"Post-Int: Male={post_male_prop:.1%}, Female={post_female_prop:.1%}")

        # Binomial tests for post-intervention representation
        post_male_binom = binomtest(male_count, total_positions, expected_male_prop, alternative='two-sided')
        post_female_binom = binomtest(female_count, total_positions, expected_female_prop, alternative='two-sided')

        # Effect sizes
        post_male_effect = post_male_prop - expected_male_prop
        post_female_effect = post_female_prop - expected_female_prop

        # Changes from original
        male_change = post_male_prop - orig_male_prop
        female_change = post_female_prop - orig_female_prop

        print(f"\nPost-Intervention Binomial Tests:")
        print(f"  Male: p-value = {post_male_binom.pvalue:.4f}, Effect size = {post_male_effect:+.3f}")
        print(f"  Female: p-value = {post_female_binom.pvalue:.4f}, Effect size = {post_female_effect:+.3f}")
        print(f"\nChanges from Original:")
        print(f"  Male: {male_change:+.1%}")
        print(f"  Female: {female_change:+.1%}")

        # Store results for summary table
        results_comparison.append({
            'k': k,
            'orig_male_p': 'See previous test',
            'post_male_p': post_male_binom.pvalue,
            'orig_female_p': 'See previous test',
            'post_female_p': post_female_binom.pvalue,
            'male_change': male_change,
            'female_change': female_change,
            'post_male_prop': post_male_prop,
            'post_female_prop': post_female_prop
        })

    return results_comparison


def analyze_resumes_to_jobs_representation(gender_mapping, population_proportions):
    """
    Analyze representation in resumes-to-jobs direction.

    Args:
        gender_mapping: Mapping of candidate IDs to gender
        population_proportions: Expected population proportions

    Returns:
        dict: Query representation analysis results
    """
    print(f"\nRESUMES-TO-JOBS REPRESENTATION ANALYSIS")

    # Count queries by gender
    male_queries = sum(1 for gender in gender_mapping.values() if gender == 'male')
    female_queries = sum(1 for gender in gender_mapping.values() if gender == 'female')
    total_queries = male_queries + female_queries

    if total_queries == 0:
        print("No gender mapping found in resumes_to_jobs_results")
        return None

    # Expected proportions
    expected_male_prop = population_proportions['male']
    expected_female_prop = population_proportions['female']
    observed_male_prop = male_queries / total_queries
    observed_female_prop = female_queries / total_queries

    print(f"Expected: Male={expected_male_prop:.1%}, Female={expected_female_prop:.1%}")
    print(f"Observed: Male={observed_male_prop:.1%}, Female={observed_female_prop:.1%}")

    # Binomial test for query representation
    from scipy.stats import binomtest
    male_query_test = binomtest(male_queries, total_queries, expected_male_prop, alternative='two-sided')
    female_query_test = binomtest(female_queries, total_queries, expected_female_prop, alternative='two-sided')

    print(f"\nQuery Representation Tests:")
    print(f"Male queries: p-value = {male_query_test.pvalue:.4f}")
    print(f"Female queries: p-value = {female_query_test.pvalue:.4f}")

    return {
        'male_queries': male_queries,
        'female_queries': female_queries,
        'male_query_p': male_query_test.pvalue,
        'female_query_p': female_query_test.pvalue,
    }


def comprehensive_dosage_response_analysis(original_results, counterfactual_results, gender_mapping,
                                           population_proportions):
    """
    Test intervention effectiveness across all tested flip factors.

    Args:
        original_results: Original bias analysis results
        counterfactual_results: Counterfactual analysis results
        gender_mapping: Mapping of candidate IDs to gender
        population_proportions: Expected population proportions

    Returns:
        list: Dosage-response analysis results
    """
    print(f"\nDosage-response analysis of counterfactual interventions")

    available_factors = list(counterfactual_results['flip_results'].keys())
    print(f"Available flip factors: {available_factors}")

    dosage_results = []

    for flip_factor in sorted(available_factors):
        print(f"\n Testing Flip Factor α = {flip_factor}")

        # Get post-intervention results
        post_results = counterfactual_results['flip_results'][flip_factor]['jobs_to_resumes']['search_results']

        # Test representation significance for this flip factor
        top_k_values = [5, 10, 20, 50]
        factor_results = {'flip_factor': flip_factor}

        for k in top_k_values:
            # Count representation in post-intervention results
            male_count = 0
            female_count = 0
            total_positions = 0

            for query_id, document_scores in post_results.items():
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

            if total_positions > 0:
                post_female_prop = female_count / total_positions
                expected_female_prop = population_proportions['female']

                # Binomial test
                from scipy.stats import binomtest
                female_binom = binomtest(female_count, total_positions, expected_female_prop, alternative='two-sided')

                factor_results[f'top_{k}_female_p'] = female_binom.pvalue
                factor_results[f'top_{k}_female_prop'] = post_female_prop
                factor_results[f'top_{k}_female_count'] = female_count
                factor_results[f'top_{k}_total'] = total_positions

        dosage_results.append(factor_results)

    # Create summary table
    print(f"\nDosage-Response Summary Table")
    print(f"{'Factor':<8} {'Top-5 p':<10} {'Top-10 p':<10} {'Top-20 p':<10} {'Top-50 p':<10}")
    for result in dosage_results:
        factor = result['flip_factor']
        p5 = result.get('top_5_female_p', np.nan)
        p10 = result.get('top_10_female_p', np.nan)
        p20 = result.get('top_20_female_p', np.nan)
        p50 = result.get('top_50_female_p', np.nan)

        print(f"{factor:<8.2f} {p5:<10.4f} {p10:<10.4f} {p20:<10.4f} {p50:<10.4f}")

    return dosage_results


def analyze_intervention_success_rates(counterfactual_results):
    """
    Analyze success rates of counterfactual interventions across flip factors.

    Args:
        counterfactual_results: Counterfactual analysis results

    Returns:
        list: Success metrics for each flip factor
    """
    print(f"\nAnalyzing Counterfactual Intervention Success Rates")

    if 'flip_results' not in counterfactual_results:
        print("No flip results found")
        return {}

    success_metrics = []

    for flip_factor, results in counterfactual_results['flip_results'].items():
        # Extract accuracy drop
        accuracy_drop = results.get('classifier_accuracy_drop', 0)

        # Extract bias metrics changes
        bias_stats = results.get('jobs_to_resumes', {}).get('score_distribution', {}).get('statistical_test', {})
        effect_size = abs(bias_stats.get('effect_size', 0))
        p_value = bias_stats.get('p_value', 1.0)

        # Success criteria
        significant_bias_reduction = effect_size < 0.05
        maintained_accuracy = accuracy_drop < 0.1
        statistical_significance = p_value > 0.05

        success_score = sum([significant_bias_reduction, maintained_accuracy, statistical_significance]) / 3

        success_metrics.append({
            'flip_factor': flip_factor,
            'accuracy_drop': accuracy_drop,
            'effect_size': effect_size,
            'p_value': p_value,
            'bias_reduction_success': significant_bias_reduction,
            'accuracy_preservation': maintained_accuracy,
            'statistical_success': statistical_significance,
            'overall_success_score': success_score
        })

        print(f"Flip Factor {flip_factor}:")
        print(f"  Accuracy Drop: {accuracy_drop:.4f} ({'✓' if maintained_accuracy else '✗'})")
        print(f"  Effect Size: {effect_size:.4f} ({'✓' if significant_bias_reduction else '✗'})")
        print(f"  p-value: {p_value:.4f} ({'✓' if statistical_significance else '✗'})")
        print(f"  Success Score: {success_score:.2f}/1.0")

    # Find best performing flip factor
    if success_metrics:
        best_factor = max(success_metrics, key=lambda x: x['overall_success_score'])
        print(
            f"\nBest Performing Flip Factor: {best_factor['flip_factor']} (Success: {best_factor['overall_success_score']:.2f})")

    return success_metrics
