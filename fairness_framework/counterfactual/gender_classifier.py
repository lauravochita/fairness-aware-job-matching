"""
Gender classifier module for counterfactual manipulation in fairness-aware job-matching.
Provides gender direction extraction and robust classifier training for embedding manipulation.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import copy
from fairness_framework.utils.config import Config, ModelConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_gender_classifier_with_teacher_params(embeddings, labels):
    """
    Train a logistic regression classifier to detect gender from embeddings,
    using the same hyperparameters as the semi-supervised teacher model.

    Args:
        embeddings (np.ndarray or torch.Tensor): Input embeddings
        labels (np.ndarray): Gender labels (0=female, 1=male)

    Returns:
        tuple: (trained_classifier, cv_accuracy, train_accuracy)
    """

    # Convert to numpy if tensor
    embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings.copy()

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=ModelConfig.GENDER_CLASSIFIER_CV_FOLDS, shuffle=True, random_state=42)
    C_values = [0.1, 1.0, 10.0]
    best_c, best_cv_score = 1.0, 0.0

    logger.info("Cross-validation results:")
    for C in C_values:
        clf = LogisticRegression(max_iter=1000, C=C, random_state=42)
        scores = cross_val_score(clf, embeddings_np, labels, cv=cv, scoring='accuracy')
        mean_score = scores.mean()
        std_score = scores.std()
        logger.info(f"  C={C:<4} CV Accuracy = {mean_score:.4f} ± {std_score * 2:.4f}")

        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_c = C

    logger.info(f"Best C: {best_c} (CV accuracy = {best_cv_score:.4f})")

    # Calculate class weights based on label distribution
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts.astype(np.float32) + 1e-5)
    class_weights /= np.sum(class_weights)
    weight_dict = {i: w for i, w in enumerate(class_weights)}

    # Train final classifier
    final_clf = LogisticRegression(
        max_iter=1000,
        C=best_c,
        class_weight=weight_dict,
        random_state=42
    )
    final_clf.fit(embeddings_np, labels)

    train_accuracy = final_clf.score(embeddings_np, labels)
    logger.info(f"Final classifier accuracy on training data: {train_accuracy:.4f}")
    logger.info(f"CV/Train accuracy difference (potential overfit): {train_accuracy - best_cv_score:.4f}")

    return final_clf, best_cv_score, train_accuracy


def analyze_gender_directions(classifier, embeddings):
    """
    Extract and summarize the direction in embedding space most predictive of gender.

    Args:
        classifier (LogisticRegression): Trained gender classifier
        embeddings (np.ndarray): Input embeddings for analysis

    Returns:
        tuple: (gender_weights, direction_norm)
    """
    logger.info("Analyzing gender-informative directions")

    gender_weights = classifier.coef_[0]
    abs_weights = np.abs(gender_weights)
    top_indices = np.argsort(abs_weights)[-10:][::-1]

    logger.info(f"Weight vector (dim={gender_weights.shape[0]})")
    logger.info(f"  Mean: {np.mean(gender_weights):.6f}")
    logger.info(f"  Std:  {np.std(gender_weights):.6f}")
    logger.info(f"  Min:  {np.min(gender_weights):.6f}")
    logger.info(f"  Max:  {np.max(gender_weights):.6f}")

    logger.info("Top 10 gender-informative dimensions:")
    for idx in top_indices:
        direction = "Male" if gender_weights[idx] > 0 else "Female"
        logger.info(f"  Dimension {idx:3d}: weight = {gender_weights[idx]:.6f} {direction}")

    direction_norm = np.linalg.norm(gender_weights)
    logger.info(f"Gender direction L2 norm: {direction_norm:.6f}")

    return gender_weights, direction_norm


def validate_gender_classifier_robustness(embeddings, labels, n_iter=5):
    """
    Validates the stability of a gender classifier across multiple train/test splits.
    Includes permutation test to detect chance-level performance.

    Args:
        embeddings (np.ndarray): Input embeddings
        labels (np.ndarray): Gender labels
        n_iter (int): Number of validation iterations

    Returns:
        dict: Robustness validation results including permutation test
    """
    logger.info("Classifier Robustness and Chance-Level Check")

    accuracies = []
    effect_sizes = []

    for i in range(n_iter):
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=i, stratify=labels
        )
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=i)
        clf.fit(X_train, y_train)

        acc = clf.score(X_test, y_test)
        accuracies.append(acc)

        # Calculate effect size (Cohen's d for classifier predictions)
        probs = clf.predict_proba(X_test)[:, 1]
        male_probs = probs[y_test == 1]
        female_probs = probs[y_test == 0]

        if len(male_probs) > 1 and len(female_probs) > 1:
            pooled_std = np.sqrt(
                ((len(male_probs) - 1) * np.var(male_probs) + (len(female_probs) - 1) * np.var(female_probs)) /
                (len(male_probs) + len(female_probs) - 2)
            )
            d = (np.mean(male_probs) - np.mean(female_probs)) / pooled_std
            effect_sizes.append(d)

    # Permutation test for chance-level validation
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    true_acc = clf.fit(embeddings, labels).score(embeddings, labels)

    permuted_scores = []
    for _ in range(ModelConfig.PERMUTATION_TEST_ITERATIONS):
        perm_labels = np.random.permutation(labels)
        perm_acc = clf.fit(embeddings, perm_labels).score(embeddings, perm_labels)
        permuted_scores.append(perm_acc)

    p_val = np.mean(np.array(permuted_scores) >= true_acc)

    # Compile results
    robustness_results = {
        'accuracy': {'mean': np.mean(accuracies), 'std': np.std(accuracies)},
        'effect_size': {'mean': np.mean(effect_sizes), 'std': np.std(effect_sizes)} if effect_sizes else {'mean': 0,
                                                                                                          'std': 0},
        'permutation_p_value': p_val,
        'individual_accuracies': accuracies,
        'individual_effect_sizes': effect_sizes,
        'permuted_scores': permuted_scores,
        'true_accuracy': true_acc
    }

    # Log summary
    logger.info(f"  Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    if effect_sizes:
        logger.info(f"  Mean Effect Size (Cohen's d): {np.mean(effect_sizes):.4f} ± {np.std(effect_sizes):.4f}")
    logger.info(f"  Permutation Test p-value: {p_val:.4f}")

    # Validation assessment
    if p_val < 0.05:
        logger.info("Classifier performance significantly exceeds chance")
    else:
        logger.warning("Classifier performance may be at chance level")

    return robustness_results


def analyze_gender_direction_interpretability(clf, embeddings, labels):
    """
    Identifies top contributing dimensions to gender classification.
    Measures weight sparsity and distribution across embedding space.

    Args:
        clf (LogisticRegression): Trained classifier
        embeddings (np.ndarray): Input embeddings
        labels (np.ndarray): Gender labels

    Returns:
        dict: Interpretability analysis results
    """
    logger.info("Gender Direction Interpretability Analysis")

    weights = clf.coef_[0]
    top_dims = np.argsort(np.abs(weights))[-20:][::-1]

    logger.info("Top 20 most gender-informative dimensions:")
    top_dimensions_info = []
    for i, idx in enumerate(top_dims):
        direction = "→ Male" if weights[idx] > 0 else "→ Female"
        dimension_info = {
            'rank': i + 1,
            'dimension': idx,
            'weight': weights[idx],
            'direction': direction
        }
        top_dimensions_info.append(dimension_info)
        logger.info(f"    Dim {idx:4d}: {weights[idx]:8.4f} {direction}")

    # Sparsity and strong dimensions analysis
    sparsity = np.sum(np.abs(weights) < 0.01) / len(weights) * 100
    strong = np.sum(np.abs(weights) > 1.0)
    very_strong = np.sum(np.abs(weights) > 5.0)

    logger.info(f"Weight Sparsity (<0.01): {sparsity:.1f}%")
    logger.info(f"Strong Weights (|w| > 1.0): {strong}")
    logger.info(f"Very Strong Weights (|w| > 5.0): {very_strong}")

    # Distribution analysis across embedding space
    chunk_size = len(weights) // 10
    chunk_norms = [np.linalg.norm(weights[i:i + chunk_size]) for i in range(0, len(weights), chunk_size)]
    cv = np.std(chunk_norms) / np.mean(chunk_norms) if np.mean(chunk_norms) > 0 else 0

    logger.info(f"Coefficient of Variation (chunk norms): {cv:.4f}")

    interpretability_results = {
        'top_dimensions': top_dimensions_info[:10],
        'sparsity_pct': sparsity,
        'strong_weights_count': strong,
        'very_strong_weights_count': very_strong,
        'concentration_cv': cv,
        'weight_distribution': {
            'mean': np.mean(weights),
            'std': np.std(weights),
            'min': np.min(weights),
            'max': np.max(weights)
        },
        'chunk_norms': chunk_norms
    }

    return interpretability_results


def visualize_gender_classifier_analysis(clf, embeddings, labels, robustness_results, interpretability_results):
    """
    Create visualizations for gender classifier analysis.

    Args:
        clf (LogisticRegression): Trained classifier
        embeddings (np.ndarray): Input embeddings
        labels (np.ndarray): Gender labels
        robustness_results (dict): Results from robustness validation
        interpretability_results (dict): Results from interpretability analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Permutation test histogram
    axes[0, 0].hist(robustness_results['permuted_scores'], bins=20, alpha=0.7,
                    edgecolor='black', label='Permuted Accuracies')
    axes[0, 0].axvline(robustness_results['true_accuracy'], color='red',
                       linestyle='--', label='True Accuracy')
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Permutation Test for Classifier Robustness')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Weight distribution histogram
    weights = clf.coef_[0]
    axes[0, 1].hist(weights, bins=50, edgecolor='black', alpha=0.75)
    axes[0, 1].axvline(0, linestyle='--', color='red')
    axes[0, 1].set_xlabel("Weight Value")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Gender Classifier Weights")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Gender information across embedding space
    chunk_norms = interpretability_results['chunk_norms']
    axes[1, 0].plot(chunk_norms, 'o-', linewidth=2)
    axes[1, 0].set_xlabel("Chunk Index")
    axes[1, 0].set_ylabel("L2 Norm")
    axes[1, 0].set_title("Gender Information Across Embedding Space")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Top feature importance
    top_dims = interpretability_results['top_dimensions'][:10]
    weights_top = [dim['weight'] for dim in top_dims]
    dims_top = [f"Dim {dim['dimension']}" for dim in top_dims]

    colors = ['red' if w < 0 else 'blue' for w in weights_top]
    axes[1, 1].barh(range(len(weights_top)), weights_top, color=colors, alpha=0.7)
    axes[1, 1].set_yticks(range(len(dims_top)))
    axes[1, 1].set_yticklabels(dims_top)
    axes[1, 1].set_xlabel("Weight Value")
    axes[1, 1].set_title("Top 10 Gender-Informative Dimensions")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def comprehensive_gender_classifier_analysis(embeddings, labels, visualize=True):
    """
    Run comprehensive gender classifier analysis pipeline.

    Args:
        embeddings (np.ndarray or torch.Tensor): Input embeddings
        labels (np.ndarray): Gender labels
        visualize (bool): Whether to create visualizations

    Returns:
        dict: Complete gender classifier analysis results
    """
    logger.info("Gender classifier analysis")

    # Convert to numpy if needed
    embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings.copy()

    # 1. Train classifier
    classifier, cv_accuracy, train_accuracy = train_gender_classifier_with_teacher_params(
        embeddings_np, labels
    )

    # 2. Analyze gender directions
    gender_weights, direction_norm = analyze_gender_directions(classifier, embeddings_np)

    # 3. Validate robustness
    robustness_results = validate_gender_classifier_robustness(embeddings_np, labels)

    # 4. Analyze interpretability
    interpretability_results = analyze_gender_direction_interpretability(
        classifier, embeddings_np, labels
    )

    # 5. Create visualizations if requested
    if visualize:
        try:
            visualize_gender_classifier_analysis(
                classifier, embeddings_np, labels, robustness_results, interpretability_results
            )
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

    # Compile results
    comprehensive_results = {
        'classifier': classifier,
        'gender_weights': gender_weights,
        'direction_norm': direction_norm,
        'cv_accuracy': cv_accuracy,
        'train_accuracy': train_accuracy,
        'robustness_validation': robustness_results,
        'interpretability_analysis': interpretability_results,
        'analysis_summary': {
            'classifier_reliable': robustness_results['permutation_p_value'] < 0.05,
            'mean_cv_accuracy': robustness_results['accuracy']['mean'],
            'accuracy_stability': robustness_results['accuracy']['std'],
            'weight_sparsity': interpretability_results['sparsity_pct'],
            'strong_dimensions': interpretability_results['strong_weights_count']
        }
    }

    # Log summary
    logger.info("Analysis summary:")
    logger.info(f"  Classifier CV accuracy: {cv_accuracy:.4f}")
    logger.info(
        f"  Robustness (mean ± std): {robustness_results['accuracy']['mean']:.4f} ± {robustness_results['accuracy']['std']:.4f}")
    logger.info(f"  Permutation test p-value: {robustness_results['permutation_p_value']:.4f}")
    logger.info(f"  Weight sparsity: {interpretability_results['sparsity_pct']:.1f}%")
    logger.info(f"  Strong dimensions (|w| > 1.0): {interpretability_results['strong_weights_count']}")

    if comprehensive_results['analysis_summary']['classifier_reliable']:
        logger.info("Gender classifier is reliable for counterfactual manipulation")
    else:
        logger.warning("Gender classifier may not be reliable")

    return comprehensive_results
