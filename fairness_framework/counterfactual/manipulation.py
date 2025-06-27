"""
Counterfactual manipulation module for fairness-aware job-matching framework.
Provides embedding manipulation functions for gender bias intervention.
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def flip_embeddings_along_gender_direction(
        embeddings: np.ndarray, gender_weights: np.ndarray, flip_factor: float = 1.0
) -> np.ndarray:
    """
    Flip/rescale embeddings along the gender-informative direction.

    Args:
        embeddings (np.ndarray): Original input embeddings.
        gender_weights (np.ndarray): Learned gender direction.
        flip_factor (float): Degree to which gender direction is removed (1.0 = full flip).
    """
    print(f"\nApplying flip along gender direction (flip factor = {flip_factor:.2f})")

    embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings.copy()
    gender_direction = gender_weights / np.linalg.norm(gender_weights)

    projections = (embeddings_np @ gender_direction[:, np.newaxis]) * gender_direction
    flipped = embeddings_np - (2 * flip_factor * projections)

    diff_norms = np.linalg.norm(flipped - embeddings_np, axis=1)
    print(f"  Avg embedding change (L2): {np.mean(diff_norms):.6f}")
    print(f"  Max change: {np.max(diff_norms):.6f}")
    print(f"  Min change: {np.min(diff_norms):.6f}")

    return flipped


def test_flip_factor_optimality(embeddings, labels, gender_weights, flip_factors=None):
    """
    Tests a range of flip factors to determine an optimal debiasing level.
    Measures classifier performance degradation and embedding distortion.
    """
    print("\nFlip Factor Optimization")

    if flip_factors is None:
        flip_factors = np.arange(0.0, 1.1, 0.1)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(embeddings, labels)
    baseline_acc = clf.score(embeddings, labels)

    results = []

    for f in flip_factors:
        modified = flip_embeddings_along_gender_direction(embeddings, gender_weights, flip_factor=f)
        modified_acc = clf.score(modified, labels)
        drop = baseline_acc - modified_acc
        distortion = np.mean(np.linalg.norm(modified - embeddings, axis=1))
        results.append(
            {'flip_factor': f, 'accuracy_drop': drop, 'distortion': distortion, 'classifier_acc': modified_acc})

    df = pd.DataFrame(results)

    # Print summary
    print(f"{'Flip':<8} {'Acc Drop':<10} {'Distortion':<12} {'Post-Flip Acc':<15}")

    for row in df.itertuples(index=False):
        print(f"{row.flip_factor:<8.1f} {row.accuracy_drop:<10.4f} {row.distortion:<12.6f} {row.classifier_acc:<15.4f}")

    # Plot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(df['flip_factor'], df['accuracy_drop'], 'o-', linewidth=2)
    plt.xlabel("Flip Factor")
    plt.ylabel("Accuracy Drop")
    plt.title("Classifier Performance Degradation")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(df['flip_factor'], df['distortion'], 'o-', linewidth=2, color='orange')
    plt.xlabel("Flip Factor")
    plt.ylabel("Avg Embedding Distortion")
    plt.title("Embedding Distortion")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return df
