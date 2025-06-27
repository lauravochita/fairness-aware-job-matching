"""
Model architectures and training modules.
"""

from .vae_baseline import VAE_Gender, Encoder, Decoder, train_baseline_vae, evaluate_baseline_vae, extract_latent_attributes
from .semi_supervised_vae import (
    ImprovedSemiSupervisedVAE,
    train_semi_supervised_vae_improved,
    evaluate_semi_supervised_vae,
    extract_semi_supervised_predictions,
    evaluate_gender_inference,
    run_semi_supervised_gender_vae_fixed
)

__all__ = [
    'VAE_Gender', 'Encoder', 'Decoder',
    'train_baseline_vae', 'evaluate_baseline_vae', 'extract_latent_attributes',
    'ImprovedSemiSupervisedVAE',
    'train_semi_supervised_vae_improved',
    'evaluate_semi_supervised_vae',
    'extract_semi_supervised_predictions',
    'evaluate_gender_inference',
    'run_semi_supervised_gender_vae_fixed'
]