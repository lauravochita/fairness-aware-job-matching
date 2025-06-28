"""
VAE baseline architectures for fairness-aware job-matching framework.
Contains the foundational encoder-decoder components and VAE models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils.config import Config, ModelConfig


class Encoder(torch.nn.Module):
    """Basic encoder architecture for VAE models."""

    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return F.relu(self.linear3(x))


class Decoder(torch.nn.Module):
    """Basic decoder architecture for VAE models."""

    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class VAE_Gender(torch.nn.Module):
    """
    VAE specifically adapted for CV embeddings and binary gender classification.
    Based on VAE_fair but modified for the specific dataset requirements.
    """

    def __init__(self, feat_d, hidden_dim, z_dim, a_dim, custom=True):
        super(VAE_Gender, self).__init__()

        # Encoders - adapted for CV embedding data format
        self.encoder_A = Encoder(feat_d + 1, hidden_dim, hidden_dim)  # embeddings + bias
        self.encoder_Z = Encoder(feat_d * 2 + 1, hidden_dim, hidden_dim)  # embeddings*2 + bias

        # Decoders
        self.decoder_r = Decoder(z_dim + a_dim, hidden_dim, feat_d)  # reconstruct embeddings
        self.decoder_z = Decoder(z_dim, hidden_dim, feat_d)  # content reconstruction

        # Gender prediction head (binary classification)
        self.gender_decoder = torch.nn.Sequential(
            torch.nn.Linear(z_dim + a_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2)  # Binary classification (0=female, 1=male)
        )

        self.custom = custom

        # Latent space parameters
        self._enc_mu_Z = torch.nn.Linear(hidden_dim, z_dim)
        self._enc_log_sigma_Z = torch.nn.Linear(hidden_dim, z_dim)
        self._enc_mu_A = torch.nn.Linear(hidden_dim, a_dim)
        self._enc_log_sigma_A = torch.nn.Linear(hidden_dim, a_dim)

    def _sample_latent(self, h_enc, _enc_mu, _enc_log_sigma, mode='test'):
        """Sample from latent distribution."""
        mu = _enc_mu(h_enc)
        log_sigma = _enc_log_sigma(h_enc)

        if self.custom:
            sigma = torch.exp(log_sigma)
        else:
            sigma = torch.exp(F.normalize(log_sigma, dim=1))

        std_z = torch.randn_like(sigma).to(mu.device)

        if mode == 'train':
            return mu + sigma * std_z
        return mu + sigma * std_z, mu, sigma

    def sampling(self, state_A, state_Z):
        """Sample latent variables."""
        h_enc_A = self.encoder_A(state_A)
        h_enc_Z = self.encoder_Z(state_Z)

        Z, mu_Z, sigma_Z = self._sample_latent(h_enc_Z, self._enc_mu_Z, self._enc_log_sigma_Z)
        A, mu_A, sigma_A = self._sample_latent(h_enc_A, self._enc_mu_A, self._enc_log_sigma_A)

        # Store for loss calculation
        self.mean = mu_Z
        self.sigma = sigma_Z
        self.mean_A = mu_A
        self.sigma_A = sigma_A

        return A, Z

    def forward(self, state_A, state_Z):
        """Forward pass."""
        A, Z = self.sampling(state_A, state_Z)

        # Reconstructions
        X_r = self.decoder_r(torch.cat((A, Z), dim=1))  # Full reconstruction
        X_z = self.decoder_z(Z)  # Content reconstruction

        # Gender prediction (binary classification)
        gender_logits = self.gender_decoder(torch.cat((A, Z), dim=1))

        return X_r, X_z, gender_logits


def latent_loss(mu, sigma):
    """
    Calculate KL divergence loss for latent variables.

    Args:
        mu (torch.Tensor): Mean of latent distribution
        sigma (torch.Tensor): Standard deviation of latent distribution

    Returns:
        torch.Tensor: KL divergence loss
    """
    kl_loss = -0.5 * torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
    return kl_loss


def train_baseline_vae(train_data, train_labels, test_data, test_labels,
                       num_epochs=100, batch_size=32, learning_rate=0.0002,
                       device=None):
    """
    Train the baseline VAE model for CV embeddings and gender classification.

    Args:
        train_data (torch.Tensor): Training embedding data
        train_labels (np.ndarray): Training gender labels
        test_data (torch.Tensor): Test embedding data
        test_labels (np.ndarray): Test gender labels
        num_epochs (int): Number of training epochs
        batch_size (int): Training batch size
        learning_rate (float): Learning rate for optimizer
        device (str): Device for training ('cuda' or 'cpu')

    Returns:
        tuple: (trained_model, train_losses, test_losses, test_accuracies)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training baseline VAE for CV embeddings")

    input_dim = train_data.shape[1]

    # Create baseline VAE
    baseline_vae = VAE_Gender(
        feat_d=input_dim,
        hidden_dim=ModelConfig.VAE_HIDDEN_DIM,
        z_dim=ModelConfig.VAE_Z_DIM,
        a_dim=ModelConfig.VAE_A_DIM,
        custom=True
    ).to(device)

    optimizer = torch.optim.Adam(
        baseline_vae.parameters(),
        lr=learning_rate,
        weight_decay=ModelConfig.VAE_WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    train_losses = []
    test_losses = []
    test_accuracies = []

    train_dataset = torch.utils.data.TensorDataset(train_data)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    best_test_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 15

    # KL annealing
    kl_weight = 0.0
    kl_anneal_rate = 1.0 / (num_epochs // 4)

    for epoch in range(num_epochs):
        baseline_vae.train()
        epoch_loss = 0

        # Update KL weight
        if kl_weight < 1.0:
            kl_weight += kl_anneal_rate
            kl_weight = min(kl_weight, 1.0)

        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)

            # Prepare inputs (same format as semi-supervised pipeline)
            ones_column = torch.ones(data.shape[0], 1).to(device)
            state_A = torch.cat([data, ones_column], dim=1)
            state_Z = torch.cat([data, data, ones_column], dim=1)

            optimizer.zero_grad()

            # Forward pass
            X_r, X_z, gender_logits = baseline_vae(state_A, state_Z)

            # Get labels for current batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + data.size(0), len(train_labels))
            batch_labels = train_labels[start_idx:end_idx]

            # Losses
            recon_loss = F.mse_loss(X_r, data)
            content_loss = F.mse_loss(X_z, data)
            kl_loss_z = latent_loss(baseline_vae.mean, baseline_vae.sigma)
            kl_loss_a = latent_loss(baseline_vae.mean_A, baseline_vae.sigma_A)

            # Gender classification loss (properly sized)
            if len(batch_labels) > 0:
                batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long).to(device)
                gender_loss = F.cross_entropy(gender_logits[:len(batch_labels)], batch_labels_tensor)
            else:
                gender_loss = torch.tensor(0.0).to(device)

            # Total loss
            total_loss = (recon_loss + 0.5 * content_loss +
                          kl_weight * (kl_loss_z + 0.1 * kl_loss_a) + 0.5 * gender_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline_vae.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)

        # Evaluate
        test_loss, test_accuracy = evaluate_baseline_vae(baseline_vae, test_data, test_labels, batch_size, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        scheduler.step(test_loss)

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = baseline_vae.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Baseline Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {avg_epoch_loss:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  KL Weight: {kl_weight:.2f}")

        if patience_counter >= patience:
            print(f"Baseline early stopping at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        baseline_vae.load_state_dict(best_model_state)
        print(f"Loaded best baseline model with test loss: {best_test_loss:.4f}")

    return baseline_vae, train_losses, test_losses, test_accuracies


def evaluate_baseline_vae(vae, test_data, test_labels, batch_size=128, device=None):
    """
    Evaluate baseline VAE model.

    Args:
        vae (VAE_Gender): Trained VAE model
        test_data (torch.Tensor): Test embedding data
        test_labels (np.ndarray): Test gender labels
        batch_size (int): Evaluation batch size
        device (str): Device for evaluation

    Returns:
        tuple: (average_loss, accuracy)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae.eval()
    test_dataset = torch.utils.data.TensorDataset(test_data)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    total_loss = 0
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (data,) in enumerate(test_loader):
            data = data.to(device)
            ones_column = torch.ones(data.shape[0], 1).to(device)
            state_A = torch.cat([data, ones_column], dim=1)
            state_Z = torch.cat([data, data, ones_column], dim=1)

            X_r, X_z, gender_logits = vae(state_A, state_Z)

            # Get predictions
            gender_probs = F.softmax(gender_logits, dim=1)
            predictions = torch.argmax(gender_probs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())

            # Calculate losses
            recon_loss = F.mse_loss(X_r, data)
            content_loss = F.mse_loss(X_z, data)
            kl_loss_z = latent_loss(vae.mean, vae.sigma)
            kl_loss_a = latent_loss(vae.mean_A, vae.sigma_A)

            loss = recon_loss + content_loss + kl_loss_z + kl_loss_a
            total_loss += loss.item()

    # Calculate accuracy
    accuracy = None
    if test_labels is not None:
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(test_labels, all_predictions)

    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy


def extract_latent_attributes(vae, data, device=None):
    """
    Extract latent attributes from trained VAE.

    Args:
        vae (VAE_Gender): Trained VAE model
        data (torch.Tensor): Input embedding data
        device (str): Device for computation

    Returns:
        np.ndarray: Extracted latent attribute representations
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae.eval()
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=0
    )

    all_A = []

    with torch.no_grad():
        for batch_data, in loader:
            batch_data = batch_data.to(device)
            ones_column = torch.ones(batch_data.shape[0], 1).to(device)
            state_A = torch.cat([batch_data, ones_column], dim=1)

            h_enc_A = vae.encoder_A(state_A)
            A, mu_A, _ = vae._sample_latent(h_enc_A, vae._enc_mu_A, vae._enc_log_sigma_A, mode='test')
            all_A.append(A.cpu().numpy())

    return np.vstack(all_A)