"""
Semi-supervised VAE module for fairness-aware job-matching framework.
Contains the enhanced VAE with teacher-student learning for gender inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from fairness_framework.models.vae_baseline import VAE_Gender, latent_loss
from fairness_framework.utils.config import ModelConfig
from fairness_framework.utils.proprietary_imports import load_fairws_vae_components

# Load the external VAE_fair from FairWS framework
VAE_fair, fairws_latent_loss = load_fairws_vae_components()


class ImprovedSemiSupervisedVAE(nn.Module):
    """
    Semi-supervised VAE with enhanced bridge network and balanced consistency loss.
    Integrates teacher-student learning for improved gender inference in high-dimensional embeddings.
    """

    def __init__(self, feat_d, feat_related_d, label_dim, hidden_dim, z_dim, a_dim,
                 temperature=1.0, custom=False):
        super(ImprovedSemiSupervisedVAE, self).__init__()

        # Initialize base VAE
        self.latent_A = None
        self.vae = VAE_fair(feat_d, feat_related_d, label_dim, hidden_dim, z_dim, a_dim,
                            temperature, custom=custom)

        # Teacher classifier (supervised)
        self.teacher = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        self.teacher_trained = False

        # Bridge network
        self.bridge = nn.Sequential(
            nn.Linear(a_dim, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),

            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),

            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1)  # LogSoftmax for numerical stability
        )

        # Consistency loss weight
        self.consistency_weight = 0.5
        self.teacher_preds_cache = None

    def train_teacher(self, embeddings, labels):
        """
        Train the teacher classifier with cross-validation.
        """
        print("Training teacher classifier with cross-validation...")

        # Convert to numpy
        if torch.is_tensor(embeddings):
            embeddings_np = embeddings.cpu().numpy()
        else:
            embeddings_np = embeddings

        # Perform cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Test different C values
        C_values = [0.1, 1.0, 10.0]

        print("Cross-validation results:")
        best_c = 1.0
        best_cv_score = 0

        for C in C_values:
            teacher_cv = LogisticRegression(max_iter=1000, C=C, random_state=42)
            cv_scores = cross_val_score(teacher_cv, embeddings_np, labels, cv=cv, scoring='accuracy')
            mean_cv_score = cv_scores.mean()
            std_cv_score = cv_scores.std()

            print(f"  C={C}: CV Accuracy = {mean_cv_score:.4f} (+/- {std_cv_score * 2:.4f})")

            if mean_cv_score > best_cv_score:
                best_cv_score = mean_cv_score
                best_c = C

        print(f"Best C value: {best_c} with CV accuracy: {best_cv_score:.4f}")

        # Add class weights based on label distribution
        class_counts = np.bincount(labels)
        class_weights = 1.0 / (class_counts.astype(np.float32) + 1e-5)
        class_weights = class_weights / np.sum(class_weights)

        # Train final model with best C and class weights
        self.teacher = LogisticRegression(
            max_iter=1000,
            C=best_c,
            class_weight={i: w for i, w in enumerate(class_weights)},
            random_state=42
        )
        self.teacher.fit(embeddings_np, labels)

        # Report training accuracy
        train_acc = self.teacher.score(embeddings_np, labels)

        print(f"Teacher CV accuracy: {best_cv_score:.4f}")
        print(f"Teacher train accuracy: {train_acc:.4f}")
        print(f"Difference (overfitting indicator): {train_acc - best_cv_score:.4f}")

        # Get teacher predictions for consistency loss
        teacher_probs = self.teacher.predict_proba(embeddings_np)
        self.teacher_preds_cache = torch.FloatTensor(teacher_probs)

        self.teacher_trained = True
        return best_cv_score

    def forward(self, state_A, state_Z, use_teacher=True):
        """
        Forward pass with optional teacher guidance.
        """
        # VAE forward pass
        X_r, X_z, y = self.vae(state_A, state_Z)

        # Get latent A representations
        h_enc_A = self.vae.encoder_A(state_A)

        # Check what _sample_latent returns and handle accordingly
        sample_latent_output = self.vae._sample_latent(h_enc_A, self.vae._enc_mu_A,
                                                       self.vae._enc_log_sigma_A, mode='train')

        # Check returns tuple 2/ 3 elements
        if isinstance(sample_latent_output, tuple) and len(sample_latent_output) == 3:
            A, mu_A, sigma_A = sample_latent_output
        elif isinstance(sample_latent_output, tuple) and len(sample_latent_output) == 2:
            A, mu_A = sample_latent_output
            sigma_A = None
        else:
            # Use the output directly if it's not a tuple
            A = sample_latent_output
            mu_A = self.vae.mean_A  # Access from the VAE
            sigma_A = self.vae.sigma_A if hasattr(self.vae, 'sigma_A') else None

        # Bridge network: Map latent A to gender space
        gender_logits = self.bridge(A)

        # Store for loss calculation
        self.latent_A = A
        self.gender_logits = gender_logits

        return X_r, X_z, y, A, gender_logits

    def get_teacher_predictions(self, embeddings):
        """
        Get teacher predictions for current batch.
        """
        if not self.teacher_trained:
            raise ValueError("Teacher must be trained first")

        if torch.is_tensor(embeddings):
            embeddings_np = embeddings.cpu().numpy()
        else:
            embeddings_np = embeddings

        teacher_probs = self.teacher.predict_proba(embeddings_np)
        return torch.FloatTensor(teacher_probs).to(embeddings.device)

    def compute_consistency_loss(self, embeddings):
        """
        Compute consistency loss between teacher and VAE predictions.
        """
        teacher_probs = self.get_teacher_predictions(embeddings)

        # KL divergence between teacher and student (using log-softmax outputs)
        consistency_loss = F.kl_div(
            self.gender_logits,
            teacher_probs,
            reduction='batchmean',
            log_target=False
        )

        return consistency_loss


def train_semi_supervised_vae_improved(model, train_data, train_labels, test_data, test_labels=None,
                                       num_epochs=150, batch_size=32, learning_rate=0.0002):
    """
    Train the semi-supervised VAE with improvements for better convergence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Train teacher classifier
    print("1. Training Teacher Classifier")
    teacher_acc = model.train_teacher(train_data, train_labels)

    # 2. Train VAE with teacher guidance
    print("2. Training Semi-Supervised VAE")

    # Lower learning weight decay for better stability
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    # Scheduler with more patience
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,
        min_lr=1e-6
    )

    train_losses = []
    test_losses = []
    test_accuracies = [] if test_labels is not None else None
    consistency_losses = []

    train_dataset = torch.utils.data.TensorDataset(train_data)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    best_test_loss = float('inf')
    best_model_state = None
    best_accuracy = 0
    patience = 25
    early_stop_counter = 0

    # Create class weights for balancing
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / (class_counts.astype(np.float32) + 1e-5)
    class_weights = class_weights / np.sum(class_weights)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    # Slower KL annealing
    kl_weight = 0.0
    kl_anneal_rate = 1.0 / (num_epochs // 4)

    # Gradual consistency weight increase
    consistency_schedule = np.linspace(0.05, 1.0, num_epochs // 2)

    # Training epochs
    for epoch in tqdm(range(num_epochs), desc="Training Semi-Supervised VAE"):
        model.train()
        epoch_loss = 0
        epoch_consistency_loss = 0

        # Update KL weight
        if kl_weight < 1.0:
            kl_weight += kl_anneal_rate
            kl_weight = min(kl_weight, 1.0)

        # Update consistency weight
        if epoch < len(consistency_schedule):
            model.consistency_weight = consistency_schedule[epoch]
        else:
            model.consistency_weight = 1.0

        # Training loop
        for batch_idx, (data,) in enumerate(train_loader):
            # Ensure data is on correct device
            data = data.to(device)

            # Prepare inputs
            ones_column = torch.ones(data.shape[0], 1).to(device)
            state_A = torch.cat([data, ones_column], dim=1)
            state_Z = torch.cat([data, data, ones_column], dim=1)

            optimizer.zero_grad()

            # Forward pass
            X_r, X_z, y, A, gender_logits = model(state_A, state_Z)

            # VAE losses - use the FairWS latent_loss function
            recon_loss = F.mse_loss(X_r, data)
            content_loss = F.mse_loss(X_z, data)
            kl_loss_z = fairws_latent_loss(model.vae.mean, model.vae.sigma)
            kl_loss_a = fairws_latent_loss(model.vae.mean_A, model.vae.sigma_A)

            # Get teacher predictions for current batch
            teacher_probs = model.get_teacher_predictions(data)

            # Weight teacher probabilities by inverse class frequency
            batch_weights = torch.zeros_like(teacher_probs)
            for c in range(len(class_weights)):
                batch_weights[:, c] = class_weights_tensor[c]

            weighted_teacher_probs = teacher_probs * batch_weights
            weighted_teacher_probs = weighted_teacher_probs / (weighted_teacher_probs.sum(dim=1, keepdim=True) + 1e-8)

            # Consistency loss with weighted teacher probabilities
            consistency_loss = F.kl_div(
                gender_logits,
                weighted_teacher_probs,
                reduction='batchmean'
            )

            # Total loss (with adjusted weights)
            vae_loss = recon_loss + 0.5 * content_loss + kl_weight * (kl_loss_z + 0.5 * kl_loss_a)
            total_loss = vae_loss + model.consistency_weight * consistency_loss

            # Small direct supervision on a random subset for stability
            if torch.rand(1).item() < 0.1:
                # Get a small subset of batch indices
                subset_size = min(8, data.size(0))
                subset_indices = torch.randperm(data.size(0))[:subset_size]

                # Get corresponding labels
                batch_indices = batch_idx * batch_size + subset_indices.cpu().numpy()
                batch_indices = batch_indices[batch_indices < len(train_labels)]

                if len(batch_indices) > 0:
                    subset_labels = torch.tensor(train_labels[batch_indices]).to(device)
                    subset_logits = gender_logits[subset_indices[:len(batch_indices)]]

                    # Small direct supervision signal
                    direct_loss = F.cross_entropy(
                        subset_logits,
                        subset_labels,
                        weight=class_weights_tensor
                    )
                    total_loss = total_loss + 0.05 * direct_loss

            total_loss.backward()
            # Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_consistency_loss += consistency_loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_consistency_loss = epoch_consistency_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        consistency_losses.append(avg_consistency_loss)

        # Evaluate
        test_loss, test_accuracy = evaluate_semi_supervised_vae(model, test_data, test_labels, batch_size)
        test_losses.append(test_loss)
        if test_accuracies is not None:
            test_accuracies.append(test_accuracy)

        # Use test accuracy to decide early stopping
        if test_accuracy is not None and test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
            print(f"New best accuracy: {best_accuracy:.4f}")
        else:
            early_stop_counter += 1

        # Use test loss for learning rate scheduling
        scheduler.step(test_loss)

        # Logging
        if (epoch + 1) % 10 == 0 or epoch < 5 or early_stop_counter > 10:
            if test_accuracy is not None:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"  Train Loss: {avg_epoch_loss:.4f}")
                print(f"  Consistency Loss: {avg_consistency_loss:.4f}")
                print(f"  Test Loss: {test_loss:.4f}")
                print(f"  Test Accuracy: {test_accuracy:.4f}")
                print(f"  KL Weight: {kl_weight:.2f}")
                print(f"  Consistency Weight: {model.consistency_weight:.2f}")
                print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
            else:
                print(f"Epoch {epoch + 1}/{num_epochs}, Train: {avg_epoch_loss:.4f}, Test: {test_loss:.4f}")

        # Early stopping based on accuracy plateau
        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1} - no accuracy improvement for {patience} epochs")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with accuracy: {best_accuracy:.4f}")

    return model, train_losses, test_losses, test_accuracies, consistency_losses


def evaluate_semi_supervised_vae(model, test_data, test_labels=None, batch_size=128):
    """
    Evaluate the semi-supervised VAE.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    test_dataset = torch.utils.data.TensorDataset(test_data)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    total_loss = 0
    all_latent_a = []
    all_gender_probs = []

    with torch.no_grad():
        for batch_idx, (data,) in enumerate(test_loader):
            # Ensure data is on correct device
            data = data.to(device)

            ones_column = torch.ones(data.shape[0], 1).to(device)
            state_A = torch.cat([data, ones_column], dim=1)
            state_Z = torch.cat([data, data, ones_column], dim=1)

            # Forward pass
            X_r, X_z, y, A, gender_logits = model(state_A, state_Z, use_teacher=False)

            all_latent_a.append(A.cpu().numpy())
            all_gender_probs.append(F.softmax(gender_logits, dim=1).cpu().numpy())

            # Calculate losses using FairWS latent_loss
            recon_loss = F.mse_loss(X_r, data)
            content_loss = F.mse_loss(X_z, data)
            kl_loss_z = fairws_latent_loss(model.vae.mean, model.vae.sigma)
            kl_loss_a = fairws_latent_loss(model.vae.mean_A, model.vae.sigma_A)

            loss = recon_loss + content_loss + kl_loss_z + kl_loss_a
            total_loss += loss.item()

    # Combine results
    latent_a = np.vstack(all_latent_a)
    gender_probs = np.vstack(all_gender_probs)

    # Evaluate bridge network accuracy
    accuracy = None
    if test_labels is not None:
        bridge_preds = (gender_probs[:, 1] > 0.5).astype(int)
        accuracy = accuracy_score(test_labels, bridge_preds)

    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy


def extract_semi_supervised_predictions(model, data):
    """
    Extract all predictions from the semi-supervised VAE.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=0
    )

    all_latent_a = []
    all_gender_probs = []
    all_teacher_probs = []

    with torch.no_grad():
        for batch_data, in loader:
            # Ensure data is on correct device
            batch_data = batch_data.to(device)

            ones_column = torch.ones(batch_data.shape[0], 1).to(device)
            state_A = torch.cat([batch_data, ones_column], dim=1)
            state_Z = torch.cat([batch_data, batch_data, ones_column], dim=1)

            # VAE predictions
            X_r, X_z, y, A, gender_logits = model(state_A, state_Z, use_teacher=False)
            all_latent_a.append(A.cpu().numpy())
            all_gender_probs.append(F.softmax(gender_logits, dim=1).cpu().numpy())

            # Teacher predictions
            teacher_probs = model.get_teacher_predictions(batch_data)
            all_teacher_probs.append(teacher_probs.cpu().numpy())

    latent_a = np.vstack(all_latent_a)
    gender_probs = np.vstack(all_gender_probs)
    teacher_probs = np.vstack(all_teacher_probs)

    return latent_a, gender_probs, teacher_probs


def evaluate_gender_inference(latent_A, true_genders):
    """
    Evaluate gender inference using multiple methods (threshold, linear combo, logistic regression).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    print("Evaluating gender inference with multiple methods...")

    methods = {}

    # 1. Simple threshold on first dimension
    dim1_threshold = np.mean(latent_A[:, 0])
    pred_dim1 = (latent_A[:, 0] > dim1_threshold).astype(int)
    acc_dim1 = accuracy_score(true_genders, pred_dim1)
    f1_dim1 = f1_score(true_genders, pred_dim1)
    methods["First dimension threshold"] = {
        "predictions": pred_dim1,
        "accuracy": acc_dim1,
        "f1": f1_dim1
    }

    # 2. If at least 2 dimensions, try linear combination
    if latent_A.shape[1] >= 2:
        # 2a. Optimal linear combination
        linear_combo = 0.8 * latent_A[:, 0] + 0.6 * latent_A[:, 1]
        combo_threshold = np.mean(linear_combo)
        pred_combo = (linear_combo > combo_threshold).astype(int)
        acc_combo = accuracy_score(true_genders, pred_combo)
        f1_combo = f1_score(true_genders, pred_combo)
        methods["Linear combination"] = {
            "predictions": pred_combo,
            "accuracy": acc_combo,
            "f1": f1_combo
        }

        # 2b. Use logistic regression with cross-validation
        clf = LogisticRegression(max_iter=1000, C=1.0)
        pred_lr = cross_val_predict(clf, latent_A[:, :2], true_genders, cv=5, method='predict')
        proba_lr = cross_val_predict(clf, latent_A[:, :2], true_genders, cv=5, method='predict_proba')[:, 1]
        acc_lr = accuracy_score(true_genders, pred_lr)
        f1_lr = f1_score(true_genders, pred_lr)
        methods["Logistic regression"] = {
            "predictions": pred_lr,
            "accuracy": acc_lr,
            "f1": f1_lr,
            "probabilities": proba_lr
        }

    # Find the best method based on accuracy
    best_method_name = max(methods.items(), key=lambda x: x[1]["accuracy"])[0]
    best_method = methods[best_method_name]
    predictions = best_method["predictions"]
    accuracy = best_method["accuracy"]
    f1 = best_method["f1"]

    # Calculate additional metrics
    precision = precision_score(true_genders, predictions)
    recall = recall_score(true_genders, predictions)

    # Report results
    print(f"\nGender inference results:")
    for name, method in methods.items():
        print(f"  {name}: Accuracy = {method['accuracy']:.4f}, F1 = {method['f1']:.4f}")

    print(f"\nBest method: {best_method_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Show confusion matrix
    conf_matrix = confusion_matrix(true_genders, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Female (0)', 'Male (1)'],
                yticklabels=['Female (0)', 'Male (1)'])
    plt.xlabel('Predicted Gender')
    plt.ylabel('True Gender')
    plt.title(f'Gender Inference Confusion Matrix (Method: {best_method_name})')
    plt.show()

    # Visualize the latent space
    plt.figure(figsize=(10, 8))

    if latent_A.shape[1] > 1:  # 2D or higher latent space
        correct_pred = predictions == true_genders
        plt.scatter(latent_A[correct_pred & (true_genders == 0), 0], latent_A[correct_pred & (true_genders == 0), 1],
                    color='blue', alpha=0.7, marker='o', label='Female (correct)')
        plt.scatter(latent_A[correct_pred & (true_genders == 1), 0], latent_A[correct_pred & (true_genders == 1), 1],
                    color='red', alpha=0.7, marker='o', label='Male (correct)')
        plt.scatter(latent_A[~correct_pred & (true_genders == 0), 0], latent_A[~correct_pred & (true_genders == 0), 1],
                    color='blue', alpha=0.7, marker='x', label='Female (incorrect)')
        plt.scatter(latent_A[~correct_pred & (true_genders == 1), 0], latent_A[~correct_pred & (true_genders == 1), 1],
                    color='red', alpha=0.7, marker='x', label='Male (incorrect)')

        plt.title(f'Latent Attribute Space (A) for Gender - Accuracy: {accuracy:.4f}')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # For logistic regression, plot decision boundary
        if "Logistic regression" in methods:
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(latent_A[:, :2], true_genders)

            x_min, x_max = latent_A[:, 0].min() - 0.5, latent_A[:, 0].max() + 0.5
            y_min, y_max = latent_A[:, 1].min() - 0.5, latent_A[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                 np.arange(y_min, y_max, 0.02))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, colors='black', linestyles='-', levels=[0.5])

    else:  # 1D latent space
        plt.hist([latent_A[true_genders == 0, 0], latent_A[true_genders == 1, 0]],
                 bins=20, alpha=0.7, label=['Female', 'Male'])
        plt.axvline(dim1_threshold, color='black', linestyle='--',
                    label=f'Decision Boundary ({dim1_threshold:.3f})')
        plt.title('Latent Attribute Space (A) for Gender')
        plt.xlabel('Attribute Value')
        plt.ylabel('Count')
        plt.legend()

    plt.show()

    return predictions, accuracy, f1, conf_matrix


def run_semi_supervised_gender_vae_fixed():
    """
    Train custom baseline first, then semi-supervised, then compare.
    """
    from ..data.data_loader import load_embeddings, load_oracle_data, prepare_data_with_oracle
    from .vae_baseline import train_baseline_vae, extract_latent_attributes
    from sklearn.model_selection import train_test_split
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Semi-Supervised VAE Pipeline")
    embeddings, candidate_indices = load_embeddings()
    oracle_df = load_oracle_data()
    valid_embeddings, valid_indices, valid_genders = prepare_data_with_oracle(
        embeddings, candidate_indices, oracle_df
    )

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        valid_embeddings, valid_genders, test_size=0.2, random_state=42, stratify=valid_genders
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    all_data_tensor = torch.FloatTensor(valid_embeddings).to(device)

    print(f"Training set: {X_train_tensor.shape}")
    print(f"Testing set: {X_test_tensor.shape}")
    print(f"Train gender distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Test gender distribution: {pd.Series(y_test).value_counts().to_dict()}")

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Step 1: Train Custom Baseline VAE
    print("Step 1: Training Custom VAE Baseline")
    baseline_vae, baseline_train_losses, baseline_test_losses, baseline_test_accuracies = train_baseline_vae(
        X_train_tensor, y_train, X_test_tensor, y_test,
        num_epochs=100,
        batch_size=32,
        learning_rate=0.0002,
        device=device
    )

    # Extract custom baseline predictions
    print("\n Evaluating Custom Baseline")
    clean_baseline_latent_a = extract_latent_attributes(baseline_vae, all_data_tensor, device)
    baseline_predictions, baseline_accuracy, baseline_f1, baseline_conf_matrix = evaluate_gender_inference(
        clean_baseline_latent_a, valid_genders
    )

    print(f"\n Custom BASELINE VAE training completed")
    print(f"Custom Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"Custom Baseline F1: {baseline_f1:.4f}")

    # Step2: Train Semi-Supervised VAE
    print("Step 2: Training Semi-Supervised VAE")

    input_dim = X_train_tensor.shape[1]
    model = ImprovedSemiSupervisedVAE(
        feat_d=input_dim,
        feat_related_d=input_dim,
        label_dim=1,
        hidden_dim=512,
        z_dim=32,
        a_dim=2,
        temperature=1.0,
        custom=True
    ).to(device)

    try:
        trained_model, train_losses, test_losses, test_accuracies, consistency_losses = train_semi_supervised_vae_improved(
            model, X_train_tensor, y_train, X_test_tensor, y_test,
            num_epochs=150,
            batch_size=32,
            learning_rate=0.0002
        )

        # Step 3: Final Evaluation and Comparison

        print("Step 3: Final Evaluation and Comparison")

        # Extract predictions from semi-supervised model
        print("Extracting semi-supervised predictions...")
        semi_latent_a, gender_probs, teacher_probs = extract_semi_supervised_predictions(trained_model, all_data_tensor)

        # Get predictions from different methods
        bridge_preds = (gender_probs[:, 1] > 0.5).astype(int)
        teacher_preds = (teacher_probs[:, 1] > 0.5).astype(int)

        # Calculate accuracies
        bridge_acc = accuracy_score(valid_genders, bridge_preds)
        teacher_acc = accuracy_score(valid_genders, teacher_preds)

        print("\nMETHOD COMPARISON")
        print(f"Custom Baseline VAE:      {baseline_accuracy:.4f}")
        print(f"Teacher (supervised):     {teacher_acc:.4f}")
        print(f"Bridge (semi-supervised): {bridge_acc:.4f}")

        # Calculate TRUE improvement
        true_improvement = bridge_acc - baseline_accuracy
        print(f"\n True Improvement: {true_improvement:.4f} ({true_improvement * 100:.1f}%)")

        # Detailed evaluation of semi-supervised latent space
        print("\nSemi-Supervised Latent Space Analysis")
        semi_predictions, semi_accuracy, semi_f1, semi_conf_matrix = evaluate_gender_inference(
            semi_latent_a, valid_genders
        )

        # Step4: Save Results
        print("\Saving Results...")
        torch.save({
            'baseline_model_state_dict': baseline_vae.state_dict(),
            'semisupervised_model_state_dict': trained_model.state_dict(),
            'baseline_accuracy': baseline_accuracy,
            'teacher_accuracy': teacher_acc,
            'bridge_accuracy': bridge_acc,
            'semi_supervised_accuracy': semi_accuracy,
            'true_improvement': true_improvement,
            'baseline_train_losses': baseline_train_losses,
            'semisupervised_train_losses': train_losses,
            'consistency_losses': consistency_losses,
        }, 'semi_supervised_comparison.pt')

        # Save detailed results
        results_df = pd.DataFrame({
            'candidate_index': valid_indices,
            'true_gender': valid_genders,
            'baseline_prediction': baseline_predictions,
            'teacher_prediction': teacher_preds,
            'bridge_prediction': bridge_preds,
            'semi_supervised_prediction': semi_predictions,
            'teacher_prob': teacher_probs[:, 1],
            'bridge_prob': gender_probs[:, 1],
            'baseline_latent_dim_1': clean_baseline_latent_a[:, 0],
            'baseline_latent_dim_2': clean_baseline_latent_a[:, 1] if clean_baseline_latent_a.shape[1] > 1 else np.nan,
            'semisupervised_latent_dim_1': semi_latent_a[:, 0],
            'semisupervised_latent_dim_2': semi_latent_a[:, 1] if semi_latent_a.shape[1] > 1 else np.nan
        })

        results_df.to_csv('semi_supervised_results.csv', index=False)

        print("\nPipeline Complete")
        print(f"Results saved to: semi_supervised_results.csv")

        return trained_model, baseline_vae, results_df, true_improvement

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
