"""Train a practical, bounded MLP to predict vendor-error fraction (0..0.04) from quote-time features,
with 5-fold CV, early stopping, baseline comparison, and business metrics/plots.
"""

import os
import json
import math
import random
import argparse
import time
from decimal import Decimal
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from app.core.prediction.model.architecture import BoundedMultiLayerPerceptron

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_CSV_PATH = "app/core/prediction/data/trades.csv"
WEIGHTS_DIRECTORY = "app/core/prediction/weights"
ARITFACTS_DIRECTORY = "app/core/prediction/artifacts"
WEIGHTS_FILE_PATH = os.path.join(WEIGHTS_DIRECTORY, "vendor_error_mlp.pt")
METADATA_FILE_PATH = os.path.join(ARITFACTS_DIRECTORY, "vendor_error_meta.json")

RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.2
CROSS_VALIDATION_FOLDS = 5
TRAINING_BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY_FACTOR = 1e-4
MAXIMUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 8
HIDDEN_LAYER_SIZES = [32, 16]
DROPOUT_RATE = 0.10
DATA_LOADER_WORKERS = 0
MAXIMUM_ADJUSTMENT_FRACTION = 0.04  # bound for predicted adjustment factor
RELATIVE_TOLERANCE = 1e-6


# -----------------------------
# Utility Functions
# -----------------------------
def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducible results across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


TOKEN_DECIMALS = {"ETH": 18, "WETH": 18, "USDC": 6, "USDT": 6}


def to_units(amount_str: str, token: str) -> float:
    """Convert a stringified token amount to proper units based on token decimals."""
    return float(Decimal(amount_str) / Decimal(10 ** TOKEN_DECIMALS.get(token, 18)))


class FeatureStandardizer:
    """Standard scaler that fits on training data only to prevent data leakage."""

    def __init__(self):
        self.feature_means = None
        self.feature_stds = None

    def fit(self, features: np.ndarray) -> None:
        """Compute mean and standard deviation for each feature."""
        self.feature_means = features.mean(axis=0)
        self.feature_stds = features.std(axis=0)
        # Prevent division by zero for constant features
        self.feature_stds[self.feature_stds == 0] = 1e-6

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Standardize features using computed statistics."""
        return (features - self.feature_means) / self.feature_stds

    def to_dictionary(self) -> Dict[str, list]:
        """Export standardizer parameters for saving."""
        return {"mean": self.feature_means.tolist(), "std": self.feature_stds.tolist()}

    @staticmethod
    def from_dictionary(parameters_dict: Dict[str, list]) -> "FeatureStandardizer":
        """Load standardizer from saved parameters."""
        standardizer = FeatureStandardizer()
        standardizer.feature_means = np.array(parameters_dict["mean"], dtype=np.float32)
        standardizer.feature_stds = np.array(parameters_dict["std"], dtype=np.float32)
        return standardizer


# -----------------------------
# Feature Engineering
# -----------------------------
BASE_FEATURE_COLUMNS = [
    "vendor_onehot_odos",
    "vendor_onehot_1inch",
    "log_amount_in_usd",
    "log_output_amount_quote",
    "quote_gas_usd",
    "hop_count",
    "log_min_pool_tvl_usd",
]

# Lightweight interactions that capture business intuition
FEATURE_INTERACTIONS = [
    # Trade size × route complexity
    ("log_amount_in_usd", "hop_count", "log_amount_x_hops"),
    # Vendor-specific size effects
    ("vendor_onehot_odos", "log_amount_in_usd", "odos_x_log_amount"),
    ("vendor_onehot_1inch", "log_amount_in_usd", "inch_x_log_amount"),
    # Vendor-specific route complexity effects
    ("vendor_onehot_odos", "hop_count", "odos_x_hops"),
    ("vendor_onehot_1inch", "hop_count", "inch_x_hops"),
]

TARGET_COLUMN_NAME = "predicted_adjustment_factor"


def build_feature_matrix(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict]:
    """Extract and engineer features from the raw dataset."""

    # Create vendor one-hot encodings
    dataframe["vendor_onehot_odos"] = (dataframe["vendor"] == "odos").astype(float)
    dataframe["vendor_onehot_1inch"] = (dataframe["vendor"] == "1inch").astype(float)

    # Guard against unknown vendors
    known_vendors = {"odos", "1inch"}
    actual_vendors = set(dataframe["vendor"].unique())
    assert known_vendors.issuperset(
        actual_vendors
    ), f"Unknown vendors found: {actual_vendors - known_vendors}"

    # Apply log transformations to skewed numerical features
    dataframe["log_amount_in_usd"] = np.log1p(dataframe["amount_in_usd"].astype(float))
    dataframe["log_min_pool_tvl_usd"] = np.log1p(
        dataframe["min_pool_tvl_usd"].astype(float)
    )

    # Convert token amounts using proper decimals and apply log transform
    output_amount_quote_units = dataframe.apply(
        lambda r: to_units(str(r["quote_out_token"]), r["out_token"]), axis=1
    ).astype(float)
    dataframe["log_output_amount_quote"] = np.log1p(output_amount_quote_units)

    # Keep remaining features as simple floats
    dataframe["quote_gas_usd"] = dataframe["quote_gas_usd"].astype(float)
    dataframe["hop_count"] = dataframe["hop_count"].astype(float)

    # Create interaction features
    for feature_a, feature_b, interaction_name in FEATURE_INTERACTIONS:
        dataframe[interaction_name] = dataframe[feature_a].astype(float) * dataframe[
            feature_b
        ].astype(float)

    # Assemble final feature list
    all_feature_columns = BASE_FEATURE_COLUMNS + [
        interaction_name for _, _, interaction_name in FEATURE_INTERACTIONS
    ]

    # Extract target variable
    target_values = (
        dataframe[TARGET_COLUMN_NAME].astype(float).values.astype(np.float32)
    )

    # Validate target is within expected business bounds
    if not (
        np.nanmax(target_values) <= MAXIMUM_ADJUSTMENT_FRACTION + 1e-8
        and np.nanmin(target_values) >= -1e-8
    ):
        raise ValueError(
            "Target values outside expected [0, 0.04] range; check data generation."
        )

    # Assemble feature matrix
    feature_matrix = dataframe[all_feature_columns].values.astype(np.float32)

    # Create metadata for model tracking
    feature_metadata = {
        "feature_cols": all_feature_columns,
        "target_col": TARGET_COLUMN_NAME,
        "interactions": [
            interaction_name for _, _, interaction_name in FEATURE_INTERACTIONS
        ],
        "vendor_map": {"odos": 0, "1inch": 1},
        "note": "All features are from quote-time; token amounts transformed with log1p using proper decimals; interactions capture vendor/size/hops effects.",
    }

    # Final data quality check
    if np.any(~np.isfinite(feature_matrix)) or np.any(~np.isfinite(target_values)):
        raise ValueError("Found NaN/Inf values in features or target.")

    return dataframe, feature_matrix, target_values, feature_metadata


# -----------------------------
# Dataset and Model Classes
# -----------------------------
class TrainingDataset(Dataset):
    """PyTorch dataset wrapper for numpy feature and target arrays."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).float().view(-1, 1)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


# -----------------------------
# Training and Evaluation Functions
# -----------------------------
def run_training_epoch(
    model, data_loader, loss_criterion, optimizer=None, device="cpu"
):
    """Run one epoch of training or validation."""
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_samples = 0

    for feature_batch, target_batch in data_loader:
        feature_batch = feature_batch.to(device)
        target_batch = target_batch.to(device)

        predictions = model(feature_batch)
        loss = loss_criterion(predictions, target_batch)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = target_batch.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def perform_cross_validation_training(
    features: np.ndarray,
    targets: np.ndarray,
    num_folds: int,
    device: str = "cpu",
    max_epochs: int = MAXIMUM_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
):
    """Train model using k-fold CV, with per-epoch logging and fold timing; returns histories and best fold pack."""
    cross_validator = KFold(n_splits=num_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_training_histories = []
    best_model_candidates = []

    for fold_number, (train_idx, val_idx) in enumerate(
        cross_validator.split(features), 1
    ):
        fold_start = time.time()

        Xtr, ytr = features[train_idx], targets[train_idx]
        Xva, yva = features[val_idx], targets[val_idx]

        # Normalize targets to [0, 1] for better training
        ytr_scaled = ytr / MAXIMUM_ADJUSTMENT_FRACTION
        yva_scaled = yva / MAXIMUM_ADJUSTMENT_FRACTION

        scaler = FeatureStandardizer()
        scaler.fit(Xtr)
        Xtr_s = scaler.transform(Xtr).astype(np.float32)
        Xva_s = scaler.transform(Xva).astype(np.float32)

        # Print sanity checks
        print(
            f"[Fold {fold_number}] Target range: y=[{ytr.min():.6f}, {ytr.max():.6f}]"
        )
        print(
            f"[Fold {fold_number}] Features shape: {Xtr_s.shape}, first 5 vals: {Xtr_s[0, :5]}"
        )

        train_loader = DataLoader(
            TrainingDataset(Xtr_s, ytr_scaled),
            batch_size=TRAINING_BATCH_SIZE,
            shuffle=True,
            num_workers=DATA_LOADER_WORKERS,
        )
        val_loader = DataLoader(
            TrainingDataset(Xva_s, yva_scaled),
            batch_size=TRAINING_BATCH_SIZE,
            shuffle=False,
            num_workers=DATA_LOADER_WORKERS,
        )

        model = BoundedMultiLayerPerceptron(
            features.shape[1], HIDDEN_LAYER_SIZES, DROPOUT_RATE
        ).to(device)
        opt = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY_FACTOR
        )
        crit = nn.L1Loss()  # Use L1Loss to align with MAE business metric

        best_val = math.inf
        best_state = None
        hist = {"train": [], "val": []}
        patience_left = patience

        for epoch in range(1, max_epochs + 1):
            ep_start = time.time()
            tr_loss = run_training_epoch(model, train_loader, crit, opt, device)
            va_loss = run_training_epoch(model, val_loader, crit, None, device)
            hist["train"].append(tr_loss)
            hist["val"].append(va_loss)

            improved = va_loss < best_val - 1e-6
            if improved:
                best_val = va_loss
                best_state = {
                    k: v.cpu().detach().clone() for k, v in model.state_dict().items()
                }
                patience_left = patience
            else:
                patience_left -= 1

            # Calculate MAE in business units for validation
            with torch.no_grad():
                model.eval()
                val_mae_scaled = va_loss  # This is already L1Loss on scaled targets
                val_mae_bps = val_mae_scaled * MAXIMUM_ADJUSTMENT_FRACTION * 10000
                model.train()

            print(
                f"[Fold {fold_number}][Epoch {epoch:03d}] "
                f"train={tr_loss:.6f} val={va_loss:.6f} val_mae_bps={val_mae_bps:.2f} "
                f"{'(best)' if improved else ''} patience={patience_left} "
                f"elapsed={time.time()-ep_start:.2f}s"
            )

            if patience_left == 0:
                break

        # Check for prediction saturation
        model.eval()
        with torch.no_grad():
            pred_s = (
                model(torch.from_numpy(Xva_s[:100]).to(device)).cpu().numpy().ravel()
            )
        pct_bounds_s = np.mean((pred_s <= 0.01) | (pred_s >= 0.99)) * 100
        pct_bounds = (
            np.mean(
                (pred_s * MAXIMUM_ADJUSTMENT_FRACTION <= 0.0004)
                | (pred_s * MAXIMUM_ADJUSTMENT_FRACTION >= 0.04 - 0.0004)
            )
            * 100
        )
        print(
            f"[Fold {fold_number}] preds@bounds scaled(0/1): {pct_bounds_s:.1f}%  unscaled([0,0.04]): {pct_bounds:.1f}%"
        )

        fold_training_histories.append(hist)
        best_model_candidates.append((best_val, best_state, scaler))
        print(
            f"[Fold {fold_number}] best_validation_loss={best_val:.6f} "
            f"epochs={len(hist['val'])} fold_time={time.time()-fold_start:.2f}s"
        )

    best_idx = int(np.argmin([bv for bv, _, _ in best_model_candidates]))
    return fold_training_histories, best_idx, best_model_candidates[best_idx]


def plot_training_learning_curve(training_histories, best_fold_index, output_path):
    """Create and save learning curve plot for the best performing fold."""
    best_history = training_histories[best_fold_index]

    plt.figure()
    plt.plot(best_history["train"], label="train")
    plt.plot(best_history["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("L1Loss")
    plt.title(f"Learning Curve (Fold {best_fold_index + 1})")
    plt.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def categorize_trade_amount(amount_usd: float) -> str:
    """Categorize trade amounts into buckets for stratified analysis."""
    if amount_usd < 10:
        return "1"
    elif amount_usd < 100:
        return "10"
    elif amount_usd < 1_000:
        return "100"
    elif amount_usd < 10_000:
        return "1k"
    elif amount_usd < 100_000:
        return "10k"
    elif amount_usd < 1_000_000:
        return "100k"
    else:
        return "1m+"


def evaluate_business_metrics(
    dataframe: pd.DataFrame,
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    model_tag: str,
):
    """Compute and print business-relevant performance metrics."""
    prediction_errors = predicted_values - true_values
    mean_absolute_error = np.mean(np.abs(prediction_errors))
    mean_absolute_error_basis_points = mean_absolute_error * 10_000.0

    # Coverage within tolerance bands
    coverage_25_bps = np.mean(np.abs(prediction_errors) <= 0.0025)
    coverage_50_bps = np.mean(np.abs(prediction_errors) <= 0.0050)
    coverage_100_bps = np.mean(np.abs(prediction_errors) <= 0.0100)

    print(
        f"[{model_tag}] MAE={mean_absolute_error:.6f}  MAE_bps={mean_absolute_error_basis_points:.2f}  |err|<=25bps:{coverage_25_bps:.2%}  <=50bps:{coverage_50_bps:.2%}  <=100bps:{coverage_100_bps:.2%}"
    )

    # Stratified analysis by vendor and trade size
    analysis_dataframe = dataframe.copy()
    analysis_dataframe["_y_true"] = true_values
    analysis_dataframe["_y_pred"] = predicted_values
    analysis_dataframe["_bucket"] = (
        analysis_dataframe["amount_in_usd"].astype(float).apply(categorize_trade_amount)
    )

    # By vendor
    for vendor in ["odos", "1inch"]:
        vendor_data = analysis_dataframe[analysis_dataframe["vendor"] == vendor]
        if len(vendor_data) > 0:
            vendor_errors = np.abs(vendor_data["_y_pred"] - vendor_data["_y_true"])
            vendor_mae_bps = np.mean(vendor_errors) * 10_000
            print(
                f"  [{model_tag}][{vendor}] n={len(vendor_data)}  MAE_bps={vendor_mae_bps:.2f}"
            )

    # By trade size bucket
    bucket_order = ["1", "10", "100", "1k", "10k", "100k", "1m+"]
    for bucket in sorted(
        analysis_dataframe["_bucket"].unique(), key=lambda x: bucket_order.index(x)
    ):
        bucket_data = analysis_dataframe[analysis_dataframe["_bucket"] == bucket]
        bucket_errors = np.abs(bucket_data["_y_pred"] - bucket_data["_y_true"])
        bucket_mae_bps = np.mean(bucket_errors) * 10_000
        print(
            f"  [{model_tag}][size={bucket}] n={len(bucket_data)}  MAE_bps={bucket_mae_bps:.2f}"
        )


# -----------------------------
# Baseline Model
# -----------------------------
def evaluate_linear_regression_baseline(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    test_dataframe: pd.DataFrame,
):
    """Train and evaluate a linear regression baseline for comparison."""
    baseline_regressor = LinearRegression()
    baseline_regressor.fit(train_features, train_targets)
    baseline_predictions = baseline_regressor.predict(test_features)

    # Enforce business bounds
    baseline_predictions = np.clip(
        baseline_predictions, 0.0, MAXIMUM_ADJUSTMENT_FRACTION
    )

    evaluate_business_metrics(
        test_dataframe, test_targets, baseline_predictions, model_tag="BaselineLinear"
    )


# -----------------------------
# Main Training Pipeline
# -----------------------------
def main():
    """Main training pipeline with cross-validation and evaluation."""
    set_random_seed(RANDOM_SEED)

    # Parse command line arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--csv", default=DEFAULT_CSV_PATH)
    argument_parser.add_argument("--device", default="cpu")
    argument_parser.add_argument("--max-epochs", type=int, default=MAXIMUM_EPOCHS)
    argument_parser.add_argument(
        "--patience", type=int, default=EARLY_STOPPING_PATIENCE
    )
    argument_parser.add_argument("--kfolds", type=int, default=CROSS_VALIDATION_FOLDS)
    arguments = argument_parser.parse_args()

    # Track total training time
    run_start = time.time()

    # Load and prepare data
    raw_dataframe = pd.read_csv(
        arguments.csv, dtype={"quote_out_token": str, "actual_out_token": str}
    )
    processed_dataframe, feature_matrix, target_values, feature_metadata = (
        build_feature_matrix(raw_dataframe)
    )

    # Split data into train/validation and test sets
    (
        train_val_features,
        test_features,
        train_val_targets,
        test_targets,
        train_val_dataframe,
        test_dataframe,
    ) = train_test_split(
        feature_matrix,
        target_values,
        processed_dataframe,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    # Perform cross-validation to select best model - NOW USING CLI FLAGS
    (
        training_histories,
        best_fold_index,
        (best_validation_loss, best_model_state, best_feature_standardizer),
    ) = perform_cross_validation_training(
        train_val_features,
        train_val_targets,
        num_folds=arguments.kfolds,
        device=arguments.device,
        max_epochs=arguments.max_epochs,
        patience=arguments.patience,
    )

    # Evaluate selected model on test set
    standardized_test_features = best_feature_standardizer.transform(
        test_features
    ).astype(np.float32)
    test_targets_scaled = test_targets / MAXIMUM_ADJUSTMENT_FRACTION
    test_dataset = TrainingDataset(standardized_test_features, test_targets_scaled)
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=False,
        num_workers=DATA_LOADER_WORKERS,
    )

    test_model = BoundedMultiLayerPerceptron(
        feature_matrix.shape[1], HIDDEN_LAYER_SIZES, DROPOUT_RATE
    ).to(arguments.device)
    test_model.load_state_dict(best_model_state)
    test_model.eval()  # CRITICAL: Set to eval mode to disable dropout
    loss_criterion = nn.L1Loss()

    # Collect test predictions
    test_loss_accumulator = 0.0
    total_test_samples = 0
    test_predictions_list = []

    with torch.no_grad():
        for feature_batch, target_batch in test_data_loader:
            feature_batch = feature_batch.to(arguments.device)
            target_batch = target_batch.to(arguments.device)

            pred_scaled = test_model(feature_batch)
            test_predictions_list.append(pred_scaled.cpu().numpy().reshape(-1))

            batch_loss = loss_criterion(
                pred_scaled, target_batch
            )  # Use pred_scaled directly
            test_loss_accumulator += float(batch_loss) * target_batch.size(0)
            total_test_samples += target_batch.size(0)

    # Convert predictions back to original scale for evaluation
    final_test_predictions_scaled = np.concatenate(test_predictions_list, axis=0)
    final_test_predictions = final_test_predictions_scaled * MAXIMUM_ADJUSTMENT_FRACTION
    final_test_loss = test_loss_accumulator / max(total_test_samples, 1)

    print(f"[TEST] L1Loss={final_test_loss:.6f}")

    # Evaluate baseline model for comparison
    standardized_train_val_features = best_feature_standardizer.transform(
        train_val_features
    ).astype(np.float32)
    evaluate_linear_regression_baseline(
        standardized_train_val_features,
        train_val_targets,
        standardized_test_features,
        test_targets,
        test_dataframe,
    )

    # Evaluate neural network with business metrics
    evaluate_business_metrics(
        test_dataframe, test_targets, final_test_predictions, model_tag="MLP"
    )

    # Save model artifacts
    os.makedirs(WEIGHTS_DIRECTORY, exist_ok=True)
    torch.save(best_model_state, WEIGHTS_FILE_PATH)

    metadata_output = {
        **feature_metadata,
        "scaler": best_feature_standardizer.to_dictionary(),
        "loss": "L1Loss",
        "hidden_sizes": HIDDEN_LAYER_SIZES,
        "dropout": DROPOUT_RATE,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY_FACTOR,
        "kfolds": arguments.kfolds,
        "max_epochs": arguments.max_epochs,
        "patience": arguments.patience,
        "seed": RANDOM_SEED,
        "test_loss_l1": float(final_test_loss),
        "bounded_output_max": MAXIMUM_ADJUSTMENT_FRACTION,
        "target_normalization": True,
        "target_norm_divisor": MAXIMUM_ADJUSTMENT_FRACTION,
        "interactions_added": True,
        "weights_path": WEIGHTS_FILE_PATH,
    }

    with open(METADATA_FILE_PATH, "w") as metadata_file:
        json.dump(metadata_output, metadata_file, indent=2)

    # Generate and save learning curve plot
    learning_curve_path = os.path.join(ARITFACTS_DIRECTORY, "learning_curve.png")
    plot_training_learning_curve(
        training_histories, best_fold_index, learning_curve_path
    )

    print(f"Saved weights -> {WEIGHTS_FILE_PATH}")
    print(f"Saved meta    -> {METADATA_FILE_PATH}")
    print(f"Saved curve   -> {learning_curve_path}")
    print(f"Total training time: {time.time() - run_start:.2f}s")


if __name__ == "__main__":
    main()
