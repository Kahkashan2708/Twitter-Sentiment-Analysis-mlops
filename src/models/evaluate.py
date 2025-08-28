import os
import json
import pickle
import mlflow
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Registry model names
LOGREG_REGISTRY_NAME = "twitter-sentiment-logreg"
LGBM_REGISTRY_NAME = "twitter-sentiment-lgbm"

# Paths to feature & label files
FEATURES_PATH = "data/processed/features.pkl"
LABELS_PATH = "data/processed/labels.pkl"

# Output metrics file
METRICS_PATH = "metrics.json"


def load_data(features_path, labels_path):
    with open(features_path, "rb") as f:
        X = pickle.load(f)
    with open(labels_path, "rb") as f:
        y = pickle.load(f)
    return X, y


def evaluate_model(model_name, model, X, y):
    preds = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1_score": f1_score(y, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y, preds).tolist()
    }
    print(f"Evaluation results for {model_name}:")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1 Score : {metrics['f1_score']:.4f}")
    return metrics


def plot_and_log_comparison(all_metrics):
    # Extract metrics for both models
    log_acc = all_metrics["logisticregression"]["accuracy"]
    lgb_acc = all_metrics["lightgbm"]["accuracy"]

    log_f1 = all_metrics["logisticregression"]["f1_score"]
    lgb_f1 = all_metrics["lightgbm"]["f1_score"]

    models = ["Logistic Regression", "LightGBM"]
    x = [log_acc, lgb_acc]
    y = [log_f1, lgb_f1]

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, color=["blue", "green"], s=120)

    for i, model in enumerate(models):
        plt.text(x[i] + 0.005, y[i], model, fontsize=10)

    plt.xlabel("Accuracy")
    plt.ylabel("F1 Score")
    plt.title("Model Comparison: Logistic Regression vs LightGBM")
    plt.grid(True)

    # Save & log with MLflow
    plot_path = "model_comparison.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    print(f"Scatter plot saved and logged as {plot_path}")


def main():
    # Load test data
    X, y = load_data(FEATURES_PATH, LABELS_PATH)

    all_metrics = {}

    mlflow.set_experiment("Twitter Sentiment Evaluation")
    with mlflow.start_run(run_name="evaluation"):
        # Load models from registry (latest version or stage)
        logreg = mlflow.pyfunc.load_model(f"models:/{LOGREG_REGISTRY_NAME}/latest")
        lgbm = mlflow.pyfunc.load_model(f"models:/{LGBM_REGISTRY_NAME}/latest")

        models = {
            "logisticregression": logreg,
            "lightgbm": lgbm
        }

        for model_name, model in models.items():
            # Evaluate
            metrics = evaluate_model(model_name, model, X, y)
            all_metrics[model_name] = metrics

            # Log metrics
            for metric_name, metric_value in metrics.items():
                if metric_name != "confusion_matrix":
                    mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)

        # Save metrics.json
        with open(METRICS_PATH, "w") as f:
            json.dump(all_metrics, f, indent=2)
        mlflow.log_artifact(METRICS_PATH)
        print(f"\nAll evaluation metrics saved to {METRICS_PATH}")

        # Plot comparison
        plot_and_log_comparison(all_metrics)


if __name__ == "__main__":
    main()
