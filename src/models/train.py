import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.lightgbm

# Registry names
LOGREG_REGISTRY_NAME = "twitter-sentiment-logreg"
LGBM_REGISTRY_NAME = "twitter-sentiment-lgbm"

def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test, model_output_path, registry_name):
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)

    print(f"{model_name} Train Accuracy: {train_acc:.4f}")
    print(f"{model_name} Test Accuracy: {test_acc:.4f}")
    print(f"{model_name} Test F1 Score: {test_f1:.4f}")

    # Log with MLflow
    mlflow.set_experiment("Twitter Sentiment Training")
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_score", test_f1)

        # Log and register model
        if model_name.lower() == "logisticregression":
            mlflow.sklearn.log_model(
                model, 
                artifact_path="logreg_model",
                registered_model_name=registry_name,
                input_example=X_train[:5]
            )
        elif model_name.lower() == "lightgbm":
            mlflow.lightgbm.log_model(
                model, 
                artifact_path="lgbm_model",
                registered_model_name=registry_name,
                input_example=X_train[:5]
            )

    # Save model locally too
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"{model_name} saved to {model_output_path}")


def main():
    features_path = "data/processed/features.pkl"
    labels_path = "data/processed/labels.pkl"
    
    # Load data
    with open(features_path, 'rb') as f:
        X = pickle.load(f)
    with open(labels_path, 'rb') as f:
        y = pickle.load(f)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # LogisticRegression + LightGBM
    models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=42), LOGREG_REGISTRY_NAME),
        "LightGBM": (lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=-1,
            random_state=42
        ), LGBM_REGISTRY_NAME)
    }
    
    # Train and log each model
    for model_name, (model, registry_name) in models.items():
        model_output_path = f"models/{model_name.lower()}.pkl"
        train_and_log_model(model_name, model, X_train, X_test, y_train, y_test, model_output_path, registry_name)


if __name__ == "__main__":
    main()