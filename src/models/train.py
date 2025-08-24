import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import mlflow
import mlflow.sklearn

def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test, model_output_path):
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
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.sklearn.log_model(model, artifact_path="model")
    
    # Save model locally
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
    
    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }
    
    # Train and log each model
    for model_name, model in models.items():
        model_output_path = f"models/{model_name.lower()}.pkl"
        train_and_log_model(model_name, model, X_train, X_test, y_train, y_test, model_output_path)


if __name__ == "__main__":
    main()