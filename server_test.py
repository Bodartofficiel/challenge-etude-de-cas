# Server setup (run on the designated tracking server)
import mlflow
import os
import numpy as np


# Client code (run on each team member's computer)
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


def use_mlflow_client(server_uri, experiment_name, run_name):
    """
    Example of how to use MLflow client to log and load models.
    Args:
        server_uri: URI of the MLflow tracking server (e.g., "http://138.195.51.144:5001")
        experiment_name: Name of the experiment in MLflow
        run_name: Name of the specific run
    """

    # Set the tracking URI to point to the shared server
    mlflow.set_tracking_uri(server_uri)

    # Set the experiment (creates it if it doesn't exist)
    mlflow.set_experiment(experiment_name)

    # Load dataset and split data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    # Start a new run
    with mlflow.start_run(run_name=run_name):
        # Train a simple RandomForest model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("random_state", model.random_state)
        mlflow.log_metric("accuracy", accuracy)

        # Log the model
        input_example = np.array([X_test[0]])  # Example input for the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            input_example=input_example,
            registered_model_name="RandomForestClassifier",  # Optional: Register the model
        )

        print(f"Model logged to MLflow with accuracy: {accuracy}")

    

def main():
    server_uri = "http://138.195.243.136:5001"
    experiment_name = "/dior-test"
    run_name = "run_test"
    use_mlflow_client(server_uri=server_uri, experiment_name=experiment_name, run_name=run_name)
    
if __name__=="__main__":
    main()