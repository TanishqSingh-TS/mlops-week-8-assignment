Project_Id = "keen-phalanx-473718-p1"
Location = "us-central1"  
Bucket_URI = "gs://mlops-course-keen-phalanx-473718-p1"

import os
import pandas as pd
import numpy as np  # Added for random choices
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
import joblib
from zoneinfo import ZoneInfo
from sklearn import metrics
import mlflow
import requests
import sys

# --- Data Poisoning Function ---
def poison_data(y_train_df, poison_percent):
    """
    Flips the labels of a specified percentage of the training data.
    Returns a copy of the label data.
    """
    # Return an unchanged copy if poison_percent is 0
    if poison_percent == 0:
        print("Training on 0% poisoned (original) data.")
        return y_train_df.copy()

    y_poisoned = y_train_df.copy()
    num_to_poison = int(len(y_poisoned) * (poison_percent / 100.0))
    
    # Get all unique classes from the original labels
    classes = y_train_df.unique()
    
    # Select random indices to poison, without replacement
    poison_indices = np.random.choice(y_poisoned.index, num_to_poison, replace=False)
    
    poisoned_count = 0
    for idx in poison_indices:
        original_label = y_poisoned.loc[idx]
        
        # Find all classes *except* the original one
        possible_new_labels = [c for c in classes if c != original_label]
        
        # Randomly pick one of the *incorrect* labels
        if possible_new_labels: # Ensure there is more than one class
            new_label = np.random.choice(possible_new_labels)
            y_poisoned.loc[idx] = new_label
            poisoned_count += 1
        
    print(f"Successfully poisoned {poisoned_count} samples ({poison_percent}%)")
    return y_poisoned

# --- MLflow Setup ---
mlflow.set_tracking_uri("http://34.46.30.185:5000")
mlflow.sklearn.autolog(
    max_tuning_runs=15,
    registered_model_name="poisioned-iris-classifier",
    log_models=True  # Ensure models are logged
)

# --- Load and Split Data ---
# This assumes your DVC setup has placed 'iris.csv' in the './data/' directory
try:
    data = pd.read_csv("./data/iris.csv")
except FileNotFoundError:
    print("Error: './data/iris.csv' not found.")
    print("Please ensure your DVC data is pulled (e.g., run 'dvc pull').")
    sys.exit(1)

train, test = train_test_split(data, test_size=0.2, stratify=data['species'], random_state=55)

X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train_original = train.species 
X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = test.species

# --- Hyperparameter Grid ---
param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1],
    'class_weight': [None]
}

# --- Define Poisoning Levels ---
poison_levels = [0, 5, 10, 15, 25, 50, 75]

print("Starting training loops with data poisoning...")

# --- Main Training Loop ---
for percent in poison_levels:
    # Set a descriptive run name
    run_name = f"Poison_{percent}pct"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n--- Starting Run: {run_name} (Run ID: {run.info.run_id}) ---")
        
        # 1. Create poisoned training labels for this run
        y_train_poisoned = poison_data(y_train_original, percent)
        
        # 2. Set MLflow tags to identify the run
        mlflow.set_tag("poison_percent_tag", f"{percent}%")
        mlflow.set_tag("model_type", "DecisionTreeClassifier")

        # 3. Log the poison percent as a metric for easy graphing in MLflow UI
        mlflow.log_metric("poison_percent", percent)

        # 4. Train the model
        model = DecisionTreeClassifier(random_state=1)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
        
        # Fit on poisoned labels
        grid_search.fit(X_train, y_train_poisoned) 

        # 5. Evaluate the *best* model on the *clean* test data
        test_score = grid_search.score(X_test, y_test)
        
        # 6. Manually log key metrics (autolog will also get them, but this is explicit)
        mlflow.log_metric("test_accuracy", test_score)
        mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)
        mlflow.log_params(grid_search.best_params_)

        print(f"Run {run_name} finished.")
        print(f"Best Parameters for Model: {grid_search.best_params_}")
        print(f"Best Cross Validation Score (on {percent}% poisoned data): {grid_search.best_score_:.4f}")
        print(f"Test Score (on clean test data): {test_score:.3f}")

print("\n--- All training runs complete. ---")
print(f"Check the MLflow UI at http://34.45.167.124:5000 to compare all {len(poison_levels)} runs.")




    