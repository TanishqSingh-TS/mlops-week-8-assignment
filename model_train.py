Project_Id = "keen-phalanx-473718-p1"
Location = "us-central1"  
Bucket_URI = "gs://mlops-course-keen-phalanx-473718-p1"

import os
import pandas as pd
import numpy as np  # Added for random choices and noise
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
import joblib
from zoneinfo import ZoneInfo
from sklearn import metrics
import mlflow
import requests
import sys

# --- Data Poisoning Function (Labels) ---
def poison_labels(y_train_df, poison_percent):
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
        
    print(f"Successfully poisoned {poisoned_count} labels ({poison_percent}%)")
    return y_poisoned

# --- NEW: Data Poisoning Function (Features) ---
def poison_features(X_train_df, poison_percent):
    """
    Adds extreme outlier noise to a percentage of the features.
    Returns a copy of the feature data.
    """
    print(f"\nPoisoning {poison_percent}% of features...")
    X_poisoned = X_train_df.copy()
    num_to_poison = int(len(X_poisoned) * (poison_percent / 100.0))
    
    # Select random indices to poison, without replacement
    poison_indices = np.random.choice(X_poisoned.index, num_to_poison, replace=False)
    
    # Select a random feature to poison for this attack
    # We'll poison 'sepal_length' and 'sepal_width' to make them outliers
    features_to_poison = ['sepal_length', 'sepal_width']
    
    for idx in poison_indices:
        for feature in features_to_poison:
            # Add significant noise - e.g., multiply by 5 to 10
            noise_factor = np.random.uniform(5, 10)
            X_poisoned.loc[idx, feature] *= noise_factor
            
    print(f"Successfully poisoned features for {len(poison_indices)} samples.")
    return X_poisoned

# --- MLflow Setup ---
mlflow.set_tracking_uri("http://34.46.30.185:5000")
mlflow.sklearn.autolog(
    max_tuning_runs=15,
    registered_model_name="poisoned-iris-classifier",
    log_models=True  # Ensure models are logged
)

# --- Load and Split Data ---
try:
    data = pd.read_csv("./data/iris.csv")
except FileNotFoundError:
    print("Error: './data/iris.csv' not found.")
    print("Please ensure your DVC data is pulled (e.g., run 'dvc pull').")
    sys.exit(1)

train, test = train_test_split(data, test_size=0.2, stratify=data['species'], random_state=55)

X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train_original = train.species  # Keep the original labels safe
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

# ====================================================================
# EXPERIMENT 1: LABEL POISONING
# ====================================================================
print("============================================")
print(" STARTING EXPERIMENT 1: LABEL POISONING ")
print("============================================")

poison_levels = [0, 5, 10, 15, 25, 50, 75]

for percent in poison_levels:
    run_name = f"DecisionTree_LabelPoison_{percent}pct"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n--- Starting Run: {run_name} (Run ID: {run.info.run_id}) ---")
        
        # 1. Create poisoned training labels
        y_train_poisoned = poison_labels(y_train_original, percent)
        
        # 2. Set MLflow tags
        mlflow.set_tag("poison_percent_tag", f"{percent}%")
        mlflow.set_tag("poison_type", "label") # NEW TAG
        mlflow.set_tag("model_type", "DecisionTreeClassifier")

        # 3. Log the poison percent as a metric
        mlflow.log_metric("poison_percent", percent)

        # 4. Train the model
        model = DecisionTreeClassifier(random_state=1)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
        
        # Fit on original features and *poisoned labels*
        grid_search.fit(X_train, y_train_poisoned) 

        # 5. Evaluate on the *clean* test data
        test_score = grid_search.score(X_test, y_test)
        
        # 6. Manually log key metrics
        mlflow.log_metric("test_accuracy", test_score)
        mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)
        mlflow.log_params(grid_search.best_params_)

        print(f"Run {run_name} finished.")
        print(f"Best CV Score (on {percent}% poisoned labels): {grid_search.best_score_:.4f}")
        print(f"Test Score (on clean test data): {test_score:.3f}")

print("\n--- Label Poisoning experiment complete. ---")


# ====================================================================
# NEW EXPERIMENT 2: FEATURE POISONING
# ====================================================================
print("\n============================================")
print(" STARTING EXPERIMENT 2: FEATURE POISONING ")
print("============================================")

feature_poison_percent = 25
run_name = f"DecisionTree_FeaturePoison_{feature_poison_percent}pct"

with mlflow.start_run(run_name=run_name) as run:
    print(f"\n--- Starting Run: {run_name} (Run ID: {run.info.run_id}) ---")
    
    # 1. Create poisoned training features
    X_train_poisoned = poison_features(X_train, feature_poison_percent)
    
    # 2. Set MLflow tags
    mlflow.set_tag("poison_percent_tag", f"{feature_poison_percent}%")
    mlflow.set_tag("poison_type", "feature") # NEW TAG
    mlflow.set_tag("model_type", "DecisionTreeClassifier")

    # 3. Log the poison percent as a metric
    mlflow.log_metric("poison_percent", feature_poison_percent)

    # 4. Train the model
    model = DecisionTreeClassifier(random_state=1)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
    
    # Fit on *poisoned features* and *original labels*
    grid_search.fit(X_train_poisoned, y_train_original) 

    # 5. Evaluate on the *clean* test data
    test_score = grid_search.score(X_test, y_test)
    
    # 6. Manually log key metrics
    mlflow.log_metric("test_accuracy", test_score)
    mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)
    mlflow.log_params(grid_search.best_params_)

    print(f"Run {run_name} finished.")
    print(f"Best CV Score (on {feature_poison_percent}% poisoned features): {grid_search.best_score_:.4f}")
    print(f"Test Score (on clean test data): {test_score:.3f}")

print("\n--- All training runs complete. ---")
print(f"Check the MLflow UI at http://3.239.123.11:5000 to compare all runs.")