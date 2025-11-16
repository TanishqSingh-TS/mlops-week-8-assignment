# MLOps Week 8 Assignment: Impact of Data Poisoning on Model Performance

This repository contains the files for an MLOps experiment focused on quantifying the impact of data poisoning on a model's performance. The pipeline is designed to automatically train multiple models with varying levels of "poisoned" (intentionally mislabeled) data and log all results to MLflow for comparison.

## Overview

The core goal of this project is to demonstrate and measure the "Garbage In, Garbage Out" principle. The experiment automatically:

1.  **Defines Poison Levels**: Sets a list of percentages (e.g., 0%, 5%, 10%... 75%) to represent the amount of data to poison.
2.  **Poisons Data**: For each level, it creates a new "poisoned" version of the training dataset by randomly flipping a percentage of the labels.
3.  **Trains & Evaluates**: It trains a `DecisionTreeClassifier` using `GridSearchCV` on each poisoned dataset.
4.  **Logs Results**: It logs all hyperparameters, metrics, and models from every run to an MLflow tracking server.
5.  **Compares Performance**: The final, clean test accuracy of each model is logged, allowing for a clear comparison of how performance degrades as the data quality gets worse.
 
## 📂 Repository Structure

Here's an overview of the key files and directories in this project:

* **`model_train.py`**: The main Python script that defines the poisoning function, sets up MLflow, loops through the poison levels, and runs the training and evaluation.
* **`data/iris.csv`**: The original, clean source dataset (managed by DVC).
* **`requirements.txt`**: Lists all Python dependencies for the project (e.g., `mlflow`, `scikit-learn`, `pandas`).
* **`README.md`**: This file, explaining the project's purpose and how to interpret the results.

## ⚙️ How It Works

The entire experiment is orchestrated by the `model_train.py` script:

1.  **Setup**: The script connects to the MLflow tracking server (e.g., `http://34.45.167.124:5000`) and enables `mlflow.sklearn.autolog()`.
2.  **Load Data**: The clean `iris.csv` is loaded and split into a training set and a clean, untouched test set.
3.  **Poisoning Loop**: The script iterates through the `poison_levels` list.
4.  **Inside the Loop (For each poison level):**
    * A new, named MLflow run is started (e.g., `DecisionTree_Poison_15pct`).
    * The `poison_data` function creates a poisoned copy of the training labels.
    * The run is **tagged** with the `poison_percent_tag` for easy filtering.
    * A `GridSearchCV` model is trained on the `X_train` features and the **poisoned** `y_train` labels.
    * The best model from the grid search is evaluated against the **clean** `X_test` and `y_test` data.
    * `mlflow.sklearn.autolog()` logs all `GridSearchCV` runs, and we manually log the final `test_accuracy` and `poison_percent` as metrics.

## 📈 How to Analyze the Results

The primary output of this experiment is the **MLflow UI**.

1.  Go to the MLflow server URL: `http://34.45.167.124:5000`.
2.  You will see all the runs (e.g., `DecisionTree_Poison_0pct`, `..._5pct`, `..._10pct`, etc.).
3.  **Select all** the runs from this experiment.
4.  Click the **"Compare"** button.
5.  **Generate Comparison Plots:**
    * Go to the "Parallel Coordinates Plot" to see how the best hyperparameters (like `max_depth`) change.
    * Go to the "Scatter Plot" and set:
        * **X-axis**: `poison_percent` (the metric we logged)
        * **Y-axis**: `test_accuracy`
