import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix

FEATURE_NAMES = [
    "ctx-lh-inferiorparietal", "ctx-lh-inferiortemporal", "ctx-lh-isthmuscingulate", "ctx-lh-middletemporal",
    "ctx-lh-posteriorcingulate", "ctx-lh-precuneus", "ctx-rh-isthmuscingulate", "ctx-rh-posteriorcingulate",
    "ctx-rh-inferiorparietal", "ctx-rh-middletemporal", "ctx-rh-precuneus", "ctx-rh-inferiortemporal",
    "ctx-lh-entorhinal", "ctx-lh-supramarginal"
]

def load_data(data_dir):
    """Load and preprocess training and test datasets."""
    sMCI_train = pd.read_csv(f"{data_dir}/train.fdg_pet.sMCI.csv", names=FEATURE_NAMES)
    pMCI_train = pd.read_csv(f"{data_dir}/train.fdg_pet.pMCI.csv", names=FEATURE_NAMES)
    sMCI_test = pd.read_csv(f"{data_dir}/test.fdg_pet.sMCI.csv", names=FEATURE_NAMES)
    pMCI_test = pd.read_csv(f"{data_dir}/test.fdg_pet.pMCI.csv", names=FEATURE_NAMES)

    X_train = pd.concat([sMCI_train, pMCI_train], axis=0).values
    y_train = np.array([0] * len(sMCI_train) + [1] * len(pMCI_train))
    X_test = pd.concat([sMCI_test, pMCI_test], axis=0).values
    y_test = np.array([0] * len(sMCI_test) + [1] * len(pMCI_test))

    return X_train, y_train, X_test, y_test

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn / (tn + fp)

    print(f"{model_name} Evaluation:")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall (Sensitivity):", rec)
    print("Specificity:", spec)
    print("Balanced Accuracy:", bal_acc)

    return acc, prec, rec

def train_decision_tree(X_train, y_train, X_test, y_test):
    param_grid = {'criterion': ['gini', 'entropy', 'log_loss']}
    clf = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
    clf.fit(X_train, y_train)

    print("Best Decision Tree Criterion:", clf.best_params_)

    best_tree = clf.best_estimator_
    evaluate_model(best_tree, X_test, y_test, "Decision Tree")

    plt.figure(figsize=(20, 10))
    plot_tree(best_tree, filled=True, feature_names=FEATURE_NAMES, class_names=['sMCI', 'pMCI'])
    plt.show()

    return best_tree

def train_random_forest(X_train, y_train, X_test, y_test):
    param_grid = {'criterion': ['gini', 'entropy', 'log_loss'], 'n_estimators': [100, 200]}
    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    clf.fit(X_train, y_train)

    print("Best Random Forest Parameters:", clf.best_params_)

    best_rf = clf.best_estimator_
    evaluate_model(best_rf, X_test, y_test, "Random Forest")

    return best_rf


def predictMCIconverters(Xtest, data_dir):
    """
    Returns a vector of predictions with elements "0" for sMCI and "1" for pMCI,
    corresponding to each of the N_test feature vectors in Xtest.
    """

    # Load datasets
    train_sMCI = pd.read_csv(f"{data_dir}/train.fdg_pet.sMCI.csv", header=None)
    train_pMCI = pd.read_csv(f"{data_dir}/train.fdg_pet.pMCI.csv", header=None)
    test_sMCI = pd.read_csv(f"{data_dir}/test.fdg_pet.sMCI.csv", header=None)
    test_pMCI = pd.read_csv(f"{data_dir}/test.fdg_pet.pMCI.csv", header=None)

    # Combine training and test for richer training
    df_sMCI = pd.concat([train_sMCI, test_sMCI], ignore_index=True)
    df_pMCI = pd.concat([train_pMCI, test_pMCI], ignore_index=True)

    # Add labels
    df_sMCI["label"] = 0
    df_pMCI["label"] = 1

    # Combine datasets
    df = pd.concat([df_sMCI, df_pMCI], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split features and labels
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # Split train/validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Grid search for best Random Forest
    param_grid = {
        "n_estimators": [200],
        "max_depth": [20],
        "criterion": ["entropy"],
        'min_samples_split': [5],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt'],
        'class_weight': ['balanced']
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate on validation
    y_valid_pred = best_model.predict(X_valid)
    balanced_acc = balanced_accuracy_score(y_valid, y_valid_pred)

    # Retrain on full data
    best_model.fit(X, y)

    # Predict on new test set
    y_pred = best_model.predict(Xtest)

    return y_pred


if __name__ == "__main__":
    data_dir = "Data"
    X_train, y_train, X_test, y_test = load_data(data_dir)

    print("\nTraining Decision Tree...")
    train_decision_tree(X_train, y_train, X_test, y_test)

    print("\nTraining Random Forest...")
    train_random_forest(X_train, y_train, X_test, y_test)

    # test_sMCI = pd.read_csv("Data/test.fdg_pet.sMCI.csv", header=None)
    # test_pMCI = pd.read_csv("Data/test.fdg_pet.pMCI.csv", header=None)
    # Xtest = pd.concat([test_sMCI, test_pMCI], ignore_index=True).values
    # print("\nGenerating predictions with the best model...")
    # predictions = predictMCIconverters(X_test, data_dir)
    # print("Predictions:", predictions)
