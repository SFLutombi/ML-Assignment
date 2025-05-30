import os
from pathlib import Path
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from model import HARModel
import matplotlib.pyplot as plt
import sys
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, GroupShuffleSplit
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import argparse
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

def setup_plotting_style():
    """Set up matplotlib style for better visualizations."""
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 12

def plot_confusion_matrix(y_true, y_pred, labels, save_path, title_suffix=""):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix {title_suffix}")
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path / f'confusion_matrix{title_suffix.lower().replace(" ", "_")}.png')
    plt.close()

def plot_performance_metrics(accuracies, f1_scores, save_path, title_suffix=""):
    """Plot performance metrics across folds."""
    plt.figure(figsize=(15, 6))
    
    # Plot accuracies
    plt.subplot(1, 2, 1)
    plt.boxplot(accuracies)
    plt.title(f'Accuracy Distribution {title_suffix}')
    plt.ylabel('Accuracy')
    
    # Plot F1 scores
    plt.subplot(1, 2, 2)
    plt.boxplot(f1_scores)
    plt.title(f'F1-Score Distribution {title_suffix}')
    plt.ylabel('Macro F1-Score')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path / f'performance_metrics{title_suffix.lower().replace(" ", "_")}.png')
    plt.close()

def save_results_to_file(results_dict, save_path):
    """Save numerical results to a text file."""
    with open(save_path / 'results.txt', 'w') as f:
        for metric, value in results_dict.items():
            f.write(f"{metric}:\n")
            f.write(f"{value}\n\n")

def get_model(model_type='rf', **kwargs):
    """Return a classifier based on type and kwargs."""
    if model_type == 'rf':
        if kwargs.get('results_subdir') == 'bayesian_opt':
            # Define the search space for Bayesian Optimization
            search_space = {
                'n_estimators': Integer(50, 500),
                'max_depth': Integer(5, 50),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
                'max_features': Categorical(['sqrt', 'log2'])
            }
            
            # Create base estimator
            base_rf = RandomForestClassifier(random_state=42)
            
            # Create BayesSearchCV object
            optimizer = BayesSearchCV(
                base_rf,
                search_space,
                n_iter=50,  # Number of optimization iterations
                cv=5,       # 5-fold CV for optimization
                n_jobs=-1,  # Use all available cores
                random_state=42
            )
            return optimizer
        elif kwargs.get('results_subdir') == 'halving_grid':
            # Define the parameter grid for HalvingGridSearchCV
            param_grid = {
                'n_estimators': [50, 100, 200, 300, 400, 500],
                'max_depth': [5, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10, 15, 20],
                'min_samples_leaf': [1, 2, 4, 6, 8, 10],
                'max_features': ['sqrt', 'log2']
            }
            
            # Create base estimator
            base_rf = RandomForestClassifier(random_state=42)
            
            # Create HalvingGridSearchCV object
            optimizer = HalvingGridSearchCV(
                base_rf,
                param_grid,
                cv=5,           # 5-fold CV
                factor=3,       # Reduction factor for iterations
                n_jobs=-1,      # Use all available cores
                random_state=42,
                aggressive_elimination=True
            )
            return optimizer
        else:
            # Base model with default parameters
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_features='sqrt'
            )
    # Add more models here as needed
    raise ValueError(f"Unknown model_type: {model_type}")

def evaluate_model(clf, X, y, groups, output_dir, title_suffix=""):
    """Evaluate model using LOSO-CV and return results."""
    logo = LeaveOneGroupOut()
    
    # Initialize metrics storage
    accuracies = []
    f1_scores = []
    all_y_true = []
    all_y_pred = []
    
    # Perform LOSO-CV
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Store metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='macro'))
        
        # Store predictions for confusion matrix
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
    
    # Calculate results
    mean_cv_accuracy = np.mean(accuracies)
    mean_cv_f1 = np.mean(f1_scores)
    
    # Generate plots
    labels = sorted(np.unique(y))
    plot_confusion_matrix(all_y_true, all_y_pred, labels, output_dir, title_suffix)
    plot_performance_metrics(accuracies, f1_scores, output_dir, title_suffix)
    
    # Prepare results dictionary
    results = {
        'Mean CV Accuracy': mean_cv_accuracy,
        'Mean CV F1-Score': mean_cv_f1,
        'Individual Fold Accuracies': accuracies,
        'Individual Fold F1-Scores': f1_scores,
        'Classification Report': classification_report(all_y_true, all_y_pred)
    }
    
    return results, clf

def main():
    parser = argparse.ArgumentParser(description='HAR Model Training and Evaluation')
    parser.add_argument('--results_subdir', type=str, default='base', help='Subfolder for results (e.g., base, bayesian_opt)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of subjects to hold out for testing')
    args = parser.parse_args()

    try:
        # Set up paths
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        data_dir = current_dir.parent / 'motion-sense' / 'data'
        output_dir = current_dir.parent / 'results' / args.results_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Human Activity Recognition Model Training ({args.results_subdir}) ===\n")

        print("1. Loading and preprocessing data...")
        # Load and preprocess data
        loader = DataLoader(data_dir)
        merged_data = loader.merge_sensor_data()
        normalized_data = loader.normalize_per_subject(merged_data)
        print(f"Data loaded successfully! Shape: {normalized_data.shape}")

        print("\n2. Extracting features...")
        # Extract features
        extractor = FeatureExtractor(window_size=100, stride=50)
        X, y, groups = extractor.extract_features(normalized_data)
        
        # Convert to numpy arrays if they aren't already
        X = np.array(X)
        y = np.array(y)
        groups = np.array(groups)
        
        print(f"Features extracted successfully! Shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of samples: {X.shape[0]}")
        print(f"Number of unique activities: {len(np.unique(y))}")
        print(f"Number of subjects: {len(np.unique(groups))}")

        # Split subjects into train and test
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        
        # Split the data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        print(f"\nSplit data into:")
        print(f"Training: {len(np.unique(groups_train))} subjects")
        print(f"Testing:  {len(np.unique(groups_test))} subjects")

        print("\n3. Training and evaluating model...")
        # Get model
        clf = get_model('rf', results_subdir=args.results_subdir)
        
        # Perform cross-validation on training data
        print("\nPerforming cross-validation on training data...")
        cv_results, trained_clf = evaluate_model(
            clf, X_train, y_train, groups_train, 
            output_dir, 
            title_suffix="(Cross-Validation)"
        )
        
        # If using Bayesian Optimization, get the best parameters
        if args.results_subdir == 'bayesian_opt':
            print("\nBest parameters found:")
            for param, value in trained_clf.best_params_.items():
                print(f"{param}: {value}")
            # Use the best estimator for final testing
            final_clf = trained_clf.best_estimator_
        elif args.results_subdir == 'halving_grid':
            # Use the best estimator from HalvingGridSearchCV
            final_clf = trained_clf.best_estimator_
        else:
            final_clf = trained_clf

        # Evaluate on held-out test set
        print("\nEvaluating on held-out test set...")
        final_clf.fit(X_train, y_train)  # Retrain on full training data
        y_pred = final_clf.predict(X_test)
        
        # Calculate test metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='macro')
        test_report = classification_report(y_test, y_pred)
        
        print(f"\nTest Set Results:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Macro F1-Score: {test_f1:.4f}")
        print("\nClassification Report:")
        print(test_report)
        
        # Generate test set plots
        labels = sorted(np.unique(y))
        plot_confusion_matrix(y_test, y_pred, labels, output_dir, title_suffix="(Test Set)")
        
        # Save all results
        results_dict = {
            'Cross-Validation Results': cv_results,
            'Test Set Results': {
                'Accuracy': test_accuracy,
                'Macro F1-Score': test_f1,
                'Classification Report': test_report
            }
        }
        
        # Add best parameters if using Bayesian Optimization or HalvingGridSearchCV
        if args.results_subdir == 'bayesian_opt' or args.results_subdir == 'halving_grid':
            results_dict['Best Parameters'] = trained_clf.best_params_
            results_dict['Optimization History'] = str(trained_clf.optimizer_results_[0])
            
        save_results_to_file(results_dict, output_dir)

        print(f"\nDone! Results have been saved to: {output_dir}")

    except Exception as e:
        print("\nError occurred during execution:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 