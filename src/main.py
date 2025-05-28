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
from sklearn.model_selection import LeaveOneGroupOut

def setup_plotting_style():
    """Set up matplotlib style for better visualizations."""
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 12

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
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
    plt.title("Confusion Matrix (LOSO-CV)")
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrix.png')
    plt.close()

def plot_performance_metrics(accuracies, f1_scores, save_path):
    """Plot performance metrics across folds."""
    plt.figure(figsize=(15, 6))
    
    # Plot accuracies
    plt.subplot(1, 2, 1)
    plt.boxplot(accuracies)
    plt.title('Accuracy Distribution Across Subjects')
    plt.ylabel('Accuracy')
    
    # Plot F1 scores
    plt.subplot(1, 2, 2)
    plt.boxplot(f1_scores)
    plt.title('F1-Score Distribution Across Subjects')
    plt.ylabel('Macro F1-Score')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path / 'performance_metrics.png')
    plt.close()

def save_results_to_file(results_dict, save_path):
    """Save numerical results to a text file."""
    with open(save_path / 'results.txt', 'w') as f:
        for metric, value in results_dict.items():
            f.write(f"{metric}:\n")
            f.write(f"{value}\n\n")

def main():
    try:
        # Set up paths
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        data_dir = current_dir.parent / 'motion-sense' / 'data'
        output_dir = current_dir.parent / 'results'
        output_dir.mkdir(exist_ok=True)

        print("\n=== Human Activity Recognition Model Training ===\n")

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

        print("\n3. Training and evaluating model...")
        # Train and evaluate model
        logo = LeaveOneGroupOut()
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
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
        
        # Calculate and display results
        print("\n4. Results:")
        mean_cv_accuracy = np.mean(accuracies)
        mean_cv_f1 = np.mean(f1_scores)
        
        print(f"Mean CV Accuracy: {mean_cv_accuracy:.4f}")
        print(f"Individual fold scores: {[f'{score:.4f}' for score in accuracies]}\n")
        
        # Generate classification report
        print("Classification Report:")
        class_report = classification_report(all_y_true, all_y_pred)
        print(class_report)

        print("\n5. Generating plots...")
        # Generate and save plots
        labels = sorted(np.unique(y))
        
        # Plot confusion matrix
        plot_confusion_matrix(all_y_true, all_y_pred, labels, output_dir)
        
        # Plot performance metrics
        plot_performance_metrics(accuracies, f1_scores, output_dir)
        
        # Save numerical results
        results_dict = {
            'Mean CV Accuracy': mean_cv_accuracy,
            'Mean CV F1-Score': mean_cv_f1,
            'Individual Fold Accuracies': accuracies,
            'Individual Fold F1-Scores': f1_scores,
            'Classification Report': class_report
        }
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