from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class HARModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1
        )
        self.logo = LeaveOneGroupOut()
        self.feature_importances_ = None

    def train_and_evaluate(self, X, y, groups):
        """Train and evaluate the model using LOSO-CV."""
        predictions = []
        true_labels = []
        scores = []

        # Perform Leave-One-Subject-Out Cross-Validation
        for train_idx, test_idx in self.logo.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train model
            self.model.fit(X_train, y_train)

            # Make predictions
            y_pred = self.model.predict(X_test)

            # Store results
            predictions.extend(y_pred)
            true_labels.extend(y_test)
            scores.append(np.mean(y_pred == y_test))

        # Calculate and store feature importances
        self.feature_importances_ = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return predictions, true_labels, scores

    def plot_confusion_matrix(self, y_true, y_pred, output_path=None):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['dws', 'ups', 'sit', 'std', 'wlk', 'jog'],
            yticklabels=['dws', 'ups', 'sit', 'std', 'wlk', 'jog']
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

    def plot_feature_importance(self, top_n=20, output_path=None):
        """Plot feature importance."""
        plt.figure(figsize=(12, 6))
        top_features = self.feature_importances_.head(top_n)
        
        sns.barplot(
            x='importance',
            y='feature',
            data=top_features
        )
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

    def print_classification_report(self, y_true, y_pred):
        """Print classification report."""
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred)) 