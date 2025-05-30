# Base Random Forest Model Specification

## Model Overview
- **Algorithm**: Random Forest Classifier
- **Implementation**: scikit-learn RandomForestClassifier
- **Purpose**: Human Activity Recognition (HAR) using MotionSense dataset

## Model Configuration
### Base Parameters
```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_features='sqrt'
    # All other parameters use sklearn defaults
)
```

### Default sklearn Parameters (Implicit)
- `criterion='gini'`: The function to measure the quality of a split
- `max_depth=None`: Nodes are expanded until all leaves are pure
- `min_samples_split=2`: Minimum samples required to split an internal node
- `min_samples_leaf=1`: Minimum samples required to be at a leaf node
- `min_weight_fraction_leaf=0.0`: Minimum weighted fraction of the sum total of weights
- `max_leaf_nodes=None`: Grow trees with unlimited number of leaf nodes
- `min_impurity_decrease=0.0`: A node will be split if this split induces a decrease in impurity greater than or equal to this value
- `bootstrap=True`: Use bootstrap samples when building trees
- `oob_score=False`: Whether to use out-of-bag samples to estimate generalization score
- `n_jobs=None`: The number of jobs to run in parallel
- `verbose=0`: Controls the verbosity when fitting and predicting
- `warm_start=False`: When set to True, reuse the solution of the previous call
- `class_weight=None`: Weights associated with classes
- `ccp_alpha=0.0`: Complexity parameter used for Minimal Cost-Complexity Pruning
- `max_samples=None`: Draw n_samples samples for each base estimator

## Data Processing
### Input Features
- Source: Accelerometer and Gyroscope data
- Window Size: 100 samples
- Stride: 50 samples (50% overlap)

### Feature Extraction
- **Statistical Features**:
  - Mean
  - Standard deviation
  - Min/Max values
  - Root Mean Square (RMS)
  - Signal Magnitude Area
- **Spectral Features**:
  - Fast Fourier Transform (FFT) based features

### Data Normalization
- Per-subject normalization
- Applied independently to each sensor axis

## Training Configuration
### Data Split
- Test Size: 20% of subjects (group-based split)
- Training Size: 80% of subjects
- Split Method: GroupShuffleSplit
- Random State: 42

### Cross-Validation
- Method: Leave-One-Subject-Out (LOSO)
- Evaluation Metrics:
  - Accuracy
  - Macro F1-Score
  - Per-class precision, recall, and F1-score
  - Confusion Matrix

## Output
### Model Artifacts
- Trained model state
- Confusion matrices (both CV and test set)
- Performance metric plots
- Classification reports

### Results Storage
Directory Structure:
```
results/base/
├── confusion_matrix(cross-validation).png
├── confusion_matrix(test_set).png
├── performance_metrics(cross-validation).png
└── results.txt
```

## Performance Monitoring
### Metrics Tracked
- Cross-validation metrics:
  - Mean CV Accuracy
  - Mean CV Macro F1-score
  - Individual fold accuracies
  - Individual fold F1-scores
- Test set metrics:
  - Final accuracy
  - Final macro F1-score
  - Per-class metrics
  - Confusion matrix

### Visualization
- Confusion Matrix:
  - Normalized values
  - Raw counts as annotations
  - Blues colormap
- Performance Distribution:
  - Box plots for accuracy
  - Box plots for F1-scores

## Dependencies
- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- scipy>=1.7.0
- seaborn>=0.12.0
- matplotlib>=3.7.0

## Usage
```bash
python src/main.py --results_subdir base --test_size 0.2
``` 