# Human Activity Recognition (HAR) using MotionSense Dataset

This project implements a machine learning pipeline for human activity recognition using the MotionSense dataset. The system uses accelerometer and gyroscope data to classify different activities like walking, jogging, sitting, etc.

## Project Structure

```
ML-Assignment/
│
├── src/                    # Source code
│   ├── main.py            # Main script for training and evaluation
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── feature_extractor.py # Feature extraction
│   └── model.py           # Model implementation
│
├── motion-sense/          # Dataset directory
│   └── data/             
│       ├── B_Accelerometer_data.zip
│       └── C_Gyroscope_data.zip
│
├── results/               # Results directory (created automatically)
│   ├── base/             # Results for base model
│   └── bayesian_opt/     # Results for optimized model
│
├── requirements.txt       # Project dependencies
└── tuning-documentation.md # Detailed documentation of model tuning process
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ML-Assignment
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the MotionSense dataset is in the correct location:
- Place `B_Accelerometer_data.zip` and `C_Gyroscope_data.zip` in the `motion-sense/data/` directory

## Model Training and Evaluation

The project implements a robust evaluation strategy using group holdout validation:

1. **Data Splitting**:
   - Subjects are randomly split into training (80%) and test (20%) sets
   - This ensures complete separation of subjects between training and testing
   - Test subjects are never seen during model development

2. **Training Phase**:
   - Cross-validation is performed on training subjects only
   - For base model: Leave-One-Subject-Out CV
   - For optimized model: Bayesian Optimization with internal CV

3. **Final Evaluation**:
   - Models are evaluated on the held-out test subjects
   - Provides unbiased estimate of real-world performance

### Running the Models

#### Base Model
```bash
python src/main.py --results_subdir base [--test_size 0.2]
```

#### Optimized Model (Bayesian Optimization)
```bash
python src/main.py --results_subdir bayesian_opt [--test_size 0.2]
```

#### Optimized Model (Successive Halving Grid Search)
```bash
python src/main.py --results_subdir halving_grid [--test_size 0.2]
```

Parameters:
- `--results_subdir`: Output directory for results (options: base, bayesian_opt, halving_grid)
- `--test_size`: Proportion of subjects to hold out (default: 0.2)

## Pipeline Components

### 1. Data Loading (`data_loader.py`)
- Loads accelerometer and gyroscope data from zip files
- Merges sensor data based on measurement IDs
- Applies per-subject normalization
- Handles data cleanup and temporary file management

### 2. Feature Extraction (`feature_extractor.py`)
- Implements sliding window approach
- Extracts statistical features:
  * Mean, standard deviation, min, max
  * RMS (Root Mean Square)
  * Signal Magnitude Area
- Extracts spectral features using FFT
- Handles data segmentation and labeling

### 3. Model Training and Evaluation (`main.py`)
- Implements group holdout validation
- Performs Leave-One-Subject-Out CV on training data
- Supports both base and optimized model training
- Generates comprehensive evaluation metrics:
  * Confusion matrix (for both CV and test set)
  * Classification report
  * Per-subject performance metrics
  * Cross-validation scores

## Results Organization

Each experiment's results are stored in a separate subdirectory under `results/`:

### Base Model (`results/base/`)
- Cross-validation results:
  * Confusion matrix
  * Performance metrics plots
  * Classification report
- Test set results:
  * Confusion matrix
  * Performance metrics
  * Classification report

### Optimized Model (`results/bayesian_opt/`)
- Same metrics as base model
- Additional information:
  * Best parameters found
  * Optimization history
  * Comparative performance metrics

### Optimized Model (`results/halving_grid/`)
- Same metrics as base model
- Additional information:
  * Best parameters found
  * Optimization history
  * Comparative performance metrics
  * Successive halving iterations summary

## Model Performance Metrics

The evaluation includes:
- Cross-validation metrics:
  * Mean CV Accuracy
  * Macro F1-score
  * Per-class metrics
- Test set metrics:
  * Final accuracy
  * Final F1-score
  * Confusion matrix
  * Detailed classification report

## Tuning Process Documentation

The model tuning process is documented in `tuning-documentation.md`, which includes:
- Base model configuration and performance
- Bayesian Optimization approach and rationale
- Detailed results and analysis
- Performance comparisons

