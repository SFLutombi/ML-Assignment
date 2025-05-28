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

The project implements two versions of the Random Forest classifier:

### 1. Base Model
- Basic implementation with default parameters
- Used as a baseline for performance comparison

To run the base model:
```bash
python src/main.py --results_subdir base
```

### 2. Optimized Model (Bayesian Optimization)
- Uses Bayesian Optimization to find optimal hyperparameters
- Optimizes the following parameters:
  * n_estimators (range: 50-500)
  * max_depth (range: 5-50)
  * min_samples_split (range: 2-20)
  * min_samples_leaf (range: 1-10)
  * max_features (options: 'sqrt', 'log2')

To run the optimized model:
```bash
python src/main.py --results_subdir bayesian_opt
```

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
- Implements Leave-One-Subject-Out Cross-Validation (LOSO-CV)
- Supports both base and optimized model training
- Generates comprehensive evaluation metrics:
  * Confusion matrix
  * Classification report
  * Per-subject performance metrics
  * Cross-validation scores

## Results Organization

Each experiment's results are stored in a separate subdirectory under `results/`:

### Base Model (`results/base/`)
- Confusion matrix visualization
- Performance metrics plots
- Classification report
- Detailed metrics in results.txt

### Optimized Model (`results/bayesian_opt/`)
- Same metrics as base model
- Additional information:
  * Best parameters found
  * Optimization history
  * Comparative performance metrics

## Model Performance Metrics

The evaluation includes:
- Mean CV Accuracy
- Macro F1-score
- Per-class precision, recall, and F1-scores
- Confusion matrix
- Performance distribution across subjects

## Tuning Process Documentation

The model tuning process is documented in `tuning-documentation.md`, which includes:
- Base model configuration and performance
- Bayesian Optimization approach and rationale
- Detailed results and analysis
- Performance comparisons

## Contributing

To contribute to the project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add your license information here] 