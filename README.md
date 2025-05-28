# Human Activity Recognition (HAR) using MotionSense Dataset

This project implements a machine learning classifier for human activity recognition using the MotionSense dataset. The model uses both accelerometer and gyroscope data to classify six different activities: walking downstairs, walking upstairs, sitting, standing, walking, and jogging.

## Features

- Data preprocessing and normalization
- Feature extraction (statistical and spectral features)
- Random Forest classifier
- Leave-One-Subject-Out Cross-Validation (LOSO-CV)
- Performance visualization (confusion matrix and feature importance)

## Technical Implementation Details

### Data Preprocessing
1. **Data Loading**
   - Loads accelerometer and gyroscope data from separate ZIP files
   - Merges sensor data based on timestamp and subject ID
   - Handles data from multiple subjects and activities

2. **Normalization**
   - Applies z-score normalization (μ=0, σ=1) per subject
   - Normalizes each sensor axis independently
   - Formula: `(x - μ)/σ` where μ and σ are calculated per subject
   - This helps account for individual differences in sensor readings

### Feature Extraction
1. **Windowing**
   - Window size: 100 samples (2 seconds of data)
   - Stride: 50 samples (50% overlap between windows)
   - Each window becomes one training instance

2. **Statistical Features** (computed for each axis: x, y, z of both sensors)
   - Mean
   - Standard deviation
   - Minimum value
   - Maximum value
   - Root Mean Square (RMS)
   - Energy (sum of squares)
   - Signal Magnitude Area (SMA)

3. **Spectral Features**
   - FFT peak magnitude
   - FFT dominant frequency bin
   - Helps capture periodic patterns in activities

### Model Architecture
- **Classifier**: Random Forest
  - Number of trees: 100
  - Max depth: None (trees grow until leaves are pure)
  - Uses all CPU cores for parallel training
  - Provides feature importance rankings

### Validation Strategy
1. **Leave-One-Subject-Out Cross-Validation (LOSO-CV)**
   - Training: Uses data from all subjects except one
   - Testing: Evaluates on the held-out subject
   - Repeats for each subject
   - Better estimates real-world performance
   - Prevents data leakage between subjects

2. **Performance Metrics**
   - Per-class precision, recall, and F1-score
   - Confusion matrix for detailed error analysis
   - Mean cross-validation accuracy across all subjects
   - Feature importance visualization

## Project Structure

```
ML-Assignment/
├── data/                      # Data directory
├── src/
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── feature_extractor.py  # Feature extraction functions
│   ├── model.py              # Model definition and training
│   └── main.py              # Main execution script
├── results/                  # Output directory for plots
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Make sure the MotionSense dataset is in the `motion-sense` directory.

## Running the Code

To train and evaluate the model:

```bash
python src/main.py
```

This will:
1. Load and preprocess the data
2. Extract features from the sensor data
3. Train and evaluate the model using LOSO-CV
4. Generate performance plots in the `results` directory

## Results

The results will be saved in the `results` directory:
- `confusion_matrix.png`: Confusion matrix showing classification performance
- `feature_importance.png`: Bar plot of the most important features

The console will display:
- Mean cross-validation accuracy
- Detailed classification report with precision, recall, and F1-score for each activity

## Implementation Notes

1. **Data Quality**
   - Missing values are handled during data loading
   - Timestamps are used to synchronize accelerometer and gyroscope data
   - Majority voting is used for window labels

2. **Performance Considerations**
   - Uses parallel processing for Random Forest training
   - Efficient numpy operations for feature extraction
   - Memory-efficient data loading with generators

3. **Extensibility**
   - Modular design allows easy addition of new features
   - Can be extended to support different classifiers
   - Configurable window size and stride 