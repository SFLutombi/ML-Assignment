# Human Activity Recognition Model Tuning Documentation

## Base Model (Random Forest)
- **Configuration**:
  - n_estimators: 100
  - random_state: 42
  - All other parameters: default values

- **Key Characteristics**:
  - Basic Random Forest implementation
  - No parameter tuning
  - Serves as baseline for comparison

- **Results**:
  - Results can be found in `results/base/`
  - Baseline for comparing future improvements

## Bayesian Optimization Tuning (results/bayesian_opt/)
- **Approach**:
  - Using Bayesian Optimization to automatically find optimal hyperparameters
  - Optimizing parameters:
    * n_estimators (range: 50-500)
    * max_depth (range: 5-50)
    * min_samples_split (range: 2-20)
    * min_samples_leaf (range: 1-10)
    * max_features (options: 'sqrt', 'log2')

- **Rationale**:
  - Bayesian Optimization efficiently explores hyperparameter space
  - More systematic than manual tuning
  - Can find non-intuitive parameter combinations
  - Optimizes based on cross-validation performance

- **Results**:
  - Results can be found in `results/bayesian_opt/`
  - [To be updated after running]

- **Analysis**:
  - [To be updated after running]

Note: Using Bayesian Optimization allows us to systematically explore the hyperparameter space and find the optimal configuration based on model performance. 