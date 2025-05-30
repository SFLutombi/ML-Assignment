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

## Successive Halving Grid Search (results/halving_grid/)
- **Approach**:
  - Using HalvingGridSearchCV for efficient hyperparameter optimization
  - Implements successive halving algorithm to efficiently allocate resources
  - Optimizing parameters:
    * n_estimators: [50, 100, 200, 300, 400, 500]
    * max_depth: [5, 10, 20, 30, 40, 50]
    * min_samples_split: [2, 5, 10, 15, 20]
    * min_samples_leaf: [1, 2, 4, 6, 8, 10]
    * max_features: ['sqrt', 'log2']

- **Rationale**:
  - More efficient than traditional grid search
  - Uses successive halving to quickly eliminate poor performing configurations
  - Allocates more resources to promising parameter combinations
  - Particularly effective for large parameter spaces

- **Key Features**:
  - Factor=3: Reduces candidates by a factor of 3 at each iteration
  - Aggressive elimination: Quickly removes poor performing configurations
  - 5-fold cross-validation for robust evaluation
  - Parallel processing using all available cores

- **Results**:
  - Results can be found in `results/halving_grid/`
  - [To be updated after running]

- **Analysis**:
  - [To be updated after running]

Note: The successive halving approach provides a good balance between exploration of the parameter space and computational efficiency. It is particularly effective when dealing with large parameter spaces where traditional grid search would be computationally prohibitive. 