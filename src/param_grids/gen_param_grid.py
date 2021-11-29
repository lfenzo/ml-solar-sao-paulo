"""
Generate Hyperparemeter Search Space (Params Grids)

The grids are generate from python dicts, serialized into .pkl files
and loaded at the moment of grid-search in '../scikit_gym.py'
"""

import os
import pickle

if __name__ == '__main__':

    # default values for each h-param are conventionaly placed as fist
    # values in each h-param

    param_grids = {

        # Multi-Layer Perceptron Regressor
        'mlp': {
            'hidden_layer_sizes': [
                (100, ), (200, ),
                (100, 100), (150, 150),
                (50, 50, 20), (30, 30, 15),
            ],
            'learning_rate': ['constant', 'adaptative'],
            'solver': ['adam', 'sgd', 'lbfgs'],
            'activation': ['logistic', 'relu'],
        },

        # Random Forest Regressor
        'rf': {
            'criterion': ['mse'],
            'min_samples_split': [2, 20, 100, 250, 500],
            'min_samples_leaf': [1, 10, 50, 150, 500],
            'max_features': ['auto', 'sqrt'],
            'n_estimators': [100, 150, 200, 400],
        },

        # Extra Trees Regressor
        'extr': {
            'criterion': ['mse'],
            'min_samples_split': [2, 20, 100, 250, 500],
            'min_samples_leaf': [1, 10, 50, 150, 500],
            'max_features': ['auto', 'sqrt'],
            'n_estimators': [100, 150, 200, 400],
        },

        # Extreme Gradient Boosting Regressor
        'xgb': {
            'eta': [0.3, 0.1, 0.15, 0.35],
            'gamma': [0, 0.05, 0.01],
            'max_depth': [6, 8, 10, 12],
            'sampling_method': ['uniform', 'gradient_based']
        },

        # Support Vector Regressor
        'svr': {
            'epsilon': [0.1, 0.15, 0.2, 0.4],
            'gamma': ['scale', 'auto'],
            'C': [1.0, 2.0, 5.0]
        },
    }

    current_directory = os.path.basename(os.getcwd())

    # when running from '../training_pipeline.py' one must change the cwd
    if not current_directory == 'param_grids':
        os.chdir('param_grids')

    # serializes each model h-param search space inside the current directory
    for alg, param_grid in param_grids.items():
        with open(f'./{alg}_param_grid.pkl', 'wb') as file:
            pickle.dump(param_grid, file)

    os.chdir('..')
