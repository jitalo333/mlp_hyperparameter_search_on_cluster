# mlp_hyperparameter_search_on_cluster
EEG-based emotion classification using Scattering Transform and MLP, with hyperparameter tuning via Optuna on a GPU cluster.

This code uses Kymatio to extract the Scattering Transform (ST) from EEG signals, with the goal of training a Multilayer Perceptron (MLP) emotion classifier across three categories: negative, positive, and neutral. Hyperparameter selection is performed using a Bayesian optimization algorithm with Optuna. Both the ST extraction and the MLP training are executed on a GPU cluster.

