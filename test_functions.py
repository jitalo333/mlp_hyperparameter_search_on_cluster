from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import optuna
import torch
import pickle
import os
from optuna_MLP import MLP, optuna_objective



# ----------- Dataset simulado -----------
X, y = make_classification(n_samples=500, n_features=20, n_classes=3,
                           n_informative=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ----------- Lanzar la optimización -----------
study = optuna.create_study(direction="maximize")
opt_model = optuna_objective(X_train, X_test, y_train, y_test)
del X_train, X_test, y_train, y_test

study.optimize(opt_model.objective, n_trials=3)

# ----------- Mostrar mejores resultados -----------
print("Mejor accuracy:", study.best_value)
print("Mejores hiperparámetros:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

"""""
#load best model
results_best_model = opt_model.get_results()

#Save dictionary
path = '/content/drive/MyDrive/Tesis_code/Adultos/wavelet_scattering'
filepath = os.path.join(path, 'results_best_model.pkl')
with open(filepath, 'wb') as f:
    pickle.dump(results_best_model, f)

metrics = {k: results_best_model[k] for k in ['accuracy', 'precision', 'recall', 'f1_score', 'epoch_number', 'best_params' ]}
print(metrics)

#Load model
path = '/content/drive/MyDrive/Tesis_code/Adultos/wavelet_scattering'
filepath = os.path.join(path, 'results_best_model.pkl')

with open(filepath, 'rb') as f:
    results_best_model = pickle.load(f)

"""