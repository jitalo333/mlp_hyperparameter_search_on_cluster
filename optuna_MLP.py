import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import optuna
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ----------- Modelo MLP -----------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout, n_classes):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class optuna_objective:
    def __init__(self):
        self.results = {}
    # ----------- Método objetivo para Optuna -----------
    def objective(self, trial):
        global X_train, X_test, y_train, y_test

        # ----------- Hiperparámetros a optimizar -----------
        n_layers = trial.suggest_int("n_layers", 1, 4)
        #hidden_sizes = []
        #max_units = 528
        #for i in range(n_layers):
        #    units = trial.suggest_int(f"n_units_l{i}", 16, max_units, step=64)
        #    hidden_sizes.append(units)
        #    max_units = units  # forzar disminución
        hidden_sizes = [trial.suggest_int(f"n_units_l{i}", 2, 514, step=64) for i in range(n_layers)]
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 5e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        scaler = trial.suggest_categorical("scaler", ['None', 'standard', 'minmax'])

        # ----------- Escalado de datos -----------
        if scaler == 'standard':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif scaler == 'minmax':
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # ----------- Datasets y Dataloaders -----------
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        n_features = train_dataset[0][0].shape[0]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # ----------- Modelo y entrenamiento -----------
        model = MLP(input_dim=n_features, hidden_sizes=hidden_sizes, dropout=dropout, n_classes=3)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # ----------- Early stopping params -----------
        max_epochs = 200
        patience = 10
        min_delta = 1e-4
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(max_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                output = model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()

            # ----------- Validación -----------
            model.eval()
            correct, total_samples, val_loss = 0, 0, 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    output = model(xb)
                    loss = criterion(output, yb)
                    val_loss += loss.item() * xb.size(0)
                    pred = output.argmax(dim=1)
                    correct += (pred == yb).sum().item()
                    total_samples += xb.size(0)

            avg_val_loss = val_loss / total_samples
            acc = correct / total_samples

            # ----------- Optuna pruning -----------
            trial.report(acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # ----------- Early stopping manual -----------
            """
            if avg_val_loss + min_delta < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
           """

        # ----------- Guardar mejor modelo del trial -----------
        if trial.number == 0 or acc > trial.study.best_value:
            self.eval_model(
                model_dict=model.state_dict(),
                X_test=X_test,
                y_test=y_test,
                best_params=trial.params,
                epoch_number=epoch
            )

        return acc

    # ----------- Método para evaluar el modelo con los mejores hiperparámetros -----------
    def eval_model(self, model_dict, X_test, y_test, best_params, epoch_number, model_class=MLP, device=None):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reconstruir el modelo con los mejores hiperparámetros
        input_dim = X_test.shape[1]
        hidden_sizes = [best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])]
        dropout = best_params["dropout"]
        n_classes = len(set(y_test))  # o usa 3 si lo sabes de antemano

        model = model_class(input_dim=input_dim,
                            hidden_sizes=hidden_sizes,
                            dropout=dropout,
                            n_classes=n_classes)

        # Cargar los pesos
        model.load_state_dict(model_dict)
        model.to(device)
        model.eval()

        # Convertir a tensores si vienen como arrays
        if isinstance(X_test, np.ndarray):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        if isinstance(y_test, np.ndarray):
            y_test = torch.tensor(y_test, dtype=torch.long)

        X_test = X_test.to(device)
        y_test = y_test.to(device)

        with torch.no_grad():
            outputs = model(X_test)
            preds = torch.argmax(outputs, dim=1)

        y_true = y_test.cpu().numpy()
        y_pred = preds.cpu().numpy()

        model_cpu = model.to('cpu')

        self.results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'cm': confusion_matrix(y_true, y_pred),
            'best_params': best_params,
            'model_state_dict': model_cpu.state_dict(),
            'epoch_number': epoch_number
        }

    # ----------- Método para obtener los resultados -----------
    def get_results(self):
      return self.results


 