import numpy as np
from collections import Counter

def create_segment_label(value, N):
    ones = np.ones(N)
    labels = ones*value
    return labels

def generate_datasets(X_train_C, X_test_C, y_train_C, y_test_C):
  y_train = []
  y_test = []
  for idx, y in enumerate(y_train_C):
    N = X_train_C[idx].shape[0]
    y_train.append(create_segment_label(y, N))

  for idx, y in enumerate(y_test_C):
    N = X_test_C[idx].shape[0]
    y_test.append(create_segment_label(y, N))

  y_train = np.concatenate(y_train, axis=0)
  y_test = np.concatenate(y_test, axis=0)
  #print(y_train.shape, y_test.shape)

  X_train = np.vstack(X_train_C)
  X_test = np.vstack(X_test_C)
  #print(X_train.shape, X_test.shape)

  return X_train, X_test, y_train, y_test

def count_labels(y_tensor):
  # Cuenta cuántas veces aparece cada etiqueta
  labels = y_tensor.tolist()
  label_counts = Counter(labels)
  print(label_counts)

class results():
  def __init__(self, code_name, epochs, learning_rate, batch_size):
    self.parameters = {
        'code_name': code_name,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
    }

    self.best_metrics = {
    'epoch': None,
    'test_loss': None,
    'test_acc': 0,
    'precision': 0,
    'recall': None,
    'f1': 0,      # arranca muy bajo
    'confusion_df': None
    }

  def update_results(self, test_loss, test_acc, precision, recall, instant_f1, confusion_df, metric = 'f1'):
    #Update metrics considering best f1-score
    if instant_f1 > self.best_metrics[metric]:
      self.best_metrics = {
          'test_loss': test_loss,
          'test_acc': test_acc,
          'precision': precision,
          'recall': recall,
          'f1': instant_f1,
          'confusion_df': confusion_df
      }

  def get_parameters(self):
    return self.parameters

  def get_best_metrics(self):
    return self.best_metrics