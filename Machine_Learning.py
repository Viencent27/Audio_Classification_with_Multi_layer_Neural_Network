import numpy as np
import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def create_model(input_shape, num_classes, num_hidden_layers, num_hidden_neurons, activation):
  """
  Membuat model neural network dengan jumlah hidden layer, jumlah neuron, dan fungsi aktivasi yang ditentukan.
  """
  model = Sequential()
  model.add(Dense(num_hidden_neurons, activation=activation, input_shape=input_shape))

  for _ in range(num_hidden_layers - 1):
    model.add(Dense(num_hidden_neurons, activation=activation))

  model.add(Dense(num_classes, activation='softmax'))
  return model

def train_model(model, X_train, y_train, lr, num_epochs):
  """
  Melatih model dengan data train menggunakan learning rate dan jumlah epochs yang ditentukan.
  """
  optimizer = SGD(learning_rate=lr)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, verbose=0)
  return model

def evaluate_model(model, X_test, y_test):
  """
  Mengevaluasi model menggunakan data test dan menghitung metrik evaluasi seperti akurasi, presisi, recall, dan F1-Score.
  """
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

  return accuracy, precision, recall, f1