import flwr as fl
import tensorflow as tf
import time
import sklearn
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow.keras as keras
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, BatchNormalization, Activation, Add, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

dataset=pd.read_csv("D:/FL_IDS/Processed_Combined_IoT_dataset.csv")
properties = list(dataset.columns.values)
properties.remove('label')
X = dataset[properties]
y = dataset['label']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))

y_train = enc.transform(y_train.values.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.values.reshape(-1, 1)).toarray()
y_true = np.argmax(y_test, axis=1)

if len(x_train.shape) == 2: 
        x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))
        input_shape = x_train.shape[1:]


# FCN 
input_layer = keras.layers.Input(input_shape)

# Fully connected layer 1
conv1 = keras.layers.Conv1D(filters=32, kernel_size=8, padding='same')(input_layer)
conv1 = keras.layers.BatchNormalization()(conv1)
conv1 = keras.layers.Activation(activation='relu')(conv1)

# Fully connected layer 2
conv2 = keras.layers.Conv1D(filters=64, kernel_size=5, padding='same')(conv1)
conv2 = keras.layers.BatchNormalization()(conv2)
conv2 = keras.layers.Activation('relu')(conv2)

# Fully connected layer 3
conv3 = keras.layers.Conv1D(32, kernel_size=3,padding='same')(conv2)
conv3 = keras.layers.BatchNormalization()(conv3)
conv3 = keras.layers.Activation('relu')(conv3)

# Gap lager
gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

# Output layer
output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

def precision(y_true, y_pred):
    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
    precision = tp / (tp + fp + 1e-16)
    return tf.reduce_mean(precision)

def recall(y_true, y_pred):
    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
    recall = tp / (tp + fn + 1e-16)
    return tf.reduce_mean(recall)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = 2 * prec * rec / (prec + rec + 1e-16)
    return tf.reduce_mean(f1)

model = keras.models.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01, decay=0.005), loss='categorical_crossentropy', metrics=['accuracy', precision, recall, f1_score])

NAME = "FCN on IoT Combined Dataset"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME), histogram_freq = 1, profile_batch = 5)

# Define Flower client
class IoTClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    return model.get_weights()

  def fit(self, parameters, config):
    model.set_weights(parameters)
    batch_size = 512
    mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))
    start_time1 = time.time()
    model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=5, validation_data=(x_test, y_test),callbacks=[tensorboard])
    duration1 = time.time() - start_time1
    return model.get_weights(), len(x_train), {}

  def evaluate(self,parameters, config):
    model.set_weights(parameters)
    start_time2 = time.time()
    re = model.evaluate(x_test, y_test)
    duration2 = time.time() - start_time2
    y_pred = np.argmax(model.predict(x_test), axis=1)
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)
    accuracy = re[1]  # get accuracy value
    precision = re[2]  # get precision value
    f1 = model.metrics[-1].result().numpy()  # get F1 score value
    return re[0], len(x_test), {"Accuracy": float(accuracy), "Precision": float(precision), "F1 score": float(f1)}
	
# Start Flower client
fl.client.start_numpy_client(server_address="10.31.0.26:8082", client=IoTClient(),)