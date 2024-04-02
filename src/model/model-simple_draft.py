# path and os packages and setup (if needed)


# external packages and apis
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.api._v2.keras import activations
#from scipy import ndimage
import numpy as np
from keras.models import load_model

# developed packages


# packages setup (if needed)


dataset = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = dataset.load_data()
#X_train.shape #60k imgs with 28px x 28px

# Labels
y_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'bag',
    9: 'Ankle boot'
}


# Model def and creation
# Model = Sequence(Input -> Processing -> Output)
X_train = X_train/float(255) # normalization
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape = (28, 28)), # input layer
        keras.layers.Dense(256, activation=tensorflow.nn.relu), # 256 = nunber of nodes and ReLU (non linear function) as the activation function  # processing layer (Hidden layers)
        keras.layers.Dropout(0.2), # turn 20% of nodes to "standby"
        #keras.layers.Dense(128, activation=tensorflow.nn.relu),
        #keras.layers.Dense(64, activation=tensorflow.nn.relu),
        keras.layers.Dense(10, activation=tensorflow.nn.softmax) # output layer
    ]
)

# Before we fit our model, we need to compile it
# adam is the most recommended optimizer for classifications w/ labels>2 and sperse categorical crossentropy for measure the loss
adam = keras.optimizers.Adam(learning_rate=0.002)
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss'),
    keras.callbacks.ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True)
]

model.compile(
    optimizer=adam,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'] # metric to measure the model performance
)
hist = model.fit(
    X_train, y_train,
    batch_size=480,
    epochs=5, # something like cvs in sklearn
    validation_split=.2, # % for model validation
    callbacks=callbacks
) # same as sklearn models syntax


# Model summary
model_summary = model.summary()
model_summary

# Weights
DenseLayer_Weights = model.layers[1].get_weights()[0]
DenseLayer_WeightsZeros = np.zeros(DenseLayer_Weights.shape)
DenseLayer_WeightsRandom = np.random.rand(DenseLayer_Weights.shape[0], DenseLayer_Weights.shape[1])

# Bias
DenseLayer_Bias = model.layers[1].get_weights()[1]
DenseLayer_BiasZeros = np.zeros(DenseLayer_Bias.shape)

# Testing different weights and bias
#model.layers[1].set_weights([DenseLayer_WeightsZeros, DenseLayer_Bias])
#model.layers[1].set_weights([DenseLayer_WeightsRandom, DenseLayer_Bias])
#model.layers[1].set_weights([DenseLayer_WeightsZeros, DenseLayer_BiasZeros])
#model.layers[1].set_weights([DenseLayer_WeightsRandom, DenseLayer_BiasZeros])

model.get_config()


# lets test our model performance on the test dataset
y_pred = model.predict(X_test)
print(f"\nPREDICTION SAMPLE\nPredicted label: {y_labels[np.argmax(y_pred[0])]}\nReal label: {y_labels[y_test[0]]}")

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Performance\nLoss: {loss}\nAccuracy: {accuracy}")

# Accuracy plot per evaluation epoch
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Epochs Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'])

# Loss plot per evaluation epoch
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Epochs Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'])


model.save('fmnist_model.keras')
#loaded_model = load_model('fminist_model.keras')