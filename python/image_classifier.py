import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import *
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
import matplotlib.pyplot as plt


class ImageClassifier:
    def __init__(self,
                 X_train,
                 y_train,
                 batchnorm_flag: bool,
                 epochs=10
                 ):
        self.X_train = X_train
        self.y_train = y_train
        self.batchnorm_flag = batchnorm_flag
        self.epochs = epochs

    def _obj_function(self,
                      params):
        if self.batchnorm_flag == True:
            model_arch = self._get_batchnorm_arch()
        else:
            model_arch = self._get_arch()

        # Select Optimizer && Learning rate
        optimizer_call = getattr(tf.keras.optimizers, params["optimizer"])
        optimizer = optimizer_call(learning_rate=params["learning_rate"])

        # Get callbacks
        callbacks = self._get_callbacks()

        # Compile model
        model_arch.compile(loss="binary_crossentropy",
                           optimizer=optimizer,
                           metrics=["acc"]
                           )
        history = model_arch.fit(self.X_train, self.y_train, validation_split=.2, epochs=self.epochs, verbose=2,
                                 callbacks=callbacks)

        # Evaluate our model
        obj_metric = history.history["val_acc"][-1]
        return {"loss": obj_metric, "status": STATUS_OK}

    def train_model_hyperopt(self,
                             objective_funtion,
                             space: dict,
                             max_evals: int,
                             seed=int
                             ):

        best_hyperparams = fmin(fn=objective_funtion,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=max_evals,
                                trials=Trials(),
                                rstate=seed
                                )
        return best_hyperparams

    def train_model(self,
                    optimizer,
                    learning_rate: float
                    ):

        # Get model arch
        if self.batchnorm_flag == True:
            model_arch = _get_batchnorm_arch()
        else:
            model_arch = self._get_arch()

        # Set optimizer and LR
        optimizer.learning_rate.assign(learning_rate)

        # Get callbacks
        callbacks = self._get_callbacks()

        # Compile model
        model_arch.compile(loss="binary_crossentropy",
                           optimizer=optimizer,
                           metrics=["acc"]
                           )
        # Retrain
        history = model_arch.fit(self.X_train, self.y_train, validation_split=.2, epochs=self.epochs, verbose=2,
                                 callbacks=callbacks)

        # View model loss
        plot = view_model_loss(history)

        return history, model_arch

    @staticmethod
    def view_model_loss(history):
        plt.clf()
        plt.plot(history.history["loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()

    @staticmethod
    def _get_batchnorm_arch():
        model = models.Sequential()
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(
        300, 180, 3)))  # input shape must be the match the input image tensor shape
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    @staticmethod
    def _get_arch():
        model = models.Sequential()
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(
        300, 180, 3)))  # input shape must be the match the input image tensor shape
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    @staticmethod
    def _get_callbacks():

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3)

        # Reduce LR on plateau
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                      patience=5, min_lr=0.001)

        # Learning Rate scheduler TO DO

        callbacks = [early_stopping, reduce_lr]

        return callbacks