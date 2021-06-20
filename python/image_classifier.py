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

    def obj_function(self,
                     params):
        """

        :param params:
        :return:

        """
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
                             objective_function,
                             space: dict,
                             max_evals: int,
                             seed=int
                             ):
        """

        :param objective_function: given objective function to minimzie
        :param space: search space
        :param max_evals: max number of evaluations
        :param seed: random state
        :return:
            best_hyperparameters
        """

        best_hyperparams = fmin(fn=objective_function,
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
        """

        :param optimizer: the tf.keras.optimizer used when compiling model
        :param learning_rate: LR optimizer will use to find minima
        :return: history, model_architecture
        """

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

        return history, model_arch

    @staticmethod
    def _get_batchnorm_arch():
        """

        :return: model architecture using batch normalization
        """
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
        """

        :return: model architecture using dropout
        """
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
        """

        :return: callbacks, list of configured callbacks to be used for model training
        """

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3)

        # Reduce LR on plateau
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                      patience=5, min_lr=0.001)

        # Learning Rate scheduler TO DO

        callbacks = [early_stopping, reduce_lr]

        return callbacks
