import glob
import os
from skimage import io, transform
import numpy as np
from sklearn.model_selection import train_test_split
from python.image_classifier import ImageClassifier
from hyperopt import hp
import tensorflow as tf


path= "/Users/marygrace.moesta/Desktop/bee_classifier/data/PollenDataset/images"
im_list= glob.glob(os.path.join(path, '*.jpg'))


def dataset(file_list,size=(300,180),flattened=False):
    data = []
    for i, file in enumerate(file_list):
        image = io.imread(file)
        image = transform.resize(image, size, mode='constant')
        if flattened:
            image = image.flatten()

        data.append(image)

    labels = [1 if f.split("/")[-1][0] == 'P' else 0 for f in file_list]

    return np.array(data), np.array(labels)


(X,y) = dataset(im_list)

print(X.shape)
print(len(X))
print(y.shape)

# Initial train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20)

# Verifying
print(X_train.shape)

# Instatntiate classifier object
img_classifier = ImageClassifier(X_train=X_train,
                                 y_train=y_train,
                                 batchnorm_flag=False
                                 )
obj_function = img_classifier.obj_function
space = {
    "optimizer": hp.choice("optimizer", ["Adadelta", "Adam"]),
    "learning_rate": hp.loguniform("learning_rate", -5, 0)
}

best_params = img_classifier.train_model_hyperopt(space=space,
                                                  max_evals=4,
                                                  objective_function=obj_function,
                                                  seed=np.random.RandomState(42)
                                                  )

# Re-train model with new params
opt = tf.keras.optimizers.Adadelta()
history, model_arch = img_classifier.train_model(optimizer=opt,
                                                 learning_rate=best_params.get('learning_rate')
                                                 )

