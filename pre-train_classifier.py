'''(MODIFIED. pre-train_classifier.py)'''

'''This script is heavily modified from the blog post
"Building powerful image classification models using very little data" from blog.keras.io. We construct a classifier for our purposes using a pre-trained (VGG16) model.

It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data

This is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
Additionally, the script requires a "Locations.py" file. See README for more info.
'''

Pre_train = False  # need to pre-train conv layers?
Internet = True  # connected to the internet (Slack)?
Jupyter = True  # using Jupyter?
Heras = False  # use Heras?
model_name = "pre-trained_top_VGG16"

# let's name our model
model_name_complete = model_name + "_complete"
arch_name = model_name + ".json"


# dimensions of our images.
img_width, img_height = 150, 150

# number train samples, number validation samples, number of epochs
nb_train_samples = 17000  # remember to set batch sizes correctly!
nb_validation_samples = 8000
nb_epoch = 50


print("Pre-train:", Pre_train)
print("Using Internet:", Internet)
print("Using Heras:", Heras)
print("Using Jupyter:", Jupyter)


# dependencies
import os
import h5py
import numpy as np
from vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.regularizers import l2
from keras.optimizers import SGD, rmsprop
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from heraspy.model import HeraModel


# access locations file
exec(open("Locations.py").read())
print("locations successfully read.")


# define various saving modules for reproducibility
def save_weights(which_model, filename):
    run = os.path.join(runs_dir, filename)
    which_model.save_weights(run, overwrite=True)
    print("weights saved as", filename)


def save_architecture(archname, string):
    arch = os.path.join(arch_dir, archname)
    open(arch, 'w').write(string)
    print("architecture saved as", archname)


def save_plot(which_model, plotname=model_name):
    plots = os.path.join(plot_dir, plotname)
    plot(which_model, to_file=plots, show_layer_names=True, show_shapes=True)
    print("plot saved as", plotname)


# construct our neural network from the "VGG16.py" script. See Keras
# documentation for more information.
def construct_pretrained_model(image_width,  image_height):
    # building VGG16 model
    model = VGG16(weights='imagenet', include_top=False,
                  def_input=(image_width, image_height, 3))

    print('Model loaded.')
    return model


# this is the augmentation configuration (only rescaling) we will use for
# testing our classifier. Note that there is no need to create a
# configuration for the training data since the model is already
# pre-trained.
def construct_image_generators(image_width, image_height, train_data_loc):
    bottleneck_datagen = ImageDataGenerator(rescale=1. / 255)

    # define bottleneck generator
    bottleneck_generator = bottleneck_datagen.flow_from_directory(
        directory=train_data_loc,
        target_size=(image_width, image_height),
        color_mode="rgb",
        classes=['cats', 'dogs'],
        batch_size=20,
        shuffle=False,
        class_mode='binary')

    print("bottleneck generator constructed.")
    return bottleneck_generator


# Callbacks
# set early stopping requirements. stop the training process once the
# validation error fails to decrease for some specified time
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=20,  # num. of epochs with no improvement
                               mode='auto')

# save the model after every epoch. helps to mitigate the effects of
# training interruptions
save_model_name = os.path.join(runs_dir, model_name)
checkpointer = ModelCheckpoint(filepath=save_model_name, monitor='val_loss',
                               verbose=0, save_best_only=True)  # save_weights_only = True

# TensorBoard, optional use (currently not working)
#tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)

# initialize heraspy (see Keras Callbacks)
if Heras:
    hera_model = HeraModel(
        {
            'id': model_name  # any ID you want to use to identify your model
        },
        {
            # location of the local hera server, out of the box it's the
            # following
            'domain': 'localhost',
            'port': 4000
        }
    )


# now we define the bottleneck features
def bottleneck_features():
    # construct model
    model = construct_pretrained_model(img_width, img_height)

    # construct image generators
    generator_train = construct_image_generators(
        img_width, img_height, train_data_dir)
    generator_validation = construct_image_generators(
        img_width, img_height, validation_data_dir)

    # predict and save bottleneck features
    bottleneck_features_train = model.predict_generator(
        generator_train, nb_train_samples)
    np.save(open(train_bottleneck, 'wb'), bottleneck_features_train)

    bottleneck_features_validation = model.predict_generator(
        generator_validation, nb_validation_samples)
    np.save(open(validation_bottleneck, 'wb'), bottleneck_features_validation)


# we create the fully-connected model that will be stacked on top of the
# pre-trained model
def construct_top_model(input_data):
    top_model = Sequential()
    top_model.add(Flatten(input_shape=input_data.shape[1:]))
    top_model.add(Dense(output_dim=256,
                        input_dim=True,
                        init='uniform',  # initialize the weights
                        bias=True,
                        W_regularizer=l2(0.01)
                        ))
    top_model.add(Activation('relu'))
    top_model.add(Dropout(0.5))
    # model.add(GaussianNoise(0.1))
    top_model.add(Dense(output_dim=1,
                        init='uniform'))
    top_model.add(Activation('sigmoid'))  # tanhu, sigmoid

    # save architecture
    json_string = top_model.to_json()
    save_architecture(arch_name, json_string)

    # compile model
    top_model.compile(loss='binary_crossentropy',  # cost function, mse, categorical_crossentropy
                      # SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    # model.compile(loss='categorical_crossentropy',
    # optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
    return top_model


# train the top model from the bottleneck features
def train_top_model():
    train_data = np.load(open(runs_dir + "/bottleneck_VGG16_train.npy", 'rb'))
    train_labels = np.array(
        [0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))

    validation_data = np.load(
        open(runs_dir + "/bottleneck_VGG16_validation.npy", 'rb'))
    validation_labels = np.array(
        [0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))

    # construct top model
    top_model = construct_top_model(train_data)

    if Heras:
        # train model
        top_model.fit(  # use with ImageDataGenerator
            train_data,
            train_labels,
            nb_epoch=nb_epoch,  # number of epochs
            validation_data=(validation_data, validation_labels),
            callbacks=[early_stopping, checkpointer, hera_model.callback  # tensorboard
                       ])
    else:
        top_model.fit(  # use with ImageDataGenerator
            train_data,
            train_labels,
            nb_epoch=nb_epoch,  # number of epochs
            validation_data=(validation_data, validation_labels),
            callbacks=[early_stopping, checkpointer  # tensorboard
                       ])
    save_weights(which_model=top_model, filename=model_name_complete)
    if Internet:
        # Send message to Slack, signaling the end of training
        slck = os.path.join(slack_dir, "Slack.py")
        exec(open(slck).read())
        print("message sent to Slack")
    return top_model


# optionally, we can run the file from within Jupyter (see
# "pre-train_classifier.ipynb"). If we are not using Jupyter, run the
# following modules.
if not Jupyter:
    if Pre_train:
        bottleneck_features()
    trained_model = train_top_model()
    trained_model.summary()


print("pre-train_classifier.py complete.")
