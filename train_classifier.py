'''(MODIFIED. train_classifier.py)'''

'''This script is heavily modifed from the blog post
"Building powerful image classification models using very little data" from blog.keras.io. We build and train a classifer from scratch (using Keras)

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
'''


Internet = True #connected to the internet (Slack)?
Jupyter = True #using Jupyter?
Heras = False #use Heras?
model_name = "original_model"

#let's name our model
model_name_complete = model_name + "_complete"
arch_name = model_name + ".json"


# dimensions of our images.
img_width, img_height = 150, 150

#number train samples, number validation samples, number of epochs
nb_train_samples = 200 #17000 #remember to set batch sizes correctly!
nb_validation_samples = 40 #8000
nb_epoch = 1


print("Using Internet:", Internet)
print("Using Heras:", Heras)
print("Using Jupyter:", Jupyter)


#dependencies
import os
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


#define various saving modules for reproducibility
def save_weights(which_model, filename):
    run = os.path.join(runs_dir, filename)
    which_model.save_weights(run, overwrite = True)
    print("weights saved as", filename)

def save_architecture(archname, string):
    arch = os.path.join(arch_dir, archname)
    open(arch, 'w').write(string)
    print("architecture saved as", archname)

def save_plot(which_model, plotname = model_name):
    plots = os.path.join(plot_dir, plotname)
    plot(which_model, to_file = plots, show_layer_names = True, show_shapes = True)
    print("plot saved as", plotname)


#construct our neural network model
def construct_model(image_width, image_height, name):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(image_width, image_height, 3)))
    model.add(Convolution2D(nb_filter = 32, #number of filters
                            nb_row = 3, #number of rows in kernel
                            nb_col = 3,
                            init = 'uniform',
                            border_mode = 'valid', #padded with zeroes
                            W_regularizer = None,
                            activity_regularizer = None,
                            bias = True,
                            name = 'conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (3, 3),
                            strides = (2, 2),
                            border_mode = 'valid'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3,
                            init = 'uniform',
                            name = 'conv2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (3, 3),
                            strides = (2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3,
                            init = 'uniform',
                            name = 'conv3'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (3, 3),
                            strides = (2, 2)))

    model.add(Flatten())
    model.add(Dense(output_dim = 64,
                    input_dim = True,
                    init = 'uniform', #initialize the weights
                    bias = True,
                    W_regularizer = l2(0.01)
                    ))
    model.add(Activation('relu'))

    model.add(Dense(output_dim = 1,
                    init = 'uniform'))
    model.add(Activation('sigmoid')) #tanhu, sigmoid

    json_string = model.to_json()
    save_architecture(name, json_string)
    print("model constructed.")
    return model


# this is the augmentation configuration we will use for the training data (data augmentation)
def construct_image_generators(image_width, image_height, train_data_loc, validation_data_loc):
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            rotation_range=25,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            zoom_range=0.2,
            fill_mode = 'nearest',
            vertical_flip = True,
            horizontal_flip=True)

    # this is the augmentation configuration we will use for the testing data (only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)

    #define train generator
    train_generator = train_datagen.flow_from_directory(
            directory = train_data_loc,
            target_size=(image_width, image_height),
            color_mode = "rgb",
            classes = ['cats', 'dogs'],
            batch_size=20,
            shuffle = True,
            class_mode='binary')

    #define validation generator
    validation_generator = test_datagen.flow_from_directory(
            validation_data_loc,
            target_size=(image_width, image_height),
            color_mode = "rgb",
            classes = ['cats', 'dogs'],
            batch_size=20,
            shuffle = True,
            class_mode='binary')
    print("image generators constructed.")
    return train_generator, validation_generator


#Callbacks
#set early stopping requirements. stop the training process once the validation error fails to decrease for some specified time
early_stopping = EarlyStopping(monitor='val_loss',
                            patience = 20, #num. of epochs with no improvement
                            mode = 'auto')

#save the model after every epoch. helps to mitigate the effects of training interruptions
save_model_name = os.path.join(runs_dir, model_name)
checkpointer = ModelCheckpoint(filepath = save_model_name, monitor = 'val_loss', verbose = 0, save_best_only = True) #save_weights_only = True)

#TensorBoard, optional use (currently not working)
#tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)

#initialize heraspy (see Keras Callbacks)
if Heras:
    hera_model = HeraModel(
        {
            'id': model_name # any ID you want to use to identify your model
        },
        {
            # location of the local hera server, out of the box it's the following
            'domain': 'localhost',
            'port': 4000
        }
    )



#now we train the model
def train_model():
#train the model (with/without Heras)
    #construct model
    model = construct_model(img_width, img_height, arch_name)

    #construct image generators
    generators = construct_image_generators(img_width, img_height, train_data_dir, validation_data_dir)

    #compile model
    model.compile(loss='binary_crossentropy', #cost function, mse, categorical_crossentropy
                  optimizer='rmsprop', #SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
                  metrics=['accuracy'])


    if Heras:
        model.fit_generator( #use with ImageDataGenerator
                generators[0],
                samples_per_epoch=nb_train_samples,
                nb_epoch=nb_epoch, #number of epochs
                validation_data=generators[1],
                nb_val_samples=nb_validation_samples,
                callbacks = [early_stopping, checkpointer, hera_model.callback #tensorboard
                ])
    else:
        model.fit_generator(
                generators[0],
                samples_per_epoch=nb_train_samples,
                nb_epoch=nb_epoch,
                validation_data=generators[1],
                nb_val_samples=nb_validation_samples,
                callbacks = [early_stopping, checkpointer #tensorboard
                ])
    save_weights(which_model = model, filename = model_name_complete)
    save_plot(which_model = model)
    if Internet:
        #Send message to Slack, signaling the end of training
        slck = os.path.join(slack_dir, "Slack.py")
        exec(open(slck).read())
        print("message sent to Slack")
    return model


#optionally, we can run the file from within Jupyter (see "train_classifier.ipynb"). If we are not using Jupyter, run the following modules.
if not Jupyter:
    trained_model = train_model()
    trained_model.summary()


print("train_classifier.py complete.")
