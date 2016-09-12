'''(MODIFIED. train_classifier.py)'''

'''This script is heavily modified from the blog post
"Building powerful image classification models using very little data" from blog.keras.io. We now fine-tune the entire pre-trained model + fully-connected top model. Careful to use an extremely small learning rate.

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
model_name = "pre-trained_top_VGG16"
full_model_name = "fully_tuned_VGG16"

#let's name our model
model_name_complete = model_name + "_complete"
full_model_name_complete = full_model_name + "_complete"
full_arch_name = full_model_name + ".json"
arch_name = model_name + ".json"


# dimensions of our images.
img_width, img_height = 150, 150

#number of trian samples, number of validation samples, number of epochs
nb_train_samples = 17000 #remember to set batch sizes correctly!
nb_validation_samples = 8000
tune_nb_epoch = 10


print("Using Internet:", Internet)
print("Using Heras:", Heras)
print("Using Jupyter:", Jupyter)


#dependencies
import os
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


#construct our model (pre-trained + top_model)
def construct_model(image_width, image_height):
    #building VGG16 model
    model = VGG16(weights='imagenet', include_top=False, def_input = (image_width, image_height, 3))

    #load top_model
    top_model = model_from_json(open(model_location).read())
    weights_location = os.path.join(runs_dir, model_name) #or model_name_complete
    top_model.load_weights(weights_location)

    #combine VGG16 and top_model
    model.add(top_model)

    #save architecture
    json_string = model.to_json()
    save_architecture(full_arch_name, json_string)

    print("model constructed.")
    return model


#this is the augmentation configuration we will use for the trianing data (data augmentation)
def construct_tuning_generators(image_width, image_height, train_data_loc, validation_data_loc):
    tuning_test_datagen = ImageDataGenerator(rescale=1./255)

    #this is the augmentation configuration we will use for the training data
    tuning_train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            rotation_range=25,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            zoom_range=0.2,
            fill_mode = 'nearest',
            vertical_flip = True,
            horizontal_flip=True)

    #define train generator
    tuning_train_generator = tuning_train_datagen.flow_from_directory(
            train_data_loc,
            target_size=(image_width, image_height),
            color_mode = "rgb",
            classes = ['cats', 'dogs'],
            batch_size=20,
            shuffle = True,
            class_mode='binary')

    #define validation generator
    tuning_validation_generator = tuning_test_datagen.flow_from_directory(
            validation_data_loc,
            target_size=(image_width, image_height),
            color_mode = "rgb",
            classes = ['cats', 'dogs'],
            batch_size=20,
            shuffle = True,
            class_mode='binary')

    print("tuning generators constructed.")
    return tuning_train_generator, tuning_validation_generator


#Callbacks
#set early stopping requirements. stop the training process once the validation error fails to decrease for some specified time
early_stopping = EarlyStopping(monitor='val_loss',
                            patience = 20, #num. of epochs with no improvement
                            mode = 'auto')

#save the model after every epoch. helps to mitigate the effects of training interruptions
save_model_name2 = os.path.join(runs_dir, full_model_name)
checkpointer2 = ModelCheckpoint(filepath = save_model_name2, monitor = 'val_loss', verbose = 0, save_best_only = True) #save_weights_only = True

#TensorBoard, optional use (currently not working)
#tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)

#initialize heraspy
if Heras:
    hera_model = HeraModel(
        {
            'id': full_model_name # any ID you want to use to identify your model
        },
        {
            # location of the local hera server, out of the box it's the following
            'domain': 'localhost',
            'port': 4000
        }
    )


#now we fine-tune the model
def fine_tuning():
#train model (with/without Heras)
    #add pre-trained model to top_model
    full_model = construct_model(img_width, img_height)

    #"freeze" lowers layers (will only fine-tune final conv layer and fully connected)
    for layer in full_model.layers[:25]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    full_model.compile(loss = 'binary_crossentropy',
                  optimizer = SGD(lr=1e-4, momentum=0.9),
                  metrics = ['accuracy'])

    #construct tuning generators
    generators = construct_tuning_generators(img_width, img_height, train_data_dir, validation_data_dir)

    #fit full_model
    if Heras:
        full_model.fit_generator(
            generators[0],
            samples_per_epoch=nb_train_samples,
            nb_epoch=tune_nb_epoch, #number of epochs
            validation_data=generators[1],
            nb_val_samples=nb_validation_samples,
            callbacks = [early_stopping, checkpointer2, hera_model.callback #tensorboard
            ])
    else:
        full_model.fit_generator(
            generators[0],
            samples_per_epoch=nb_train_samples,
            nb_epoch=tune_nb_epoch,
            validation_data=generators[1],
            nb_val_samples=nb_validation_samples,
            callbacks = [early_stopping, checkpointer2 #tensorboard
            ])
    save_weights(which_model = full_model, filename = full_model_name_complete)
    save_plot(which_model = full_model)
    if Internet:
        #Send message to Slack, signaling the end of the training
        slck = os.path.join(slack_dir, "Slack.py")
        exec(open(slck).read())
        print("message sent to Slack")
    return full_model


#optionally, we can run the file from within Jupyter (see "train_classifier.ipynb"). If we are not using Jupyter, run the following modules.
if not Jupyter:
    fully_tuned_model = fine_tuning()
    fully_tuned_model.summary()


print("fine-tune_classifier.py complete.")
