# Classifying Cats and Dogs using CNN and Transfer Learning

In this blog post, we'll build a model to try and identify whether images contain a dog or a cat. The dataset contains 25,000 images of cats and dogs and they have already been split into train and test sets. You can download the data [here](https://www.kaggle.com/c/dogs-vs-cats).  



```python
# import required libraries
import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
```

## Explore the Data
We will first download the dataset using the code block below. The data also needs to be split into a training and testing set.  
The code block below downloads the full Cats-v-Dogs dataset and stores it as `cats-and-dogs.zip`. It then unzips it to `/tmp`, which will create a `tmp/PetImages` directory containing subdirectories called `Cat` and `Dog`.


```python
!wget --no-check-certificate \
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" \
    -O "/tmp/cats-and-dogs.zip"

local_zip = '/tmp/cats-and-dogs.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

```

    --2020-08-20 14:13:38--  https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
    Resolving download.microsoft.com (download.microsoft.com)... 23.35.76.84, 2600:1407:d800:29e::e59, 2600:1407:d800:2a2::e59, ...
    Connecting to download.microsoft.com (download.microsoft.com)|23.35.76.84|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 824894548 (787M) [application/octet-stream]
    Saving to: ‘/tmp/cats-and-dogs.zip’
    
    /tmp/cats-and-dogs. 100%[===================>] 786.68M  43.7MB/s    in 23s     
    
    2020-08-20 14:14:02 (34.0 MB/s) - ‘/tmp/cats-and-dogs.zip’ saved [824894548/824894548]
    
    


```python
print(len(os.listdir('/tmp/PetImages/Cat/')))
print(len(os.listdir('/tmp/PetImages/Dog/')))
```

    12501
    12501
    

We'll use `os.mkdir` to create your directories. We need a directory for `cats-v-dogs`, and subdirectories for `training` and `testing`. These in turn will need subdirectories for `cats` and `dogs`.


```python
to_create = [
             '/tmp/cats-v-dogs',
             '/tmp/cats-v-dogs/training',
             '/tmp/cats-v-dogs/testing',
             '/tmp/cats-v-dogs/training/cats',
             '/tmp/cats-v-dogs/training/dogs',
             '/tmp/cats-v-dogs/testing/cats',
             '/tmp/cats-v-dogs/testing/dogs'
            ]

for directory in to_create:
    try:
        os.mkdir(directory)
        print(directory, 'created')
    except OSError:
        pass
```

    /tmp/cats-v-dogs created
    /tmp/cats-v-dogs/training created
    /tmp/cats-v-dogs/testing created
    /tmp/cats-v-dogs/training/cats created
    /tmp/cats-v-dogs/training/dogs created
    /tmp/cats-v-dogs/testing/cats created
    /tmp/cats-v-dogs/testing/dogs created
    

We will write a Python function called `split_data` which takes a `SOURCE` directory containing the files, `TRAINING` and `TESTING` directories that a portion of the files will be copied to, and a `SPLIT_SIZE` to determine the portion of the split.  
The files should also be randomized, so that the training set is a random X% of the files, and the test set is the reamining files.  
Also, all images should be checked, and if they have a zero files length, they will not be copied over.

`os.listdir(DIRECTORY)` gives you a listing of the contents of that directory.  

`os.path.getsize(PATH)` gives you the size of the file.  

`copyfile(source, destination)` copies a file from source to destination.  

`random.sample(list, len(list))` shuffles a list.


```python
# os.listdir(DIRECTORY) gives you a listing of the contents of that directory
# os.path.getsize(PATH) gives you the size of the file
# copyfile(source, destination) copies a file from source to destination
# random.sample(list, len(list)) shuffles a list
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    all_files = []
    
    for file_name in os.listdir(SOURCE):
        file_path = SOURCE + file_name

        if os.path.getsize(file_path):
            all_files.append(file_name)
        else:
            print('{} is zero length, so ignoring'.format(file_name))
    
    n_files = len(all_files)
    split_point = int(n_files * SPLIT_SIZE)
    
    shuffled = random.sample(all_files, n_files)
    
    train_set = shuffled[:split_point]
    test_set = shuffled[split_point:]
    
    for file_name in train_set:
        copyfile(SOURCE + file_name, TRAINING + file_name)
        
    for file_name in test_set:
        copyfile(SOURCE + file_name, TESTING + file_name)


CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
```

    666.jpg is zero length, so ignoring
    11702.jpg is zero length, so ignoring
    


```python
print(len(os.listdir('/tmp/cats-v-dogs/training/cats/')))
print(len(os.listdir('/tmp/cats-v-dogs/training/dogs/')))
print(len(os.listdir('/tmp/cats-v-dogs/testing/cats/')))
print(len(os.listdir('/tmp/cats-v-dogs/testing/dogs/')))

```

    11250
    11250
    1250
    1250
    

There are 11,250 images each in the training set of the cats and dogs directories.  
There are 1,250 images each in the test set of the cats and dogs directories.

Let's see what the file names look like in the cats and dogs training directories. We will see that there are no labels in the dataset.


```python
train_cat_fnames = os.listdir('/tmp/cats-v-dogs/training/cats/')
train_dog_fnames = os.listdir('/tmp/cats-v-dogs/training/dogs/')

print(train_cat_fnames[:10])
print(train_dog_fnames[:10])
```

    ['333.jpg', '8919.jpg', '1366.jpg', '2691.jpg', '3556.jpg', '3354.jpg', '11491.jpg', '3167.jpg', '3796.jpg', '12240.jpg']
    ['333.jpg', '8919.jpg', '2691.jpg', '3556.jpg', '3354.jpg', '11491.jpg', '3796.jpg', '12240.jpg', '10219.jpg', '5107.jpg']
    

Now let's take a look at a few pictures to get a better sense of what the cat and dog datasets look like. We'll display a batch of 8 cat and 8 dog pictures. We can rerun the cell to see a fresh batch each time.


```python
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
%matplotlib inline

# parameters for our graph
nrows = 4
ncols = 4

# index for iterating over images
pic_index = 0

# set up matplotlib fig and size it to fit 4X4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_cat_pix = [os.path.join('/tmp/cats-v-dogs/training/cats/', fname) 
                for fname in train_cat_fnames[ pic_index-8:pic_index] 
               ]

next_dog_pix = [os.path.join('/tmp/cats-v-dogs/training/dogs/', fname) 
                for fname in train_dog_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_cat_pix+next_dog_pix):
  # set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()
```


![png](output_14_0.png)


## Building a model
We'll define a Sequential layer and add some convolutional layers. The input shape parameter will be 150 X 150 for the size and 3 (bytes) for the color depth.  
We then add a few convolutional and pooling layers, and flatten the final result to feed into the densely connected layers.  
Next, we'll configure the specifications for model training. We'll train our model with the binary_crossentropy loss because it's a binary classification problem and our final activation is a sigmoid, so that the output of our network will be a single scalar between 0 and 1, encoding the probability that the current image is class 1 (as opposed to class 0). We'll use the rmsprop optimizer with a learning rate of 0.001. During training, we will monitor the classification accuracy.


```python
# define a keras model to classify cats-v-dogs
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape=(150, 150, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
```

The `model.summary()` method prints a summary of the neural network.


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 148, 148, 32)      896       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 74, 74, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 72, 72, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 36, 36, 64)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 34, 34, 128)       73856     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 17, 17, 128)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 15, 15, 128)       147584    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 7, 7, 128)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 6272)              0         
    _________________________________________________________________
    dense (Dense)                (None, 512)               3211776   
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               65664     
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 129       
    =================================================================
    Total params: 3,518,401
    Trainable params: 3,518,401
    Non-trainable params: 0
    _________________________________________________________________
    

The *output shape* column shows how the size of our feature map evolves in each successive layer. The convolution layers reduce the size of the feature maps by a bit due to padding, and each pooling layer halves the dimensions.

## Data Preprocessing

Let's set up the data generators that will read the pictures in our source folders, convert them to float32 tensors, and feed them (with their labels) to our network. We'll have one generator for the training images and one for the validation images. Our generators will yield batches of 20 images of size 150 X 150 and their labels (binary).  
We will preprocess our images by normalizing the pixel values to be in the [0, 1] range (originally all values are in the [0, 255] range).  
In Keras this can be done via the `keras.preprocessing.image.ImageDataGenerator` class using the `rescale` parameter. This `ImageDataGenerator` class allows us to instantiate generators of augmented image batches (and their labels) via `.flow(data, labels)` or `.flow_from_directory(directory)`. These generators can then be used with the Keras model methods that accept data generators as inputs: `fit`, `evaluate_generator`, and `predict_generator`.


```python
TRAINING_DIR = '/tmp/cats-v-dogs/training' 
train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    fill_mode='nearest'
) 
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=64,
    class_mode='binary',
    target_size=(150, 150)
) 

VALIDATION_DIR = '/tmp/cats-v-dogs/testing' 
validation_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    fill_mode='nearest'
) 
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=64,
    class_mode='binary',
    target_size=(150, 150)
) 
```

    Found 22498 images belonging to 2 classes.
    Found 2500 images belonging to 2 classes.
    

Considering time and GPU constraints, we will only be training the dataset for 25 epochs.


```python
history = model.fit(train_generator,
                    epochs=25,
                    verbose=1,
                    validation_data=validation_generator)

# The expectation here is that the model will train, and that accuracy will be > 95% on both training and validation
# i.e. acc:A1 and val_acc:A2 will be visible, and both A1 and A2 will be > .9
```

    Epoch 1/25
    316/352 [=========================>....] - ETA: 19s - loss: 0.6918 - accuracy: 0.5576

    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:770: UserWarning: Possibly corrupt EXIF data.  Expecting to read 32 bytes but only got 0. Skipping tag 270
      " Skipping tag %s" % (size, len(data), tag)
    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:770: UserWarning: Possibly corrupt EXIF data.  Expecting to read 5 bytes but only got 0. Skipping tag 271
      " Skipping tag %s" % (size, len(data), tag)
    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:770: UserWarning: Possibly corrupt EXIF data.  Expecting to read 8 bytes but only got 0. Skipping tag 272
      " Skipping tag %s" % (size, len(data), tag)
    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:770: UserWarning: Possibly corrupt EXIF data.  Expecting to read 8 bytes but only got 0. Skipping tag 282
      " Skipping tag %s" % (size, len(data), tag)
    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:770: UserWarning: Possibly corrupt EXIF data.  Expecting to read 8 bytes but only got 0. Skipping tag 283
      " Skipping tag %s" % (size, len(data), tag)
    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:770: UserWarning: Possibly corrupt EXIF data.  Expecting to read 20 bytes but only got 0. Skipping tag 306
      " Skipping tag %s" % (size, len(data), tag)
    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:770: UserWarning: Possibly corrupt EXIF data.  Expecting to read 48 bytes but only got 0. Skipping tag 532
      " Skipping tag %s" % (size, len(data), tag)
    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:788: UserWarning: Corrupt EXIF data.  Expecting to read 2 bytes but only got 0. 
      warnings.warn(str(msg))
    

    352/352 [==============================] - 216s 613ms/step - loss: 0.6919 - accuracy: 0.5646 - val_loss: 0.6709 - val_accuracy: 0.5984
    Epoch 2/25
    352/352 [==============================] - 215s 611ms/step - loss: 0.6265 - accuracy: 0.6577 - val_loss: 0.5989 - val_accuracy: 0.6844
    Epoch 3/25
    352/352 [==============================] - 217s 617ms/step - loss: 0.5871 - accuracy: 0.6933 - val_loss: 0.5487 - val_accuracy: 0.7084
    Epoch 4/25
    352/352 [==============================] - 217s 616ms/step - loss: 0.5511 - accuracy: 0.7233 - val_loss: 0.5326 - val_accuracy: 0.7376
    Epoch 5/25
    352/352 [==============================] - 216s 614ms/step - loss: 0.5141 - accuracy: 0.7477 - val_loss: 0.5099 - val_accuracy: 0.7396
    Epoch 6/25
    352/352 [==============================] - 215s 612ms/step - loss: 0.4857 - accuracy: 0.7664 - val_loss: 0.4842 - val_accuracy: 0.7924
    Epoch 7/25
    352/352 [==============================] - 217s 616ms/step - loss: 0.4651 - accuracy: 0.7821 - val_loss: 0.4328 - val_accuracy: 0.8136
    Epoch 8/25
    352/352 [==============================] - 218s 619ms/step - loss: 0.4351 - accuracy: 0.7965 - val_loss: 0.4019 - val_accuracy: 0.8288
    Epoch 9/25
    352/352 [==============================] - 215s 612ms/step - loss: 0.4165 - accuracy: 0.8092 - val_loss: 0.4253 - val_accuracy: 0.7980
    Epoch 10/25
    352/352 [==============================] - 215s 610ms/step - loss: 0.3971 - accuracy: 0.8219 - val_loss: 0.3628 - val_accuracy: 0.8448
    Epoch 11/25
    352/352 [==============================] - 216s 614ms/step - loss: 0.3777 - accuracy: 0.8331 - val_loss: 0.3563 - val_accuracy: 0.8440
    Epoch 12/25
    352/352 [==============================] - 216s 615ms/step - loss: 0.3610 - accuracy: 0.8457 - val_loss: 0.3587 - val_accuracy: 0.8528
    Epoch 13/25
    352/352 [==============================] - 216s 613ms/step - loss: 0.3466 - accuracy: 0.8521 - val_loss: 0.3320 - val_accuracy: 0.8584
    Epoch 14/25
    352/352 [==============================] - 216s 615ms/step - loss: 0.3312 - accuracy: 0.8555 - val_loss: 0.3045 - val_accuracy: 0.8748
    Epoch 15/25
    352/352 [==============================] - 216s 614ms/step - loss: 0.3296 - accuracy: 0.8640 - val_loss: 0.3036 - val_accuracy: 0.8804
    Epoch 16/25
    352/352 [==============================] - 217s 615ms/step - loss: 0.3208 - accuracy: 0.8623 - val_loss: 0.3066 - val_accuracy: 0.8664
    Epoch 17/25
    352/352 [==============================] - 216s 614ms/step - loss: 0.3126 - accuracy: 0.8670 - val_loss: 0.3310 - val_accuracy: 0.8620
    Epoch 18/25
    352/352 [==============================] - 217s 616ms/step - loss: 0.3096 - accuracy: 0.8678 - val_loss: 0.2933 - val_accuracy: 0.8760
    Epoch 19/25
    352/352 [==============================] - 217s 615ms/step - loss: 0.3049 - accuracy: 0.8733 - val_loss: 0.2483 - val_accuracy: 0.9008
    Epoch 20/25
    352/352 [==============================] - 217s 616ms/step - loss: 0.3020 - accuracy: 0.8724 - val_loss: 0.3135 - val_accuracy: 0.8552
    Epoch 21/25
    352/352 [==============================] - 217s 617ms/step - loss: 0.2977 - accuracy: 0.8748 - val_loss: 0.2472 - val_accuracy: 0.8968
    Epoch 22/25
    352/352 [==============================] - 217s 617ms/step - loss: 0.2961 - accuracy: 0.8747 - val_loss: 0.2655 - val_accuracy: 0.8888
    Epoch 23/25
    352/352 [==============================] - 217s 617ms/step - loss: 0.2964 - accuracy: 0.8791 - val_loss: 0.3054 - val_accuracy: 0.8764
    Epoch 24/25
    352/352 [==============================] - 217s 617ms/step - loss: 0.2907 - accuracy: 0.8787 - val_loss: 0.2580 - val_accuracy: 0.8848
    Epoch 25/25
    352/352 [==============================] - 217s 617ms/step - loss: 0.2846 - accuracy: 0.8828 - val_loss: 0.2440 - val_accuracy: 0.8968
    

We can see four values per epoch - *loss*, *accuracy*, *validation loss* and *validation accuracy*.  
The *loss* and *accuracy* are a great indication of progress of training. It's making a guess as to the classification of the training data, and then measuring it against the known label, calculating the result. The *accuracy* is the portion of correct guesses. The *validation accuracy* is the measurement with the data that has not been used in training. As expected, this would be a bit lower.

After 25 epochs, the *training loss* = 0.28 and *training accuracy* = 0.88. The *validation loss* = 0.24 and *validation accuracy* = 0.89.

## Evaluating accuracy and loss for the model
Let's plot the training/validation accuracy and loss, during training.


```python
# PLOT LOSS AND ACCURACY
%matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')

# Desired output. Charts with training and validation metrics. No crash :)
```




    Text(0.5, 1.0, 'Training and validation loss')




![png](output_27_1.png)



![png](output_27_2.png)


The graphs suggest that the validation accuracy is seen rising synchronously with the training accuracy. The validation loss is decreasing synchronously with the training accuracy.  
However, the validation accuracy and validation loss lines are wavy and are not smooth like the training lines.  
Let's see how the model performs after applying transfer learning.

## Transfer Learning
In Transfer Learning, we take an existing model that's trained on far more data, and use the features that that model learned.  
We'll be using the keras `layers` API, to pick at the layers, and to understand which ones we want to use, and which ones we want to retrain. A copy of the pretrained weights for the inception neural network is saved at the below URL. Keras has the model definition built-in. It's the parameters that can then get loaded into the skeleton of the model, to turn it back into a trained model. So now if we want to use Inception, it's fortunate that keras has the model definition built in.


```python
from tensorflow.keras import layers
from tensorflow.keras import Model 

!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

from tensorflow.keras.applications.inception_v3 import InceptionV3
```

    --2020-08-20 15:45:07--  https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
    Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.69.128, 172.217.212.128, 172.217.214.128, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.69.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 87910968 (84M) [application/x-hdf]
    Saving to: ‘/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5’
    
    /tmp/inception_v3_w 100%[===================>]  83.84M   199MB/s    in 0.4s    
    
    2020-08-20 15:45:08 (199 MB/s) - ‘/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5’ saved [87910968/87910968]
    
    

The inception V3 has a fully-connected layer at the top. So by setting `include_top=false`, we're specifying that we want to ignore this and get straight to the convolutions. Now that we have our pretrained model instantiated, we can iterate through its layers and lock them, saying that they're not going to be trainable with this code. 


```python
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable=False

pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
```

    Model: "inception_v3"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 150, 150, 3) 0                                            
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, 74, 74, 32)   864         input_1[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization (BatchNorma (None, 74, 74, 32)   96          conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    activation (Activation)         (None, 74, 74, 32)   0           batch_normalization[0][0]        
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, 72, 72, 32)   9216        activation[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, 72, 72, 32)   96          conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    activation_1 (Activation)       (None, 72, 72, 32)   0           batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    conv2d_6 (Conv2D)               (None, 72, 72, 64)   18432       activation_1[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, 72, 72, 64)   192         conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    activation_2 (Activation)       (None, 72, 72, 64)   0           batch_normalization_2[0][0]      
    __________________________________________________________________________________________________
    max_pooling2d_4 (MaxPooling2D)  (None, 35, 35, 64)   0           activation_2[0][0]               
    __________________________________________________________________________________________________
    conv2d_7 (Conv2D)               (None, 35, 35, 80)   5120        max_pooling2d_4[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_3 (BatchNor (None, 35, 35, 80)   240         conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    activation_3 (Activation)       (None, 35, 35, 80)   0           batch_normalization_3[0][0]      
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, 33, 33, 192)  138240      activation_3[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_4 (BatchNor (None, 33, 33, 192)  576         conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    activation_4 (Activation)       (None, 33, 33, 192)  0           batch_normalization_4[0][0]      
    __________________________________________________________________________________________________
    max_pooling2d_5 (MaxPooling2D)  (None, 16, 16, 192)  0           activation_4[0][0]               
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, 16, 16, 64)   12288       max_pooling2d_5[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_8 (BatchNor (None, 16, 16, 64)   192         conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    activation_8 (Activation)       (None, 16, 16, 64)   0           batch_normalization_8[0][0]      
    __________________________________________________________________________________________________
    conv2d_10 (Conv2D)              (None, 16, 16, 48)   9216        max_pooling2d_5[0][0]            
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, 16, 16, 96)   55296       activation_8[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_6 (BatchNor (None, 16, 16, 48)   144         conv2d_10[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_9 (BatchNor (None, 16, 16, 96)   288         conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    activation_6 (Activation)       (None, 16, 16, 48)   0           batch_normalization_6[0][0]      
    __________________________________________________________________________________________________
    activation_9 (Activation)       (None, 16, 16, 96)   0           batch_normalization_9[0][0]      
    __________________________________________________________________________________________________
    average_pooling2d (AveragePooli (None, 16, 16, 192)  0           max_pooling2d_5[0][0]            
    __________________________________________________________________________________________________
    conv2d_9 (Conv2D)               (None, 16, 16, 64)   12288       max_pooling2d_5[0][0]            
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, 16, 16, 64)   76800       activation_6[0][0]               
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, 16, 16, 96)   82944       activation_9[0][0]               
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, 16, 16, 32)   6144        average_pooling2d[0][0]          
    __________________________________________________________________________________________________
    batch_normalization_5 (BatchNor (None, 16, 16, 64)   192         conv2d_9[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_7 (BatchNor (None, 16, 16, 64)   192         conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_10 (BatchNo (None, 16, 16, 96)   288         conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_11 (BatchNo (None, 16, 16, 32)   96          conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    activation_5 (Activation)       (None, 16, 16, 64)   0           batch_normalization_5[0][0]      
    __________________________________________________________________________________________________
    activation_7 (Activation)       (None, 16, 16, 64)   0           batch_normalization_7[0][0]      
    __________________________________________________________________________________________________
    activation_10 (Activation)      (None, 16, 16, 96)   0           batch_normalization_10[0][0]     
    __________________________________________________________________________________________________
    activation_11 (Activation)      (None, 16, 16, 32)   0           batch_normalization_11[0][0]     
    __________________________________________________________________________________________________
    mixed0 (Concatenate)            (None, 16, 16, 256)  0           activation_5[0][0]               
                                                                     activation_7[0][0]               
                                                                     activation_10[0][0]              
                                                                     activation_11[0][0]              
    __________________________________________________________________________________________________
    conv2d_19 (Conv2D)              (None, 16, 16, 64)   16384       mixed0[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_15 (BatchNo (None, 16, 16, 64)   192         conv2d_19[0][0]                  
    __________________________________________________________________________________________________
    activation_15 (Activation)      (None, 16, 16, 64)   0           batch_normalization_15[0][0]     
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, 16, 16, 48)   12288       mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_20 (Conv2D)              (None, 16, 16, 96)   55296       activation_15[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_13 (BatchNo (None, 16, 16, 48)   144         conv2d_17[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_16 (BatchNo (None, 16, 16, 96)   288         conv2d_20[0][0]                  
    __________________________________________________________________________________________________
    activation_13 (Activation)      (None, 16, 16, 48)   0           batch_normalization_13[0][0]     
    __________________________________________________________________________________________________
    activation_16 (Activation)      (None, 16, 16, 96)   0           batch_normalization_16[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_1 (AveragePoo (None, 16, 16, 256)  0           mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, 16, 16, 64)   16384       mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, 16, 16, 64)   76800       activation_13[0][0]              
    __________________________________________________________________________________________________
    conv2d_21 (Conv2D)              (None, 16, 16, 96)   82944       activation_16[0][0]              
    __________________________________________________________________________________________________
    conv2d_22 (Conv2D)              (None, 16, 16, 64)   16384       average_pooling2d_1[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_12 (BatchNo (None, 16, 16, 64)   192         conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_14 (BatchNo (None, 16, 16, 64)   192         conv2d_18[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_17 (BatchNo (None, 16, 16, 96)   288         conv2d_21[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_18 (BatchNo (None, 16, 16, 64)   192         conv2d_22[0][0]                  
    __________________________________________________________________________________________________
    activation_12 (Activation)      (None, 16, 16, 64)   0           batch_normalization_12[0][0]     
    __________________________________________________________________________________________________
    activation_14 (Activation)      (None, 16, 16, 64)   0           batch_normalization_14[0][0]     
    __________________________________________________________________________________________________
    activation_17 (Activation)      (None, 16, 16, 96)   0           batch_normalization_17[0][0]     
    __________________________________________________________________________________________________
    activation_18 (Activation)      (None, 16, 16, 64)   0           batch_normalization_18[0][0]     
    __________________________________________________________________________________________________
    mixed1 (Concatenate)            (None, 16, 16, 288)  0           activation_12[0][0]              
                                                                     activation_14[0][0]              
                                                                     activation_17[0][0]              
                                                                     activation_18[0][0]              
    __________________________________________________________________________________________________
    conv2d_26 (Conv2D)              (None, 16, 16, 64)   18432       mixed1[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_22 (BatchNo (None, 16, 16, 64)   192         conv2d_26[0][0]                  
    __________________________________________________________________________________________________
    activation_22 (Activation)      (None, 16, 16, 64)   0           batch_normalization_22[0][0]     
    __________________________________________________________________________________________________
    conv2d_24 (Conv2D)              (None, 16, 16, 48)   13824       mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_27 (Conv2D)              (None, 16, 16, 96)   55296       activation_22[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_20 (BatchNo (None, 16, 16, 48)   144         conv2d_24[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_23 (BatchNo (None, 16, 16, 96)   288         conv2d_27[0][0]                  
    __________________________________________________________________________________________________
    activation_20 (Activation)      (None, 16, 16, 48)   0           batch_normalization_20[0][0]     
    __________________________________________________________________________________________________
    activation_23 (Activation)      (None, 16, 16, 96)   0           batch_normalization_23[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_2 (AveragePoo (None, 16, 16, 288)  0           mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_23 (Conv2D)              (None, 16, 16, 64)   18432       mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_25 (Conv2D)              (None, 16, 16, 64)   76800       activation_20[0][0]              
    __________________________________________________________________________________________________
    conv2d_28 (Conv2D)              (None, 16, 16, 96)   82944       activation_23[0][0]              
    __________________________________________________________________________________________________
    conv2d_29 (Conv2D)              (None, 16, 16, 64)   18432       average_pooling2d_2[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_19 (BatchNo (None, 16, 16, 64)   192         conv2d_23[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_21 (BatchNo (None, 16, 16, 64)   192         conv2d_25[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_24 (BatchNo (None, 16, 16, 96)   288         conv2d_28[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_25 (BatchNo (None, 16, 16, 64)   192         conv2d_29[0][0]                  
    __________________________________________________________________________________________________
    activation_19 (Activation)      (None, 16, 16, 64)   0           batch_normalization_19[0][0]     
    __________________________________________________________________________________________________
    activation_21 (Activation)      (None, 16, 16, 64)   0           batch_normalization_21[0][0]     
    __________________________________________________________________________________________________
    activation_24 (Activation)      (None, 16, 16, 96)   0           batch_normalization_24[0][0]     
    __________________________________________________________________________________________________
    activation_25 (Activation)      (None, 16, 16, 64)   0           batch_normalization_25[0][0]     
    __________________________________________________________________________________________________
    mixed2 (Concatenate)            (None, 16, 16, 288)  0           activation_19[0][0]              
                                                                     activation_21[0][0]              
                                                                     activation_24[0][0]              
                                                                     activation_25[0][0]              
    __________________________________________________________________________________________________
    conv2d_31 (Conv2D)              (None, 16, 16, 64)   18432       mixed2[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_27 (BatchNo (None, 16, 16, 64)   192         conv2d_31[0][0]                  
    __________________________________________________________________________________________________
    activation_27 (Activation)      (None, 16, 16, 64)   0           batch_normalization_27[0][0]     
    __________________________________________________________________________________________________
    conv2d_32 (Conv2D)              (None, 16, 16, 96)   55296       activation_27[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_28 (BatchNo (None, 16, 16, 96)   288         conv2d_32[0][0]                  
    __________________________________________________________________________________________________
    activation_28 (Activation)      (None, 16, 16, 96)   0           batch_normalization_28[0][0]     
    __________________________________________________________________________________________________
    conv2d_30 (Conv2D)              (None, 7, 7, 384)    995328      mixed2[0][0]                     
    __________________________________________________________________________________________________
    conv2d_33 (Conv2D)              (None, 7, 7, 96)     82944       activation_28[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_26 (BatchNo (None, 7, 7, 384)    1152        conv2d_30[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_29 (BatchNo (None, 7, 7, 96)     288         conv2d_33[0][0]                  
    __________________________________________________________________________________________________
    activation_26 (Activation)      (None, 7, 7, 384)    0           batch_normalization_26[0][0]     
    __________________________________________________________________________________________________
    activation_29 (Activation)      (None, 7, 7, 96)     0           batch_normalization_29[0][0]     
    __________________________________________________________________________________________________
    max_pooling2d_6 (MaxPooling2D)  (None, 7, 7, 288)    0           mixed2[0][0]                     
    __________________________________________________________________________________________________
    mixed3 (Concatenate)            (None, 7, 7, 768)    0           activation_26[0][0]              
                                                                     activation_29[0][0]              
                                                                     max_pooling2d_6[0][0]            
    __________________________________________________________________________________________________
    conv2d_38 (Conv2D)              (None, 7, 7, 128)    98304       mixed3[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_34 (BatchNo (None, 7, 7, 128)    384         conv2d_38[0][0]                  
    __________________________________________________________________________________________________
    activation_34 (Activation)      (None, 7, 7, 128)    0           batch_normalization_34[0][0]     
    __________________________________________________________________________________________________
    conv2d_39 (Conv2D)              (None, 7, 7, 128)    114688      activation_34[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_35 (BatchNo (None, 7, 7, 128)    384         conv2d_39[0][0]                  
    __________________________________________________________________________________________________
    activation_35 (Activation)      (None, 7, 7, 128)    0           batch_normalization_35[0][0]     
    __________________________________________________________________________________________________
    conv2d_35 (Conv2D)              (None, 7, 7, 128)    98304       mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_40 (Conv2D)              (None, 7, 7, 128)    114688      activation_35[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_31 (BatchNo (None, 7, 7, 128)    384         conv2d_35[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_36 (BatchNo (None, 7, 7, 128)    384         conv2d_40[0][0]                  
    __________________________________________________________________________________________________
    activation_31 (Activation)      (None, 7, 7, 128)    0           batch_normalization_31[0][0]     
    __________________________________________________________________________________________________
    activation_36 (Activation)      (None, 7, 7, 128)    0           batch_normalization_36[0][0]     
    __________________________________________________________________________________________________
    conv2d_36 (Conv2D)              (None, 7, 7, 128)    114688      activation_31[0][0]              
    __________________________________________________________________________________________________
    conv2d_41 (Conv2D)              (None, 7, 7, 128)    114688      activation_36[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_32 (BatchNo (None, 7, 7, 128)    384         conv2d_36[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_37 (BatchNo (None, 7, 7, 128)    384         conv2d_41[0][0]                  
    __________________________________________________________________________________________________
    activation_32 (Activation)      (None, 7, 7, 128)    0           batch_normalization_32[0][0]     
    __________________________________________________________________________________________________
    activation_37 (Activation)      (None, 7, 7, 128)    0           batch_normalization_37[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_3 (AveragePoo (None, 7, 7, 768)    0           mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_34 (Conv2D)              (None, 7, 7, 192)    147456      mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_37 (Conv2D)              (None, 7, 7, 192)    172032      activation_32[0][0]              
    __________________________________________________________________________________________________
    conv2d_42 (Conv2D)              (None, 7, 7, 192)    172032      activation_37[0][0]              
    __________________________________________________________________________________________________
    conv2d_43 (Conv2D)              (None, 7, 7, 192)    147456      average_pooling2d_3[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_30 (BatchNo (None, 7, 7, 192)    576         conv2d_34[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_33 (BatchNo (None, 7, 7, 192)    576         conv2d_37[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_38 (BatchNo (None, 7, 7, 192)    576         conv2d_42[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_39 (BatchNo (None, 7, 7, 192)    576         conv2d_43[0][0]                  
    __________________________________________________________________________________________________
    activation_30 (Activation)      (None, 7, 7, 192)    0           batch_normalization_30[0][0]     
    __________________________________________________________________________________________________
    activation_33 (Activation)      (None, 7, 7, 192)    0           batch_normalization_33[0][0]     
    __________________________________________________________________________________________________
    activation_38 (Activation)      (None, 7, 7, 192)    0           batch_normalization_38[0][0]     
    __________________________________________________________________________________________________
    activation_39 (Activation)      (None, 7, 7, 192)    0           batch_normalization_39[0][0]     
    __________________________________________________________________________________________________
    mixed4 (Concatenate)            (None, 7, 7, 768)    0           activation_30[0][0]              
                                                                     activation_33[0][0]              
                                                                     activation_38[0][0]              
                                                                     activation_39[0][0]              
    __________________________________________________________________________________________________
    conv2d_48 (Conv2D)              (None, 7, 7, 160)    122880      mixed4[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_44 (BatchNo (None, 7, 7, 160)    480         conv2d_48[0][0]                  
    __________________________________________________________________________________________________
    activation_44 (Activation)      (None, 7, 7, 160)    0           batch_normalization_44[0][0]     
    __________________________________________________________________________________________________
    conv2d_49 (Conv2D)              (None, 7, 7, 160)    179200      activation_44[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_45 (BatchNo (None, 7, 7, 160)    480         conv2d_49[0][0]                  
    __________________________________________________________________________________________________
    activation_45 (Activation)      (None, 7, 7, 160)    0           batch_normalization_45[0][0]     
    __________________________________________________________________________________________________
    conv2d_45 (Conv2D)              (None, 7, 7, 160)    122880      mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_50 (Conv2D)              (None, 7, 7, 160)    179200      activation_45[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_41 (BatchNo (None, 7, 7, 160)    480         conv2d_45[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_46 (BatchNo (None, 7, 7, 160)    480         conv2d_50[0][0]                  
    __________________________________________________________________________________________________
    activation_41 (Activation)      (None, 7, 7, 160)    0           batch_normalization_41[0][0]     
    __________________________________________________________________________________________________
    activation_46 (Activation)      (None, 7, 7, 160)    0           batch_normalization_46[0][0]     
    __________________________________________________________________________________________________
    conv2d_46 (Conv2D)              (None, 7, 7, 160)    179200      activation_41[0][0]              
    __________________________________________________________________________________________________
    conv2d_51 (Conv2D)              (None, 7, 7, 160)    179200      activation_46[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_42 (BatchNo (None, 7, 7, 160)    480         conv2d_46[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_47 (BatchNo (None, 7, 7, 160)    480         conv2d_51[0][0]                  
    __________________________________________________________________________________________________
    activation_42 (Activation)      (None, 7, 7, 160)    0           batch_normalization_42[0][0]     
    __________________________________________________________________________________________________
    activation_47 (Activation)      (None, 7, 7, 160)    0           batch_normalization_47[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_4 (AveragePoo (None, 7, 7, 768)    0           mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_44 (Conv2D)              (None, 7, 7, 192)    147456      mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_47 (Conv2D)              (None, 7, 7, 192)    215040      activation_42[0][0]              
    __________________________________________________________________________________________________
    conv2d_52 (Conv2D)              (None, 7, 7, 192)    215040      activation_47[0][0]              
    __________________________________________________________________________________________________
    conv2d_53 (Conv2D)              (None, 7, 7, 192)    147456      average_pooling2d_4[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_40 (BatchNo (None, 7, 7, 192)    576         conv2d_44[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_43 (BatchNo (None, 7, 7, 192)    576         conv2d_47[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_48 (BatchNo (None, 7, 7, 192)    576         conv2d_52[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_49 (BatchNo (None, 7, 7, 192)    576         conv2d_53[0][0]                  
    __________________________________________________________________________________________________
    activation_40 (Activation)      (None, 7, 7, 192)    0           batch_normalization_40[0][0]     
    __________________________________________________________________________________________________
    activation_43 (Activation)      (None, 7, 7, 192)    0           batch_normalization_43[0][0]     
    __________________________________________________________________________________________________
    activation_48 (Activation)      (None, 7, 7, 192)    0           batch_normalization_48[0][0]     
    __________________________________________________________________________________________________
    activation_49 (Activation)      (None, 7, 7, 192)    0           batch_normalization_49[0][0]     
    __________________________________________________________________________________________________
    mixed5 (Concatenate)            (None, 7, 7, 768)    0           activation_40[0][0]              
                                                                     activation_43[0][0]              
                                                                     activation_48[0][0]              
                                                                     activation_49[0][0]              
    __________________________________________________________________________________________________
    conv2d_58 (Conv2D)              (None, 7, 7, 160)    122880      mixed5[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_54 (BatchNo (None, 7, 7, 160)    480         conv2d_58[0][0]                  
    __________________________________________________________________________________________________
    activation_54 (Activation)      (None, 7, 7, 160)    0           batch_normalization_54[0][0]     
    __________________________________________________________________________________________________
    conv2d_59 (Conv2D)              (None, 7, 7, 160)    179200      activation_54[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_55 (BatchNo (None, 7, 7, 160)    480         conv2d_59[0][0]                  
    __________________________________________________________________________________________________
    activation_55 (Activation)      (None, 7, 7, 160)    0           batch_normalization_55[0][0]     
    __________________________________________________________________________________________________
    conv2d_55 (Conv2D)              (None, 7, 7, 160)    122880      mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_60 (Conv2D)              (None, 7, 7, 160)    179200      activation_55[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_51 (BatchNo (None, 7, 7, 160)    480         conv2d_55[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_56 (BatchNo (None, 7, 7, 160)    480         conv2d_60[0][0]                  
    __________________________________________________________________________________________________
    activation_51 (Activation)      (None, 7, 7, 160)    0           batch_normalization_51[0][0]     
    __________________________________________________________________________________________________
    activation_56 (Activation)      (None, 7, 7, 160)    0           batch_normalization_56[0][0]     
    __________________________________________________________________________________________________
    conv2d_56 (Conv2D)              (None, 7, 7, 160)    179200      activation_51[0][0]              
    __________________________________________________________________________________________________
    conv2d_61 (Conv2D)              (None, 7, 7, 160)    179200      activation_56[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_52 (BatchNo (None, 7, 7, 160)    480         conv2d_56[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_57 (BatchNo (None, 7, 7, 160)    480         conv2d_61[0][0]                  
    __________________________________________________________________________________________________
    activation_52 (Activation)      (None, 7, 7, 160)    0           batch_normalization_52[0][0]     
    __________________________________________________________________________________________________
    activation_57 (Activation)      (None, 7, 7, 160)    0           batch_normalization_57[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_5 (AveragePoo (None, 7, 7, 768)    0           mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_54 (Conv2D)              (None, 7, 7, 192)    147456      mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_57 (Conv2D)              (None, 7, 7, 192)    215040      activation_52[0][0]              
    __________________________________________________________________________________________________
    conv2d_62 (Conv2D)              (None, 7, 7, 192)    215040      activation_57[0][0]              
    __________________________________________________________________________________________________
    conv2d_63 (Conv2D)              (None, 7, 7, 192)    147456      average_pooling2d_5[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_50 (BatchNo (None, 7, 7, 192)    576         conv2d_54[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_53 (BatchNo (None, 7, 7, 192)    576         conv2d_57[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_58 (BatchNo (None, 7, 7, 192)    576         conv2d_62[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_59 (BatchNo (None, 7, 7, 192)    576         conv2d_63[0][0]                  
    __________________________________________________________________________________________________
    activation_50 (Activation)      (None, 7, 7, 192)    0           batch_normalization_50[0][0]     
    __________________________________________________________________________________________________
    activation_53 (Activation)      (None, 7, 7, 192)    0           batch_normalization_53[0][0]     
    __________________________________________________________________________________________________
    activation_58 (Activation)      (None, 7, 7, 192)    0           batch_normalization_58[0][0]     
    __________________________________________________________________________________________________
    activation_59 (Activation)      (None, 7, 7, 192)    0           batch_normalization_59[0][0]     
    __________________________________________________________________________________________________
    mixed6 (Concatenate)            (None, 7, 7, 768)    0           activation_50[0][0]              
                                                                     activation_53[0][0]              
                                                                     activation_58[0][0]              
                                                                     activation_59[0][0]              
    __________________________________________________________________________________________________
    conv2d_68 (Conv2D)              (None, 7, 7, 192)    147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_64 (BatchNo (None, 7, 7, 192)    576         conv2d_68[0][0]                  
    __________________________________________________________________________________________________
    activation_64 (Activation)      (None, 7, 7, 192)    0           batch_normalization_64[0][0]     
    __________________________________________________________________________________________________
    conv2d_69 (Conv2D)              (None, 7, 7, 192)    258048      activation_64[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_65 (BatchNo (None, 7, 7, 192)    576         conv2d_69[0][0]                  
    __________________________________________________________________________________________________
    activation_65 (Activation)      (None, 7, 7, 192)    0           batch_normalization_65[0][0]     
    __________________________________________________________________________________________________
    conv2d_65 (Conv2D)              (None, 7, 7, 192)    147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_70 (Conv2D)              (None, 7, 7, 192)    258048      activation_65[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_61 (BatchNo (None, 7, 7, 192)    576         conv2d_65[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_66 (BatchNo (None, 7, 7, 192)    576         conv2d_70[0][0]                  
    __________________________________________________________________________________________________
    activation_61 (Activation)      (None, 7, 7, 192)    0           batch_normalization_61[0][0]     
    __________________________________________________________________________________________________
    activation_66 (Activation)      (None, 7, 7, 192)    0           batch_normalization_66[0][0]     
    __________________________________________________________________________________________________
    conv2d_66 (Conv2D)              (None, 7, 7, 192)    258048      activation_61[0][0]              
    __________________________________________________________________________________________________
    conv2d_71 (Conv2D)              (None, 7, 7, 192)    258048      activation_66[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_62 (BatchNo (None, 7, 7, 192)    576         conv2d_66[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_67 (BatchNo (None, 7, 7, 192)    576         conv2d_71[0][0]                  
    __________________________________________________________________________________________________
    activation_62 (Activation)      (None, 7, 7, 192)    0           batch_normalization_62[0][0]     
    __________________________________________________________________________________________________
    activation_67 (Activation)      (None, 7, 7, 192)    0           batch_normalization_67[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_6 (AveragePoo (None, 7, 7, 768)    0           mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_64 (Conv2D)              (None, 7, 7, 192)    147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_67 (Conv2D)              (None, 7, 7, 192)    258048      activation_62[0][0]              
    __________________________________________________________________________________________________
    conv2d_72 (Conv2D)              (None, 7, 7, 192)    258048      activation_67[0][0]              
    __________________________________________________________________________________________________
    conv2d_73 (Conv2D)              (None, 7, 7, 192)    147456      average_pooling2d_6[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_60 (BatchNo (None, 7, 7, 192)    576         conv2d_64[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_63 (BatchNo (None, 7, 7, 192)    576         conv2d_67[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_68 (BatchNo (None, 7, 7, 192)    576         conv2d_72[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_69 (BatchNo (None, 7, 7, 192)    576         conv2d_73[0][0]                  
    __________________________________________________________________________________________________
    activation_60 (Activation)      (None, 7, 7, 192)    0           batch_normalization_60[0][0]     
    __________________________________________________________________________________________________
    activation_63 (Activation)      (None, 7, 7, 192)    0           batch_normalization_63[0][0]     
    __________________________________________________________________________________________________
    activation_68 (Activation)      (None, 7, 7, 192)    0           batch_normalization_68[0][0]     
    __________________________________________________________________________________________________
    activation_69 (Activation)      (None, 7, 7, 192)    0           batch_normalization_69[0][0]     
    __________________________________________________________________________________________________
    mixed7 (Concatenate)            (None, 7, 7, 768)    0           activation_60[0][0]              
                                                                     activation_63[0][0]              
                                                                     activation_68[0][0]              
                                                                     activation_69[0][0]              
    __________________________________________________________________________________________________
    conv2d_76 (Conv2D)              (None, 7, 7, 192)    147456      mixed7[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_72 (BatchNo (None, 7, 7, 192)    576         conv2d_76[0][0]                  
    __________________________________________________________________________________________________
    activation_72 (Activation)      (None, 7, 7, 192)    0           batch_normalization_72[0][0]     
    __________________________________________________________________________________________________
    conv2d_77 (Conv2D)              (None, 7, 7, 192)    258048      activation_72[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_73 (BatchNo (None, 7, 7, 192)    576         conv2d_77[0][0]                  
    __________________________________________________________________________________________________
    activation_73 (Activation)      (None, 7, 7, 192)    0           batch_normalization_73[0][0]     
    __________________________________________________________________________________________________
    conv2d_74 (Conv2D)              (None, 7, 7, 192)    147456      mixed7[0][0]                     
    __________________________________________________________________________________________________
    conv2d_78 (Conv2D)              (None, 7, 7, 192)    258048      activation_73[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_70 (BatchNo (None, 7, 7, 192)    576         conv2d_74[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_74 (BatchNo (None, 7, 7, 192)    576         conv2d_78[0][0]                  
    __________________________________________________________________________________________________
    activation_70 (Activation)      (None, 7, 7, 192)    0           batch_normalization_70[0][0]     
    __________________________________________________________________________________________________
    activation_74 (Activation)      (None, 7, 7, 192)    0           batch_normalization_74[0][0]     
    __________________________________________________________________________________________________
    conv2d_75 (Conv2D)              (None, 3, 3, 320)    552960      activation_70[0][0]              
    __________________________________________________________________________________________________
    conv2d_79 (Conv2D)              (None, 3, 3, 192)    331776      activation_74[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_71 (BatchNo (None, 3, 3, 320)    960         conv2d_75[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_75 (BatchNo (None, 3, 3, 192)    576         conv2d_79[0][0]                  
    __________________________________________________________________________________________________
    activation_71 (Activation)      (None, 3, 3, 320)    0           batch_normalization_71[0][0]     
    __________________________________________________________________________________________________
    activation_75 (Activation)      (None, 3, 3, 192)    0           batch_normalization_75[0][0]     
    __________________________________________________________________________________________________
    max_pooling2d_7 (MaxPooling2D)  (None, 3, 3, 768)    0           mixed7[0][0]                     
    __________________________________________________________________________________________________
    mixed8 (Concatenate)            (None, 3, 3, 1280)   0           activation_71[0][0]              
                                                                     activation_75[0][0]              
                                                                     max_pooling2d_7[0][0]            
    __________________________________________________________________________________________________
    conv2d_84 (Conv2D)              (None, 3, 3, 448)    573440      mixed8[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_80 (BatchNo (None, 3, 3, 448)    1344        conv2d_84[0][0]                  
    __________________________________________________________________________________________________
    activation_80 (Activation)      (None, 3, 3, 448)    0           batch_normalization_80[0][0]     
    __________________________________________________________________________________________________
    conv2d_81 (Conv2D)              (None, 3, 3, 384)    491520      mixed8[0][0]                     
    __________________________________________________________________________________________________
    conv2d_85 (Conv2D)              (None, 3, 3, 384)    1548288     activation_80[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_77 (BatchNo (None, 3, 3, 384)    1152        conv2d_81[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_81 (BatchNo (None, 3, 3, 384)    1152        conv2d_85[0][0]                  
    __________________________________________________________________________________________________
    activation_77 (Activation)      (None, 3, 3, 384)    0           batch_normalization_77[0][0]     
    __________________________________________________________________________________________________
    activation_81 (Activation)      (None, 3, 3, 384)    0           batch_normalization_81[0][0]     
    __________________________________________________________________________________________________
    conv2d_82 (Conv2D)              (None, 3, 3, 384)    442368      activation_77[0][0]              
    __________________________________________________________________________________________________
    conv2d_83 (Conv2D)              (None, 3, 3, 384)    442368      activation_77[0][0]              
    __________________________________________________________________________________________________
    conv2d_86 (Conv2D)              (None, 3, 3, 384)    442368      activation_81[0][0]              
    __________________________________________________________________________________________________
    conv2d_87 (Conv2D)              (None, 3, 3, 384)    442368      activation_81[0][0]              
    __________________________________________________________________________________________________
    average_pooling2d_7 (AveragePoo (None, 3, 3, 1280)   0           mixed8[0][0]                     
    __________________________________________________________________________________________________
    conv2d_80 (Conv2D)              (None, 3, 3, 320)    409600      mixed8[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_78 (BatchNo (None, 3, 3, 384)    1152        conv2d_82[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_79 (BatchNo (None, 3, 3, 384)    1152        conv2d_83[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_82 (BatchNo (None, 3, 3, 384)    1152        conv2d_86[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_83 (BatchNo (None, 3, 3, 384)    1152        conv2d_87[0][0]                  
    __________________________________________________________________________________________________
    conv2d_88 (Conv2D)              (None, 3, 3, 192)    245760      average_pooling2d_7[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_76 (BatchNo (None, 3, 3, 320)    960         conv2d_80[0][0]                  
    __________________________________________________________________________________________________
    activation_78 (Activation)      (None, 3, 3, 384)    0           batch_normalization_78[0][0]     
    __________________________________________________________________________________________________
    activation_79 (Activation)      (None, 3, 3, 384)    0           batch_normalization_79[0][0]     
    __________________________________________________________________________________________________
    activation_82 (Activation)      (None, 3, 3, 384)    0           batch_normalization_82[0][0]     
    __________________________________________________________________________________________________
    activation_83 (Activation)      (None, 3, 3, 384)    0           batch_normalization_83[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_84 (BatchNo (None, 3, 3, 192)    576         conv2d_88[0][0]                  
    __________________________________________________________________________________________________
    activation_76 (Activation)      (None, 3, 3, 320)    0           batch_normalization_76[0][0]     
    __________________________________________________________________________________________________
    mixed9_0 (Concatenate)          (None, 3, 3, 768)    0           activation_78[0][0]              
                                                                     activation_79[0][0]              
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 3, 3, 768)    0           activation_82[0][0]              
                                                                     activation_83[0][0]              
    __________________________________________________________________________________________________
    activation_84 (Activation)      (None, 3, 3, 192)    0           batch_normalization_84[0][0]     
    __________________________________________________________________________________________________
    mixed9 (Concatenate)            (None, 3, 3, 2048)   0           activation_76[0][0]              
                                                                     mixed9_0[0][0]                   
                                                                     concatenate[0][0]                
                                                                     activation_84[0][0]              
    __________________________________________________________________________________________________
    conv2d_93 (Conv2D)              (None, 3, 3, 448)    917504      mixed9[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_89 (BatchNo (None, 3, 3, 448)    1344        conv2d_93[0][0]                  
    __________________________________________________________________________________________________
    activation_89 (Activation)      (None, 3, 3, 448)    0           batch_normalization_89[0][0]     
    __________________________________________________________________________________________________
    conv2d_90 (Conv2D)              (None, 3, 3, 384)    786432      mixed9[0][0]                     
    __________________________________________________________________________________________________
    conv2d_94 (Conv2D)              (None, 3, 3, 384)    1548288     activation_89[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_86 (BatchNo (None, 3, 3, 384)    1152        conv2d_90[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_90 (BatchNo (None, 3, 3, 384)    1152        conv2d_94[0][0]                  
    __________________________________________________________________________________________________
    activation_86 (Activation)      (None, 3, 3, 384)    0           batch_normalization_86[0][0]     
    __________________________________________________________________________________________________
    activation_90 (Activation)      (None, 3, 3, 384)    0           batch_normalization_90[0][0]     
    __________________________________________________________________________________________________
    conv2d_91 (Conv2D)              (None, 3, 3, 384)    442368      activation_86[0][0]              
    __________________________________________________________________________________________________
    conv2d_92 (Conv2D)              (None, 3, 3, 384)    442368      activation_86[0][0]              
    __________________________________________________________________________________________________
    conv2d_95 (Conv2D)              (None, 3, 3, 384)    442368      activation_90[0][0]              
    __________________________________________________________________________________________________
    conv2d_96 (Conv2D)              (None, 3, 3, 384)    442368      activation_90[0][0]              
    __________________________________________________________________________________________________
    average_pooling2d_8 (AveragePoo (None, 3, 3, 2048)   0           mixed9[0][0]                     
    __________________________________________________________________________________________________
    conv2d_89 (Conv2D)              (None, 3, 3, 320)    655360      mixed9[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_87 (BatchNo (None, 3, 3, 384)    1152        conv2d_91[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_88 (BatchNo (None, 3, 3, 384)    1152        conv2d_92[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_91 (BatchNo (None, 3, 3, 384)    1152        conv2d_95[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_92 (BatchNo (None, 3, 3, 384)    1152        conv2d_96[0][0]                  
    __________________________________________________________________________________________________
    conv2d_97 (Conv2D)              (None, 3, 3, 192)    393216      average_pooling2d_8[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_85 (BatchNo (None, 3, 3, 320)    960         conv2d_89[0][0]                  
    __________________________________________________________________________________________________
    activation_87 (Activation)      (None, 3, 3, 384)    0           batch_normalization_87[0][0]     
    __________________________________________________________________________________________________
    activation_88 (Activation)      (None, 3, 3, 384)    0           batch_normalization_88[0][0]     
    __________________________________________________________________________________________________
    activation_91 (Activation)      (None, 3, 3, 384)    0           batch_normalization_91[0][0]     
    __________________________________________________________________________________________________
    activation_92 (Activation)      (None, 3, 3, 384)    0           batch_normalization_92[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_93 (BatchNo (None, 3, 3, 192)    576         conv2d_97[0][0]                  
    __________________________________________________________________________________________________
    activation_85 (Activation)      (None, 3, 3, 320)    0           batch_normalization_85[0][0]     
    __________________________________________________________________________________________________
    mixed9_1 (Concatenate)          (None, 3, 3, 768)    0           activation_87[0][0]              
                                                                     activation_88[0][0]              
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 3, 3, 768)    0           activation_91[0][0]              
                                                                     activation_92[0][0]              
    __________________________________________________________________________________________________
    activation_93 (Activation)      (None, 3, 3, 192)    0           batch_normalization_93[0][0]     
    __________________________________________________________________________________________________
    mixed10 (Concatenate)           (None, 3, 3, 2048)   0           activation_85[0][0]              
                                                                     mixed9_1[0][0]                   
                                                                     concatenate_1[0][0]              
                                                                     activation_93[0][0]              
    ==================================================================================================
    Total params: 21,802,784
    Trainable params: 0
    Non-trainable params: 21,802,784
    __________________________________________________________________________________________________
    last layer output shape:  (None, 7, 7, 768)
    

Looking at the summary, the bottom layers have concoluted to 3 X 3. But I want to use something with a little more information, so I'll move up the model description to find `mixed7`, which is the output of a lot of convolutions that are 7 X 7. We can experiment with other layers as well.  
We'll define our new model, taking the output from the inception model's mixed7 layer, which we call `last_output`.


```python
# flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)

# add a fully connected layer with 1,024 hidden units and ReLu activation
x = layers.Dense(1024, activation='relu')(x)

# add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)

# add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer= RMSprop(lr=0.0001),
              loss= 'binary_crossentropy',
              metrics= ['accuracy'])
```


```python
history = model.fit(train_generator,
                    epochs=25,
                    verbose=1,
                    validation_data=validation_generator)
```

    Epoch 1/25
    148/352 [===========>..................] - ETA: 1:54 - loss: 0.2809 - accuracy: 0.8917

    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:770: UserWarning: Possibly corrupt EXIF data.  Expecting to read 32 bytes but only got 0. Skipping tag 270
      " Skipping tag %s" % (size, len(data), tag)
    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:770: UserWarning: Possibly corrupt EXIF data.  Expecting to read 5 bytes but only got 0. Skipping tag 271
      " Skipping tag %s" % (size, len(data), tag)
    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:770: UserWarning: Possibly corrupt EXIF data.  Expecting to read 8 bytes but only got 0. Skipping tag 272
      " Skipping tag %s" % (size, len(data), tag)
    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:770: UserWarning: Possibly corrupt EXIF data.  Expecting to read 8 bytes but only got 0. Skipping tag 282
      " Skipping tag %s" % (size, len(data), tag)
    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:770: UserWarning: Possibly corrupt EXIF data.  Expecting to read 8 bytes but only got 0. Skipping tag 283
      " Skipping tag %s" % (size, len(data), tag)
    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:770: UserWarning: Possibly corrupt EXIF data.  Expecting to read 20 bytes but only got 0. Skipping tag 306
      " Skipping tag %s" % (size, len(data), tag)
    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:770: UserWarning: Possibly corrupt EXIF data.  Expecting to read 48 bytes but only got 0. Skipping tag 532
      " Skipping tag %s" % (size, len(data), tag)
    /usr/local/lib/python3.6/dist-packages/PIL/TiffImagePlugin.py:788: UserWarning: Corrupt EXIF data.  Expecting to read 2 bytes but only got 0. 
      warnings.warn(str(msg))
    

    352/352 [==============================] - 222s 631ms/step - loss: 0.2235 - accuracy: 0.9106 - val_loss: 0.1468 - val_accuracy: 0.9388
    Epoch 2/25
    352/352 [==============================] - 220s 625ms/step - loss: 0.1557 - accuracy: 0.9374 - val_loss: 0.1307 - val_accuracy: 0.9440
    Epoch 3/25
    352/352 [==============================] - 220s 624ms/step - loss: 0.1486 - accuracy: 0.9416 - val_loss: 0.1297 - val_accuracy: 0.9440
    Epoch 4/25
    352/352 [==============================] - 219s 622ms/step - loss: 0.1422 - accuracy: 0.9454 - val_loss: 0.1672 - val_accuracy: 0.9364
    Epoch 5/25
    352/352 [==============================] - 219s 621ms/step - loss: 0.1366 - accuracy: 0.9480 - val_loss: 0.1397 - val_accuracy: 0.9556
    Epoch 6/25
    352/352 [==============================] - 219s 621ms/step - loss: 0.1290 - accuracy: 0.9520 - val_loss: 0.1330 - val_accuracy: 0.9476
    Epoch 7/25
    352/352 [==============================] - 219s 621ms/step - loss: 0.1320 - accuracy: 0.9500 - val_loss: 0.1185 - val_accuracy: 0.9560
    Epoch 8/25
    352/352 [==============================] - 218s 618ms/step - loss: 0.1285 - accuracy: 0.9534 - val_loss: 0.1308 - val_accuracy: 0.9500
    Epoch 9/25
    352/352 [==============================] - 218s 621ms/step - loss: 0.1248 - accuracy: 0.9541 - val_loss: 0.1357 - val_accuracy: 0.9512
    Epoch 10/25
    352/352 [==============================] - 218s 619ms/step - loss: 0.1235 - accuracy: 0.9556 - val_loss: 0.1690 - val_accuracy: 0.9408
    Epoch 11/25
    352/352 [==============================] - 218s 620ms/step - loss: 0.1193 - accuracy: 0.9576 - val_loss: 0.1227 - val_accuracy: 0.9596
    Epoch 12/25
    352/352 [==============================] - 217s 617ms/step - loss: 0.1215 - accuracy: 0.9553 - val_loss: 0.1076 - val_accuracy: 0.9604
    Epoch 13/25
    352/352 [==============================] - 219s 622ms/step - loss: 0.1196 - accuracy: 0.9576 - val_loss: 0.1097 - val_accuracy: 0.9564
    Epoch 14/25
    352/352 [==============================] - 217s 618ms/step - loss: 0.1159 - accuracy: 0.9588 - val_loss: 0.1196 - val_accuracy: 0.9616
    Epoch 15/25
    352/352 [==============================] - 219s 623ms/step - loss: 0.1203 - accuracy: 0.9568 - val_loss: 0.1336 - val_accuracy: 0.9504
    Epoch 16/25
    352/352 [==============================] - 219s 623ms/step - loss: 0.1156 - accuracy: 0.9597 - val_loss: 0.1120 - val_accuracy: 0.9548
    Epoch 17/25
    352/352 [==============================] - 218s 620ms/step - loss: 0.1111 - accuracy: 0.9612 - val_loss: 0.1228 - val_accuracy: 0.9592
    Epoch 18/25
    352/352 [==============================] - 217s 616ms/step - loss: 0.1116 - accuracy: 0.9596 - val_loss: 0.1395 - val_accuracy: 0.9516
    Epoch 19/25
    352/352 [==============================] - 220s 626ms/step - loss: 0.1096 - accuracy: 0.9605 - val_loss: 0.1209 - val_accuracy: 0.9548
    Epoch 20/25
    352/352 [==============================] - 219s 622ms/step - loss: 0.1113 - accuracy: 0.9613 - val_loss: 0.1215 - val_accuracy: 0.9512
    Epoch 21/25
    352/352 [==============================] - 220s 626ms/step - loss: 0.1100 - accuracy: 0.9604 - val_loss: 0.1177 - val_accuracy: 0.9588
    Epoch 22/25
    352/352 [==============================] - 221s 629ms/step - loss: 0.1067 - accuracy: 0.9624 - val_loss: 0.1397 - val_accuracy: 0.9528
    Epoch 23/25
    352/352 [==============================] - 221s 628ms/step - loss: 0.1090 - accuracy: 0.9601 - val_loss: 0.1139 - val_accuracy: 0.9576
    Epoch 24/25
    352/352 [==============================] - 221s 628ms/step - loss: 0.1053 - accuracy: 0.9610 - val_loss: 0.1184 - val_accuracy: 0.9580
    Epoch 25/25
    352/352 [==============================] - 221s 627ms/step - loss: 0.1036 - accuracy: 0.9635 - val_loss: 0.1160 - val_accuracy: 0.9572
    

After 25 epochs, the *training loss* = 0.1 and *training accuracy* = 0.9, while the *validation loss* = 0.11 and *validation accuracy* = 0.95.  
The model has performed much better after applying transfer learning as compared to the CNN we used earlier.


## Evaluating accuracy and loss for the model
Let's plot the training/validation accuracy and loss, during training.


```python
# PLOT LOSS AND ACCURACY
%matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')

# Desired output. Charts with training and validation metrics. No crash :)
```




    Text(0.5, 1.0, 'Training and validation loss')




![png](output_38_1.png)



![png](output_38_2.png)


Here we can see that the validation loss and accuracy lines are much smoother for this model as compared to the earlier one. The loss values are lower, while the accuracy values are higher.  

Hence, we can conclude that the model that uses transfer learning classifies cats and dogs much better than the CNN.
