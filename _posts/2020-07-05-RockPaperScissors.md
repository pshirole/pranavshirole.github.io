---
layout: post
title: Multiclass Classifications using CNN and Tensorflow
tags: [computer vision, tensorflow, cnn]
---

In this blog post, we will identify whether an image is Rock, Paper or Scissors.

## The dataset
Rock Paper Scissors is a dataset containing about 3,000 computer-generated images from a variety of different hands, from different races, ages and genders, posed into Rock, Paper or Scissors and labelled as such. Each image is 300 X 300 pixels in 24-bit color. The images have all been generated using CGI techniques as an experiment in determining if a CGI-based dataset can be used for classification against real images. You can download the dataset [here](http://www.laurencemoroney.com/rock-paper-scissors-dataset/).


```python
# download the training and test set zip files

!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip \
    -O /tmp/rps.zip
  
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip \
    -O /tmp/rps-test-set.zip
```

    --2020-09-07 06:12:21--  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip
    Resolving storage.googleapis.com (storage.googleapis.com)... 172.217.212.128, 172.217.214.128, 108.177.111.128, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|172.217.212.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 200682221 (191M) [application/zip]
    Saving to: ‘/tmp/rps.zip’
    
    /tmp/rps.zip        100%[===================>] 191.38M   111MB/s    in 1.7s    
    
    2020-09-07 06:12:22 (111 MB/s) - ‘/tmp/rps.zip’ saved [200682221/200682221]
    
    --2020-09-07 06:12:22--  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip
    Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.124.128, 172.217.212.128, 172.217.214.128, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.124.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 29516758 (28M) [application/zip]
    Saving to: ‘/tmp/rps-test-set.zip’
    
    /tmp/rps-test-set.z 100%[===================>]  28.15M   169MB/s    in 0.2s    
    
    2020-09-07 06:12:23 (169 MB/s) - ‘/tmp/rps-test-set.zip’ saved [29516758/29516758]
    
    


```python
import os
import zipfile

# unzip the data into the tmp directory
local_zip = '/tmp/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()

local_zip = '/tmp/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()
```


```python
# subdirectories
rock_dir = os.path.join('/tmp/rps/rock')
paper_dir = os.path.join('/tmp/rps/paper')
scissors_dir = os.path.join('/tmp/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])
```

    total training rock images: 840
    total training paper images: 840
    total training scissors images: 840
    ['rock02-110.png', 'rock02-061.png', 'rock04-062.png', 'rock04-058.png', 'rock03-011.png', 'rock01-064.png', 'rock03-090.png', 'rock04-049.png', 'rock06ck02-062.png', 'rock06ck02-000.png']
    ['paper02-028.png', 'paper03-019.png', 'paper05-007.png', 'paper07-113.png', 'paper07-104.png', 'paper01-027.png', 'paper06-062.png', 'paper05-011.png', 'paper01-101.png', 'paper02-004.png']
    ['scissors04-059.png', 'scissors04-078.png', 'scissors01-029.png', 'scissors02-112.png', 'scissors04-074.png', 'testscissors02-068.png', 'scissors01-089.png', 'scissors04-041.png', 'scissors01-099.png', 'scissors01-082.png']
    

There are 840 images of each class.  
Let's see some examples of the images.


```python
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_rock = [os.path.join(rock_dir, fname) 
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) 
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()
```

    /tmp/rps/rock/rock02-110.png
    
    
<img src="/assets/img/computer_vision/rock_paper_scissors/output_5_1.png">


    /tmp/rps/rock/rock02-061.png
    


<img src="/assets/img/computer_vision/rock_paper_scissors/output_5_3.png">


    /tmp/rps/paper/paper02-028.png
    

<img src="/assets/img/computer_vision/rock_paper_scissors/output_5_5.png">


    /tmp/rps/paper/paper03-019.png
    

<img src="/assets/img/computer_vision/rock_paper_scissors/output_5_7.png">


    /tmp/rps/scissors/scissors04-059.png
    

<img src="/assets/img/computer_vision/rock_paper_scissors/output_5_9.png">


    /tmp/rps/scissors/scissors04-078.png
    

<img src="/assets/img/computer_vision/rock_paper_scissors/output_5_11.png">


## Data preprocessing

```python
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

# set up the image generator
TRAINING_DIR = "/tmp/rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "/tmp/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

```

## Modeling

```python

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # the second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # the third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # the fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("rps.h5")

```

    Found 2520 images belonging to 3 classes.
    Found 372 images belonging to 3 classes.
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 148, 148, 64)      1792      
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 74, 74, 64)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 72, 72, 64)        36928     
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
    dropout (Dropout)            (None, 6272)              0         
    _________________________________________________________________
    dense (Dense)                (None, 512)               3211776   
    _________________________________________________________________
    dense_1 (Dense)              (None, 3)                 1539      
    =================================================================
    Total params: 3,473,475
    Trainable params: 3,473,475
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/25
     2/20 [==>...........................] - ETA: 1s - loss: 5.8507 - accuracy: 0.3452WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0479s vs `on_train_batch_end` time: 0.1123s). Check your callbacks.
    20/20 [==============================] - 21s 1s/step - loss: 1.6690 - accuracy: 0.3623 - val_loss: 1.0931 - val_accuracy: 0.4919
    Epoch 2/25
    20/20 [==============================] - 21s 1s/step - loss: 1.1116 - accuracy: 0.4052 - val_loss: 1.0942 - val_accuracy: 0.3468
    Epoch 3/25
    20/20 [==============================] - 21s 1s/step - loss: 1.0647 - accuracy: 0.4385 - val_loss: 0.9382 - val_accuracy: 0.6129
    Epoch 4/25
    20/20 [==============================] - 21s 1s/step - loss: 0.9481 - accuracy: 0.5214 - val_loss: 0.9023 - val_accuracy: 0.8011
    Epoch 5/25
    20/20 [==============================] - 21s 1s/step - loss: 0.8231 - accuracy: 0.6290 - val_loss: 0.5388 - val_accuracy: 0.6371
    Epoch 6/25
    20/20 [==============================] - 21s 1s/step - loss: 0.6919 - accuracy: 0.6821 - val_loss: 0.6611 - val_accuracy: 0.5565
    Epoch 7/25
    20/20 [==============================] - 21s 1s/step - loss: 0.5531 - accuracy: 0.7726 - val_loss: 0.5019 - val_accuracy: 0.7177
    Epoch 8/25
    20/20 [==============================] - 21s 1s/step - loss: 0.4280 - accuracy: 0.8310 - val_loss: 0.1039 - val_accuracy: 1.0000
    Epoch 9/25
    20/20 [==============================] - 22s 1s/step - loss: 0.4344 - accuracy: 0.8179 - val_loss: 0.3247 - val_accuracy: 0.8522
    Epoch 10/25
    20/20 [==============================] - 21s 1s/step - loss: 0.4032 - accuracy: 0.8504 - val_loss: 0.0396 - val_accuracy: 1.0000
    Epoch 11/25
    20/20 [==============================] - 21s 1s/step - loss: 0.2661 - accuracy: 0.9052 - val_loss: 0.0393 - val_accuracy: 1.0000
    Epoch 12/25
    20/20 [==============================] - 21s 1s/step - loss: 0.2516 - accuracy: 0.8964 - val_loss: 0.1177 - val_accuracy: 0.9597
    Epoch 13/25
    20/20 [==============================] - 21s 1s/step - loss: 0.1619 - accuracy: 0.9421 - val_loss: 0.1453 - val_accuracy: 0.9543
    Epoch 14/25
    20/20 [==============================] - 21s 1s/step - loss: 0.1787 - accuracy: 0.9361 - val_loss: 0.0986 - val_accuracy: 0.9758
    Epoch 15/25
    20/20 [==============================] - 21s 1s/step - loss: 0.2301 - accuracy: 0.9119 - val_loss: 0.2302 - val_accuracy: 0.8844
    Epoch 16/25
    20/20 [==============================] - 21s 1s/step - loss: 0.1690 - accuracy: 0.9417 - val_loss: 0.0461 - val_accuracy: 0.9973
    Epoch 17/25
    20/20 [==============================] - 21s 1s/step - loss: 0.0809 - accuracy: 0.9718 - val_loss: 0.0404 - val_accuracy: 0.9812
    Epoch 18/25
    20/20 [==============================] - 21s 1s/step - loss: 0.2039 - accuracy: 0.9286 - val_loss: 0.0465 - val_accuracy: 0.9839
    Epoch 19/25
    20/20 [==============================] - 21s 1s/step - loss: 0.1677 - accuracy: 0.9421 - val_loss: 0.0850 - val_accuracy: 0.9570
    Epoch 20/25
    20/20 [==============================] - 21s 1s/step - loss: 0.0885 - accuracy: 0.9702 - val_loss: 0.0294 - val_accuracy: 0.9946
    Epoch 21/25
    20/20 [==============================] - 21s 1s/step - loss: 0.1087 - accuracy: 0.9627 - val_loss: 0.0159 - val_accuracy: 0.9946
    Epoch 22/25
    20/20 [==============================] - 21s 1s/step - loss: 0.1243 - accuracy: 0.9623 - val_loss: 0.0410 - val_accuracy: 0.9866
    Epoch 23/25
    20/20 [==============================] - 21s 1s/step - loss: 0.0897 - accuracy: 0.9683 - val_loss: 0.0441 - val_accuracy: 0.9839
    Epoch 24/25
    20/20 [==============================] - 21s 1s/step - loss: 0.1056 - accuracy: 0.9643 - val_loss: 0.0088 - val_accuracy: 1.0000
    Epoch 25/25
    20/20 [==============================] - 21s 1s/step - loss: 0.1093 - accuracy: 0.9607 - val_loss: 0.0381 - val_accuracy: 0.9919
    


```python
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()
```


![png](output_7_0.png)



    <Figure size 432x288 with 0 Axes>


We can see that the training accuracy improved over time and trends towards 1.0. The validation accuracy was unstable in the beginning but has a value between 0.9 and 1.0 over time.  

## Model evaluation
Let us now test the model with some images that it hasn't previously seen. This new dataset with 33 images can be downloaded [here](https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-validation.zip).


```python
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = fn
  img = image.load_img(path, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(fn)
  print(classes)
```



<input type="file" id="files-43a11596-8e2c-4960-99b9-8824c1fcde99" name="files[]" multiple disabled
   style="border:none" />
<output id="result-43a11596-8e2c-4960-99b9-8824c1fcde99">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 


    Saving paper1.png to paper1.png
    Saving paper2.png to paper2.png
    Saving paper3.png to paper3.png
    Saving paper4.png to paper4.png
    Saving paper5.png to paper5.png
    Saving paper6.png to paper6.png
    Saving paper7.png to paper7.png
    Saving paper8.png to paper8.png
    Saving paper9.png to paper9.png
    Saving paper-hires1.png to paper-hires1.png
    Saving paper-hires2.png to paper-hires2.png
    Saving rock1.png to rock1.png
    Saving rock2.png to rock2.png
    Saving rock3.png to rock3.png
    Saving rock4.png to rock4.png
    Saving rock5.png to rock5.png
    Saving rock6.png to rock6.png
    Saving rock7.png to rock7.png
    Saving rock8.png to rock8.png
    Saving rock9.png to rock9.png
    Saving rock-hires1.png to rock-hires1.png
    Saving rock-hires2.png to rock-hires2.png
    Saving scissors1.png to scissors1.png
    Saving scissors2.png to scissors2.png
    Saving scissors3.png to scissors3.png
    Saving scissors4.png to scissors4.png
    Saving scissors5.png to scissors5.png
    Saving scissors6.png to scissors6.png
    Saving scissors7.png to scissors7.png
    Saving scissors8.png to scissors8.png
    Saving scissors9.png to scissors9.png
    Saving scissors-hires1.png to scissors-hires1.png
    Saving scissors-hires2.png to scissors-hires2.png
    paper1.png
    [[1. 0. 0.]]
    paper2.png
    [[1. 0. 0.]]
    paper3.png
    [[1. 0. 0.]]
    paper4.png
    [[1. 0. 0.]]
    paper5.png
    [[1. 0. 0.]]
    paper6.png
    [[1. 0. 0.]]
    paper7.png
    [[1. 0. 0.]]
    paper8.png
    [[1. 0. 0.]]
    paper9.png
    [[0. 1. 0.]]
    paper-hires1.png
    [[1. 0. 0.]]
    paper-hires2.png
    [[1. 0. 0.]]
    rock1.png
    [[0. 1. 0.]]
    rock2.png
    [[0. 1. 0.]]
    rock3.png
    [[0. 1. 0.]]
    rock4.png
    [[0. 1. 0.]]
    rock5.png
    [[0. 1. 0.]]
    rock6.png
    [[0. 1. 0.]]
    rock7.png
    [[0. 1. 0.]]
    rock8.png
    [[0. 1. 0.]]
    rock9.png
    [[0. 1. 0.]]
    rock-hires1.png
    [[0. 1. 0.]]
    rock-hires2.png
    [[0. 1. 0.]]
    scissors1.png
    [[0. 0. 1.]]
    scissors2.png
    [[0. 0. 1.]]
    scissors3.png
    [[0. 0. 1.]]
    scissors4.png
    [[0. 0. 1.]]
    scissors5.png
    [[0. 0. 1.]]
    scissors6.png
    [[0. 0. 1.]]
    scissors7.png
    [[0. 0. 1.]]
    scissors8.png
    [[0. 0. 1.]]
    scissors9.png
    [[0. 0. 1.]]
    scissors-hires1.png
    [[0. 0. 1.]]
    scissors-hires2.png
    [[0. 0. 1.]]
    

When using the image generator, the classes come from directories and thus were sorted in alphabetical order. So the first value is for paper, then rock, and then scissors.  
The model guessed 32 out of 33 images correctly. It got only 1 image wrong, the `paper9.png`. If you download the images for yourself, you can see why the model got confused.  
We can conclude that the model is highly accurate.
