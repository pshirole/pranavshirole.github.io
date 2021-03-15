---
layout: post
title: Predicting text for a Harry Potter book
tags: [computer vision, pytorch, fastai, classification]
---

## Classifying Pet Breeds 
In this blog post, we'll try to figure out what breed of pet is shown in each image of a dataset.  
The dataset we'll be using is the Oxford-IIIT Pet dataset. It's a 37 category pet dataset with roughly 200 images for each class. The images have a large variations in scale, pose and lighting. The dataset can be [downloaded here](https://www.robots.ox.ac.uk/~vgg/data/pets/).

First, let's install fastai and import all its modules.


```python
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```

    [K     |████████████████████████████████| 727kB 17.1MB/s 
    [K     |████████████████████████████████| 1.2MB 47.1MB/s 
    [K     |████████████████████████████████| 51kB 6.8MB/s 
    [K     |████████████████████████████████| 194kB 56.0MB/s 
    [K     |████████████████████████████████| 61kB 8.7MB/s 
    [K     |████████████████████████████████| 51kB 7.8MB/s 
    [K     |████████████████████████████████| 12.8MB 251kB/s 
    [K     |████████████████████████████████| 776.8MB 23kB/s 
    [31mERROR: torchtext 0.9.0 has requirement torch==1.8.0, but you'll have torch 1.7.1 which is incompatible.[0m
    [?25hMounted at /content/gdrive
    


```python
from fastbook import *
```


```python
from fastai.vision.all import *
```

We'll use `untar_data` to download files from a URL. We'll set a path to this dataset.


```python
path = untar_data(URLs.PETS)
```





The below code helps us by showing only the relevant paths, and not showing the parent path folders.


```python
Path.BASE_PATH = path
```

Let's see what's in the directory.


```python
path.ls()
```




    (#2) [Path('images'),Path('annotations')]



The dataset provides us with the *images* and *annotations* directories. The [website](https://www.robots.ox.ac.uk/~vgg/data/pets/) for the dataset tells us that the *annotations* directory contains information about where the pets are rather than what they are. Hence, we will ignore the *annotations* directory.  
Let's look at the *images* directory.


```python
(path/'images').ls()
```




    (#7393) [Path('images/german_shorthaired_99.jpg'),Path('images/boxer_88.jpg'),Path('images/great_pyrenees_119.jpg'),Path('images/Abyssinian_146.jpg'),Path('images/newfoundland_197.jpg'),Path('images/english_cocker_spaniel_161.jpg'),Path('images/British_Shorthair_269.jpg'),Path('images/newfoundland_139.jpg'),Path('images/Egyptian_Mau_221.jpg'),Path('images/havanese_108.jpg')...]



There are 7,393 images in the folder. The breed of the pet is embedded in the image name along with a number and the extension. So, we can use a regular expression to extract the valuable information.  
Let's try out an example first.  
We'll pick one the filenames.


```python
fname = (path/'images').ls()[0]
```


```python
re.findall(r'(.+)_\d+.jpg', fname.name)
```




    ['german_shorthaired']



This regular expression plucks out all the characters leading up to the last underscore character, as long as the subsequence chracters are numerical digits and then the JPEG file extension.  
Now that we have confirmed that the regular expression works for the example, let's use it to label the whole dataset. For labeling with regular expressions, we'll use the `RegexLabeller` class.


```python
pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75))

dls = pets.dataloaders(path/'images')
```

Let's find out what breeds are present in the dataset.


```python
dls.vocab
```




    ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']




```python
dls.show_batch(max_n=9, figsize=(6,7))
```


<img src="/assets/img/computer_vision/pet_breeds_classifier/output_19_0.png">



### Training a simple model

Let's train a simple model. We'll be using transfer learning to train the model using the ImageNet dataset. The fastai method `fine_tune` will help us with this.


```python
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(15)
```

    Downloading: "https://download.pytorch.org/models/resnet34-333f7ec4.pth" to /root/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth
    


    HBox(children=(FloatProgress(value=0.0, max=87306240.0), HTML(value='')))


    
    


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.541179</td>
      <td>1.021038</td>
      <td>0.307848</td>
      <td>01:09</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.476121</td>
      <td>0.846739</td>
      <td>0.263870</td>
      <td>01:12</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.359381</td>
      <td>0.908805</td>
      <td>0.273342</td>
      <td>01:12</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.305826</td>
      <td>0.898781</td>
      <td>0.263870</td>
      <td>01:12</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.285199</td>
      <td>0.767794</td>
      <td>0.236130</td>
      <td>01:12</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.242656</td>
      <td>0.810228</td>
      <td>0.234100</td>
      <td>01:12</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.187951</td>
      <td>0.887635</td>
      <td>0.228687</td>
      <td>01:12</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.151485</td>
      <td>0.777132</td>
      <td>0.217862</td>
      <td>01:12</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.136154</td>
      <td>0.763384</td>
      <td>0.197564</td>
      <td>01:13</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.105534</td>
      <td>0.732230</td>
      <td>0.200271</td>
      <td>01:12</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.084376</td>
      <td>0.664226</td>
      <td>0.173884</td>
      <td>01:11</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.058460</td>
      <td>0.670688</td>
      <td>0.181326</td>
      <td>01:12</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.038214</td>
      <td>0.669005</td>
      <td>0.180650</td>
      <td>01:11</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.023151</td>
      <td>0.693375</td>
      <td>0.177267</td>
      <td>01:11</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.024510</td>
      <td>0.665506</td>
      <td>0.170501</td>
      <td>01:11</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.023619</td>
      <td>0.637721</td>
      <td>0.161028</td>
      <td>01:11</td>
    </tr>
  </tbody>
</table>


Let's look at a graph of the training and validation loss.


```python
learn.recorder.plot_loss()
```


<img src="/assets/img/computer_vision/pet_breeds_classifier/output_23_0.png">



Even our simple model gives us really good accuracy. Let's use a classification matrix to interpret our model; see where it's doing well and where it's doing badly.


```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
```



<img src="/assets/img/computer_vision/pet_breeds_classifier/output_25_1.png">



We have 37 different breeds of pet, which means we have 37 X 37 entries in this matrix, which makes it hard to interpret. We'll use the `most_confused` method, which just shows us the cells of the confusion matrix with the most incorrect predictions.


```python
# cells with 5 or more incorrect predictions
interp.most_confused(min_val=5)
```




    [('staffordshire_bull_terrier', 'american_pit_bull_terrier', 19),
     ('Maine_Coon', 'Persian', 7),
     ('boxer', 'american_pit_bull_terrier', 6),
     ('english_setter', 'english_cocker_spaniel', 6),
     ('Egyptian_Mau', 'Bengal', 5),
     ('american_bulldog', 'american_pit_bull_terrier', 5),
     ('american_pit_bull_terrier', 'boxer', 5),
     ('staffordshire_bull_terrier', 'american_bulldog', 5)]



These are the breeds of cats and dogs that are hardest to predict by our model. Coincidently, these are also pretty common comparisons that even experts get confused about, for e.g. the differences in appearance between the Staffordshire Bull Terrier and the American Pit Bull Terrier (i.e. the comparison that confused our model the most) are so minute that most humans will find it hard to distinguish them.

### Picking the Learning Rate
Let's try to improve our model by picking an appropriate learning rate.


```python
interp.plot_top_losses(8, nrows=2)
```


<img src="/assets/img/computer_vision/pet_breeds_classifier/output_29_0.png">



```python
learn.show_results()
```



<img src="/assets/img/computer_vision/pet_breeds_classifier/output_30_1.png">



Let's save this model.


```python
learn.save('model1')
```




    Path('models/model1.pth')



Let's use fastai's learning rate finder to find a suitable learning rate.


```python
learn2 = cnn_learner(dls, resnet34, metrics=error_rate)
lr_min, lr_steep = learn2.lr_find()
```



<img src="/assets/img/computer_vision/pet_breeds_classifier/output_34_1.png">




```python
print(f'Minimum/10: {lr_min: .2e}, steepest_point: {lr_steep: .2e}')
```

    Minimum/10:  1.00e-02, steepest_point:  5.25e-03
    

Let's now pick the learning rate at the steepest point in the graph.


```python
learn2 = cnn_learner(dls, resnet34, metrics=error_rate)
learn2.fine_tune(15, 5.25e-3)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.105870</td>
      <td>1.111952</td>
      <td>0.329499</td>
      <td>01:13</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.409039</td>
      <td>0.954296</td>
      <td>0.280785</td>
      <td>01:16</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.338957</td>
      <td>1.197951</td>
      <td>0.326116</td>
      <td>01:17</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.374680</td>
      <td>1.036932</td>
      <td>0.278755</td>
      <td>01:17</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.444182</td>
      <td>1.355077</td>
      <td>0.357240</td>
      <td>01:16</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.408214</td>
      <td>1.485182</td>
      <td>0.407307</td>
      <td>01:16</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.353418</td>
      <td>1.045590</td>
      <td>0.292963</td>
      <td>01:17</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.321617</td>
      <td>1.315597</td>
      <td>0.347767</td>
      <td>01:16</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.277381</td>
      <td>0.998754</td>
      <td>0.297023</td>
      <td>01:16</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.186780</td>
      <td>0.957907</td>
      <td>0.265900</td>
      <td>01:17</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.117437</td>
      <td>0.792308</td>
      <td>0.219215</td>
      <td>01:17</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.099135</td>
      <td>0.933071</td>
      <td>0.249662</td>
      <td>01:17</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.060645</td>
      <td>0.753601</td>
      <td>0.200947</td>
      <td>01:16</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.040250</td>
      <td>0.736492</td>
      <td>0.197564</td>
      <td>01:16</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.034840</td>
      <td>0.737214</td>
      <td>0.198241</td>
      <td>01:16</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.023275</td>
      <td>0.721530</td>
      <td>0.194181</td>
      <td>01:16</td>
    </tr>
  </tbody>
</table>


Let's plot the losses on a graph.


```python
learn2.recorder.plot_loss()
```

<img src="/assets/img/computer_vision/pet_breeds_classifier/output_39_0.png">



```python
learn2.save('model2')
```




    Path('models/model2.pth')



### Unfreezing and Transfer Learning

When fine-tuning, we aim to replace the random weights in our added linear layers with weights that correctly achieve our desired task of classifying pet breeds without breaking the carefully pretrained weights and the other layers. We'll tell the optimizer to only update the weights in those randomly added final layers, and not change the weights in the rest of the neural network. This is called *freezing* the pretrained layers.  
When we create a model from a pretrained network, fastai automatically freexzes all of the pretrained layers for us.

We can try and improve our model by changing the parameters of the `fine_tune` method.  
First, we'll train the randomly added layers for three epochs, using `fit_one_cycle`, which is the suggested way to train models without using `fine_tune`.


```python
learn3 = cnn_learner(dls, resnet34, metrics=error_rate)
learn3.fit_one_cycle(4, 5.25e-3)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.082502</td>
      <td>1.659888</td>
      <td>0.414073</td>
      <td>01:11</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.648850</td>
      <td>1.761306</td>
      <td>0.445873</td>
      <td>01:12</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.421008</td>
      <td>1.299498</td>
      <td>0.379567</td>
      <td>01:11</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.278318</td>
      <td>0.959298</td>
      <td>0.298376</td>
      <td>01:11</td>
    </tr>
  </tbody>
</table>


Now, we'll unfreeze the model.


```python
learn3.unfreeze()
```

We'll run `lr_find` again because having more layers to train, and weights that have already been trained for three epochs, means our previously found learning rate isn't appropriate anymore.


```python
learn3.lr_find()
```








    SuggestedLRs(lr_min=1.58489319801447e-07, lr_steep=1.3182567499825382e-06)




![png](output_46_2.png)


We can see that we have a somewhat flat area before a sharp increase, and we should take a point well before that sharp increase. But the deepest layers of our pretrained model might not need as high a learning rate as the last ones, so we should use different learning rates for different layers, which is known as *discriminative learning rates*.  
We'll use a lower learning rate for the early layers of the neural network, and a higher learning rate for the later layers (and especially the randomly added layers).  
fastai lets you pass a Python `slice` object anywhere that a learning rate is expected. The first value passed will be the learning rate in the earliest layer of the neural network, and the second value will be the learning rate in the final layer. The layers in between will have learning rates that are multiplicatively equidistant throughout that range.


```python
learn3 = cnn_learner(dls, resnet34, metrics=error_rate)
learn3.fit_one_cycle(4, 5.25e-3)
learn3.unfreeze()
learn3.fit_one_cycle(12, lr_max=slice(1e-6,1e-4))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.090194</td>
      <td>1.388986</td>
      <td>0.393099</td>
      <td>01:12</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.649271</td>
      <td>1.753144</td>
      <td>0.456698</td>
      <td>01:12</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.439216</td>
      <td>1.004193</td>
      <td>0.300406</td>
      <td>01:12</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.294163</td>
      <td>0.960151</td>
      <td>0.284168</td>
      <td>01:12</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.233927</td>
      <td>0.932793</td>
      <td>0.276725</td>
      <td>01:17</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.212731</td>
      <td>0.892106</td>
      <td>0.260487</td>
      <td>01:18</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.203497</td>
      <td>0.919420</td>
      <td>0.273342</td>
      <td>01:18</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.197853</td>
      <td>0.822041</td>
      <td>0.249662</td>
      <td>01:17</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.183791</td>
      <td>0.815729</td>
      <td>0.244249</td>
      <td>01:18</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.161308</td>
      <td>0.875792</td>
      <td>0.261840</td>
      <td>01:17</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.146391</td>
      <td>0.858144</td>
      <td>0.253721</td>
      <td>01:17</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.129562</td>
      <td>0.838909</td>
      <td>0.255751</td>
      <td>01:16</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.126738</td>
      <td>0.844979</td>
      <td>0.251691</td>
      <td>01:17</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.123629</td>
      <td>0.839640</td>
      <td>0.253721</td>
      <td>01:16</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.124569</td>
      <td>0.803057</td>
      <td>0.238836</td>
      <td>01:17</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.112034</td>
      <td>0.806164</td>
      <td>0.238836</td>
      <td>01:18</td>
    </tr>
  </tbody>
</table>



```python
learn3.recorder.plot_loss()
```

<img src="/assets/img/computer_vision/pet_breeds_classifier/output_49_0.png">



```python
learn3.save('model3')
```




    Path('models/model3.pth')



### Using Deeper Architectures
Let's try to improve our model's performance using deeper architectures.

Deeper architectures like ResNet50 take longer to train. We can speed things up by using mixed-precision training, wherein we can use less-precise numbers like fp16 (half-precision floating point). To enable this feature, we just add `to_fp16()` to the `Learner`.


```python
# import the fp16 module
from fastai.callback.fp16 import *
```


```python
learn4 = cnn_learner(dls, resnet50, metrics=error_rate).to_fp16()
learn4.fine_tune(15)
```

    Downloading: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to /root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
    


    HBox(children=(FloatProgress(value=0.0, max=102502400.0), HTML(value='')))


    
    


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.994141</td>
      <td>1.132718</td>
      <td>0.330176</td>
      <td>01:07</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.350162</td>
      <td>0.960766</td>
      <td>0.280785</td>
      <td>01:09</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.258597</td>
      <td>0.936031</td>
      <td>0.276725</td>
      <td>01:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.256309</td>
      <td>0.948686</td>
      <td>0.282815</td>
      <td>01:09</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.253626</td>
      <td>1.049184</td>
      <td>0.276725</td>
      <td>01:08</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.251305</td>
      <td>1.222372</td>
      <td>0.322057</td>
      <td>01:09</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.179732</td>
      <td>1.153965</td>
      <td>0.279432</td>
      <td>01:09</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.142337</td>
      <td>1.300296</td>
      <td>0.315968</td>
      <td>01:09</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.121877</td>
      <td>0.982640</td>
      <td>0.244249</td>
      <td>01:10</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.097359</td>
      <td>1.028661</td>
      <td>0.261840</td>
      <td>01:10</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.071566</td>
      <td>0.761003</td>
      <td>0.196211</td>
      <td>01:09</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.036808</td>
      <td>0.773423</td>
      <td>0.196888</td>
      <td>01:10</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.029443</td>
      <td>0.626214</td>
      <td>0.165088</td>
      <td>01:09</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.022480</td>
      <td>0.700238</td>
      <td>0.184709</td>
      <td>01:09</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.018170</td>
      <td>0.660072</td>
      <td>0.177267</td>
      <td>01:09</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.013801</td>
      <td>0.684406</td>
      <td>0.177943</td>
      <td>01:09</td>
    </tr>
  </tbody>
</table>



```python
learn4.recorder.plot_loss()
```


<img src="/assets/img/computer_vision/pet_breeds_classifier/output_54_0.png">



```python
learn4.save('model4')
```




    Path('models/model4.pth')



### Exporting our model
Comparing the training results and the graphs of the losses, we can conclude that the first model we trained was the best one. So we will now export this model and create a GUI and build a classifer within our notebook itself.


```python
learn.export()
```


```python
path = Path()
path.ls(file_exts='.pkl')
```




    (#1) [Path('export.pkl')]




```python
learn_inf = load_learner('export.pkl')
```


```python
learn_inf.dls.vocab
```




    ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']




```python
from fastai.vision.widgets import *
```


```python
def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)
```


```python
VBox([widgets.Label('Select your pet!'), 
      btn_upload, btn_run, out_pl, lbl_pred])
```


    VBox(children=(Label(value='Select your pet!'), FileUpload(value={'pet_dog.jpg': {'metadata': {'lastModified':…

