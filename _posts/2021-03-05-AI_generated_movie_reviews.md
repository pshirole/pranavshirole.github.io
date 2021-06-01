---
layout: post
title: AI-generated Movie Reviews
tags: [nlp, pytorch, fastai]
---


In this blog post, we will create a language model that will generate its own movie reviews.

This blog post is basically a continuation of my previous post and you should definitely read that if you want to better understand the methodology behind the process used in this task.

The dataset we'll be using is the [IMDb Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/), which contains 25,000 highly polarized movie reviews for training, and 25,000 for testing.


```python
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```


```python
from fastbook import *
```

Let's download the dataset.


```python
from fastai.text.all import *
path = untar_data(URLs.IMDB)
```






```python
Path.BASE_PATH = path
path.ls()
```




    (#7) [Path('imdb.vocab'),Path('train'),Path('README'),Path('tmp_clas'),Path('test'),Path('tmp_lm'),Path('unsup')]



We'll grab the text files using `get_text_files`, which gets all the text files in a pth. We can optionally pass `folders` to restrict the search to a particular list of subfolders.


```python
files = get_text_files(path, folders=['train', 'test', 'unsup'])
```

Here's a review we can look at.


```python
txt = files[0].open().read()
txt
```




    "Dressed to Kill (1980) is a mystery horror film from Brian De Palma and it really works.The atmosphere is right there.The atmosphere that makes you scared.And isn't that what a horror film is supposed to do.All the actors are in the right places.Michael Caine is perfect as Dr. Robert Elliott, the shrink with a little secret.Angie Dickinson as Kate Miller, the sexually frustrated mature woman is terrific.Keith Gordon as her son Peter is brilliant.Nancy Allen as Liz Blake the call girl is fantastic.Dennis Franz does his typical detective role.His Detective Marino is one of the most colorful in this movie.There are plenty of creepy scenes in this movie.The elevator scene is one of them.There have been made comparisons between this and Alfred Hitchcock's Psycho (1960).There are some similarities between these two movies.Both of these movies may cause some sleepless nights."



---
## Training a Text Classifier

### Language Model using DataBlock
Fastai handles tokenization and numericalization automatically when `TextBlock` is passed to `DataBlock`.  
Let's create a language model using `TextBlock`.


```python
get_imdb = partial(get_text_files, folders=['train', 'test', 'unsup'])

dls_lm = DataBlock(
    blocks=TextBlock.from_folder(path, is_lm=True),
    get_items=get_imdb, splitter=RandomSplitter(0.1)
).dataloaders(path, path=path, bs=128, seq_len=72)
```





The `from_folder` tells `TextBlock` how to access the texts for the initial preprocessing.

We can look at a couple of examples in the model.


```python
dls_lm.show_batch(max_n=2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>text_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xxbos xxmaj being that i am not a fan of xxmaj snoop xxmaj dogg , as an actor , that made me even more anxious to check out this flick . i remember he was interviewed on " jay xxmaj leno , " and said that he turned down a role in the big - budget xxmaj adam xxmaj sandler comedy " the xxmaj longest xxmaj yard " to be in this</td>
      <td>xxmaj being that i am not a fan of xxmaj snoop xxmaj dogg , as an actor , that made me even more anxious to check out this flick . i remember he was interviewed on " jay xxmaj leno , " and said that he turned down a role in the big - budget xxmaj adam xxmaj sandler comedy " the xxmaj longest xxmaj yard " to be in this film</td>
    </tr>
    <tr>
      <th>1</th>
      <td>viewer , the first number in the series does provide an unexpected element of suspense in addition to capable costuming from xxmaj ha xxmaj nguyen , fine stunt performing , and a polished turn from xxmaj carr . xxmaj an unrated version is available that seemingly promises to provide additional footage of the ardent romantic actions shared by the mismatched lovers . xxbos xxmaj the xxmaj minion is about … well ,</td>
      <td>, the first number in the series does provide an unexpected element of suspense in addition to capable costuming from xxmaj ha xxmaj nguyen , fine stunt performing , and a polished turn from xxmaj carr . xxmaj an unrated version is available that seemingly promises to provide additional footage of the ardent romantic actions shared by the mismatched lovers . xxbos xxmaj the xxmaj minion is about … well , a</td>
    </tr>
  </tbody>
</table>


Now that our data is ready, we can fine-tune the pretrained language model.


---
## Fine-tuning the Language Model

To convert the integer word indices into activations that we can use for our neural network, we will use embeddings. We'll feed those embeddings into a *recurrent neural network* (RNN), using an architecture called *AWD-LSTM*.  
The embeddings in the pretrained model are merged with random embeddings added for words that weren't in the pretraining vocabulary. This is handled automatically inside `language_model_learner`.


```python
learn = language_model_learner(
    dls_lm, AWD_LSTM, drop_mult=0.3,
    metrics=[accuracy, Perplexity()]
).to_fp16()
```






```python
learn.fit_one_cycle(3, 2e-2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>perplexity</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>4.128321</td>
      <td>4.070849</td>
      <td>0.284800</td>
      <td>58.606724</td>
      <td>29:54</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.995339</td>
      <td>3.938066</td>
      <td>0.296213</td>
      <td>51.319229</td>
      <td>29:57</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.860701</td>
      <td>3.867283</td>
      <td>0.303124</td>
      <td>47.812309</td>
      <td>30:00</td>
    </tr>
  </tbody>
</table>



```python
learn.unfreeze()
learn.fit_one_cycle(10, 2e-3)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>perplexity</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.675387</td>
      <td>3.746690</td>
      <td>0.317715</td>
      <td>42.380569</td>
      <td>32:10</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.645742</td>
      <td>3.704438</td>
      <td>0.322705</td>
      <td>40.627209</td>
      <td>32:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.605402</td>
      <td>3.664308</td>
      <td>0.327991</td>
      <td>39.029121</td>
      <td>31:54</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.535574</td>
      <td>3.633687</td>
      <td>0.331826</td>
      <td>37.852131</td>
      <td>31:51</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.451682</td>
      <td>3.618303</td>
      <td>0.334019</td>
      <td>37.274242</td>
      <td>31:41</td>
    </tr>
    <tr>
      <td>5</td>
      <td>3.417034</td>
      <td>3.603825</td>
      <td>0.336183</td>
      <td>36.738476</td>
      <td>31:49</td>
    </tr>
    <tr>
      <td>6</td>
      <td>3.359589</td>
      <td>3.594853</td>
      <td>0.337721</td>
      <td>36.410355</td>
      <td>31:44</td>
    </tr>
    <tr>
      <td>7</td>
      <td>3.266180</td>
      <td>3.592850</td>
      <td>0.338945</td>
      <td>36.337505</td>
      <td>31:36</td>
    </tr>
    <tr>
      <td>8</td>
      <td>3.213485</td>
      <td>3.597207</td>
      <td>0.339176</td>
      <td>36.496162</td>
      <td>31:34</td>
    </tr>
    <tr>
      <td>9</td>
      <td>3.178523</td>
      <td>3.602469</td>
      <td>0.339008</td>
      <td>36.688713</td>
      <td>31:36</td>
    </tr>
  </tbody>
</table>


---
## Text Generation
Let's use our model to generate random reviews. Since it is trained to guess what the next word of the sentence is, we can use the model to write new reviews.


```python
TEXT = 'I like this movie because'
N_WORDS = 70
N_SENTENCES = 3
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75)
for _ in range(N_SENTENCES)]
```




```python
print('\n\n'.join(preds))
```

    i like this movie because it has an amazing cast and the story is what made it so funny . Tom Skerrit is wonderful in this movie and Gena Rowlands , who i honestly wished she would have been better recognized for her work in Love , Caution , Never Been Kissed . It is also one of the great movies i have seen in
    
    i like this movie because it shows a side of British Realism and how it affects these American People . It displays the trials and tribulations of a couple British People , struggling through the struggles and oppression of their own British Intelligence , as well as the growing , growing awareness it brings to the British we have lived through all the past years
    
    i like this movie because it is so funny and I 'm afraid i ca n't get over what is going to happen to the characters . The ending is a bit of a surprise but you must see it , because you will not be disappointed . It is very funny and shows us how not to do a job . i think it is great to see it on



```python

```


```python
TEXT = 'This movie is terrible'
N_WORDS = 70
N_SENTENCES = 3
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75)
for _ in range(N_SENTENCES)]
```



```python
print('\n\n'.join(preds))
```

    This movie is terrible . My friend went to see it and we were so disappointed . I 'm not usually a fan of the book but i had earlier read that Chris Columbus wrote some of the best writing , directing and directing since , well , there are no words to describe how bad this piece of garbage was . It was a complete waste of time
    
    This movie is terrible . Not only is it offensive in spots , it only gets worse . It has no story line . No acting and dead and cheap special effects . What a waste of talent . My 3 year old son was laughing , not laughing . Well , i really loved the first film . This one is clearly one of the
    
    This movie is terrible , i do n't know why i could n't find it , it was so awful that i had to leave the room after this horrible film was finished .



```python

```



```python
TEXT = 'This movie has great action'
N_WORDS = 70
N_SENTENCES = 3
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75)
for _ in range(N_SENTENCES)]
```



```python
print('\n\n'.join(preds))
```

    This movie has great action scenes . However , the ending is so predictable at times that i wonder if it was n't for the fact that the director did n't care for the ending . If you 're really interested in seeing Hollywood films , see this one . Some scenes are too contrived , and some of the scenes are just too long , not too long .
    
    This movie has great action sequences . The action is well acted , and Dakota Fanning was surprisingly good . The story is predictable . It is a little too long . The story is very stupid and cliché . The camera work is bad . The movie has it 's moments , but they are mostly annoying . Second , the music is n't the

    This movie has great action sequences , the one liners and a few catchy highlights are the film 's nice moments . The scenes with the Black Snake are good , but the Snake Hunter is a bit over the top , especially the Snake has a nice body . Overall , this film was great , but i have to say that Snake People was


```python

```
