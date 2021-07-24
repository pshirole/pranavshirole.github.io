---
layout: post
title: Classifying movie reviews using Sentiment Analysis and ULMFit
feature-img: "assets/img/nlp/ai_reviews/erik-witsoe-GF8VvBgcJ4o-unsplash.jpg"
tags: [nlp, pytorch, fastai]
---


In this blog post, we will create a language model that will classify movie reviews into positive reviews and negative reviews, based on their sentiment.

A language model is basically a model that can guess the next word in a text, while having read the ones before. This kind of task is called self-supervised learning, wherein we train a model using labels that are embeeded in the independent variable, rather than requiring external labels.  

The dataset we'll be using is the [IMDb Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/), which contains 25,000 highly polarized movie reviews for training, and 25,000 for testing.  

For this task, we will be using the **Universal Language Model Fine-Tuning (ULMFit)** approach. First, we will train our language model using a model pretrained on Wikipedia. Then we will go one step further by fine-tuning our pretrained language model to the IMBD corpus, and then use *that* as the base for our classifer. Basically, we'll be fine-tuning the sequence-based language model prior to fine-tuning the classification model.
Even if our language model knows the basics of the language we are using in the task, it is benefitial to understand the style of the corpus we are targeting. For e.g., in the IMDB dataset, there will be lots of names of movie directors and actors, and often a less formal style of language than that seen in Wikipedia. 



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




    (#7) [Path('tmp_lm'),Path('test'),Path('train'),Path('README'),Path('tmp_clas'),Path('unsup'),Path('imdb.vocab')]



We'll grab the text files using `get_text_files`, which gets all the text files in a pth. We can optionally pass `folders` to restrict the search to a particular list of subfolders.


```python
files = get_text_files(path, folders=['train', 'test', 'unsup'])
```

Here's a random review we can look at.


```python
txt = files[0].open().read()
txt
```




    'I saw this "movie" partly because of the sheer number of good reviews at Netflix, and from it I leaned a valuable lesson. Not a lesson about ethnic diversity however...the lesson I learned is "Don\'t trust reviews".<br /><br />Yes, racism sucks and people are complicated, but the people who actually need to see this movie are going to be the ones who are the least drawn to it and least affected by it if they DO see it. The only reason that I can think of for the number of good reviews is that it\'s being reviewed by people who aren\'t used to thinking, or who\'ve seen their first thought-provoking movie and somehow think that Haggis invented the concept. In fact, he basically made this film, which should be called "Racism For Dummies", as emotionally wrenching as possible, seemingly to give people who don\'t spend a lot of time thinking the impression that they\'ve discovered some fundamental truth that\'s never been covered in a film before. Zen and the Art of Motorcycle Maintanence it\'s not... An after-school special for the unthinking masses, cut into bite-sized overwrought ham-fisted pieces to make it easier to swallow without too much introspection.<br /><br />It\'s as if they portrayed everyone as being the worst possible extreme, simply to make us happy that we\'re such good people because we don\'t identify with the characters. Let\'s face it people. NOBODY identifies with these characters because they\'re all cardboard cutouts and stereotypes (or predictably reverse-stereotypes). It\'s well acted (even if the dialog is atrocious) and cleverly executed, so much that you don\'t think to ask "where\'s the beef?" until you can tell the film is winding down. The flaming car scene was well executed, like much of the movie, but went nowhere in the end. <br /><br />The messages are very heavy-handed, and from the "behind the scenes" blurb, the producers were clearly watching a different movie, because there is very little to laugh about in this movie, even during the intended funny parts. I have to stress that this is NOT entertainment, more like a high school diversity lesson...call it the "Blood on the Highway" of racism. They could even show this in high schools if it weren\'t for the "side-nude" shot of Jennifer Esposito.<br /><br />In this film, everyone\'s a jerk and everyone learns a lesson (except for Michael Pena who gets the best role, but the most predictable storyline).<br /><br />This is a bad film, with bad writing, and good actors....an ugly cartoon crafted by Paul Haggis for people who can\'t handle anything but the bold strokes in storytelling....a picture painted with crayons.<br /><br />Crash is a depressing little nothing, that provokes emotion, but teaches you nothing if you already know racism and prejudice are bad things.'



---
## Preprocessing the text

We'll be using the following operations to preprocess the text:
1. Tokenization
2. Numericalization
3. Language-model DataLoader creation
4. Language-model creation


### Tokenization
Tokenization converts the text into a list of words or characters or substrings, depending on the granularity of the model.  
There are three main approaches: word-based, subword based, and character-based.  
We'll use word tokenization here since we're just dealing with the plain English language and words are separated by spaces.

#### Word Tokenization
We'll use `WordTokenizer` for word tokenization. It uses fastai's current default word tokenizer, which is *spaCy* for now. The *spaCy* library has a sophisticated rules engine with special rules for URLs, individual special English words, and much more.  
We'll use fastai's `coll_repr(collection, n)` function to display the results. This displays the first *n* items of a *collection*, along with the full size. Also, fastai's tokenizers take a collection of documents to tokenize, so we have to wrap `txt` in a list.


```python
spacy = WordTokenizer()
toks = first(spacy([txt]))
print(coll_repr(toks, 30))
```

    (#560) ['I','saw','this','"','movie','"','partly','because','of','the','sheer','number','of','good','reviews','at','Netflix',',','and','from','it','I','leaned','a','valuable','lesson','.','Not','a','lesson'...]


As you can see, the words and punctuations have been separated. 

Let's use the `Tokenizer` class to add some additional functionality to the tokenization process.


```python
tkn = Tokenizer(spacy)
print(coll_repr(tkn(txt), 35))
```

    (#598) ['xxbos','i','saw','this','"','movie','"','partly','because','of','the','sheer','number','of','good','reviews','at','xxmaj','netflix',',','and','from','it','i','leaned','a','valuable','lesson','.','xxmaj','not','a','lesson','about','ethnic'...]


Everything is now lower-cased. There are now some tokens that start with the characters "xx". These are *special tokens*. Fastai adds these tokens by default, by applying a number of rules when preprocessing text, which are designed to make it easier for a model to recognize the important parts of a sentence.  
Some of the main special tokens are:
- `xxbos`: indicates the beginning of a text
- `xxmaj`: indicates the next word begins with a capital
- `xxunk`: indicates the next word is unknown


### Numericalization
*Numericalization* is the process of mapping tokens to integers. It makes a list of all the unique words that appear (the vocab), and convert each word into a number, by looking up its index in the vocab.

Let's take a look at this in action. We need to call `setup` (a special fastai method) on `Numericalize` to create the vocab. 


```python
# corpus of the first 2000 movie reviews
txts = L(o.open().read() for o in files[:2000])
txts[0][:]
```




    'I saw this "movie" partly because of the sheer number of good reviews at Netflix, and from it I leaned a valuable lesson. Not a lesson about ethnic diversity however...the lesson I learned is "Don\'t trust reviews".<br /><br />Yes, racism sucks and people are complicated, but the people who actually need to see this movie are going to be the ones who are the least drawn to it and least affected by it if they DO see it. The only reason that I can think of for the number of good reviews is that it\'s being reviewed by people who aren\'t used to thinking, or who\'ve seen their first thought-provoking movie and somehow think that Haggis invented the concept. In fact, he basically made this film, which should be called "Racism For Dummies", as emotionally wrenching as possible, seemingly to give people who don\'t spend a lot of time thinking the impression that they\'ve discovered some fundamental truth that\'s never been covered in a film before. Zen and the Art of Motorcycle Maintanence it\'s not... An after-school special for the unthinking masses, cut into bite-sized overwrought ham-fisted pieces to make it easier to swallow without too much introspection.<br /><br />It\'s as if they portrayed everyone as being the worst possible extreme, simply to make us happy that we\'re such good people because we don\'t identify with the characters. Let\'s face it people. NOBODY identifies with these characters because they\'re all cardboard cutouts and stereotypes (or predictably reverse-stereotypes). It\'s well acted (even if the dialog is atrocious) and cleverly executed, so much that you don\'t think to ask "where\'s the beef?" until you can tell the film is winding down. The flaming car scene was well executed, like much of the movie, but went nowhere in the end. <br /><br />The messages are very heavy-handed, and from the "behind the scenes" blurb, the producers were clearly watching a different movie, because there is very little to laugh about in this movie, even during the intended funny parts. I have to stress that this is NOT entertainment, more like a high school diversity lesson...call it the "Blood on the Highway" of racism. They could even show this in high schools if it weren\'t for the "side-nude" shot of Jennifer Esposito.<br /><br />In this film, everyone\'s a jerk and everyone learns a lesson (except for Michael Pena who gets the best role, but the most predictable storyline).<br /><br />This is a bad film, with bad writing, and good actors....an ugly cartoon crafted by Paul Haggis for people who can\'t handle anything but the bold strokes in storytelling....a picture painted with crayons.<br /><br />Crash is a depressing little nothing, that provokes emotion, but teaches you nothing if you already know racism and prejudice are bad things.'




```python
toks200 = txts[:200].map(tkn)
toks200[0]
```




    (#598) ['xxbos','i','saw','this','"','movie','"','partly','because','of'...]



We can pass this to `setup` to create our vocab.


```python
num = Numericalize()
num.setup(toks200)
coll_repr(num.vocab, 20)
```




    "(#1920) ['xxunk','xxpad','xxbos','xxeos','xxfld','xxrep','xxwrep','xxup','xxmaj','the','.',',','a','and','of','to','is','it','in','this'...]"



The special tokens appear first and then every word appears once, in the descending order of frequency.

Once we've created our `Numericalize` object, we can use it as if it were a function.


```python
nums = num(toks)[:20]
nums
```




    TensorText([  0, 268,  19,  22,  24,  22,   0, 101,  14,   9, 915, 475,  14,  73, 719,  45,   0,  11,  13,  51])



Our tokens have been converted to a tensor of integers that our model can receive. We can check if they map back to the original text.


```python
' '.join(num.vocab[o] for o in nums)
```




    'xxunk saw this " movie " xxunk because of the sheer number of good reviews at xxunk , and from'



Now that we have numbers, we need to put them in batches for our model.

### Putting our Texts into Batches for a Language Model
Fastai provides an `LMDataLoader` class which automatically handles creating a dependent variable that is offset from the independent variable by one token. It automatically shuffles the collection of documents at every epoch and concatenates them into a stream of tokens. It then cuts that stream into a batch of fixed-size consecutive mini-streams. Our model will then read the mini-streams in order, and thanks to an inner state, it will product the same activation whatever sequence length we picked.


```python
# apply Numericalize object to tokenized texts
nums200 = toks200.map(num)
```


```python
# pass to LMDataLoader
dl = LMDataLoader(nums200)
```

Let's confirm that this gives the expected results, by grabbing the first batch.


```python
x, y = first(dl)
x.shape, y.shape
```




    (torch.Size([64, 72]), torch.Size([64, 72]))



Here `64` is the batch size, and `72` is the sequence length.  
Let's look at the first row of the independent variable, which should be the start of the first text.


```python
' '.join(num.vocab[o] for o in x[0][:200])
```




    'xxbos i saw this " movie " xxunk because of the sheer number of good reviews at xxmaj xxunk , and from it i xxunk a xxunk lesson . xxmaj not a lesson about xxunk xxunk however … the lesson i learned is " do n\'t xxunk reviews " . \n\n xxmaj yes , racism sucks and people are xxunk , but the people who actually need to see this movie are'



This concludes all the preprocessing steps we need to apply to our data. We are now ready to train our text classifier.

---
## Training a Text Classifier
As we discussed earlier, there are two steps to training a state-of-the-art text classifier using transfer learning: first we need to fine-tune our language model pretrained on Wikipedia to the corpus of IMDB reviews, and then we can use that model to train a classifier.


### Language Model using DataBlock
Fastai handles tokenization and numericalization automatically when `TextBlock` is passed to `DataBlock`.  
Let's create a language model using `TextBlock`.


```python
get_imdb = partial(get_text_files, folders=['train', 'test', 'unsup'])

dls_lm = DataBlock(
    blocks=TextBlock.from_folder(path, is_lm=True),
    get_items=get_imdb, splitter=RandomSplitter(0.1)
).dataloaders(path, path=path, bs=128, seq_len=80)
```





Setting up the numericalizer's vocab can take a long time. But `TextBlock` performs a few optimizations:
- it saves the tokenized documents in a temporary folder, so it doesn't have to tokenize them more than once.
- it runs multiple tokenization processes in parallel, to take advantage of your computer's CPU.

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
      <td>xxbos xxmaj airwolf is a classic , action adventure with a great story , good actors and of course good effects . xxmaj this is what i believe could be one of the best movies ever made for television . xxmaj the idea of a supersonic helicopter with 14 firepower options and bulletproof body , just seem to go straight into people 's hearts . xxmaj the movie keeps us watching from start to end and that with great style</td>
      <td>xxmaj airwolf is a classic , action adventure with a great story , good actors and of course good effects . xxmaj this is what i believe could be one of the best movies ever made for television . xxmaj the idea of a supersonic helicopter with 14 firepower options and bulletproof body , just seem to go straight into people 's hearts . xxmaj the movie keeps us watching from start to end and that with great style .</td>
    </tr>
    <tr>
      <th>1</th>
      <td>after his breakout success in xxmaj captain xxmaj blood . xxmaj still i attribute this film to the well known xxmaj aussie irreverence for trashing the reputation of one of their own . \n\n xxmaj part of the problem in telling xxmaj errol xxmaj flynn 's life story was that he told enough tall tales in his life right up to the very end in his memoir , xxmaj my xxmaj wicked xxmaj wicked xxmaj ways . i could see</td>
      <td>his breakout success in xxmaj captain xxmaj blood . xxmaj still i attribute this film to the well known xxmaj aussie irreverence for trashing the reputation of one of their own . \n\n xxmaj part of the problem in telling xxmaj errol xxmaj flynn 's life story was that he told enough tall tales in his life right up to the very end in his memoir , xxmaj my xxmaj wicked xxmaj wicked xxmaj ways . i could see that</td>
    </tr>
  </tbody>
</table>


Now that our data is ready, we can fine-tune the pretrained language model.

### Fine-tuning the Language Model

To convert the integer word indices into activations that we can use for our neural network, we will use embeddings. We'll feed those embeddings into a *recurrent neural network* (RNN), using an architecture called *AWD-LSTM*.  
The embeddings in the pretrained model are merged with random embeddings added for words that weren't in the pretraining vocabulary. This is handled automatically inside `language_model_learner`.


```python
learn = language_model_learner(
    dls_lm, AWD_LSTM, drop_mult=0.3,
    metrics=[accuracy, Perplexity()]
).to_fp16()
```





Since this is a classification problem, the loss function used by default is *cross-entropy loss*. The `Perplexity` metric is the exponential of the loss (i.e. `torch.exp(cross_entropy)`) and is often used in NLP problems. The `accuracy` metric is to see how many times our model is right while trying to predict the next word.

The first stage of the process is over i.e. we have fine-tuned our language model pretrained on Wikipedia to the corpus of IMDB reviews, and we've built the `DataLoaders` and `Learner` for the second stage.

It takes a long time to train each epoch, so we'll be saving the intermediate model results during the training process. Since `fine_tune` doesn't do that for us, we'll use `fit_one_cycle`. `language_model_learner` automatically calls `freeze` when using a pretrained model (which is the default), so this will only train the embeddings (the only part of the model that contains randomly initialized weights - i.e. embeddings for words that are in our IMDB vocab, but aren't in the pretrained model vocab).


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
      <td>4.133163</td>
      <td>4.047858</td>
      <td>0.286861</td>
      <td>57.274658</td>
      <td>28:37</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.989873</td>
      <td>3.921189</td>
      <td>0.297198</td>
      <td>50.460423</td>
      <td>28:44</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.858948</td>
      <td>3.857302</td>
      <td>0.303759</td>
      <td>47.337460</td>
      <td>28:56</td>
    </tr>
  </tbody>
</table>


Since the model took a long time to train, let's save the state of our model.


```python
learn.save('lang_model')
```




    Path('models/lang_model.pth')



Once the initial training has completed, we can continue fine-tuning the model after unfreezing.


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
      <td>3.678183</td>
      <td>3.737603</td>
      <td>0.318232</td>
      <td>41.997200</td>
      <td>31:25</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.632840</td>
      <td>3.689973</td>
      <td>0.323845</td>
      <td>40.043770</td>
      <td>31:12</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.614814</td>
      <td>3.653279</td>
      <td>0.328249</td>
      <td>38.601032</td>
      <td>31:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.529696</td>
      <td>3.627186</td>
      <td>0.332271</td>
      <td>37.606827</td>
      <td>31:15</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.467677</td>
      <td>3.610096</td>
      <td>0.334249</td>
      <td>36.969608</td>
      <td>31:18</td>
    </tr>
    <tr>
      <td>5</td>
      <td>3.400183</td>
      <td>3.594222</td>
      <td>0.337025</td>
      <td>36.387383</td>
      <td>31:17</td>
    </tr>
    <tr>
      <td>6</td>
      <td>3.336390</td>
      <td>3.589247</td>
      <td>0.338615</td>
      <td>36.206810</td>
      <td>31:24</td>
    </tr>
    <tr>
      <td>7</td>
      <td>3.285172</td>
      <td>3.585617</td>
      <td>0.339453</td>
      <td>36.075596</td>
      <td>31:19</td>
    </tr>
    <tr>
      <td>8</td>
      <td>3.223539</td>
      <td>3.590128</td>
      <td>0.339571</td>
      <td>36.238712</td>
      <td>31:25</td>
    </tr>
    <tr>
      <td>9</td>
      <td>3.181863</td>
      <td>3.595526</td>
      <td>0.339376</td>
      <td>36.434860</td>
      <td>31:10</td>
    </tr>
  </tbody>
</table>


Once this is done, we save all of our model except the final layer that converts activation to probabilities of picking each token in our vocabulary. The model not including the final layer is called *encoder*. We can save it with `save_encoder`.


```python
learn.save_encoder('finetuned')
```

This completes the second stage of the process i.e. fine-tuning the language model.

---
## Creating the Classifier DataLoaders
Now that we have fine-tuned the language model, we need to fine-tune the classifier. The language model only predicts the next word of a document so it doesn't need any external labels. Our classifier, however, needs to predict the sentiment of a moview review.

Let's create a `DataBlock` for our classifier.


```python
dls_clas = DataBlock(
    blocks=(TextBlock.from_folder(path, vocab=dls_lm.vocab), CategoryBlock),
    get_y=parent_label,
    get_items=partial(get_text_files, folders=['train', 'test']),
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path, path=path, bs=128, seq_len=72)
```

In this DataBlock, `TextBlock.from_folder` no longer has the `is_lm=True` parameter. Instead we pass the `vocab` we created for the language model fine-tuning. The reason we pass the `vocab` of the language model is to make sure that we use the same correspondence of token to index. Otherwise the embeddings we learned in our fine-tuned language model won't make any sense to this model, and the fine-tuning step won't be of any use.  
By not passing `is_lm`, we tell `TextBlock` that we have regular labeled data, rather than using the next tokens as labels.  

`show_batch` can show us the dependent variable or sentiment in this case, with each indepedent variable or movie review.


```python
dls_clas.show_batch(max_n=3)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xxbos xxmaj match 1 : xxmaj tag xxmaj team xxmaj table xxmaj match xxmaj bubba xxmaj ray and xxmaj spike xxmaj dudley vs xxmaj eddie xxmaj guerrero and xxmaj chris xxmaj benoit xxmaj bubba xxmaj ray and xxmaj spike xxmaj dudley started things off with a xxmaj tag xxmaj team xxmaj table xxmaj match against xxmaj eddie xxmaj guerrero and xxmaj chris xxmaj benoit . xxmaj according to the rules of the match , both opponents have to go through tables in order to get the win . xxmaj benoit and xxmaj guerrero heated up early on by taking turns hammering first xxmaj spike and then xxmaj bubba xxmaj ray . a xxmaj german xxunk by xxmaj benoit to xxmaj bubba took the wind out of the xxmaj dudley brother . xxmaj spike tried to help his brother , but the referee restrained him while xxmaj benoit and xxmaj guerrero</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>xxbos xxmaj some have praised xxunk xxmaj lost xxmaj xxunk as a xxmaj disney adventure for adults . i do n't think so -- at least not for thinking adults . \n\n xxmaj this script suggests a beginning as a live - action movie , that struck someone as the type of crap you can not sell to adults anymore . xxmaj the " crack staff " of many older adventure movies has been done well before , ( think xxmaj the xxmaj dirty xxmaj dozen ) but xxunk represents one of the worse films in that motif . xxmaj the characters are weak . xxmaj even the background that each member trots out seems stock and awkward at best . xxmaj an xxup md / xxmaj medicine xxmaj man , a tomboy mechanic whose father always wanted sons , if we have not at least seen these before ,</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>xxbos xxmaj warning : xxmaj does contain spoilers . \n\n xxmaj open xxmaj your xxmaj eyes \n\n xxmaj if you have not seen this film and plan on doing so , just stop reading here and take my word for it . xxmaj you have to see this film . i have seen it four times so far and i still have n't made up my mind as to what exactly happened in the film . xxmaj that is all i am going to say because if you have not seen this film , then stop reading right now . \n\n xxmaj if you are still reading then i am going to pose some questions to you and maybe if anyone has any answers you can email me and let me know what you think . \n\n i remember my xxmaj grade 11 xxmaj english teacher quite well . xxmaj</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>


We do have to collate multiple documents into a mini-batch. We will expand the shortest texts to make them all the same size using a special padding token that will be ignored by our model. Also, to avoid memory issues and improve performance, we will batch together texts that are roughly the same lengths by sorting the documents in the training set by length prior to each epoch. Hence, the documents collated into a single batch will tend to be of similar lengths. We won't pad every batch to the same size, but will instead use the size of the largest document in each batch as the target size.

The sorting and padding are done automatically by the data block API when using a `TextBlock`, with `is_lm=False`.  
Let's now create a model to classify our texts.


```python
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5,
                                metrics=accuracy).to_fp16()
```

The final step prior to training the classifer is to load the encoder from our fine-tuned model. We use `load_encoder` instead of `load` because we only have pretrained weights available for the encoder; `load` be default raises an exception if an incomplete model is loaded.


```python
learn = learn.load_encoder('finetuned')
```

---
## Fine-Tuning the Classifier
The last step is to train with discriminative learning rates and gradual unfreezing. Unfreezing a few layers at a time gives better results for NLP tasks.


```python
learn.fit_one_cycle(1, 2e-2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.247250</td>
      <td>0.183979</td>
      <td>0.928880</td>
      <td>01:41</td>
    </tr>
  </tbody>
</table>


We can pass `-2` to `freeze_to` to freeze all except the last two parameter groups.


```python
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4), 5e-3))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.226782</td>
      <td>0.169867</td>
      <td>0.934840</td>
      <td>01:54</td>
    </tr>
  </tbody>
</table>


We can unfreeze a bit more and continue training.


```python
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4), 5e-3))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.203316</td>
      <td>0.154957</td>
      <td>0.941880</td>
      <td>02:40</td>
    </tr>
  </tbody>
</table>


And now we'll finally unfreeze the whole model.


```python
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4), 1e-3))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.164804</td>
      <td>0.151554</td>
      <td>0.943000</td>
      <td>03:15</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.151896</td>
      <td>0.150165</td>
      <td>0.943280</td>
      <td>03:15</td>
    </tr>
  </tbody>
</table>


Considering the resources we used and the time taken, a 94.3% accuracy is amazing! Let's check some of the results achieved by our model. 


```python
learn.show_results()
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>category</th>
      <th>category_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xxbos xxmaj there 's a sign on xxmaj the xxmaj lost xxmaj highway that says : \n\n * major xxup spoilers xxup ahead * \n\n ( but you already knew that , did n't you ? ) \n\n xxmaj since there 's a great deal of people that apparently did not get the point of this movie , xxmaj i 'd like to contribute my interpretation of why the plot makes perfect sense . xxmaj as others have pointed out , one single viewing of this movie is not sufficient . xxmaj if you have the xxup dvd of xxup md , you can " cheat " by looking at xxmaj david xxmaj lynch 's " top 10 xxmaj hints to xxmaj unlocking xxup md " ( but only upon second or third viewing , please . ) ;) \n\n xxmaj first of all , xxmaj mulholland xxmaj drive is</td>
      <td>pos</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>xxbos ( some spoilers included : ) \n\n xxmaj although , many commentators have called this film surreal , the term fits poorly here . xxmaj to quote from xxmaj encyclopedia xxmaj xxunk 's , surreal means : \n\n " fantastic or incongruous imagery " : xxmaj one need n't explain to the unimaginative how many ways a plucky ten - year - old boy at large and seeking his fortune in the driver 's seat of a red xxmaj mustang could be fantastic : those curious might read xxmaj james xxmaj kincaid ; but if you asked said lad how he were incongruous behind the wheel of a sports car , he 'd surely protest , " no way ! " xxmaj what fantasies and incongruities the film offers mostly appear within the first fifteen minutes . xxmaj thereafter we get more iterations of the same , in an</td>
      <td>pos</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>xxbos xxmaj tony xxmaj hawk 's xxmaj pro xxmaj skater 2x , is n't much different at all from the previous games ( excluding xxmaj tony xxmaj hawk 3 ) . xxmaj the only thing new that is featured in xxmaj tony xxmaj hawk 's xxmaj pro xxmaj skater 2x , is the new selection of levels , and tweaked out graphics . xxmaj tony xxmaj hawk 's xxmaj pro xxmaj skater 2x offers a new career mode , and that is the 2x career . xxmaj the 2x career is basically xxmaj tony xxmaj hawk 1 career , because there is only about five challenges per level . xxmaj if you missed xxmaj tony xxmaj hawk 1 and 2 , i suggest that you buy xxmaj tony xxmaj hawk 's xxmaj pro xxmaj skater 2x , but if you have played the first two games , you should still</td>
      <td>pos</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>3</th>
      <td>xxbos xxmaj based on the excellent novel , xxmaj watchers by xxmaj dean xxmaj koontz , is this extremely awful motion picture that probably should n't be viewed by anyone . xxmaj not since " the xxmaj running xxmaj man " have i seen a book butchered so far beyond recognition . xxmaj the difference , however , is that " the xxmaj running xxmaj man " film was still enjoyable as an amusing action film laden down a million catch phrases . xxmaj this film ▁ xxmaj nope , nothing remotely amusing . xxmaj in fact , if you love the book , as i do , you 'll hate this bastardization even more . \n\n * * xxunk xxup spoilers * * xxmaj xxunk , xxmaj i 'm basically going to tell you the story here , almost in it 's entirety . xxmaj why ? xxmaj because</td>
      <td>neg</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>xxbos xxmaj hollywood movies since the 1930s have treated gays as lepers . xxmaj in condemning homosexuality , the film industry has reflected only what the repressive society of its day espoused as an ideology . xxmaj for example , in the 1962 xxmaj otto xxmaj preminger melodrama " advise and xxmaj consent , " straight actor xxmaj don xxmaj murray was cast as a queer congressman who commits suicide rather than confess his alternative lifestyle . xxmaj gay movie characters have covered a lot of ground since " advise and xxmaj consent . " xxmaj in the 1997 movie " in &amp; xxmaj out , " ( * * 1 / 2 out of xxrep 4 * ) , heterosexual actor xxmaj kevin xxmaj kline is cast as a homosexual teacher who comes out of the closet on his wedding day . xxmaj while the conservative xxmaj hollywood of</td>
      <td>pos</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>5</th>
      <td>xxbos " empire xxmaj strikes xxmaj back " director xxmaj irvin xxmaj kershner 's " never xxmaj say xxmaj never xxmaj again , " a remake of the 1965 xxmaj james xxmaj bond movie " thunderball , " does n't surpasses the xxmaj terence xxmaj young original , but this non - harry xxmaj xxunk &amp; xxmaj albert xxup r. xxmaj broccoli film is well worth watching if you call yourself a 007 aficionado . xxmaj nevertheless , despite its shortage of clever gadgets and the lack of a vibrant musical score , " never xxmaj say xxmaj never xxmaj again " rates as an above - average , suspenseful doomsday thriller with top - flight performances by a seasoned cast including xxmaj sean xxmaj connery , xxmaj kim xxmaj basinger , xxmaj klaus xxmaj maria xxmaj brandauer , xxmaj max xxmaj von xxmaj sydow , xxmaj barbara xxmaj carrera</td>
      <td>pos</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>6</th>
      <td>xxbos i really wanted to be able to give this film a 10 . xxmaj i 've long thought it was my favorite of the four modern live - action xxmaj batman films to date ( and maybe it still will be -- i have yet to watch the xxmaj schumacher films again ) . xxmaj i 'm also starting to become concerned about whether xxmaj i 'm somehow subconsciously being contrarian . xxmaj you see , i always liked the xxmaj schumacher films . xxmaj as far as i can remember , they were either 9s or 10s to me . xxmaj but the conventional wisdom is that the two xxmaj tim xxmaj burton directed films are far superior . i had serious problems with the first xxmaj burton xxmaj batman this time around -- i ended up giving it a 7 - -and apologize as i might ,</td>
      <td>pos</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>7</th>
      <td>xxbos xxmaj may 2nd : someone clicked 11 nos , and then proceeded to do 15 more on my previous 15 comments : almost as funny as this turkey ! \n\n xxmaj may 1st : \n\n xxmaj as i write this , xxmaj i 'm still very much under the impression of what must be the funniest thriller xxmaj i 've ever seen . xxmaj i 've got a major case of the giggles , but xxmaj i 'll try and calm down . ( it 's kind of hard to write when your nose spills snot and the mouth ejects sporadic drool onto the keyboard . ) \n\n a pair of young women who just returned from a vacation take a ride on a shuttle bus . a couple of young guys join them . xxmaj but the bus is n't really a taxi service : it 's a</td>
      <td>neg</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>8</th>
      <td>xxbos " the belief in the xxmaj big xxmaj other as an invisible power structure which exists in the xxmaj real is the most succinct definition of paranoia . "  xxmaj slavoj xxmaj zizek \n\n xxmaj this is a review of " marathon xxmaj man " and " the xxmaj falcon and the xxmaj snowman " , two films by director xxmaj john xxmaj schlesinger . \n\n xxmaj though xxmaj hitchcock and xxmaj lang brought the " conspiracy thriller " to xxmaj hollywood , the genre only blossomed in the late 60s and 70s , with films like " the xxmaj parallax xxmaj view " , " z " , " marathon xxmaj man " , " capricorn xxmaj one " , " the xxmaj manchurian xxmaj candidate " , " three xxmaj days of the xxmaj condor " and " all xxmaj the xxmaj president 's xxmaj men</td>
      <td>pos</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>


Here we can see that out of the random nine reviews that our model picked, it classified eight of them correctly, which in itself is impressive. But look more closely at the one it got wrong (the fifth one, which the model predicted to be negative but it's actually positive). With the way the review is worded, a lot of humans would consider those statements as negative. This just shows how good our model is.
