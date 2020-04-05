Recommendation systems are a collection of algorithms used to recommend items to users based on information taken from the user. These systems have become ubiquitous, and can commonly be seen in online stores, movie databases, and job finders. In this blog post, we will explore **content-based** and **colaborative filtering** recommendation systems.

The dataset we'll be working on has been acquired from [GroupLens](https://grouplens.org/datasets/movielens/). It consists of 27 million ratings and 1.1 million tag applications applied to 58,000 movies by 280,000 users.


```python
# import libraries
import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# store the movie information into a pandas dataframe
movies_df = pd.read_csv('movies1.csv')

# store the ratings information into a pandas dataframe
ratings_df = pd.read_csv('ratings.csv')
```


```python
movies_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



Each movie has a unique ID, a title with its release year along with it (which may contain unicode characters) and several different genres in the same field.


```python
# dimensions of the dataframes
print(movies_df.shape)
print(ratings_df.shape)
```

    (58097, 3)
    (27753444, 4)
    

### Preprocessing the data
Let's remove the year from the 'title' column and store it in a new 'year' column.


```python
# use regular expressions to find a year stored between parantheses
# we specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)

# remove the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)

# remove the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')

# apply the strip finction to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
movies_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>Adventure|Children|Fantasy</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>Comedy|Romance</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>Comedy|Drama|Romance</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>Comedy</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>



Let's also split the values in the 'genres' column into a 'list of genres' to simplify future use. Apply Python's split string function on the genres column.


```python
# every genre is separated by a |. So call the split function on |.
movies_df['genres'] = movies_df.genres.str.split('|')
movies_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>[Adventure, Animation, Children, Comedy, Fantasy]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>[Adventure, Children, Fantasy]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>[Comedy, Romance]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>[Comedy]</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>



Since keeping genres in a list format isn't optimal for the content-based recommendation system technique, we will use the One Hot Encoding technique to convert the list of genres to a vector where each column corresponds to one possible value of the feature. This encoding is needed for feeding categorical data.  
In this case, we store every differrent genre in columns that contain either 1 or 0. 1 shows that a movie has that genre and 0 shows that it doesn't. Let's also store this dataframe in another variable since genres won't be important for our first recommendation system.


```python
# copy the movie dataframe into a new one
moviesWithGenres_df = movies_df.copy()

# for every row in the dataframe, iterate through the list of genres and place a 1 in the corresponding column
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
        
# fill in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>...</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>Sci-Fi</th>
      <th>IMAX</th>
      <th>Documentary</th>
      <th>War</th>
      <th>Musical</th>
      <th>Western</th>
      <th>Film-Noir</th>
      <th>(no genres listed)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>[Adventure, Animation, Children, Comedy, Fantasy]</td>
      <td>1995</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>[Adventure, Children, Fantasy]</td>
      <td>1995</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>[Comedy, Romance]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>[Comedy]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



Now, let's focus on the ratings dataframe.


```python
ratings_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>307</td>
      <td>3.5</td>
      <td>1256677221</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>481</td>
      <td>3.5</td>
      <td>1256677456</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1091</td>
      <td>1.5</td>
      <td>1256677471</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1257</td>
      <td>4.5</td>
      <td>1256677460</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1449</td>
      <td>4.5</td>
      <td>1256677264</td>
    </tr>
  </tbody>
</table>
</div>



Every row in the ratings dataframe has a userId  associated with at least one movie, a rating and a timestamp showing when they reviewed it. We won't be needing the timestamp column, so let's drop it.


```python
ratings_df = ratings_df.drop('timestamp',1)
ratings_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>307</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>481</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1091</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1257</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1449</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>
</div>



## Content-based recommendation system
This technique attempts to figure out what a user's favorite aspects of an item are, and then recommends items that present those aspects. In our case, we're going to try to figure out the input's favorite genres from the movies and ratings given. 

Advantages of content-based filtering:
- it learns the user's preferences.
- it's highly personalized for the user.   

Disadvantages of content-based filtering:
- it doesn't take into account what others think of the item, so low quality item recommendations might happen. 
- Extracting data is not always intuitive.
- Determining what characteristics of the item the user dislikes or likes is not always obvious.

Create an input to recommend movies to.


```python
userInput = [
    {'title':'Mission: Impossible - Fallout', 'rating':5},
    {'title':'Top Gun', 'rating':4.5},
    {'title':'Jerry Maguire', 'rating':3},
    {'title':'Vanilla Sky', 'rating':2.5},
    {'title':'Minority Report', 'rating':4},
]
inputMovies = pd.DataFrame(userInput)
inputMovies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mission: Impossible - Fallout</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Top Gun</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jerry Maguire</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Vanilla Sky</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Minority Report</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



Add movieId to input user.  
Extract the input movie's ID from the movies dataframe and add it to the input.


```python
# filter the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

# merge it to get the movieId
inputMovies = pd.merge(inputId, inputMovies)

# drop information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

# final input dataframe
inputMovies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1101</td>
      <td>Top Gun</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1393</td>
      <td>Jerry Maguire</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4975</td>
      <td>Vanilla Sky</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5445</td>
      <td>Minority Report</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>189333</td>
      <td>Mission: Impossible - Fallout</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



We will learn the input's preferences. So let's get the subset of movies that the input has watched from the dataframe containing genres defined with binary values.


```python
# filter out the movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>...</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>Sci-Fi</th>
      <th>IMAX</th>
      <th>Documentary</th>
      <th>War</th>
      <th>Musical</th>
      <th>Western</th>
      <th>Film-Noir</th>
      <th>(no genres listed)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1079</th>
      <td>1101</td>
      <td>Top Gun</td>
      <td>[Action, Romance]</td>
      <td>1986</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1361</th>
      <td>1393</td>
      <td>Jerry Maguire</td>
      <td>[Drama, Romance]</td>
      <td>1996</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4879</th>
      <td>4975</td>
      <td>Vanilla Sky</td>
      <td>[Mystery, Romance, Sci-Fi, Thriller]</td>
      <td>2001</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5348</th>
      <td>5445</td>
      <td>Minority Report</td>
      <td>[Action, Crime, Mystery, Sci-Fi, Thriller]</td>
      <td>2002</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>56349</th>
      <td>189333</td>
      <td>Mission: Impossible - Fallout</td>
      <td>[Action, Adventure, Thriller]</td>
      <td>2018</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



We only need the actual genre table. Reset the index and drop the unnecessary columns.


```python
# reset the index
userMovies = userMovies.reset_index(drop=True)

# drop unnecessary columns
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
userGenreTable
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>Drama</th>
      <th>Action</th>
      <th>Crime</th>
      <th>Thriller</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>Sci-Fi</th>
      <th>IMAX</th>
      <th>Documentary</th>
      <th>War</th>
      <th>Musical</th>
      <th>Western</th>
      <th>Film-Noir</th>
      <th>(no genres listed)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Now we learn the input preferences.  
We turn each genre into weights using the input's reviews and multiplying them into the input's genre table, and then summing up the resulting table by column.


```python
inputMovies['rating']
```




    0    4.5
    1    3.0
    2    2.5
    3    4.0
    4    5.0
    Name: rating, dtype: float64




```python
# dot product to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

# the user profile
userProfile
```




    Adventure              5.0
    Animation              0.0
    Children               0.0
    Comedy                 0.0
    Fantasy                0.0
    Romance               10.0
    Drama                  3.0
    Action                13.5
    Crime                  4.0
    Thriller              11.5
    Horror                 0.0
    Mystery                6.5
    Sci-Fi                 6.5
    IMAX                   0.0
    Documentary            0.0
    War                    0.0
    Musical                0.0
    Western                0.0
    Film-Noir              0.0
    (no genres listed)     0.0
    dtype: float64



Now we have the weights for each of the user's preferences. This is the **User Profile**. Using this, we can recommend movies that satisfy the user's preferences.  
Let's start by extracting the genre table from the original dataframe.


```python
# get the genre of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])

# drop unnecessary columns
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
genreTable.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>Drama</th>
      <th>Action</th>
      <th>Crime</th>
      <th>Thriller</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>Sci-Fi</th>
      <th>IMAX</th>
      <th>Documentary</th>
      <th>War</th>
      <th>Musical</th>
      <th>Western</th>
      <th>Film-Noir</th>
      <th>(no genres listed)</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
genreTable.shape
```




    (58097, 20)



With the input's profile and the complete list of movies and their genres in hand, we're going to take the weighted average of every movie based on the input profile and recommend the top twenty movies that most satisfy it.


```python
# multiply the genre by the weights and then take the weighted average
recommendationTable_df = ((genreTable*userProfile).sum(axis=1)) / (userProfile.sum())
recommendationTable_df.head()
```




    movieId
    1    0.083333
    2    0.083333
    3    0.166667
    4    0.216667
    5    0.000000
    dtype: float64



Here is the recommendation table.


```python
movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>[Adventure, Animation, Children, Comedy, Fantasy]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>[Adventure, Children, Fantasy]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>[Comedy, Romance]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>[Comedy]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Heat</td>
      <td>[Action, Crime, Thriller]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Sabrina</td>
      <td>[Comedy, Romance]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Tom and Huck</td>
      <td>[Adventure, Children]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Sudden Death</td>
      <td>[Action]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>GoldenEye</td>
      <td>[Action, Adventure, Thriller]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>American President, The</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>Dracula: Dead and Loving It</td>
      <td>[Comedy, Horror]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>Balto</td>
      <td>[Adventure, Animation, Children]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>Nixon</td>
      <td>[Drama]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>Cutthroat Island</td>
      <td>[Action, Adventure, Romance]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>Casino</td>
      <td>[Crime, Drama]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>Sense and Sensibility</td>
      <td>[Drama, Romance]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>Four Rooms</td>
      <td>[Comedy]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>Ace Ventura: When Nature Calls</td>
      <td>[Comedy]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>Money Train</td>
      <td>[Action, Comedy, Crime, Drama, Thriller]</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>



These are the top 20 movies to recommend to the user based on a content-based recommendation system.

## Collaborative Filtering
This technique uses other users to recommend items to the input user. It attempts to find users that have similar preferences and opinions as the input and then recommends items that they have liked to the input. there are several methods of finding similar users, and the one we will be using here is going to be based on the Pearson Correlation Function.

The process for creating a user-based recommendation system is as follows:
- Select a user with the movies the user has watched.
- Based on his ratings of movies, find the top X neighbours.
- Get the watched movie record of the user for each neighbour.
- Calculate a similarity score using some formula.
- Recommend the items with the highest score.

Advantages of collaborative filtering:
- It takes other user's ratings into consideration
- It doesn't need to study or extract information from the recommended item
- It adapts to the user's interestes which might change over time

Disadvantages of collaborative filtering:
- The approximation function can be slow.
- There might be a low amount of users to approximate
- There might be privacy issues when trying to learn the user's experiences.

Let's create an input user to recommend movies to.


```python
userInput = [
    {'title':'Mission: Impossible - Fallout', 'rating':5},
    {'title':'Top Gun', 'rating':4.5},
    {'title':'Jerry Maguire', 'rating':3},
    {'title':'Vanilla Sky', 'rating':2.5},
    {'title':'Minority Report', 'rating':4},
]
inputMovies = pd.DataFrame(userInput)
inputMovies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mission: Impossible - Fallout</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Top Gun</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jerry Maguire</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Vanilla Sky</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Minority Report</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# filter the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

# merge it to get the movieId
inputMovies = pd.merge(inputId, inputMovies)

# drop information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

# final input dataframe
inputMovies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1101</td>
      <td>Top Gun</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1393</td>
      <td>Jerry Maguire</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4975</td>
      <td>Vanilla Sky</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5445</td>
      <td>Minority Report</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>189333</td>
      <td>Mission: Impossible - Fallout</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



#### The users who have seen the same movies
Now, with the movie IDs in our input, we can get the subset of users that have watched and reviewd the movies in our input.


```python
# filter out users that have watched movies that the input has watched and storing it
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
userSubset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>214</th>
      <td>4</td>
      <td>1101</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>248</th>
      <td>4</td>
      <td>1393</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>586</th>
      <td>4</td>
      <td>4975</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>610</th>
      <td>4</td>
      <td>5445</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>935</th>
      <td>8</td>
      <td>1393</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



Group the rows by userId.


```python
# groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
userSubsetGroup = userSubset.groupby(['userId'])
```

Let's look at one of these users - userId = 4


```python
userSubsetGroup.get_group(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>214</th>
      <td>4</td>
      <td>1101</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>248</th>
      <td>4</td>
      <td>1393</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>586</th>
      <td>4</td>
      <td>4975</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>610</th>
      <td>4</td>
      <td>5445</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>
</div>



Let's sort these groups so the users that share the most movies in common with the input have higher priority. This provides a richer recommendation since we won't go through every single user.


```python
userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)
```

Now let's look at the first user.


```python
userSubsetGroup[0:3]
```




    [(214,
             userId  movieId  rating
      20548     214     1101     2.0
      20638     214     1393     3.0
      21122     214     4975     2.0
      21160     214     5445     4.0
      21933     214   189333     3.0),
     (6264,
              userId  movieId  rating
      616485    6264     1101     5.0
      616574    6264     1393     4.0
      617440    6264     4975     3.0
      617480    6264     5445     3.0
      618666    6264   189333     4.0),
     (19924,
               userId  movieId  rating
      1945179   19924     1101     3.5
      1945273   19924     1393     4.0
      1946065   19924     4975     2.0
      1946152   19924     5445     4.0
      1948193   19924   189333     3.5)]



Next, we are going to compare users to our specified user and find the one that is most similar.  
We're going to find out how similar each user is to the input through the Pearson Correlation Coefficient. It is used to measure the strength of a linear association between two variables.  

We will select a subset of users to iterate through. The limit is imposed because we don't want to waste too much time going through every single user.


```python
userSubsetGroup = userSubsetGroup[0:100]
```

Calculate the Pearson Correlation between the input user and the subset group, and store it in a dictionary, where the key is the userId and the value is the coefficient.


```python
pearsonCorrelationDict = {}

#For every user group in our subset
for name, group in userSubsetGroup:
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    #Get the N for the formula
    nRatings = len(group)
    #Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    #Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0
```


```python
pearsonCorrelationDict.items()
```




    dict_items([(214, 0.23055616708169335), (6264, 0.518751375933811), (19924, 0.48424799847909467), (21962, 0.7190233885442843), (22361, 0.6163156344279349), (24518, -0.48424799847909017), (28244, -0.22258705026211378), (30387, 0.8339502495593619), (31727, -0.6163156344279349), (32728, -0.26413527189768593), (33550, 0.3774147062120368), (36202, 0.9510441892119876), (38778, 0.5906244232186185), (43227, -0.1968748077395395), (43264, -0.9021937088963177), (48109, 0.0), (50016, 0.04402254531627891), (59611, 0.24946109012559378), (62705, 0.5187513759338097), (63353, -0.4799585206127619), (64733, -0.8524929243380921), (69860, 0.43133109281375515), (70271, -0.08524929243380922), (71857, 0.7781270639007126), (72194, 0.24112141108520613), (75629, 0.6016946526766817), (77609, 0.48224282217041226), (80398, 0.7776587696250218), (81924, -0.32283199898606263), (93997, 0.7771889263740438), (94749, 0.0), (98561, -0.5619806572616304), (99014, -0.23055616708169688), (102101, 0.2516098041413576), (104322, -0.43133109281375515), (105397, 0.8859366348279278), (112491, -0.6469966392206334), (116632, 0.20173664619648324), (117053, 0.5917813771642448), (124357, 0.7635511351031528), (125365, 0.7009130258223497), (128610, 0.45109685444815883), (131687, -0.22874785549890708), (133546, 0.6163156344279386), (148144, 0.10783277320344019), (153921, 0.12056070554260306), (161582, -0.3616821166278092), (167427, 0.10783277320343922), (167835, 0.10783277320343156), (171745, -0.6995593008237843), (173280, -0.2516098041413576), (175811, 0.616315634427937), (184822, 0.07421560439929334), (186859, -0.6163156344279386), (187056, 0.8439249387982215), (189464, 0.2017366461964786), (194365, -0.05547950410915026), (195892, 0.17049858486761843), (199011, 0.6340294594746541), (205765, 0.6163156344279422), (209798, 0.836059669922064), (210651, -0.057639041770424365), (220709, 0.8364283610093444), (221882, -0.18485618263446638), (233580, 0.7009130258223497), (240712, 0.700913025822351), (242708, 0.04876920665717847), (247867, -0.4528033232531783), (248019, 0.393749615479079), (261170, 0.518751375933811), (261224, 0.5114957546028552), (263973, -0.12888481555661682), (267699, 0.17049858486761843), (271364, 0.7043607250605002), (275841, -0.8364283610093444), (280868, 0.09843740386976975), (4, 0.4216370213557839), (56, 0.12909944487358055), (81, 0.2581988897471611), (147, 0.5502760564641688), (235, 0.0), (239, -0.7302967433402214), (313, -0.7302967433402214), (332, 0.848528137423857), (458, 0.0), (601, 0.6708203932499369), (605, 0.32071349029490925), (864, -0.31622776601683794), (930, -0.5163977794943222), (1073, -0.4242640687119285), (1153, -0.5262348115842176), (1191, 0.0), (1263, 0.0), (1312, 0.9621404708847278), (1367, -0.1414213562373095), (1419, -0.3651483716701107), (1440, 0.6708203932499369), (1513, 0.3651483716701107), (1519, -0.38138503569823695), (1523, -0.38138503569823695)])




```python
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonDF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similarityIndex</th>
      <th>userId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.230556</td>
      <td>214</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.518751</td>
      <td>6264</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.484248</td>
      <td>19924</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.719023</td>
      <td>21962</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.616316</td>
      <td>22361</td>
    </tr>
  </tbody>
</table>
</div>



#### The top x similar users to the input user
Let's get the top 50 users that are most similar to the input.


```python
topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similarityIndex</th>
      <th>userId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>93</th>
      <td>0.962140</td>
      <td>1312</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.951044</td>
      <td>36202</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.885937</td>
      <td>105397</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0.848528</td>
      <td>332</td>
    </tr>
    <tr>
      <th>54</th>
      <td>0.843925</td>
      <td>187056</td>
    </tr>
  </tbody>
</table>
</div>



Now let's start recommending movies to the input user.

#### Rating of selected users to all movies

We're going to do this by taking the weighted average of the ratings of the movies using the Pearson Correlation as the weight. But to do this, we first need to get the movies watched by the users in our pearsonDF from the ratings dataframe, and then store their correlation in a new column called 'similarityIndex'.


```python
# merge two tables
topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
topUsersRating.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similarityIndex</th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.96214</td>
      <td>1312</td>
      <td>6</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.96214</td>
      <td>1312</td>
      <td>19</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.96214</td>
      <td>1312</td>
      <td>32</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.96214</td>
      <td>1312</td>
      <td>110</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.96214</td>
      <td>1312</td>
      <td>150</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



Now we multiply the movie rating by its weight (the similarity index), then sum up the new ratings and divide it by the sum of the weights.  
We can easily do this by simply multiplying two columns, then grouping up the dataframe by movieId and then dividing two columns.  
It shows the idea of all similar users to candidate movies for the input user.


```python
# multiply the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similarityIndex</th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>weightedRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.96214</td>
      <td>1312</td>
      <td>6</td>
      <td>3.0</td>
      <td>2.886421</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.96214</td>
      <td>1312</td>
      <td>19</td>
      <td>3.5</td>
      <td>3.367492</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.96214</td>
      <td>1312</td>
      <td>32</td>
      <td>2.5</td>
      <td>2.405351</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.96214</td>
      <td>1312</td>
      <td>110</td>
      <td>2.5</td>
      <td>2.405351</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.96214</td>
      <td>1312</td>
      <td>150</td>
      <td>3.0</td>
      <td>2.886421</td>
    </tr>
  </tbody>
</table>
</div>




```python
# apply a sum to the topUsers after grouping it by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_similarityIndex</th>
      <th>sum_weightedRating</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>24.947499</td>
      <td>100.721637</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22.262128</td>
      <td>70.826453</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.242517</td>
      <td>25.223362</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.427828</td>
      <td>6.840441</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12.595882</td>
      <td>33.904291</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create an empty dataframe
recommendation_df = pd.DataFrame()

# take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weighted average recommendation score</th>
      <th>movieId</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.037344</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.181477</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.060153</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.817515</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.691696</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



Let's sort this and see the top 20 movies that the algorithm recommended.


```python
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
recommendation_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weighted average recommendation score</th>
      <th>movieId</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4863</th>
      <td>5.0</td>
      <td>4863</td>
    </tr>
    <tr>
      <th>5641</th>
      <td>5.0</td>
      <td>5641</td>
    </tr>
    <tr>
      <th>3777</th>
      <td>5.0</td>
      <td>3777</td>
    </tr>
    <tr>
      <th>3205</th>
      <td>5.0</td>
      <td>3205</td>
    </tr>
    <tr>
      <th>3847</th>
      <td>5.0</td>
      <td>3847</td>
    </tr>
  </tbody>
</table>
</div>




```python
movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(20)['movieId'].tolist())]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3118</th>
      <td>3205</td>
      <td>Black Sunday (La maschera del demonio)</td>
      <td>[Horror]</td>
      <td>1960</td>
    </tr>
    <tr>
      <th>3686</th>
      <td>3777</td>
      <td>Nekromantik</td>
      <td>[Comedy, Horror]</td>
      <td>1987</td>
    </tr>
    <tr>
      <th>3754</th>
      <td>3847</td>
      <td>Ilsa, She Wolf of the SS</td>
      <td>[Horror]</td>
      <td>1974</td>
    </tr>
    <tr>
      <th>3876</th>
      <td>3970</td>
      <td>Beyond, The (E tu vivrai nel terrore - L'aldilà)</td>
      <td>[Horror]</td>
      <td>1981</td>
    </tr>
    <tr>
      <th>4767</th>
      <td>4863</td>
      <td>Female Trouble</td>
      <td>[Comedy, Crime]</td>
      <td>1975</td>
    </tr>
    <tr>
      <th>5542</th>
      <td>5641</td>
      <td>Moderns, The</td>
      <td>[Drama]</td>
      <td>1988</td>
    </tr>
    <tr>
      <th>5681</th>
      <td>5780</td>
      <td>Polyester</td>
      <td>[Comedy]</td>
      <td>1981</td>
    </tr>
    <tr>
      <th>5810</th>
      <td>5909</td>
      <td>Visitor Q (Bizita Q)</td>
      <td>[Comedy, Drama, Horror]</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>8549</th>
      <td>26007</td>
      <td>Unknown Soldier, The (Tuntematon sotilas)</td>
      <td>[Drama, War]</td>
      <td>1955</td>
    </tr>
    <tr>
      <th>12542</th>
      <td>58425</td>
      <td>Heima</td>
      <td>[Documentary]</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>12713</th>
      <td>59684</td>
      <td>Lake of Fire</td>
      <td>[Documentary]</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>16931</th>
      <td>85181</td>
      <td>Pooh's Grand Adventure: The Search for Christo...</td>
      <td>[Adventure, Animation, Children, Musical]</td>
      <td>1997</td>
    </tr>
    <tr>
      <th>20049</th>
      <td>98198</td>
      <td>OMG Oh My God!</td>
      <td>[Comedy, Drama]</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>21237</th>
      <td>102666</td>
      <td>Ivan Vasilievich: Back to the Future (Ivan Vas...</td>
      <td>[Adventure, Comedy]</td>
      <td>1973</td>
    </tr>
    <tr>
      <th>22406</th>
      <td>106561</td>
      <td>Krrish 3</td>
      <td>[Action, Adventure, Fantasy, Sci-Fi]</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>46195</th>
      <td>167248</td>
      <td>Kedi</td>
      <td>[(no genres listed)]</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>50636</th>
      <td>176753</td>
      <td>Bingo - The King of the Mornings</td>
      <td>[Comedy, Drama]</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>51187</th>
      <td>177951</td>
      <td>Happy!</td>
      <td>[Fantasy]</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>53314</th>
      <td>182723</td>
      <td>Cosmos: A Spacetime Odissey</td>
      <td>[(no genres listed)]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>54462</th>
      <td>185227</td>
      <td>Brief History of Disbelief</td>
      <td>[Documentary]</td>
      <td>2004</td>
    </tr>
  </tbody>
</table>
</div>



These are the top 20 movies to recommend to the user based on a collaborative filtering recommendation system.
