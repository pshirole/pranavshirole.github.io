---
layout: post
title: Predicting the quality of wine
tags: [classification]
---

In this blog post, we will be analyzing the quality of red and white wines, and check which are the attributes that affect wine quality the most.  

There are two datasets, related to red and white Vinho Verde wine samples, from the north of Portugal. The datasets can be downloaded [here](https://archive.ics.uci.edu/ml/datasets/wine+quality). The goal is to model wine quality based on physicochemical tests. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).  

#### Attribute Information:

Input variables (based on physicochemical tests):
1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10. sulphates
11. alcohol 

Output variable (based on sensory data):
12. quality (score between 0 and 10)

## Red Wine
Let's first consider the red wine dataset.

### Data Analysis


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```


```python
# create a pandas dataframe
df_red = pd.read_csv('winequality-red.csv')
df_red.head()
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_red.shape
```




    (1599, 12)



There are 1,599 samples and 12 features, including our target feature - the wine quality.


```python
df_red.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1599 entries, 0 to 1598
    Data columns (total 12 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   fixed acidity         1599 non-null   float64
     1   volatile acidity      1599 non-null   float64
     2   citric acid           1599 non-null   float64
     3   residual sugar        1599 non-null   float64
     4   chlorides             1599 non-null   float64
     5   free sulfur dioxide   1599 non-null   float64
     6   total sulfur dioxide  1599 non-null   float64
     7   density               1599 non-null   float64
     8   pH                    1599 non-null   float64
     9   sulphates             1599 non-null   float64
     10  alcohol               1599 non-null   float64
     11  quality               1599 non-null   int64  
    dtypes: float64(11), int64(1)
    memory usage: 150.0 KB
    

All of our dataset is numeric and there are no missing values.


```python
df_red.describe()
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.319637</td>
      <td>0.527821</td>
      <td>0.270976</td>
      <td>2.538806</td>
      <td>0.087467</td>
      <td>15.874922</td>
      <td>46.467792</td>
      <td>0.996747</td>
      <td>3.311113</td>
      <td>0.658149</td>
      <td>10.422983</td>
      <td>5.636023</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.741096</td>
      <td>0.179060</td>
      <td>0.194801</td>
      <td>1.409928</td>
      <td>0.047065</td>
      <td>10.460157</td>
      <td>32.895324</td>
      <td>0.001887</td>
      <td>0.154386</td>
      <td>0.169507</td>
      <td>1.065668</td>
      <td>0.807569</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.600000</td>
      <td>0.120000</td>
      <td>0.000000</td>
      <td>0.900000</td>
      <td>0.012000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.990070</td>
      <td>2.740000</td>
      <td>0.330000</td>
      <td>8.400000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.100000</td>
      <td>0.390000</td>
      <td>0.090000</td>
      <td>1.900000</td>
      <td>0.070000</td>
      <td>7.000000</td>
      <td>22.000000</td>
      <td>0.995600</td>
      <td>3.210000</td>
      <td>0.550000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.900000</td>
      <td>0.520000</td>
      <td>0.260000</td>
      <td>2.200000</td>
      <td>0.079000</td>
      <td>14.000000</td>
      <td>38.000000</td>
      <td>0.996750</td>
      <td>3.310000</td>
      <td>0.620000</td>
      <td>10.200000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.200000</td>
      <td>0.640000</td>
      <td>0.420000</td>
      <td>2.600000</td>
      <td>0.090000</td>
      <td>21.000000</td>
      <td>62.000000</td>
      <td>0.997835</td>
      <td>3.400000</td>
      <td>0.730000</td>
      <td>11.100000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.900000</td>
      <td>1.580000</td>
      <td>1.000000</td>
      <td>15.500000</td>
      <td>0.611000</td>
      <td>72.000000</td>
      <td>289.000000</td>
      <td>1.003690</td>
      <td>4.010000</td>
      <td>2.000000</td>
      <td>14.900000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>



To understand how much each attribute correlates with the wine quality, we can compute the standard correlation coefficient or Pearson's r between every pair of attributes.


```python
corr_matrix = df_red.corr()
corr_matrix['quality'].sort_values(ascending=False)
```




    quality                 1.000000
    alcohol                 0.476166
    sulphates               0.251397
    citric acid             0.226373
    fixed acidity           0.124052
    residual sugar          0.013732
    free sulfur dioxide    -0.050656
    pH                     -0.057731
    chlorides              -0.128907
    density                -0.174919
    total sulfur dioxide   -0.185100
    volatile acidity       -0.390558
    Name: quality, dtype: float64



We now know the features that most affect the wine quality.  
Wine quality is directly proportional to the amount of alcohol, sulphates, citric acid.  
Wine quality is inversely proportional to the amount of volatile acidity, total sulfur dioxide, density.

### Data Visualization
Let's visualize the data by creating histograms and density plots. We can understand the distribution for separate attributes.


```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```


```python
# histograms
df_red.hist(bins=10, figsize=(20, 15))
plt.show()
```


<img src="/assets/img/wine quality/output_14_0.png">



```python
# density plots
df_red.plot(kind='density', subplots=True, figsize=(20,15),
           layout=(4,3), sharex=False)
plt.show()
```

<img src="/assets/img/wine quality/output_15_0.png">



```python
sns.distplot(df_red['quality'], hist=True, kde=True,
             bins='auto', color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.show()
```


<img src="/assets/img/wine quality/output_16_0.png">


The data distribution for the alcohol, citric acid and sulfur dioxide content atrributes is positively skewed.  
The data distribution for the density and pH attributes is quite normally distributed.  
The wine quality data distribution is a bimodal distribution and there are more wines with an average quality than wines with good or bad quality.


```python
from pandas.plotting import scatter_matrix
sm = scatter_matrix(df_red, figsize=(12, 12), diagonal='kde')
#Change label rotation
[s.xaxis.label.set_rotation(40) for s in sm.reshape(-1)]
[s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]
#May need to offset label when rotating to prevent overlap of figure
[s.get_yaxis().set_label_coords(-0.6,0.5) for s in sm.reshape(-1)]
#Hide all ticks
[s.set_xticks(()) for s in sm.reshape(-1)]
[s.set_yticks(()) for s in sm.reshape(-1)]
plt.show()
```


<img src="/assets/img/wine quality/output_18_0.png">



Let's create a pivot table that describes the median value of each feature for each quality score.


```python
# pivot table
column_names = ['fixed acidity', 'volatile acidity', 'citric acid', 
                'residual sugar', 'chlorides', 'free sulfur dioxide', 
                'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

df_red_pivot_table = df_red.pivot_table(column_names, 
                                    ['quality'],
                                    aggfunc='median')

df_red_pivot_table
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
      <th>alcohol</th>
      <th>chlorides</th>
      <th>citric acid</th>
      <th>density</th>
      <th>fixed acidity</th>
      <th>free sulfur dioxide</th>
      <th>pH</th>
      <th>residual sugar</th>
      <th>sulphates</th>
      <th>total sulfur dioxide</th>
      <th>volatile acidity</th>
    </tr>
    <tr>
      <th>quality</th>
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
      <th>3</th>
      <td>9.925</td>
      <td>0.0905</td>
      <td>0.035</td>
      <td>0.997565</td>
      <td>7.50</td>
      <td>6.0</td>
      <td>3.39</td>
      <td>2.1</td>
      <td>0.545</td>
      <td>15.0</td>
      <td>0.845</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.000</td>
      <td>0.0800</td>
      <td>0.090</td>
      <td>0.996500</td>
      <td>7.50</td>
      <td>11.0</td>
      <td>3.37</td>
      <td>2.1</td>
      <td>0.560</td>
      <td>26.0</td>
      <td>0.670</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9.700</td>
      <td>0.0810</td>
      <td>0.230</td>
      <td>0.997000</td>
      <td>7.80</td>
      <td>15.0</td>
      <td>3.30</td>
      <td>2.2</td>
      <td>0.580</td>
      <td>47.0</td>
      <td>0.580</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10.500</td>
      <td>0.0780</td>
      <td>0.260</td>
      <td>0.996560</td>
      <td>7.90</td>
      <td>14.0</td>
      <td>3.32</td>
      <td>2.2</td>
      <td>0.640</td>
      <td>35.0</td>
      <td>0.490</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11.500</td>
      <td>0.0730</td>
      <td>0.400</td>
      <td>0.995770</td>
      <td>8.80</td>
      <td>11.0</td>
      <td>3.28</td>
      <td>2.3</td>
      <td>0.740</td>
      <td>27.0</td>
      <td>0.370</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12.150</td>
      <td>0.0705</td>
      <td>0.420</td>
      <td>0.994940</td>
      <td>8.25</td>
      <td>7.5</td>
      <td>3.23</td>
      <td>2.1</td>
      <td>0.740</td>
      <td>21.5</td>
      <td>0.370</td>
    </tr>
  </tbody>
</table>
</div>



We can see just how much effect does the alcohol content and volatile acidity have on the quality of the wine.

We can plot a correlation matrix to see how two variables interact, both in direction and magnitude.


```python
column_names = ['fixed acidity', 'volatile acidity', 'citric acid', 
                'residual sugar', 'chlorides', 'free sulfur dioxide', 
                'total sulfur dioxide', 'density', 'pH', 'sulphates', 
                'alcohol', 'quality']

# plot correlation matrix
fig, ax = plt.subplots(figsize=(20, 20))
colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr_matrix, cmap=colormap, annot=True,
           fmt='.2f', annot_kws={'size': 20})
ax.set_xticklabels(column_names, 
                   rotation=45, 
                   horizontalalignment='right',
                   fontsize=20);
ax.set_yticklabels(column_names, fontsize=20);
plt.show()
```

<img src="/assets/img/wine quality/output_22_0.png">


### Data Cleaning

In our dataset, there aren't any missing values, outliers, or attributes that provide no useful information for the task. So, we could conclude than our dataset is quite clean.  
The wine preference scores vary from 3 to 8, so it's straightforward to categorize them into 'bad' or 'good' quality of wines. We will assign discrete values of 0 and 1 for the corresponding categories.


```python
# Dividing wine as good and bad by giving a limit for the quality
bins = (2, 6, 8)
group_names = ['bad', 'good']
df_red['quality'] = pd.cut(df_red['quality'], bins = bins, labels = group_names)
```


```python
from sklearn.preprocessing import LabelEncoder

# let's assign labels to our quality variable
label_quality = LabelEncoder()
# Bad becomes 0 and good becomes 1
df_red['quality'] = label_quality.fit_transform(df_red['quality'])
print(df_red['quality'].value_counts())

sns.countplot(df_red['quality'])
plt.show()
```

    0    1382
    1     217
    Name: quality, dtype: int64
    


<img src="/assets/img/wine quality/output_25_1.png">


As we can see, there are far more bad quality red wines (1,382) than good quality ones (217).

### Train/Test Split

Now we will split the dataset into a training set and a testing set.


```python
# separate the dataset
X = df_red.drop('quality', axis=1)
y = df_red['quality']
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
```

### Data Preprocessing

We will scale the features so as to get optimized results.


```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
```

### Modeling
We will be evaluating 8 different algorithms.
1. Support Vector Classifier
2. Stochastic Gradient Decent Classifier
3. Random Forest Classifier
4. Decision Tree Classifier
5. Gaussian Naive Bayes
6. K-Neighbors Classifier
7. Ada Boost Classifier
8. Logistic Regression

The key to a fair comparison of machine learning algorithms is ensuring that each algorithm is evaluated in the same way on the same data. *K-fold Cross Validation* provides a solution to this problem by dividing the data into folds and ensuring that each fold is used as a testing set at some point.


```python
# import libraries
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, cross_val_score
```


```python
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
```


```python
# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('SupportVectorClassifier', SVC()))
models.append(('StochasticGradientDecentC', SGDClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier()))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('GaussianNB', GaussianNB()))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('AdaBoostClassifier', AdaBoostClassifier()))
models.append(('LogisticRegression', LogisticRegression()))# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
   kfold = model_selection.KFold(n_splits=10, random_state=seed)
   cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
   results.append(cv_results)
   names.append(name)
   msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
   print(msg)
    
# boxplot algorithm comparison
fig = plt.figure(figsize=(40, 20))
fig.suptitle('Algorithm Comparison', fontsize=40)
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names, fontdict={'fontsize': 20})
plt.show()
```

    SupportVectorClassifier: 0.889782 (0.023210)
    StochasticGradientDecentC: 0.849151 (0.044240)
    RandomForestClassifier: 0.912457 (0.029968)
    DecisionTreeClassifier: 0.877264 (0.028120)
    GaussianNB: 0.836559 (0.022781)
    KNeighborsClassifier: 0.873364 (0.021081)
    AdaBoostClassifier: 0.885876 (0.019715)
    LogisticRegression: 0.883526 (0.031077)
    

<img src="/assets/img/wine quality/output_35_1.png">


The Box Plots of these algorithms' accuracy distribution is quite symmetrical, with negligible outliers. The adjacent box plot values are close together, which correspond to the high density of accuracy scores.

### Hyperparameter Tuning

There are several factors that can help us determine which algorithm performs best. One such factor is the performance on the cross-validation set and another factor is the choice of parameters for an algorithm.  

#### SVC
Let's fine-tune our algorithms. The first algorithm that we trained and evaluated was the *Support Vector Classifier* and the mean value for model prediction was *0.889*. We will use *GridSearchCV* for the hyperparameter tuning.


```python
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
```


```python
def svc_param_selection(X, y, nfolds):
    param = {
        'C': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4],
        'kernel': ['linear', 'rbf'],
        'gamma': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]
    }
    grid_search = GridSearchCV(svc, param_grid=param, 
                               scoring='accuracy', cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_

svc_param_selection(X_train, y_train, 10)
```




    {'C': 1.2, 'gamma': 0.9, 'kernel': 'rbf'}



Hence, the best parameters for the SVC algorithm are **{C= 1.2, gamma= 0.9 , kernel= rbf}**.  
Let's run our SVC algorithm again with the best parameters.


```python
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error

svc2 = SVC(C= 1.2, gamma= 0.9, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print('Confusion matrix')
print(confusion_matrix(y_test, pred_svc2))
print('Classification report')
print(classification_report(y_test, pred_svc2))
print('Accuracy score',accuracy_score(y_test, pred_svc2))
```

    Confusion matrix
    [[271   2]
     [ 31  16]]
    Classification report
                  precision    recall  f1-score   support
    
               0       0.90      0.99      0.94       273
               1       0.89      0.34      0.49        47
    
        accuracy                           0.90       320
       macro avg       0.89      0.67      0.72       320
    weighted avg       0.90      0.90      0.88       320
    
    Accuracy score 0.896875
    

The overall accuracy of the classifier is **89.69%**, and f1-score of the weighted avg is **0.88**, which is very good.

#### Stochastic Gradient Descent Classifier


```python
sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=60)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)
```

#### Random Forest Classifier


```python
rfc = RandomForestClassifier(n_estimators=200, max_depth=20,
                             random_state=0)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
```

#### KNeighbors Classifier


```python
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
```


```python
def knn_param_selection(X, y, nfolds):
    param = {
        'n_neighbors': [2, 3, 4, 5, 6],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2]
    }
    grid_search = GridSearchCV(knn, param_grid=param, 
                               scoring='accuracy', cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_

knn_param_selection(X_train, y_train, 10)
```




    {'algorithm': 'auto', 'n_neighbors': 4, 'p': 2, 'weights': 'distance'}



Hence, the best parameters for the KNeighborsClassifier algorithm are **{algorithm= auto, n_neighbors= 4 , p= 2, weights= distance}**.  
Let's run our knn algorithm again with the best parameters.


```python
knn2 = KNeighborsClassifier(algorithm= 'auto', 
                            n_neighbors= 5, p=2,
                           weights='distance')
knn2.fit(X_train, y_train)
pred_knn2 = knn2.predict(X_test)
print('Confusion matrix')
print(confusion_matrix(y_test, pred_knn2))
print('Classification report')
print(classification_report(y_test, pred_knn2))
print('Accuracy score',accuracy_score(y_test, pred_knn2))
```

    Confusion matrix
    [[261  12]
     [ 19  28]]
    Classification report
                  precision    recall  f1-score   support
    
               0       0.93      0.96      0.94       273
               1       0.70      0.60      0.64        47
    
        accuracy                           0.90       320
       macro avg       0.82      0.78      0.79       320
    weighted avg       0.90      0.90      0.90       320
    
    Accuracy score 0.903125
    

The overall accuracy of the classifier is **90.3%**, and f1-score of the weighted avg is **0.90**, which is very good.

#### AdaBoost Classifier


```python
ada_classifier = AdaBoostClassifier(n_estimators=100)
ada_classifier.fit(X_train, y_train)
pred_ada = ada_classifier.predict(X_test)
```


```python
# cross validation
scores = cross_val_score(ada_classifier,X_test,y_test, cv=5)
print('Accuracy score',scores.mean())
```

    Accuracy score 0.84375
    

### Model Evaluation
We can compare the models by calculating their mean absolute error and accuracy.


```python
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(
        mean_absolute_error(test_labels, predictions)))
    print('Accuracy = {:0.2f}%.'.format(accuracy_score(
        test_labels, predictions)*100))
```


```python
evaluate(svc,X_test,y_test)
evaluate(svc2,X_test,y_test)
evaluate(sgd,X_test,y_test)
evaluate(rfc,X_test,y_test)
evaluate(knn2, X_test, y_test)
evaluate(ada_classifier,X_test,y_test)
```

    Model Performance
    Average Error: 0.1250 degrees.
    Accuracy = 87.50%.
    Model Performance
    Average Error: 0.1031 degrees.
    Accuracy = 89.69%.
    Model Performance
    Average Error: 0.1688 degrees.
    Accuracy = 83.12%.
    Model Performance
    Average Error: 0.1125 degrees.
    Accuracy = 88.75%.
    Model Performance
    Average Error: 0.0969 degrees.
    Accuracy = 90.31%.
    Model Performance
    Average Error: 0.1594 degrees.
    Accuracy = 84.06%.
    

The KNeighborsClassifier model with hyperparameter tuning performs the best with an accuracy of 90.31%.

### Feature Importance
We could also analyze the feature importance for an algorithm.


```python
importance = rfc.feature_importances_

std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importance)

# plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.barh(range(X.shape[1]), importance[indices],
       color="b",  align="center")

plt.yticks(range(X.shape[1]), column_names)
plt.ylim([0, X.shape[1]])
plt.show()
```


<img src="/assets/img/wine quality/output_61_0.png">


## White Wine

Let us now consider the white wine dataset.


```python
# create a pandas dataframe
df_white = pd.read_csv('winequality-white.csv')
df_white.head()
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.0</td>
      <td>0.27</td>
      <td>0.36</td>
      <td>20.7</td>
      <td>0.045</td>
      <td>45.0</td>
      <td>170.0</td>
      <td>1.0010</td>
      <td>3.00</td>
      <td>0.45</td>
      <td>8.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.3</td>
      <td>0.30</td>
      <td>0.34</td>
      <td>1.6</td>
      <td>0.049</td>
      <td>14.0</td>
      <td>132.0</td>
      <td>0.9940</td>
      <td>3.30</td>
      <td>0.49</td>
      <td>9.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.1</td>
      <td>0.28</td>
      <td>0.40</td>
      <td>6.9</td>
      <td>0.050</td>
      <td>30.0</td>
      <td>97.0</td>
      <td>0.9951</td>
      <td>3.26</td>
      <td>0.44</td>
      <td>10.1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.2</td>
      <td>0.23</td>
      <td>0.32</td>
      <td>8.5</td>
      <td>0.058</td>
      <td>47.0</td>
      <td>186.0</td>
      <td>0.9956</td>
      <td>3.19</td>
      <td>0.40</td>
      <td>9.9</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.2</td>
      <td>0.23</td>
      <td>0.32</td>
      <td>8.5</td>
      <td>0.058</td>
      <td>47.0</td>
      <td>186.0</td>
      <td>0.9956</td>
      <td>3.19</td>
      <td>0.40</td>
      <td>9.9</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_white.shape
```




    (4898, 12)



There are 4,898 samples and 12 features, including our target feature - the wine quality.


```python
df_white.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4898 entries, 0 to 4897
    Data columns (total 12 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   fixed acidity         4898 non-null   float64
     1   volatile acidity      4898 non-null   float64
     2   citric acid           4898 non-null   float64
     3   residual sugar        4898 non-null   float64
     4   chlorides             4898 non-null   float64
     5   free sulfur dioxide   4898 non-null   float64
     6   total sulfur dioxide  4898 non-null   float64
     7   density               4898 non-null   float64
     8   pH                    4898 non-null   float64
     9   sulphates             4898 non-null   float64
     10  alcohol               4898 non-null   float64
     11  quality               4898 non-null   int64  
    dtypes: float64(11), int64(1)
    memory usage: 459.3 KB
    


```python
df_white.describe()
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.854788</td>
      <td>0.278241</td>
      <td>0.334192</td>
      <td>6.391415</td>
      <td>0.045772</td>
      <td>35.308085</td>
      <td>138.360657</td>
      <td>0.994027</td>
      <td>3.188267</td>
      <td>0.489847</td>
      <td>10.514267</td>
      <td>5.877909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.843868</td>
      <td>0.100795</td>
      <td>0.121020</td>
      <td>5.072058</td>
      <td>0.021848</td>
      <td>17.007137</td>
      <td>42.498065</td>
      <td>0.002991</td>
      <td>0.151001</td>
      <td>0.114126</td>
      <td>1.230621</td>
      <td>0.885639</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.800000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.600000</td>
      <td>0.009000</td>
      <td>2.000000</td>
      <td>9.000000</td>
      <td>0.987110</td>
      <td>2.720000</td>
      <td>0.220000</td>
      <td>8.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.300000</td>
      <td>0.210000</td>
      <td>0.270000</td>
      <td>1.700000</td>
      <td>0.036000</td>
      <td>23.000000</td>
      <td>108.000000</td>
      <td>0.991723</td>
      <td>3.090000</td>
      <td>0.410000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.800000</td>
      <td>0.260000</td>
      <td>0.320000</td>
      <td>5.200000</td>
      <td>0.043000</td>
      <td>34.000000</td>
      <td>134.000000</td>
      <td>0.993740</td>
      <td>3.180000</td>
      <td>0.470000</td>
      <td>10.400000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.300000</td>
      <td>0.320000</td>
      <td>0.390000</td>
      <td>9.900000</td>
      <td>0.050000</td>
      <td>46.000000</td>
      <td>167.000000</td>
      <td>0.996100</td>
      <td>3.280000</td>
      <td>0.550000</td>
      <td>11.400000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.200000</td>
      <td>1.100000</td>
      <td>1.660000</td>
      <td>65.800000</td>
      <td>0.346000</td>
      <td>289.000000</td>
      <td>440.000000</td>
      <td>1.038980</td>
      <td>3.820000</td>
      <td>1.080000</td>
      <td>14.200000</td>
      <td>9.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
corr_matrix2 = df_white.corr()
corr_matrix2['quality'].sort_values(ascending=False)
```




    quality                 1.000000
    alcohol                 0.435575
    pH                      0.099427
    sulphates               0.053678
    free sulfur dioxide     0.008158
    citric acid            -0.009209
    residual sugar         -0.097577
    fixed acidity          -0.113663
    total sulfur dioxide   -0.174737
    volatile acidity       -0.194723
    chlorides              -0.209934
    density                -0.307123
    Name: quality, dtype: float64



The features that have the biggest impact on wine quality are alcohol, pH, suplphates, chlorides, density, and volatile acidity.

### Data Visualization


```python
# histograms
df_white.hist(bins=10, figsize=(20, 15))
plt.show()
```


<img src="/assets/img/wine quality/output_70_0.png">



```python
# pivot table
column_names = ['fixed acidity', 'volatile acidity', 'citric acid', 
                'residual sugar', 'chlorides', 'free sulfur dioxide', 
                'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

df_white_pivot_table = df_white.pivot_table(column_names, 
                                    ['quality'],
                                    aggfunc='median')

df_white_pivot_table
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
      <th>alcohol</th>
      <th>chlorides</th>
      <th>citric acid</th>
      <th>density</th>
      <th>fixed acidity</th>
      <th>free sulfur dioxide</th>
      <th>pH</th>
      <th>residual sugar</th>
      <th>sulphates</th>
      <th>total sulfur dioxide</th>
      <th>volatile acidity</th>
    </tr>
    <tr>
      <th>quality</th>
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
      <th>3</th>
      <td>10.45</td>
      <td>0.041</td>
      <td>0.345</td>
      <td>0.994425</td>
      <td>7.3</td>
      <td>33.5</td>
      <td>3.215</td>
      <td>4.60</td>
      <td>0.44</td>
      <td>159.5</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.10</td>
      <td>0.046</td>
      <td>0.290</td>
      <td>0.994100</td>
      <td>6.9</td>
      <td>18.0</td>
      <td>3.160</td>
      <td>2.50</td>
      <td>0.47</td>
      <td>117.0</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9.50</td>
      <td>0.047</td>
      <td>0.320</td>
      <td>0.995300</td>
      <td>6.8</td>
      <td>35.0</td>
      <td>3.160</td>
      <td>7.00</td>
      <td>0.47</td>
      <td>151.0</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10.50</td>
      <td>0.043</td>
      <td>0.320</td>
      <td>0.993660</td>
      <td>6.8</td>
      <td>34.0</td>
      <td>3.180</td>
      <td>5.30</td>
      <td>0.48</td>
      <td>132.0</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11.40</td>
      <td>0.037</td>
      <td>0.310</td>
      <td>0.991760</td>
      <td>6.7</td>
      <td>33.0</td>
      <td>3.200</td>
      <td>3.65</td>
      <td>0.48</td>
      <td>122.0</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12.00</td>
      <td>0.036</td>
      <td>0.320</td>
      <td>0.991640</td>
      <td>6.8</td>
      <td>35.0</td>
      <td>3.230</td>
      <td>4.30</td>
      <td>0.46</td>
      <td>122.0</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>9</th>
      <td>12.50</td>
      <td>0.031</td>
      <td>0.360</td>
      <td>0.990300</td>
      <td>7.1</td>
      <td>28.0</td>
      <td>3.280</td>
      <td>2.20</td>
      <td>0.46</td>
      <td>119.0</td>
      <td>0.27</td>
    </tr>
  </tbody>
</table>
</div>




```python
# density plots
df_white.plot(kind='density', subplots=True, figsize=(20,15),
           layout=(4,3), sharex=False)
plt.show()
```


<img src="/assets/img/wine quality/output_72_0.png">



```python
sns.distplot(df_white['quality'], hist=True, kde=True,
             bins='auto', color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.show()
```


<img src="/assets/img/wine quality/output_73_0.png">



```python
from pandas.plotting import scatter_matrix
sm = scatter_matrix(df_white, figsize=(12, 12), diagonal='kde')
#Change label rotation
[s.xaxis.label.set_rotation(40) for s in sm.reshape(-1)]
[s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]
#May need to offset label when rotating to prevent overlap of figure
[s.get_yaxis().set_label_coords(-0.6,0.5) for s in sm.reshape(-1)]
#Hide all ticks
[s.set_xticks(()) for s in sm.reshape(-1)]
[s.set_yticks(()) for s in sm.reshape(-1)]
plt.show()
```

<img src="/assets/img/wine quality/output_74_0.png">


```python
column_names = ['fixed acidity', 'volatile acidity', 'citric acid', 
                'residual sugar', 'chlorides', 'free sulfur dioxide', 
                'total sulfur dioxide', 'density', 'pH', 'sulphates', 
                'alcohol', 'quality']

# plot correlation matrix
fig, ax = plt.subplots(figsize=(20, 20))
colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr_matrix2, cmap=colormap, annot=True,
           fmt='.2f', annot_kws={'size': 20})
ax.set_xticklabels(column_names, 
                   rotation=45, 
                   horizontalalignment='right',
                   fontsize=20);
ax.set_yticklabels(column_names, fontsize=20);
plt.show()
```


<img src="/assets/img/wine quality/output_75_0.png">



```python
# Dividing wine as good and bad by giving a limit for the quality
bins = (2, 6, 9)
group_names = ['bad', 'good']
df_white['quality'] = pd.cut(df_white['quality'], bins = bins, labels = group_names)
```


```python
# let's assign labels to our quality variable
label_quality = LabelEncoder()
# Bad becomes 0 and good becomes 1
df_white['quality'] = label_quality.fit_transform(df_white['quality'])
print(df_white['quality'].value_counts())

sns.countplot(df_white['quality'])
plt.show()
```

    0    3838
    1    1060
    Name: quality, dtype: int64
    


<img src="/assets/img/wine quality/output_77_1.png">


There are 3,838 bad quality wines, and 1,060 good quality wines.

### Train/Test Split


```python
# separate the dataset
X = df_white.drop('quality', axis=1)
y = df_white['quality']
```


```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
```

### Data Preprocessing


```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
```

### Modeling


```python
# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('SupportVectorClassifier', SVC()))
models.append(('StochasticGradientDecentC', SGDClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier()))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('GaussianNB', GaussianNB()))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('AdaBoostClassifier', AdaBoostClassifier()))
models.append(('LogisticRegression', LogisticRegression()))# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
   kfold = model_selection.KFold(n_splits=10, random_state=seed)
   cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
   results.append(cv_results)
   names.append(name)
   msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
   print(msg)
    
# boxplot algorithm comparison
fig = plt.figure(figsize=(40, 20))
fig.suptitle('Algorithm Comparison', fontsize=40)
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names, fontdict={'fontsize': 20})
plt.show()
```

    SupportVectorClassifier: 0.825930 (0.020147)
    StochasticGradientDecentC: 0.789953 (0.023700)
    RandomForestClassifier: 0.874421 (0.014710)
    DecisionTreeClassifier: 0.818526 (0.019489)
    GaussianNB: 0.729720 (0.029256)
    KNeighborsClassifier: 0.830781 (0.018867)
    AdaBoostClassifier: 0.818019 (0.017404)
    LogisticRegression: 0.804491 (0.017744)
    


<img src="/assets/img/wine quality/output_85_1.png">


### Hyperparameter Tuning

#### SVC


```python
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
```


```python
def svc_param_selection(X, y, nfolds):
    param = {
        'C': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4],
        'kernel': ['linear', 'rbf'],
        'gamma': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]
    }
    grid_search = GridSearchCV(svc, param_grid=param, 
                               scoring='accuracy', cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_

svc_param_selection(X_train, y_train, 10)
```




    {'C': 1.4, 'gamma': 1.1, 'kernel': 'rbf'}




```python
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error

svc2 = SVC(C= 1.4, gamma= 1.1, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print('Confusion matrix')
print(confusion_matrix(y_test, pred_svc2))
print('Classification report')
print(classification_report(y_test, pred_svc2))
print('Accuracy score',accuracy_score(y_test, pred_svc2))
```

    Confusion matrix
    [[730  23]
     [103 124]]
    Classification report
                  precision    recall  f1-score   support
    
               0       0.88      0.97      0.92       753
               1       0.84      0.55      0.66       227
    
        accuracy                           0.87       980
       macro avg       0.86      0.76      0.79       980
    weighted avg       0.87      0.87      0.86       980
    
    Accuracy score 0.8714285714285714
    

#### SGD Classifier


```python
sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=60)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)
```

#### Random Forest Classifier


```python
rfc = RandomForestClassifier(n_estimators=200, max_depth=20,
                             random_state=0)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
```

#### KNeighbors Classifier


```python
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
```


```python
def knn_param_selection(X, y, nfolds):
    param = {
        'n_neighbors': [2, 3, 4, 5, 6],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2]
    }
    grid_search = GridSearchCV(knn, param_grid=param, 
                               scoring='accuracy', cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_

knn_param_selection(X_train, y_train, 10)
```




    {'algorithm': 'auto', 'n_neighbors': 6, 'p': 1, 'weights': 'distance'}




```python
knn2 = KNeighborsClassifier(algorithm= 'auto', 
                            n_neighbors= 5, p=2,
                           weights='distance')
knn2.fit(X_train, y_train)
pred_knn2 = knn2.predict(X_test)
print('Confusion matrix')
print(confusion_matrix(y_test, pred_knn2))
print('Classification report')
print(classification_report(y_test, pred_knn2))
print('Accuracy score',accuracy_score(y_test, pred_knn2))
```

    Confusion matrix
    [[708  45]
     [ 75 152]]
    Classification report
                  precision    recall  f1-score   support
    
               0       0.90      0.94      0.92       753
               1       0.77      0.67      0.72       227
    
        accuracy                           0.88       980
       macro avg       0.84      0.80      0.82       980
    weighted avg       0.87      0.88      0.87       980
    
    Accuracy score 0.8775510204081632
    

#### AdaBoost Classifier


```python
ada_classifier = AdaBoostClassifier(n_estimators=100)
ada_classifier.fit(X_train, y_train)
pred_ada = ada_classifier.predict(X_test)
```


```python
# cross validation
scores = cross_val_score(ada_classifier,X_test,y_test, cv=5)
print('Accuracy score',scores.mean())
```

    Accuracy score 0.7704081632653061
    

### Model Evaluation


```python
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(
        mean_absolute_error(test_labels, predictions)))
    print('Accuracy = {:0.2f}%.'.format(accuracy_score(
        test_labels, predictions)*100))
```


```python
evaluate(svc,X_test,y_test)
evaluate(svc2,X_test,y_test)
evaluate(sgd,X_test,y_test)
evaluate(rfc,X_test,y_test)
evaluate(knn2, X_test, y_test)
evaluate(ada_classifier,X_test,y_test)
```

    Model Performance
    Average Error: 0.1796 degrees.
    Accuracy = 82.04%.
    Model Performance
    Average Error: 0.1286 degrees.
    Accuracy = 87.14%.
    Model Performance
    Average Error: 0.2235 degrees.
    Accuracy = 77.65%.
    Model Performance
    Average Error: 0.1276 degrees.
    Accuracy = 87.24%.
    Model Performance
    Average Error: 0.1224 degrees.
    Accuracy = 87.76%.
    Model Performance
    Average Error: 0.2071 degrees.
    Accuracy = 79.29%.
    

### Feature Importance


```python
importance = rfc.feature_importances_

std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importance)

# plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.barh(range(X.shape[1]), importance[indices],
       color="b",  align="center")

plt.yticks(range(X.shape[1]), column_names)
plt.ylim([0, X.shape[1]])
plt.show()
```

<img src="/assets/img/wine quality/output_105_0.png">

