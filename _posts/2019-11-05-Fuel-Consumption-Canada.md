---
layout: post
title: Analyzing Emission Ratings for all cars in Canada (model year 2019)
tags: [regression, plotting, modeling, machine learning]
---

In this blog post, we will use scikit-learn to implement different types of linear regression on our dataset. Then, we will split our data into training and testing sets, create a model using the training set, evaluate the model using a test set, and finally use the model to predict an unknown value.
The dataset is related to the Fuel Consumption and Carbon Dioxide Emission of all cars for retail sale in Canada in the year 2019. You can download the dataset here: https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64

**Understanding the data:**  
**Model:** 4WD/4X4 = Four-wheel drive, AWD = All-wheel drive, FFV = Flexible-fuel vehicle, SWB = Short wheelbase, LWB = Long wheelbase, EWB = Extended wheelbase  
**Transmission:** A = automatic, AM = automated manual, AS = automatic with select shift, AV = continuously variable, M = manual, 3 – 10 = Number of gears  
**Fuel type:** X = regular gasoline, Z = premium gasoline, D = diesel, E = ethanol (E85), N = natural gas  
**Fuel consumption:** City and highway fuel consumption ratings are shown in litres per 100 kilometres (L/100 km); the combined rating (55% city, 45% hwy) is shown in L/100 km and in miles per imperial gallon (mpg)  
**CO2 emissions:** the tailpipe emissions of carbon dioxide (in grams per kilometre) for combined city and highway driving  
**CO2 rating:** the tailpipe emissions of carbon dioxide rated on a scale from 1 (worst) to 10 (best)  
**Smog rating:** the tailpipe emissions of smog-forming pollutants rated on a scale from 1 (worst) to 10 (best)


```python
# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline
```

### Explore the data
Read the data into a pandas dataframe.


```python
df = pd.read_csv('MY2019 Fuel Consumption Ratings.csv', skipfooter=30)
df.head()
```

    C:\Users\prana\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support skipfooter; you can avoid this warning by specifying engine='python'.
      """Entry point for launching an IPython kernel.
    




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
      <th>Model Year</th>
      <th>Make</th>
      <th>Model</th>
      <th>Vehicle Class</th>
      <th>Engine Size (L)</th>
      <th>Cylinders</th>
      <th>Transmission</th>
      <th>Fuel Type</th>
      <th>Fuel Consumption City (L/100 km)</th>
      <th>Fuel Consumption Hwy (L/100 km)</th>
      <th>Fuel Consumption Comb (L/100 km)</th>
      <th>Fuel Consumption Comb (mpg)</th>
      <th>CO2 Emissions (g/km)</th>
      <th>CO2 Rating</th>
      <th>Smog Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019</td>
      <td>Acura</td>
      <td>ILX</td>
      <td>Compact</td>
      <td>2.4</td>
      <td>4</td>
      <td>AM8</td>
      <td>Z</td>
      <td>9.9</td>
      <td>7.0</td>
      <td>8.6</td>
      <td>33</td>
      <td>199</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019</td>
      <td>Acura</td>
      <td>MDX SH-AWD</td>
      <td>SUV: Small</td>
      <td>3.5</td>
      <td>6</td>
      <td>AS9</td>
      <td>Z</td>
      <td>12.2</td>
      <td>9.0</td>
      <td>10.8</td>
      <td>26</td>
      <td>252</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>Acura</td>
      <td>MDX SH-AWD A-SPEC</td>
      <td>SUV: Small</td>
      <td>3.5</td>
      <td>6</td>
      <td>AS9</td>
      <td>Z</td>
      <td>12.2</td>
      <td>9.5</td>
      <td>11.0</td>
      <td>26</td>
      <td>258</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019</td>
      <td>Acura</td>
      <td>MDX Hybrid AWD</td>
      <td>SUV: Small</td>
      <td>3.0</td>
      <td>6</td>
      <td>AM7</td>
      <td>Z</td>
      <td>9.1</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>31</td>
      <td>210</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019</td>
      <td>Acura</td>
      <td>NSX</td>
      <td>Two-seater</td>
      <td>3.5</td>
      <td>6</td>
      <td>AM9</td>
      <td>Z</td>
      <td>11.1</td>
      <td>10.8</td>
      <td>11.0</td>
      <td>26</td>
      <td>261</td>
      <td>4</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# dimensions of the dataframe
df.shape
```




    (1041, 15)




```python
# summarize the data
df.describe()
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
      <th>Model Year</th>
      <th>Engine Size (L)</th>
      <th>Cylinders</th>
      <th>Fuel Consumption City (L/100 km)</th>
      <th>Fuel Consumption Hwy (L/100 km)</th>
      <th>Fuel Consumption Comb (L/100 km)</th>
      <th>Fuel Consumption Comb (mpg)</th>
      <th>CO2 Emissions (g/km)</th>
      <th>CO2 Rating</th>
      <th>Smog Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1041.0</td>
      <td>1041.000000</td>
      <td>1041.000000</td>
      <td>1041.000000</td>
      <td>1041.000000</td>
      <td>1041.000000</td>
      <td>1041.000000</td>
      <td>1041.000000</td>
      <td>1041.000000</td>
      <td>1041.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2019.0</td>
      <td>3.114121</td>
      <td>5.603266</td>
      <td>12.382901</td>
      <td>9.041114</td>
      <td>10.876657</td>
      <td>27.633045</td>
      <td>251.396734</td>
      <td>4.557157</td>
      <td>4.128722</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>1.316896</td>
      <td>1.797741</td>
      <td>3.301165</td>
      <td>2.053922</td>
      <td>2.702855</td>
      <td>7.332346</td>
      <td>57.134349</td>
      <td>1.655076</td>
      <td>1.789056</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2019.0</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>4.200000</td>
      <td>4.000000</td>
      <td>4.100000</td>
      <td>13.000000</td>
      <td>96.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2019.0</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>10.200000</td>
      <td>7.600000</td>
      <td>9.100000</td>
      <td>22.000000</td>
      <td>212.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2019.0</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>12.100000</td>
      <td>8.800000</td>
      <td>10.600000</td>
      <td>27.000000</td>
      <td>248.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2019.0</td>
      <td>3.600000</td>
      <td>6.000000</td>
      <td>14.300000</td>
      <td>10.200000</td>
      <td>12.600000</td>
      <td>31.000000</td>
      <td>290.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2019.0</td>
      <td>8.000000</td>
      <td>16.000000</td>
      <td>26.800000</td>
      <td>17.200000</td>
      <td>22.200000</td>
      <td>69.000000</td>
      <td>522.000000</td>
      <td>10.000000</td>
      <td>7.000000</td>
    </tr>
  </tbody>
</table>
</div>



Let's rename the columns for the sake of simplicity.


```python
df.rename(columns={'Engine Size (L)':'Engine_Size', 'Fuel Consumption Comb (L/100 km)':'Fuel_Consumption', 'CO2 Emissions (g/km)':'CO2_Emissions'}, inplace=True)
df.head(2)
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
      <th>Model Year</th>
      <th>Make</th>
      <th>Model</th>
      <th>Vehicle Class</th>
      <th>Engine_Size</th>
      <th>Cylinders</th>
      <th>Transmission</th>
      <th>Fuel Type</th>
      <th>Fuel Consumption City (L/100 km)</th>
      <th>Fuel Consumption Hwy (L/100 km)</th>
      <th>Fuel_Consumption</th>
      <th>Fuel Consumption Comb (mpg)</th>
      <th>CO2_Emissions</th>
      <th>CO2 Rating</th>
      <th>Smog Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019</td>
      <td>Acura</td>
      <td>ILX</td>
      <td>Compact</td>
      <td>2.4</td>
      <td>4</td>
      <td>AM8</td>
      <td>Z</td>
      <td>9.9</td>
      <td>7.0</td>
      <td>8.6</td>
      <td>33</td>
      <td>199</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019</td>
      <td>Acura</td>
      <td>MDX SH-AWD</td>
      <td>SUV: Small</td>
      <td>3.5</td>
      <td>6</td>
      <td>AS9</td>
      <td>Z</td>
      <td>12.2</td>
      <td>9.0</td>
      <td>10.8</td>
      <td>26</td>
      <td>252</td>
      <td>4</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
cdf = df[['Engine_Size', 'Cylinders', 'Fuel_Consumption', 'CO2_Emissions']]
cdf.head()
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
      <th>Engine_Size</th>
      <th>Cylinders</th>
      <th>Fuel_Consumption</th>
      <th>CO2_Emissions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.4</td>
      <td>4</td>
      <td>8.6</td>
      <td>199</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.5</td>
      <td>6</td>
      <td>10.8</td>
      <td>252</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.5</td>
      <td>6</td>
      <td>11.0</td>
      <td>258</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>6</td>
      <td>9.0</td>
      <td>210</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.5</td>
      <td>6</td>
      <td>11.0</td>
      <td>261</td>
    </tr>
  </tbody>
</table>
</div>



We can plot each of these features to determine their frequency distribution.


```python
viz = cdf[['Engine_Size', 'Cylinders', 'Fuel_Consumption', 'CO2_Emissions']]
viz.hist()
plt.show()
```


![png](output_10_0.png)


Now let's plot each of these features vs the CO2 Emissions to see how linear is their relationship.


```python
plt.scatter(cdf.Fuel_Consumption, cdf.CO2_Emissions, color='blue')
plt.xlabel('Fuel Consumption')
plt.ylabel('CO2 Emissions')
plt.show()
```


![png](output_12_0.png)



```python
plt.scatter(cdf.Engine_Size, cdf.CO2_Emissions, color='blue')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.show()
```


![png](output_13_0.png)



```python
plt.scatter(cdf.Cylinders, cdf.CO2_Emissions, color='blue')
plt.xlabel('No. of Cylinders')
plt.ylabel('CO2 Emissions')
plt.show()
```


![png](output_14_0.png)


### Train/Test Split

Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. You train with the training set and test with the testing set.

Let's split our dataset: 80% of the data for training, and the remaining 20% for testing. Create a mask to select random rows using the `np.random.rand()` function.


```python
mask = np.random.rand(len(df)) < 0.8
train = cdf[mask]
test = cdf[~mask]
```

## Simple Linear Regression

### Modeling
Use the `sklearn` package to model the data.


```python
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Engine_Size']])
train_y = np.asanyarray(train[['CO2_Emissions']])
regr.fit(train_x, train_y)

# the coefficients
print('Coefficients: ', regr.coef_)
print('Intercept', regr.intercept_)
```

    Coefficients:  [[36.13934429]]
    Intercept [138.26102071]
    

The **coefficient** and **intercept** in simple linear regression are the only two parameters of the fit line. `sklearn` can estimate them directly from our data.

### Plot the outputs
Let's plot the fit line over the data.


```python
plt.scatter(train.Engine_Size, train.CO2_Emissions, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
```




    Text(0, 0.5, 'CO2 Emissions')




![png](output_21_1.png)


### Evaluation
We compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics provide insight to areas that require improvement.

There are different model evaluation metrics. We'll use MSE to calcualte the accuracy of our model based on the test set.
- Mean Absolute Error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just the average error.
- Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean Absolute Error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
- Root Mean Squared Error (RMSE): This is the square root of the Mean Square Error.
- R-squared: It is not an error, but is a popular metric for accuracy of your model. It represents how close the data are to the fitted regression line. The higher the R-squared, the better the model fits your data. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).


```python
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['Engine_Size']])
test_y = np.asanyarray(test[['CO2_Emissions']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y)**2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y))
```

    Mean absolute error: 23.94
    Residual sum of squares (MSE): 955.46
    R2-score: 0.57
    

## Multiple Linear Regression
In reality, there are multiple variables that predict CO2 emissions. When more than one independent variable is present, the process is called multiple linear regression.

### Modeling


```python
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Engine_Size', 'Cylinders', 'Fuel_Consumption']])
train_y = np.asanyarray(train[['CO2_Emissions']])
regr.fit(train_x, train_y)

# the coefficients
print('Coefficients: ', regr.coef_)
print('Intercept', regr.intercept_)
```

    Coefficients:  [[ 4.56544675  4.79488067 15.1778644 ]]
    Intercept [44.85260676]
    

The **coefficients** and **intercept** are the parameters of the fit line. Given that it is a mulitple linear regression, with 3 parameters, `sklearn` can estimate them from our data.

### Prediction


```python
y_hat = regr.predict(test[['Engine_Size', 'Cylinders', 'Fuel_Consumption']])
x = np.asanyarray(test[['Engine_Size', 'Cylinders', 'Fuel_Consumption']])
y = np.asanyarray(test[['CO2_Emissions']])

print('Residual sum of squares (MSE): %.2f' % np.mean((y_hat - y)**2))
print('Variance score:%.2f' % regr.score(x, y))
```

    Residual sum of squares (MSE): 288.23
    Variance score:0.92
    

If $\hat{y}$ is the estimated target output, y the corresponding (correct) target output, and Var is Variance, the square of the standard deviation, then the explained variance is estimated as follow:

$\texttt{explainedVariance}(y, \hat{y}) = 1 - \frac{Var\{ y - \hat{y}\}}{Var\{y\}}$  
The best possible score is 1.0, lower values are worse.

As suspected, the multiple linear regression model provides a higher variance score and a lower MSE. Hence, it is more accurate than the simple linear regression model.

## Polynomial Regression
Sometimes, the trend of data is not really linear, and looks curvy. In this case we can use Polynomial regression methods. In fact, many different regressions exist that can be used to fit whatever the dataset looks like, such as quadratic, cubic, and so on, and it can go on and on to infinite degrees. We can call all of these polynomial regression, where the relationship between the independent variable x and the dependent variable y is modeled as an nth degree polynomial in x.  
`PolynomialFeatures()` function in Scikit-learn library, derives a new feature set from the original feature set. That is, a matrix will be generated consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, lets say the original feature set has only one feature, *Engine_Size*. Now, if we select the degree of the polynomial to be 2, then it generates 3 features, degree=0, degree=1 and degree=2:


```python
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
```

### Modeling


```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['Engine_Size']])
train_y = np.asanyarray(train[['CO2_Emissions']])
test_x = np.asanyarray(test[['Engine_Size']])
test_y = np.asanyarray(test[['CO2_Emissions']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
train_x_poly
```




    array([[ 1.  ,  3.5 , 12.25],
           [ 1.  ,  3.5 , 12.25],
           [ 1.  ,  3.  ,  9.  ],
           ...,
           [ 1.  ,  2.  ,  4.  ],
           [ 1.  ,  2.  ,  4.  ],
           [ 1.  ,  2.  ,  4.  ]])



**fit_transform** takes our x values, and outputs a list of our data raised from power of 0 to power of 2 (since we set the degree of our polynomial to 2).

$
\begin{bmatrix}
    v_1\\
    v_2\\
    \vdots\\
    v_n
\end{bmatrix}
$
$\longrightarrow$
$
\begin{bmatrix}
    [ 1 & v_1 & v_1^2]\\
    [ 1 & v_2 & v_2^2]\\
    \vdots & \vdots & \vdots\\
    [ 1 & v_n & v_n^2]
\end{bmatrix}
$


It looks like feature sets for multiple linear regression analysis, right? Yes. It Does. 
Indeed, Polynomial regression is a special case of linear regression, with the main idea of how do you select your features. Just consider replacing the  $x$ with $x_1$, $x_1^2$ with $x_2$, and so on. Then the degree 2 equation would turn into:

$y = b + \theta_1  x_1 + \theta_2 x_2$

Now, we can deal with it as 'linear regression' problem. Therefore, this polynomial regression is considered to be a special case of traditional multiple linear regression. So, you can use the same mechanism as linear regression to solve such a problems.   
So we can use __LinearRegression()__ function to solve it:


```python
clf = linear_model.LinearRegression()
train_y = clf.fit(train_x_poly, train_y)

# the coefficients
print('Coefficients: ', clf.coef_)
print('Intercept: ', clf.intercept_)
```

    Coefficients:  [[ 0.         56.58731642 -2.65311782]]
    Intercept:  [106.13683949]
    

### Plot the model


```python
plt.scatter(train.Engine_Size, train.CO2_Emissions, color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0] + clf.coef_[0][1]*XX + clf.coef_[0][2]*np.power(XX,2)
plt.plot(XX, yy, '-r')
plt.xlabel('Engine size')
plt.ylabel('Emission')
```




    Text(0, 0.5, 'Emission')




![png](output_38_1.png)


### Evaluation


```python
from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print('Mean absolute error: %.2f' % np.mean(np.absolute(test_y_ - test_y)))
print('Residual sum of squares (MSE): %.2f' % np.mean((test_y_ - test_y)**2))
print('R2-score: %.2f' % r2_score(test_y_, test_y))
```

    Mean absolute error: 23.76
    Residual sum of squares (MSE): 970.46
    R2-score: 0.59
    

Let's use a polynomial regression with degree three to try to improve the model.


```python
#modeling
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['Engine_Size']])
train_y = np.asanyarray(train[['CO2_Emissions']])
test_x = np.asanyarray(test[['Engine_Size']])
test_y = np.asanyarray(test[['CO2_Emissions']])
poly3 = PolynomialFeatures(degree=3)
train_x_poly3 = poly3.fit_transform(train_x)
clf3 = linear_model.LinearRegression()
train_y3_ = clf3.fit(train_x_poly3, train_y)
# The coefficients
print ('Coefficients: ', clf3.coef_)
print ('Intercept: ',clf3.intercept_)
#plotting
plt.scatter(train.Engine_Size, train.CO2_Emissions,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf3.intercept_[0]+ clf3.coef_[0][1]*XX + clf3.coef_[0][2]*np.power(XX, 2) + clf3.coef_[0][3]*np.power(XX, 3)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
# evaluation
test_x_poly3 = poly3.fit_transform(test_x)
test_y3_ = clf3.predict(test_x_poly3)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y3_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y3_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y3_ , test_y))
```

    Coefficients:  [[  0.         111.3343238  -18.52676551   1.38049377]]
    Intercept:  [50.35011629]
    Mean absolute error: 23.21
    Residual sum of squares (MSE): 948.62
    R2-score: 0.59
    


![png](output_42_1.png)


The cubic regression model only results in slightly better accuracy.


```python

```
