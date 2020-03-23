---
layout: post
title: Analyzing China's GDP growth using Non-Linear Regression
tags: [regression, plotting, modeling, machine learning]
---

In this blog post, we will analyze China's GDP growth from the year 1960 to 2019. If the data shows a curvy trend, then linear regression will not produce very accurate results when compared to a non-linear regression.



```python
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# read the data into a pandas dataframe
df = pd.read_csv('/content/china_gdp.csv')
df
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
      <th>Year</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1960</td>
      <td>5.918412e+10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1961</td>
      <td>4.955705e+10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1962</td>
      <td>4.668518e+10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1963</td>
      <td>5.009730e+10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1964</td>
      <td>5.906225e+10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1965</td>
      <td>6.970915e+10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1966</td>
      <td>7.587943e+10</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1967</td>
      <td>7.205703e+10</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1968</td>
      <td>6.999350e+10</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1969</td>
      <td>7.871882e+10</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1970</td>
      <td>9.150621e+10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1971</td>
      <td>9.856202e+10</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1972</td>
      <td>1.121598e+11</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1973</td>
      <td>1.367699e+11</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1974</td>
      <td>1.422547e+11</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1975</td>
      <td>1.611625e+11</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1976</td>
      <td>1.516277e+11</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1977</td>
      <td>1.723490e+11</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1978</td>
      <td>1.483821e+11</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1979</td>
      <td>1.768565e+11</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1980</td>
      <td>1.896500e+11</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1981</td>
      <td>1.943690e+11</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1982</td>
      <td>2.035496e+11</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1983</td>
      <td>2.289502e+11</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1984</td>
      <td>2.580821e+11</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1985</td>
      <td>3.074796e+11</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1986</td>
      <td>2.988058e+11</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1987</td>
      <td>2.713498e+11</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1988</td>
      <td>3.107222e+11</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>3.459575e+11</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1990</td>
      <td>3.589732e+11</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1991</td>
      <td>3.814547e+11</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1992</td>
      <td>4.249341e+11</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1993</td>
      <td>4.428746e+11</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1994</td>
      <td>5.622611e+11</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1995</td>
      <td>7.320320e+11</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1996</td>
      <td>8.608441e+11</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1997</td>
      <td>9.581594e+11</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1998</td>
      <td>1.025277e+12</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1999</td>
      <td>1.089447e+12</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2000</td>
      <td>1.205261e+12</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2001</td>
      <td>1.332235e+12</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2002</td>
      <td>1.461906e+12</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2003</td>
      <td>1.649929e+12</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2004</td>
      <td>1.941746e+12</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2005</td>
      <td>2.268599e+12</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2006</td>
      <td>2.729784e+12</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2007</td>
      <td>3.523094e+12</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2008</td>
      <td>4.558431e+12</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2009</td>
      <td>5.059420e+12</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2010</td>
      <td>6.039659e+12</td>
    </tr>
    <tr>
      <th>51</th>
      <td>2011</td>
      <td>7.492432e+12</td>
    </tr>
    <tr>
      <th>52</th>
      <td>2012</td>
      <td>8.461623e+12</td>
    </tr>
    <tr>
      <th>53</th>
      <td>2013</td>
      <td>9.490603e+12</td>
    </tr>
    <tr>
      <th>54</th>
      <td>2014</td>
      <td>1.035483e+13</td>
    </tr>
    <tr>
      <th>55</th>
      <td>2015</td>
      <td>1.105995e+13</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2016</td>
      <td>1.123700e+13</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2017</td>
      <td>1.232317e+13</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2018</td>
      <td>1.389188e+13</td>
    </tr>
    <tr>
      <th>59</th>
      <td>2019</td>
      <td>1.436348e+13</td>
    </tr>
  </tbody>
</table>
</div>



### Plot the data


```python
plt.figure(figsize=(8,5))
x_data, y_data = (df['Year'].values, df['Value'].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()
```


<img src="/assets/img/non-linear china/output_4_0.png">


We can see that the growth starts off slow. Then, from 2005 onwards, the growth is very significant. It decelerates slightly after the period of the 2008 global recession.

### Choosing a model
Looking at the plot, a logistic function would be a good approximation, since it has the property of starting with a slow growth, increasing growth in the middle, and then decreasing again at the end.  
Let's check this assumption below:


```python
X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))

plt.plot(X, Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
```


<img src="/assets/img/non-linear china/output_7_0.png">


### Build the model
Let's build our regression model and initialize its parameters.


```python
def sigmoid(x, Beta_1, Beta_2):
  y = 1/ (1 + np.exp(-Beta_1 * (x - Beta_2)))
  return y
```

Let's look at a sample sigmoid line that might fit with the data.


```python
beta_1 = 0.1
beta_2 = 1990

# logistic function
Y_pred = sigmoid(x_data, beta_1, beta_2)

# plot initial prediction againts data points
plt.plot(x_data, Y_pred*15000000000000)
plt.plot(x_data, y_data, 'ro')
```




    [<matplotlib.lines.Line2D at 0x7f5c93a4d780>]




<img src="/assets/img/non-linear china/output_11_1.png">


Our task is to find the best parameters for the model.  
First, lets normalize our x and y.


```python
xdata = x_data / max(x_data)
ydata = y_data / max(y_data)
```

**How can we find the best parameters for our fit line?**  
We can use `curve_fit`, which uses non-linear least squares to fit our sigmoid function to the data.


```python
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)

# print the final parameters
print('beta_1=%f, beta_2=%f' % (popt[0], popt[1]))
```

    beta_1 = 571.415035, beta_2 = 0.995885
    

### Plot the model


```python
# plot the resulting regression model
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x, y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()
```


<img src="/assets/img/non-linear china/output_17_0.png">


### Train/Test Split the data
Split data into training and testing sets.


```python
msk = np.random.randn(len(df)) < 0.8
train_x = x_data[msk]
test_x = xdata[~msk]
train_y = y_data[msk]
test_y = ydata[~msk]
```

Build the model using the train set.


```python
popt, pcov = curve_fit(sigmoid, train_x, train_y)
```

    /usr/local/lib/python3.6/dist-packages/scipy/optimize/minpack.py:808: OptimizeWarning: Covariance of the parameters could not be estimated
      category=OptimizeWarning)
    

Predict GDP using the test set.


```python
y_hat = sigmoid(test_x, *popt)
```

### Evaluate the model


```python
print('Mean absolute error: %.2f' % np.mean(np.absolute(y_hat - test_y)))
print('Residual sum of error (MSE): %.2f' % np.mean((y_hat - test_y)**2))

from sklearn.metrics import r2_score
print('R2-score: %.2f' % r2_score(y_hat, test_y))
```

    Mean absolute error: 0.40
    Residual sum of error (MSE): 0.17
    R2-score: -34427.16
    


```python

```
