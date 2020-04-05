# Data Analysis with Python
We will be using a dataset about cars from back in 1985. This data set consists of three types of entities:  
- the specification of an auto in terms of various characteristics,
- its assigned insurance risk rating,
- its normalized losses in use as compared to other cars.  

The second rating corresponds to the degree to which the auto is more risky than its price indicates. Cars are initially assigned a risk factor symbol associated with its price. Then, if it is more risky (or less), this symbol is adjusted by moving it up (or down) the scale. Actuarians call this process "symboling". A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe. The third factor is the relative average loss payment per insured vehicle year. This value is normalized for all autos within a particular size classification (two-door small, station wagons, sports/specialty, etc…), and represents the average loss per car per year.

#### Attribute Information:
    symboling: -3, -2, -1, 0, 1, 2, 3
    normalized-losses: continuous from 65 to 256
    make: alfa-romero, audi, bmw, chevrolet, dodge, honda, isuzu, jaguar, mazda, mercedes-benz, mercury, mitsubishi, nissan, peugot, plymouth, porsche, renault, saab, subaru, toyota, volkswagen, volvo
    fuel-type: diesel, gas
    aspiration: std, turbo
    num-of-doors: four, two
    body-style: hardtop, wagon, sedan, hatchback, convertible
    drive-wheels: 4wd, fwd, rwd
    engine-location: front, rear
    wheel-base: continuous from 86.6 120.9
    length: continuous from 141.1 to 208.1
    width: continuous from 60.3 to 72.3
    height: continuous from 47.8 to 59.8
    curb-weight: continuous from 1488 to 4066
    engine-type: dohc, dohcv, l, ohc, ohcf, ohcv, rotor
    num-of-cylinders: eight, five, four, six, three, twelve, two
    engine-size: continuous from 61 to 326
    fuel-system: 1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi
    bore: continuous from 2.54 to 3.94
    stroke: continuous from 2.07 to 4.17
    compression-ratio: continuous from 7 to 23
    horsepower: continuous from 48 to 288
    peak-rpm: continuous from 4150 to 6600
    city-mpg: continuous from 13 to 49
    highway-mpg: continuous from 16 to 54
    price: continuous from 5118 to 45400.


## Data Acquisition


```python
import pandas as pd
import numpy as np

# read the online file and assign it to the variable 'df'
path = 'imports-85.data'
df = pd.read_csv(path, header=None)

# print the first 10 rows of the dataset
print('The first 10 rows of the dataframe')
df.head(10)
```

    The first 10 rows of the dataframe
    




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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>?</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110</td>
      <td>5500</td>
      <td>19</td>
      <td>25</td>
      <td>15250</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>158</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>105.8</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110</td>
      <td>5500</td>
      <td>19</td>
      <td>25</td>
      <td>17710</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>?</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>wagon</td>
      <td>fwd</td>
      <td>front</td>
      <td>105.8</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110</td>
      <td>5500</td>
      <td>19</td>
      <td>25</td>
      <td>18920</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>158</td>
      <td>audi</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>105.8</td>
      <td>...</td>
      <td>131</td>
      <td>mpfi</td>
      <td>3.13</td>
      <td>3.40</td>
      <td>8.3</td>
      <td>140</td>
      <td>5500</td>
      <td>17</td>
      <td>20</td>
      <td>23875</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>?</td>
      <td>audi</td>
      <td>gas</td>
      <td>turbo</td>
      <td>two</td>
      <td>hatchback</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.5</td>
      <td>...</td>
      <td>131</td>
      <td>mpfi</td>
      <td>3.13</td>
      <td>3.40</td>
      <td>7.0</td>
      <td>160</td>
      <td>5500</td>
      <td>16</td>
      <td>22</td>
      <td>?</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 26 columns</p>
</div>




```python
# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)
```

    headers
     ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
    


```python
# replace the headers in the dataframe
df.columns = headers
```


```python
# view the data types
df.dtypes
```




    symboling              int64
    normalized-losses     object
    make                  object
    fuel-type             object
    aspiration            object
    num-of-doors          object
    body-style            object
    drive-wheels          object
    engine-location       object
    wheel-base           float64
    length               float64
    width                float64
    height               float64
    curb-weight            int64
    engine-type           object
    num-of-cylinders      object
    engine-size            int64
    fuel-system           object
    bore                  object
    stroke                object
    compression-ratio    float64
    horsepower            object
    peak-rpm              object
    city-mpg               int64
    highway-mpg            int64
    price                 object
    dtype: object




```python
# get a statistical summary of each column
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
      <th>symboling</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>engine-size</th>
      <th>compression-ratio</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.834146</td>
      <td>98.756585</td>
      <td>174.049268</td>
      <td>65.907805</td>
      <td>53.724878</td>
      <td>2555.565854</td>
      <td>126.907317</td>
      <td>10.142537</td>
      <td>25.219512</td>
      <td>30.751220</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.245307</td>
      <td>6.021776</td>
      <td>12.337289</td>
      <td>2.145204</td>
      <td>2.443522</td>
      <td>520.680204</td>
      <td>41.642693</td>
      <td>3.972040</td>
      <td>6.542142</td>
      <td>6.886443</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.000000</td>
      <td>86.600000</td>
      <td>141.100000</td>
      <td>60.300000</td>
      <td>47.800000</td>
      <td>1488.000000</td>
      <td>61.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>94.500000</td>
      <td>166.300000</td>
      <td>64.100000</td>
      <td>52.000000</td>
      <td>2145.000000</td>
      <td>97.000000</td>
      <td>8.600000</td>
      <td>19.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>97.000000</td>
      <td>173.200000</td>
      <td>65.500000</td>
      <td>54.100000</td>
      <td>2414.000000</td>
      <td>120.000000</td>
      <td>9.000000</td>
      <td>24.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>102.400000</td>
      <td>183.100000</td>
      <td>66.900000</td>
      <td>55.500000</td>
      <td>2935.000000</td>
      <td>141.000000</td>
      <td>9.400000</td>
      <td>30.000000</td>
      <td>34.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>120.900000</td>
      <td>208.100000</td>
      <td>72.300000</td>
      <td>59.800000</td>
      <td>4066.000000</td>
      <td>326.000000</td>
      <td>23.000000</td>
      <td>49.000000</td>
      <td>54.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe(include='all')
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
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>wheel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>205.000000</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205.000000</td>
      <td>...</td>
      <td>205.000000</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205.000000</td>
      <td>205</td>
      <td>205</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>52</td>
      <td>22</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>8</td>
      <td>39</td>
      <td>37</td>
      <td>NaN</td>
      <td>60</td>
      <td>24</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>187</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>?</td>
      <td>toyota</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>mpfi</td>
      <td>3.62</td>
      <td>3.40</td>
      <td>NaN</td>
      <td>68</td>
      <td>5500</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>?</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>41</td>
      <td>32</td>
      <td>185</td>
      <td>168</td>
      <td>114</td>
      <td>96</td>
      <td>120</td>
      <td>202</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>94</td>
      <td>23</td>
      <td>20</td>
      <td>NaN</td>
      <td>19</td>
      <td>37</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.834146</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>98.756585</td>
      <td>...</td>
      <td>126.907317</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.142537</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25.219512</td>
      <td>30.751220</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.245307</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.021776</td>
      <td>...</td>
      <td>41.642693</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.972040</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.542142</td>
      <td>6.886443</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>86.600000</td>
      <td>...</td>
      <td>61.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.000000</td>
      <td>16.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>94.500000</td>
      <td>...</td>
      <td>97.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.600000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.000000</td>
      <td>25.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>97.000000</td>
      <td>...</td>
      <td>120.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.000000</td>
      <td>30.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>102.400000</td>
      <td>...</td>
      <td>141.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.400000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.000000</td>
      <td>34.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>120.900000</td>
      <td>...</td>
      <td>326.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>49.000000</td>
      <td>54.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 26 columns</p>
</div>




```python
# get the summary of specific columns
df[['length', 'compression-ratio']].describe()
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
      <th>length</th>
      <th>compression-ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>205.000000</td>
      <td>205.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>174.049268</td>
      <td>10.142537</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.337289</td>
      <td>3.972040</td>
    </tr>
    <tr>
      <th>min</th>
      <td>141.100000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>166.300000</td>
      <td>8.600000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>173.200000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>183.100000</td>
      <td>9.400000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>208.100000</td>
      <td>23.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get a concise summary (top 30 & bottom 30 rows)
df.info
```




    <bound method DataFrame.info of      symboling normalized-losses         make fuel-type aspiration  \
    0            3                 ?  alfa-romero       gas        std   
    1            3                 ?  alfa-romero       gas        std   
    2            1                 ?  alfa-romero       gas        std   
    3            2               164         audi       gas        std   
    4            2               164         audi       gas        std   
    ..         ...               ...          ...       ...        ...   
    200         -1                95        volvo       gas        std   
    201         -1                95        volvo       gas      turbo   
    202         -1                95        volvo       gas        std   
    203         -1                95        volvo    diesel      turbo   
    204         -1                95        volvo       gas      turbo   
    
        num-of-doors   body-style drive-wheels engine-location  wheel-base  ...  \
    0            two  convertible          rwd           front        88.6  ...   
    1            two  convertible          rwd           front        88.6  ...   
    2            two    hatchback          rwd           front        94.5  ...   
    3           four        sedan          fwd           front        99.8  ...   
    4           four        sedan          4wd           front        99.4  ...   
    ..           ...          ...          ...             ...         ...  ...   
    200         four        sedan          rwd           front       109.1  ...   
    201         four        sedan          rwd           front       109.1  ...   
    202         four        sedan          rwd           front       109.1  ...   
    203         four        sedan          rwd           front       109.1  ...   
    204         four        sedan          rwd           front       109.1  ...   
    
         engine-size  fuel-system  bore  stroke compression-ratio horsepower  \
    0            130         mpfi  3.47    2.68               9.0        111   
    1            130         mpfi  3.47    2.68               9.0        111   
    2            152         mpfi  2.68    3.47               9.0        154   
    3            109         mpfi  3.19    3.40              10.0        102   
    4            136         mpfi  3.19    3.40               8.0        115   
    ..           ...          ...   ...     ...               ...        ...   
    200          141         mpfi  3.78    3.15               9.5        114   
    201          141         mpfi  3.78    3.15               8.7        160   
    202          173         mpfi  3.58    2.87               8.8        134   
    203          145          idi  3.01    3.40              23.0        106   
    204          141         mpfi  3.78    3.15               9.5        114   
    
         peak-rpm city-mpg highway-mpg  price  
    0        5000       21          27  13495  
    1        5000       21          27  16500  
    2        5000       19          26  16500  
    3        5500       24          30  13950  
    4        5500       18          22  17450  
    ..        ...      ...         ...    ...  
    200      5400       23          28  16845  
    201      5300       19          25  19045  
    202      5500       18          23  21485  
    203      4800       26          27  22470  
    204      5400       19          25  22625  
    
    [205 rows x 26 columns]>



## Data Wrangling
Data Wrangling is the process of converting data from the initial format to a format that may be better for analysis.


```python
# replace "?" with NaN
df.replace('?', np.nan, inplace=True)
```


```python
# identify the missing data
# use ".isnull()" or ".notnull()"
missing_data = df.isnull() # True stands for missing value
missing_data.head(10)
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
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>wheel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 26 columns</p>
</div>




```python
# count the missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
```

    symboling
    False    205
    Name: symboling, dtype: int64
    
    normalized-losses
    False    164
    True      41
    Name: normalized-losses, dtype: int64
    
    make
    False    205
    Name: make, dtype: int64
    
    fuel-type
    False    205
    Name: fuel-type, dtype: int64
    
    aspiration
    False    205
    Name: aspiration, dtype: int64
    
    num-of-doors
    False    203
    True       2
    Name: num-of-doors, dtype: int64
    
    body-style
    False    205
    Name: body-style, dtype: int64
    
    drive-wheels
    False    205
    Name: drive-wheels, dtype: int64
    
    engine-location
    False    205
    Name: engine-location, dtype: int64
    
    wheel-base
    False    205
    Name: wheel-base, dtype: int64
    
    length
    False    205
    Name: length, dtype: int64
    
    width
    False    205
    Name: width, dtype: int64
    
    height
    False    205
    Name: height, dtype: int64
    
    curb-weight
    False    205
    Name: curb-weight, dtype: int64
    
    engine-type
    False    205
    Name: engine-type, dtype: int64
    
    num-of-cylinders
    False    205
    Name: num-of-cylinders, dtype: int64
    
    engine-size
    False    205
    Name: engine-size, dtype: int64
    
    fuel-system
    False    205
    Name: fuel-system, dtype: int64
    
    bore
    False    201
    True       4
    Name: bore, dtype: int64
    
    stroke
    False    201
    True       4
    Name: stroke, dtype: int64
    
    compression-ratio
    False    205
    Name: compression-ratio, dtype: int64
    
    horsepower
    False    203
    True       2
    Name: horsepower, dtype: int64
    
    peak-rpm
    False    203
    True       2
    Name: peak-rpm, dtype: int64
    
    city-mpg
    False    205
    Name: city-mpg, dtype: int64
    
    highway-mpg
    False    205
    Name: highway-mpg, dtype: int64
    
    price
    False    201
    True       4
    Name: price, dtype: int64
    
    

In this dataset, none of the columns are empty enough to drop entirely.

**Replace by mean:**  
"normalized-losses": 41 missing data, replace them with mean  
"bore": 4 missing data, replace them with mean  
"stroke": 4 missing data, replace them with mean  
"horsepower": 2 missing data, replace them with mean  
"peak-rpm": 2 missing data, replace them with mean

**Replace by frequency:**  
"num-of-doors": 2 missing data, replace them with "four"  
Reason: 84% sedans are four-door. Since four doors is most frequent, it is most likely to occur

**Drop the whole row:**  
"price": 4 missing data, simply delete the whole row  
Reason: Price is what we want to predict. Any data entry without price data cannot be used for prediction; therefore any row now without price data is not useful to us.

##### Replace by mean


```python
# normalized-losses column
# calculate average of the column. astype('float') saves the mean value in float dtype.
avg_norm_loss = df['normalized-losses'].astype('float').mean(axis=0)
print('Average of normalized-losses:', avg_norm_loss)
```

    Average of normalized-losses: 122.0
    


```python
# normalized-losses column
# replace NaN by the mean value
df['normalized-losses'].replace(np.nan, avg_norm_loss, inplace=True)
```


```python
# bore column
# calculate average of the column. astype('float') saves the mean value in float dtype.
avg_bore = df['bore'].astype('float').mean(axis=0)
print('Average of bore:', avg_bore)
```

    Average of bore: 3.3297512437810957
    


```python
# bore column
# replace NaN by the mean value
df['bore'].replace(np.nan, avg_norm_loss, inplace=True)
```


```python
# stroke column
# calculate average of the column. astype('float') saves the mean value in float dtype.
avg_stroke = df['stroke'].astype('float').mean(axis=0)
print('Average of stroke:', avg_stroke)
```

    Average of stroke: 3.2554228855721337
    


```python
# stroke column
# replace NaN by the mean value
df['stroke'].replace(np.nan, avg_stroke, inplace=True)
```


```python
# horsepower column
# calculate average of the column. astpye('float') saves the mean value in flaot dtype
avg_hp = df['horsepower'].astype('float').mean(axis=0)
print('Average of horsepower: ', avg_hp)
```

    Average of horsepower:  104.25615763546799
    


```python
# horsepower column
# replace NaN by the ean value
df['horsepower'].replace(np.nan, avg_hp, inplace=True)
```


```python
# peak-rpm column
# calculate average of the column. astype('float') saves the mean value in float dtype.
avg_peakrpm = df['peak-rpm'].astype('float').mean(axis=0)
print('Average of peak-rpm:', avg_peakrpm)
```

    Average of peak-rpm: 5125.369458128079
    


```python
# peak-rpm column
# replace NaN by the mean value
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)
```

##### Replace by Frequency


```python
# identify which values are present in a particular column
df['num-of-doors'].value_counts()
```




    four    114
    two      89
    Name: num-of-doors, dtype: int64




```python
# use the ".idxmax()" method to calculate for us the most common type automatically
df['num-of-doors'].value_counts().idxmax()
```




    'four'




```python
# replace the missing 'num-of-doors' values by the most frequent
df['num-of-doors'].replace(np.nan, 'four', inplace=True)
```

##### Drop the whole row


```python
df.dropna(subset=['price'], axis=0, inplace=True)
```


```python
# reset the index because we dropped rows
df.reset_index(drop=True, inplace=True)
df.head()
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
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>wheel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
# check the data types
df.dtypes
```




    symboling              int64
    normalized-losses     object
    make                  object
    fuel-type             object
    aspiration            object
    num-of-doors          object
    body-style            object
    drive-wheels          object
    engine-location       object
    wheel-base           float64
    length               float64
    width                float64
    height               float64
    curb-weight            int64
    engine-type           object
    num-of-cylinders      object
    engine-size            int64
    fuel-system           object
    bore                  object
    stroke                object
    compression-ratio    float64
    horsepower            object
    peak-rpm              object
    city-mpg               int64
    highway-mpg            int64
    price                 object
    dtype: object




```python
# convert the data types into proper format
# use double brackets when including multiple columns in one statement
df[['bore', 'stroke', 'price', 'peak-rpm', 'horsepower']] = df[['bore', 'stroke', 'price', 'peak-rpm', 'horsepower']].astype('float')
df['normalized-losses'] = df['normalized-losses'].astype('int')
```


```python
df.dtypes
```




    symboling              int64
    normalized-losses      int32
    make                  object
    fuel-type             object
    aspiration            object
    num-of-doors          object
    body-style            object
    drive-wheels          object
    engine-location       object
    wheel-base           float64
    length               float64
    width                float64
    height               float64
    curb-weight            int64
    engine-type           object
    num-of-cylinders      object
    engine-size            int64
    fuel-system           object
    bore                 float64
    stroke               float64
    compression-ratio    float64
    horsepower           float64
    peak-rpm             float64
    city-mpg               int64
    highway-mpg            int64
    price                float64
    dtype: object



## Data Normalization
Normalization is the process of transforming values of several variables into a similar range. Typical normalizations include scaling the variable so the variable average is 0, scaling the variable so the variance is 1, or scaling variable so the variable values range from 0 to 1.


```python
# scale the columns 'length', 'width' and 'height'
# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()

df[['length', 'width', 'height']].head()
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
      <th>length</th>
      <th>width</th>
      <th>height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.811148</td>
      <td>0.890278</td>
      <td>0.816054</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.811148</td>
      <td>0.890278</td>
      <td>0.816054</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.822681</td>
      <td>0.909722</td>
      <td>0.876254</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.848630</td>
      <td>0.919444</td>
      <td>0.908027</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.848630</td>
      <td>0.922222</td>
      <td>0.908027</td>
    </tr>
  </tbody>
</table>
</div>



### Binning
Binning is a process of transforming continuous numerical variables into discrete categorical 'bins', for grouped analysis.


```python
df['horsepower'].describe()
```




    count    201.000000
    mean     103.405534
    std       37.365700
    min       48.000000
    25%       70.000000
    50%       95.000000
    75%      116.000000
    max      262.000000
    Name: horsepower, dtype: float64



In our dataset, "horsepower" is a real valued variable ranging from 48 to 288, it has 58 unique values. What if we only care about the price difference between cars with high horsepower, medium horsepower, and little horsepower (3 types)?


```python
%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
```




    Text(0.5, 1.0, 'horsepower bins')




![png](output_40_1.png)



```python
bins = np.linspace(min(df['horsepower']), max(df['horsepower']), 4)
bins
```




    array([ 48.        , 119.33333333, 190.66666667, 262.        ])




```python
# set the group names
group_names = ['low', 'medium', 'high']
```


```python
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)
df[['horsepower', 'horsepower-binned']].head(10)
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
      <th>horsepower</th>
      <th>horsepower-binned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>111.0</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>111.0</td>
      <td>low</td>
    </tr>
    <tr>
      <th>2</th>
      <td>154.0</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>102.0</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>115.0</td>
      <td>low</td>
    </tr>
    <tr>
      <th>5</th>
      <td>110.0</td>
      <td>low</td>
    </tr>
    <tr>
      <th>6</th>
      <td>110.0</td>
      <td>low</td>
    </tr>
    <tr>
      <th>7</th>
      <td>110.0</td>
      <td>low</td>
    </tr>
    <tr>
      <th>8</th>
      <td>140.0</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>9</th>
      <td>101.0</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['horsepower-binned'].value_counts()
```




    low       153
    medium     43
    high        5
    Name: horsepower-binned, dtype: int64




```python
# plot the distribution
%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
```




    Text(0.5, 1.0, 'horsepower bins')




![png](output_45_1.png)


#### Bins Visualization
Normally, a histogram is used to visualize the distribution of bins.


```python
%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot

a = (0, 1, 2)

# draw histogram of attribute 'horsepower' with bins=3
plt.pyplot.hist(df['horsepower'], bins=3)

# set x/y labels and plot title
plt.pyplot.xlabel('horsepower')
plt.pyplot.ylabel('count')
plt.pyplot.title('horsepower bins')
```




    Text(0.5, 1.0, 'horsepower bins')




![png](output_47_1.png)


#### Indicator variable (or dummy variable)
An indicator variable (or dummy variable) is a numerical variable used to label categories. They are called 'dummies' because the numbers themselves don't have inherent meaning.


```python
df['fuel-type'].unique()
```




    array(['gas', 'diesel'], dtype=object)



We see the column "fuel-type" has two unique values, "gas" or "diesel". Regression doesn't understand words, only numbers. To use this attribute in regression analysis, we convert "fuel-type" into indicator variables.


```python
# assign numerical values to the different categories of 'fuel-tpye'
dummy_variable_1 = pd.get_dummies(df['fuel-type'])
dummy_variable_1.head()
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
      <th>diesel</th>
      <th>gas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
           'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
           'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke',
           'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',
           'highway-mpg', 'price', 'horsepower-binned'],
          dtype='object')




```python
# change column names for clarity
dummy_variable_1.rename(columns={'fuel-tpye-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
dummy_variable_1.head()
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
      <th>diesel</th>
      <th>gas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
           'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
           'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke',
           'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',
           'highway-mpg', 'price', 'horsepower-binned'],
          dtype='object')



We now have the value 0 to represent "gas" and 1 to represent "diesel" in the column "fuel-type".


```python
# merge data frame 'df' and 'dummy_variable_1'
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column 'fuel-type' from 'df'
df.drop('fuel-type', axis=1, inplace=True)
```

The last two columns are now the indicator variable representation of the fuel-type variable. It's all 0s and 1s now.


```python
# create indicator variable for the column 'aspiration'
dummy_variable_2 = pd.get_dummies(df['aspiration'])
dummy_variable_2.rename(columns={'std': 'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)
dummy_variable_2.head()
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
      <th>aspiration-std</th>
      <th>aspiration-turbo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# merge the new dataframe to the original dataframe
df = pd.concat([df, dummy_variable_2], axis=1)

# drop the column 'aspiration'
df.drop('aspiration', axis=1, inplace=True)

df.head()
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
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>...</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
      <th>horsepower-binned</th>
      <th>diesel</th>
      <th>gas</th>
      <th>aspiration-std</th>
      <th>aspiration-turbo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>0.811148</td>
      <td>0.890278</td>
      <td>...</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21</td>
      <td>27</td>
      <td>13495.0</td>
      <td>low</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>0.811148</td>
      <td>0.890278</td>
      <td>...</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21</td>
      <td>27</td>
      <td>16500.0</td>
      <td>low</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>0.822681</td>
      <td>0.909722</td>
      <td>...</td>
      <td>154.0</td>
      <td>5000.0</td>
      <td>19</td>
      <td>26</td>
      <td>16500.0</td>
      <td>medium</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>0.848630</td>
      <td>0.919444</td>
      <td>...</td>
      <td>102.0</td>
      <td>5500.0</td>
      <td>24</td>
      <td>30</td>
      <td>13950.0</td>
      <td>low</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>0.848630</td>
      <td>0.922222</td>
      <td>...</td>
      <td>115.0</td>
      <td>5500.0</td>
      <td>18</td>
      <td>22</td>
      <td>17450.0</td>
      <td>low</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>



### Analyzing Individual Feature Patterns using Visualization


```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
```

#### Continuous numerical variables
Continuous numerical variables are variables that may contain any value within some range. Continuous numerical variables can have the type "int64" or "float64". A great way to visualize these variables is by using scatterplots with fitted lines.

#### Correlation
We can calculate the correlation between variables of type 'int64' or 'float64' using the method 'corr'.


```python
df.corr() 
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
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>engine-size</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
      <th>diesel</th>
      <th>gas</th>
      <th>aspiration-std</th>
      <th>aspiration-turbo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>symboling</th>
      <td>1.000000</td>
      <td>0.466264</td>
      <td>-0.535987</td>
      <td>-0.365404</td>
      <td>-0.242423</td>
      <td>-0.550160</td>
      <td>-0.233118</td>
      <td>-0.110581</td>
      <td>0.243521</td>
      <td>-0.008153</td>
      <td>-0.182196</td>
      <td>0.075819</td>
      <td>0.279740</td>
      <td>-0.035527</td>
      <td>0.036233</td>
      <td>-0.082391</td>
      <td>-0.196735</td>
      <td>0.196735</td>
      <td>0.054615</td>
      <td>-0.054615</td>
    </tr>
    <tr>
      <th>normalized-losses</th>
      <td>0.466264</td>
      <td>1.000000</td>
      <td>-0.056661</td>
      <td>0.019424</td>
      <td>0.086802</td>
      <td>-0.373737</td>
      <td>0.099404</td>
      <td>0.112360</td>
      <td>0.124511</td>
      <td>0.055045</td>
      <td>-0.114713</td>
      <td>0.217299</td>
      <td>0.239543</td>
      <td>-0.225016</td>
      <td>-0.181877</td>
      <td>0.133999</td>
      <td>-0.101546</td>
      <td>0.101546</td>
      <td>0.006911</td>
      <td>-0.006911</td>
    </tr>
    <tr>
      <th>wheel-base</th>
      <td>-0.535987</td>
      <td>-0.056661</td>
      <td>1.000000</td>
      <td>0.876024</td>
      <td>0.814507</td>
      <td>0.590742</td>
      <td>0.782097</td>
      <td>0.572027</td>
      <td>-0.074380</td>
      <td>0.158018</td>
      <td>0.250313</td>
      <td>0.371147</td>
      <td>-0.360305</td>
      <td>-0.470606</td>
      <td>-0.543304</td>
      <td>0.584642</td>
      <td>0.307237</td>
      <td>-0.307237</td>
      <td>-0.256889</td>
      <td>0.256889</td>
    </tr>
    <tr>
      <th>length</th>
      <td>-0.365404</td>
      <td>0.019424</td>
      <td>0.876024</td>
      <td>1.000000</td>
      <td>0.857170</td>
      <td>0.492063</td>
      <td>0.880665</td>
      <td>0.685025</td>
      <td>-0.050463</td>
      <td>0.123952</td>
      <td>0.159733</td>
      <td>0.579821</td>
      <td>-0.285970</td>
      <td>-0.665192</td>
      <td>-0.698142</td>
      <td>0.690628</td>
      <td>0.211187</td>
      <td>-0.211187</td>
      <td>-0.230085</td>
      <td>0.230085</td>
    </tr>
    <tr>
      <th>width</th>
      <td>-0.242423</td>
      <td>0.086802</td>
      <td>0.814507</td>
      <td>0.857170</td>
      <td>1.000000</td>
      <td>0.306002</td>
      <td>0.866201</td>
      <td>0.729436</td>
      <td>-0.004059</td>
      <td>0.188822</td>
      <td>0.189867</td>
      <td>0.615077</td>
      <td>-0.245800</td>
      <td>-0.633531</td>
      <td>-0.680635</td>
      <td>0.751265</td>
      <td>0.244356</td>
      <td>-0.244356</td>
      <td>-0.305732</td>
      <td>0.305732</td>
    </tr>
    <tr>
      <th>height</th>
      <td>-0.550160</td>
      <td>-0.373737</td>
      <td>0.590742</td>
      <td>0.492063</td>
      <td>0.306002</td>
      <td>1.000000</td>
      <td>0.307581</td>
      <td>0.074694</td>
      <td>-0.240217</td>
      <td>-0.060663</td>
      <td>0.259737</td>
      <td>-0.087027</td>
      <td>-0.309974</td>
      <td>-0.049800</td>
      <td>-0.104812</td>
      <td>0.135486</td>
      <td>0.281578</td>
      <td>-0.281578</td>
      <td>-0.090336</td>
      <td>0.090336</td>
    </tr>
    <tr>
      <th>curb-weight</th>
      <td>-0.233118</td>
      <td>0.099404</td>
      <td>0.782097</td>
      <td>0.880665</td>
      <td>0.866201</td>
      <td>0.307581</td>
      <td>1.000000</td>
      <td>0.849072</td>
      <td>-0.029485</td>
      <td>0.167438</td>
      <td>0.156433</td>
      <td>0.757976</td>
      <td>-0.279361</td>
      <td>-0.749543</td>
      <td>-0.794889</td>
      <td>0.834415</td>
      <td>0.221046</td>
      <td>-0.221046</td>
      <td>-0.321955</td>
      <td>0.321955</td>
    </tr>
    <tr>
      <th>engine-size</th>
      <td>-0.110581</td>
      <td>0.112360</td>
      <td>0.572027</td>
      <td>0.685025</td>
      <td>0.729436</td>
      <td>0.074694</td>
      <td>0.849072</td>
      <td>1.000000</td>
      <td>-0.177698</td>
      <td>0.205928</td>
      <td>0.028889</td>
      <td>0.822676</td>
      <td>-0.256733</td>
      <td>-0.650546</td>
      <td>-0.679571</td>
      <td>0.872335</td>
      <td>0.070779</td>
      <td>-0.070779</td>
      <td>-0.110040</td>
      <td>0.110040</td>
    </tr>
    <tr>
      <th>bore</th>
      <td>0.243521</td>
      <td>0.124511</td>
      <td>-0.074380</td>
      <td>-0.050463</td>
      <td>-0.004059</td>
      <td>-0.240217</td>
      <td>-0.029485</td>
      <td>-0.177698</td>
      <td>1.000000</td>
      <td>-0.001549</td>
      <td>-0.027237</td>
      <td>0.032443</td>
      <td>0.259276</td>
      <td>-0.196827</td>
      <td>-0.170635</td>
      <td>0.005399</td>
      <td>-0.046482</td>
      <td>0.046482</td>
      <td>0.062876</td>
      <td>-0.062876</td>
    </tr>
    <tr>
      <th>stroke</th>
      <td>-0.008153</td>
      <td>0.055045</td>
      <td>0.158018</td>
      <td>0.123952</td>
      <td>0.188822</td>
      <td>-0.060663</td>
      <td>0.167438</td>
      <td>0.205928</td>
      <td>-0.001549</td>
      <td>1.000000</td>
      <td>0.187871</td>
      <td>0.098267</td>
      <td>-0.063561</td>
      <td>-0.033956</td>
      <td>-0.034636</td>
      <td>0.082269</td>
      <td>0.241064</td>
      <td>-0.241064</td>
      <td>-0.218233</td>
      <td>0.218233</td>
    </tr>
    <tr>
      <th>compression-ratio</th>
      <td>-0.182196</td>
      <td>-0.114713</td>
      <td>0.250313</td>
      <td>0.159733</td>
      <td>0.189867</td>
      <td>0.259737</td>
      <td>0.156433</td>
      <td>0.028889</td>
      <td>-0.027237</td>
      <td>0.187871</td>
      <td>1.000000</td>
      <td>-0.214514</td>
      <td>-0.435780</td>
      <td>0.331425</td>
      <td>0.268465</td>
      <td>0.071107</td>
      <td>0.985231</td>
      <td>-0.985231</td>
      <td>-0.307522</td>
      <td>0.307522</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>0.075819</td>
      <td>0.217299</td>
      <td>0.371147</td>
      <td>0.579821</td>
      <td>0.615077</td>
      <td>-0.087027</td>
      <td>0.757976</td>
      <td>0.822676</td>
      <td>0.032443</td>
      <td>0.098267</td>
      <td>-0.214514</td>
      <td>1.000000</td>
      <td>0.107885</td>
      <td>-0.822214</td>
      <td>-0.804575</td>
      <td>0.809575</td>
      <td>-0.169053</td>
      <td>0.169053</td>
      <td>-0.251127</td>
      <td>0.251127</td>
    </tr>
    <tr>
      <th>peak-rpm</th>
      <td>0.279740</td>
      <td>0.239543</td>
      <td>-0.360305</td>
      <td>-0.285970</td>
      <td>-0.245800</td>
      <td>-0.309974</td>
      <td>-0.279361</td>
      <td>-0.256733</td>
      <td>0.259276</td>
      <td>-0.063561</td>
      <td>-0.435780</td>
      <td>0.107885</td>
      <td>1.000000</td>
      <td>-0.115413</td>
      <td>-0.058598</td>
      <td>-0.101616</td>
      <td>-0.475812</td>
      <td>0.475812</td>
      <td>0.190057</td>
      <td>-0.190057</td>
    </tr>
    <tr>
      <th>city-mpg</th>
      <td>-0.035527</td>
      <td>-0.225016</td>
      <td>-0.470606</td>
      <td>-0.665192</td>
      <td>-0.633531</td>
      <td>-0.049800</td>
      <td>-0.749543</td>
      <td>-0.650546</td>
      <td>-0.196827</td>
      <td>-0.033956</td>
      <td>0.331425</td>
      <td>-0.822214</td>
      <td>-0.115413</td>
      <td>1.000000</td>
      <td>0.972044</td>
      <td>-0.686571</td>
      <td>0.265676</td>
      <td>-0.265676</td>
      <td>0.189237</td>
      <td>-0.189237</td>
    </tr>
    <tr>
      <th>highway-mpg</th>
      <td>0.036233</td>
      <td>-0.181877</td>
      <td>-0.543304</td>
      <td>-0.698142</td>
      <td>-0.680635</td>
      <td>-0.104812</td>
      <td>-0.794889</td>
      <td>-0.679571</td>
      <td>-0.170635</td>
      <td>-0.034636</td>
      <td>0.268465</td>
      <td>-0.804575</td>
      <td>-0.058598</td>
      <td>0.972044</td>
      <td>1.000000</td>
      <td>-0.704692</td>
      <td>0.198690</td>
      <td>-0.198690</td>
      <td>0.241851</td>
      <td>-0.241851</td>
    </tr>
    <tr>
      <th>price</th>
      <td>-0.082391</td>
      <td>0.133999</td>
      <td>0.584642</td>
      <td>0.690628</td>
      <td>0.751265</td>
      <td>0.135486</td>
      <td>0.834415</td>
      <td>0.872335</td>
      <td>0.005399</td>
      <td>0.082269</td>
      <td>0.071107</td>
      <td>0.809575</td>
      <td>-0.101616</td>
      <td>-0.686571</td>
      <td>-0.704692</td>
      <td>1.000000</td>
      <td>0.110326</td>
      <td>-0.110326</td>
      <td>-0.179578</td>
      <td>0.179578</td>
    </tr>
    <tr>
      <th>diesel</th>
      <td>-0.196735</td>
      <td>-0.101546</td>
      <td>0.307237</td>
      <td>0.211187</td>
      <td>0.244356</td>
      <td>0.281578</td>
      <td>0.221046</td>
      <td>0.070779</td>
      <td>-0.046482</td>
      <td>0.241064</td>
      <td>0.985231</td>
      <td>-0.169053</td>
      <td>-0.475812</td>
      <td>0.265676</td>
      <td>0.198690</td>
      <td>0.110326</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>-0.408228</td>
      <td>0.408228</td>
    </tr>
    <tr>
      <th>gas</th>
      <td>0.196735</td>
      <td>0.101546</td>
      <td>-0.307237</td>
      <td>-0.211187</td>
      <td>-0.244356</td>
      <td>-0.281578</td>
      <td>-0.221046</td>
      <td>-0.070779</td>
      <td>0.046482</td>
      <td>-0.241064</td>
      <td>-0.985231</td>
      <td>0.169053</td>
      <td>0.475812</td>
      <td>-0.265676</td>
      <td>-0.198690</td>
      <td>-0.110326</td>
      <td>-1.000000</td>
      <td>1.000000</td>
      <td>0.408228</td>
      <td>-0.408228</td>
    </tr>
    <tr>
      <th>aspiration-std</th>
      <td>0.054615</td>
      <td>0.006911</td>
      <td>-0.256889</td>
      <td>-0.230085</td>
      <td>-0.305732</td>
      <td>-0.090336</td>
      <td>-0.321955</td>
      <td>-0.110040</td>
      <td>0.062876</td>
      <td>-0.218233</td>
      <td>-0.307522</td>
      <td>-0.251127</td>
      <td>0.190057</td>
      <td>0.189237</td>
      <td>0.241851</td>
      <td>-0.179578</td>
      <td>-0.408228</td>
      <td>0.408228</td>
      <td>1.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>aspiration-turbo</th>
      <td>-0.054615</td>
      <td>-0.006911</td>
      <td>0.256889</td>
      <td>0.230085</td>
      <td>0.305732</td>
      <td>0.090336</td>
      <td>0.321955</td>
      <td>0.110040</td>
      <td>-0.062876</td>
      <td>0.218233</td>
      <td>0.307522</td>
      <td>0.251127</td>
      <td>-0.190057</td>
      <td>-0.189237</td>
      <td>-0.241851</td>
      <td>0.179578</td>
      <td>0.408228</td>
      <td>-0.408228</td>
      <td>-1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# correlation between bore, stroke, compression-ratio and horspower
df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()
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
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bore</th>
      <td>1.000000</td>
      <td>-0.001549</td>
      <td>-0.027237</td>
      <td>0.032443</td>
    </tr>
    <tr>
      <th>stroke</th>
      <td>-0.001549</td>
      <td>1.000000</td>
      <td>0.187871</td>
      <td>0.098267</td>
    </tr>
    <tr>
      <th>compression-ratio</th>
      <td>-0.027237</td>
      <td>0.187871</td>
      <td>1.000000</td>
      <td>-0.214514</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>0.032443</td>
      <td>0.098267</td>
      <td>-0.214514</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Continuos numerical variables
Continuous numerical variables are variables that may contain any value within some range. Continuous numerical variables can have the type "int64" or "float64". A great way to visualize these variables is by using scatterplots with fitted lines.

##### Positive Linear Relationship


```python
# engine size as potential predictor variable of price
sns.regplot(x='engine-size', y='price', data=df)
plt.ylim(0,)
```




    (0, 53229.620270856)




![png](output_66_1.png)


As the engine-size goes up, the price goes up: this indicates a positive direct correlation between these two variables. Engine size seems like a pretty good predictor of price since the regression line is almost a perfect diagonal line.


```python
# examine the correlation between 'engine-size' and 'price'
df[['engine-size', 'price']].corr()
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
      <th>engine-size</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>engine-size</th>
      <td>1.000000</td>
      <td>0.872335</td>
    </tr>
    <tr>
      <th>price</th>
      <td>0.872335</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



As the highway-mpg goes up, the price goes down: this indicates an inverse/negative relationship between these two variables.


```python
# examine the correlation between 'highway-mpg' and 'price'
df[['highway-mpg', 'price']].corr()
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
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>highway-mpg</th>
      <td>1.000000</td>
      <td>-0.704692</td>
    </tr>
    <tr>
      <th>price</th>
      <td>-0.704692</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



##### Weak Linear Relationship


```python
# relationship between peak-rpm and price
sns.regplot(x='peak-rpm', y='price', data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d19ad7bef0>




![png](output_72_1.png)


Peak rpm does not seem like a good predictor of the price at all since the regression line is close to horizontal. Also, the data points are very scattered and far from the fitted line, showing lots of variability. Therefore it's it is not a reliable variable.


```python
# examine the correlation between 'peak-rpm' and 'price 
df[['peak-rpm', 'price']].corr()
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
      <th>peak-rpm</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>peak-rpm</th>
      <td>1.000000</td>
      <td>-0.101616</td>
    </tr>
    <tr>
      <th>price</th>
      <td>-0.101616</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Categorical variables
These are variables that describe a 'characteristic' of a data unit, and are selected from a small group of categories. The categorical variables can have the type "object" or "int64". A good way to visualize categorical variables is by using boxplots.


```python
# relationship between body-style and price
sns.boxplot(x='body-style', y='price', data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d19bfe7c88>




![png](output_76_1.png)


We see that the distributions of price between the different body-style categories have a significant overlap, and so body-style would not be a good predictor of price.


```python
# relationship between engine location and price
sns.boxplot(x='engine-location', y='price', data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d19c0b2320>




![png](output_78_1.png)


We see that the distribution of price between these two engine-location categories, front and rear, are distinct enough to take engine-location as a potential good predictor of price.


```python
# relationship etween drive wheels and price
sns.boxplot(x='drive-wheels', y='price', data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d19c125f98>




![png](output_80_1.png)


We see that the distribution of price between the different drive-wheels categories differs; as such, drive-wheels could potentially be a predictor of price.

### Descriptive Statistical Analysis
The **describe** function automatically computes basic statistics for all continuous variables. Any NaN values are automatically skipped in these statistics.


```python
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
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>engine-size</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
      <th>diesel</th>
      <th>gas</th>
      <th>aspiration-std</th>
      <th>aspiration-turbo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>201.000000</td>
      <td>201.00000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.840796</td>
      <td>122.00000</td>
      <td>98.797015</td>
      <td>0.837102</td>
      <td>0.915126</td>
      <td>0.899108</td>
      <td>2555.666667</td>
      <td>126.875622</td>
      <td>5.692289</td>
      <td>3.256874</td>
      <td>10.164279</td>
      <td>103.405534</td>
      <td>5117.665368</td>
      <td>25.179104</td>
      <td>30.686567</td>
      <td>13207.129353</td>
      <td>0.099502</td>
      <td>0.900498</td>
      <td>0.820896</td>
      <td>0.179104</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.254802</td>
      <td>31.99625</td>
      <td>6.066366</td>
      <td>0.059213</td>
      <td>0.029187</td>
      <td>0.040933</td>
      <td>517.296727</td>
      <td>41.546834</td>
      <td>16.616706</td>
      <td>0.316048</td>
      <td>4.004965</td>
      <td>37.365700</td>
      <td>478.113805</td>
      <td>6.423220</td>
      <td>6.815150</td>
      <td>7947.066342</td>
      <td>0.300083</td>
      <td>0.300083</td>
      <td>0.384397</td>
      <td>0.384397</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.000000</td>
      <td>65.00000</td>
      <td>86.600000</td>
      <td>0.678039</td>
      <td>0.837500</td>
      <td>0.799331</td>
      <td>1488.000000</td>
      <td>61.000000</td>
      <td>2.540000</td>
      <td>2.070000</td>
      <td>7.000000</td>
      <td>48.000000</td>
      <td>4150.000000</td>
      <td>13.000000</td>
      <td>16.000000</td>
      <td>5118.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>101.00000</td>
      <td>94.500000</td>
      <td>0.801538</td>
      <td>0.890278</td>
      <td>0.869565</td>
      <td>2169.000000</td>
      <td>98.000000</td>
      <td>3.150000</td>
      <td>3.110000</td>
      <td>8.600000</td>
      <td>70.000000</td>
      <td>4800.000000</td>
      <td>19.000000</td>
      <td>25.000000</td>
      <td>7775.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>122.00000</td>
      <td>97.000000</td>
      <td>0.832292</td>
      <td>0.909722</td>
      <td>0.904682</td>
      <td>2414.000000</td>
      <td>120.000000</td>
      <td>3.310000</td>
      <td>3.290000</td>
      <td>9.000000</td>
      <td>95.000000</td>
      <td>5125.369458</td>
      <td>24.000000</td>
      <td>30.000000</td>
      <td>10295.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>137.00000</td>
      <td>102.400000</td>
      <td>0.881788</td>
      <td>0.925000</td>
      <td>0.928094</td>
      <td>2926.000000</td>
      <td>141.000000</td>
      <td>3.600000</td>
      <td>3.410000</td>
      <td>9.400000</td>
      <td>116.000000</td>
      <td>5500.000000</td>
      <td>30.000000</td>
      <td>34.000000</td>
      <td>16500.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>256.00000</td>
      <td>120.900000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4066.000000</td>
      <td>326.000000</td>
      <td>122.000000</td>
      <td>4.170000</td>
      <td>23.000000</td>
      <td>262.000000</td>
      <td>6600.000000</td>
      <td>49.000000</td>
      <td>54.000000</td>
      <td>45400.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe(include='object')
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
      <th>make</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>engine-type</th>
      <th>num-of-cylinders</th>
      <th>fuel-system</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>201</td>
      <td>201</td>
      <td>201</td>
      <td>201</td>
      <td>201</td>
      <td>201</td>
      <td>201</td>
      <td>201</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>22</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>top</th>
      <td>toyota</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>ohc</td>
      <td>four</td>
      <td>mpfi</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>32</td>
      <td>115</td>
      <td>94</td>
      <td>118</td>
      <td>198</td>
      <td>145</td>
      <td>157</td>
      <td>92</td>
    </tr>
  </tbody>
</table>
</div>



**value_counts** is a good way of understanding how many units of each characteristic/variable we have. The method "value_counts" only works on Pandas series, not Pandas Dataframes. As a result, we only include one bracket "df['drive-wheels']" not two brackets "df[['drive-wheels']]".


```python
df['drive-wheels'].value_counts()
```




    fwd    118
    rwd     75
    4wd      8
    Name: drive-wheels, dtype: int64




```python
# convert the series to a dataframe
df['drive-wheels'].value_counts().to_frame()
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
      <th>drive-wheels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fwd</th>
      <td>118</td>
    </tr>
    <tr>
      <th>rwd</th>
      <td>75</td>
    </tr>
    <tr>
      <th>4wd</th>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# rename the column 'drive-wheels' to 'value_counts'
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts
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
      <th>value_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fwd</th>
      <td>118</td>
    </tr>
    <tr>
      <th>rwd</th>
      <td>75</td>
    </tr>
    <tr>
      <th>4wd</th>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# rename the index to 'drive-wheels'
drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts
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
      <th>value_counts</th>
    </tr>
    <tr>
      <th>drive-wheels</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fwd</th>
      <td>118</td>
    </tr>
    <tr>
      <th>rwd</th>
      <td>75</td>
    </tr>
    <tr>
      <th>4wd</th>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# value_counts for engine location
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename({'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts
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
      <th>engine-location</th>
    </tr>
    <tr>
      <th>engine-location</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>front</th>
      <td>198</td>
    </tr>
    <tr>
      <th>rear</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



The value counts of the engine location would not be a good predictor variable for the price. This is because we only have 3 cars with a rear engine and 198 with an engine in the front; this result is skewed. Thus, we are not able to draw any conclusions about the engine location.

### Grouping
The 'groupby' method groups data by different categories. The data is grouped based on one or several variables and analysis is performed on the individual groups.


```python
# categories of drive wheels
df['drive-wheels'].unique()
```




    array(['rwd', 'fwd', '4wd'], dtype=object)



If we want to know on average, which type of drive wheel is most valuable, we can group 'drive-wheels' and then average them.


```python
# select columns and assign them to a variable
df_group_one = df[['drive-wheels', 'body-style', 'price']]
```


```python
# grouping
# calculate the average price for each of the different categories of data
df_group_one = df_group_one.groupby(['drive-wheels'], as_index=False).mean()
df_group_one
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
      <th>drive-wheels</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4wd</td>
      <td>10241.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fwd</td>
      <td>9244.779661</td>
    </tr>
    <tr>
      <th>2</th>
      <td>rwd</td>
      <td>19757.613333</td>
    </tr>
  </tbody>
</table>
</div>



It seems that rear-wheel drive vehicles are, on average, the most expensive, while 4-wheel drive and front-wheel drive are approximately the same price.


```python
# grouping with multiple variables
df_gptest = df[['drive-wheels', 'body-style', 'price']]
grouped_test1 = df_gptest.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
grouped_test1
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
      <th>drive-wheels</th>
      <th>body-style</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4wd</td>
      <td>hatchback</td>
      <td>7603.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4wd</td>
      <td>sedan</td>
      <td>12647.333333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4wd</td>
      <td>wagon</td>
      <td>9095.750000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fwd</td>
      <td>convertible</td>
      <td>11595.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>fwd</td>
      <td>hardtop</td>
      <td>8249.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>fwd</td>
      <td>hatchback</td>
      <td>8396.387755</td>
    </tr>
    <tr>
      <th>6</th>
      <td>fwd</td>
      <td>sedan</td>
      <td>9811.800000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>fwd</td>
      <td>wagon</td>
      <td>9997.333333</td>
    </tr>
    <tr>
      <th>8</th>
      <td>rwd</td>
      <td>convertible</td>
      <td>23949.600000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>rwd</td>
      <td>hardtop</td>
      <td>24202.714286</td>
    </tr>
    <tr>
      <th>10</th>
      <td>rwd</td>
      <td>hatchback</td>
      <td>14337.777778</td>
    </tr>
    <tr>
      <th>11</th>
      <td>rwd</td>
      <td>sedan</td>
      <td>21711.833333</td>
    </tr>
    <tr>
      <th>12</th>
      <td>rwd</td>
      <td>wagon</td>
      <td>16994.222222</td>
    </tr>
  </tbody>
</table>
</div>



This grouped data is much easier to visualize when it is made into a pivot table. A pivot table is like an Excel spreadsheet, with one variable along the column and another along the row. We can convert the dataframe to a pivot table using the method 'pivot' to create a pivot table from the groups.


```python
# leave the drive-wheel variable as the rows and pivot body-style to become the columns of the table
grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style')
grouped_pivot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">price</th>
    </tr>
    <tr>
      <th>body-style</th>
      <th>convertible</th>
      <th>hardtop</th>
      <th>hatchback</th>
      <th>sedan</th>
      <th>wagon</th>
    </tr>
    <tr>
      <th>drive-wheels</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4wd</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>7603.000000</td>
      <td>12647.333333</td>
      <td>9095.750000</td>
    </tr>
    <tr>
      <th>fwd</th>
      <td>11595.0</td>
      <td>8249.000000</td>
      <td>8396.387755</td>
      <td>9811.800000</td>
      <td>9997.333333</td>
    </tr>
    <tr>
      <th>rwd</th>
      <td>23949.6</td>
      <td>24202.714286</td>
      <td>14337.777778</td>
      <td>21711.833333</td>
      <td>16994.222222</td>
    </tr>
  </tbody>
</table>
</div>



Often, we won't have data for some of the pivot cells. We can fill these missing cells with the value 0, but any other value could potentially be used as well.


```python
# fill missing values with 0
grouped_pivot = grouped_pivot.fillna(0) 
grouped_pivot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">price</th>
    </tr>
    <tr>
      <th>body-style</th>
      <th>convertible</th>
      <th>hardtop</th>
      <th>hatchback</th>
      <th>sedan</th>
      <th>wagon</th>
    </tr>
    <tr>
      <th>drive-wheels</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4wd</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>7603.000000</td>
      <td>12647.333333</td>
      <td>9095.750000</td>
    </tr>
    <tr>
      <th>fwd</th>
      <td>11595.0</td>
      <td>8249.000000</td>
      <td>8396.387755</td>
      <td>9811.800000</td>
      <td>9997.333333</td>
    </tr>
    <tr>
      <th>rwd</th>
      <td>23949.6</td>
      <td>24202.714286</td>
      <td>14337.777778</td>
      <td>21711.833333</td>
      <td>16994.222222</td>
    </tr>
  </tbody>
</table>
</div>




```python
# groupby to find  average price of each car based on body style
df_gptest_2 = df[['body-style', 'price']]
grouped_test_bodystyle = df_gptest_2.groupby(['body-style'], as_index=False).mean()
grouped_test_bodystyle
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
      <th>body-style</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>convertible</td>
      <td>21890.500000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hardtop</td>
      <td>22208.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hatchback</td>
      <td>9957.441176</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sedan</td>
      <td>14459.755319</td>
    </tr>
    <tr>
      <th>4</th>
      <td>wagon</td>
      <td>12371.960000</td>
    </tr>
  </tbody>
</table>
</div>



Use a heat map to visualize the relationship between Body Style vs Price


```python
# use the grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()
```


![png](output_105_0.png)


The heatmap plots the target variable (price) proportional to colour with respect to the variables 'drive-wheel' and 'body-style' in the vertical and horizontal axis respectively. This allows us to visualize how the price is related to 'drive-wheel' and 'body-style'.


```python
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()
```


![png](output_107_0.png)


### Correlation and Causation
**Correlation**: a measure of the extent of interdependence between variables.

**Causation**: the relationship between cause and effect between two variables.

Correlation doesn't imply causation.

Persaon Correlation: It measures the linear dependence between two variables X and Y.

The resulting coefficient is a value between -1 and 1 inclusive, where:

*   1: Total positive linear correlation.
*   0: No linear correlation, the two variables most likely do not affect each other.
*   -1: Total negative linear correlation

Pearson Correlation is the default method of the function "corr". Like before we can calculate the Pearson Correlation of the 'int64' or 'float64' variables.




```python
# calculate the Pearson coefficient
df.corr()
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
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>engine-size</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
      <th>diesel</th>
      <th>gas</th>
      <th>aspiration-std</th>
      <th>aspiration-turbo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>symboling</th>
      <td>1.000000</td>
      <td>0.466264</td>
      <td>-0.535987</td>
      <td>-0.365404</td>
      <td>-0.242423</td>
      <td>-0.550160</td>
      <td>-0.233118</td>
      <td>-0.110581</td>
      <td>0.243521</td>
      <td>-0.008153</td>
      <td>-0.182196</td>
      <td>0.075819</td>
      <td>0.279740</td>
      <td>-0.035527</td>
      <td>0.036233</td>
      <td>-0.082391</td>
      <td>-0.196735</td>
      <td>0.196735</td>
      <td>0.054615</td>
      <td>-0.054615</td>
    </tr>
    <tr>
      <th>normalized-losses</th>
      <td>0.466264</td>
      <td>1.000000</td>
      <td>-0.056661</td>
      <td>0.019424</td>
      <td>0.086802</td>
      <td>-0.373737</td>
      <td>0.099404</td>
      <td>0.112360</td>
      <td>0.124511</td>
      <td>0.055045</td>
      <td>-0.114713</td>
      <td>0.217299</td>
      <td>0.239543</td>
      <td>-0.225016</td>
      <td>-0.181877</td>
      <td>0.133999</td>
      <td>-0.101546</td>
      <td>0.101546</td>
      <td>0.006911</td>
      <td>-0.006911</td>
    </tr>
    <tr>
      <th>wheel-base</th>
      <td>-0.535987</td>
      <td>-0.056661</td>
      <td>1.000000</td>
      <td>0.876024</td>
      <td>0.814507</td>
      <td>0.590742</td>
      <td>0.782097</td>
      <td>0.572027</td>
      <td>-0.074380</td>
      <td>0.158018</td>
      <td>0.250313</td>
      <td>0.371147</td>
      <td>-0.360305</td>
      <td>-0.470606</td>
      <td>-0.543304</td>
      <td>0.584642</td>
      <td>0.307237</td>
      <td>-0.307237</td>
      <td>-0.256889</td>
      <td>0.256889</td>
    </tr>
    <tr>
      <th>length</th>
      <td>-0.365404</td>
      <td>0.019424</td>
      <td>0.876024</td>
      <td>1.000000</td>
      <td>0.857170</td>
      <td>0.492063</td>
      <td>0.880665</td>
      <td>0.685025</td>
      <td>-0.050463</td>
      <td>0.123952</td>
      <td>0.159733</td>
      <td>0.579821</td>
      <td>-0.285970</td>
      <td>-0.665192</td>
      <td>-0.698142</td>
      <td>0.690628</td>
      <td>0.211187</td>
      <td>-0.211187</td>
      <td>-0.230085</td>
      <td>0.230085</td>
    </tr>
    <tr>
      <th>width</th>
      <td>-0.242423</td>
      <td>0.086802</td>
      <td>0.814507</td>
      <td>0.857170</td>
      <td>1.000000</td>
      <td>0.306002</td>
      <td>0.866201</td>
      <td>0.729436</td>
      <td>-0.004059</td>
      <td>0.188822</td>
      <td>0.189867</td>
      <td>0.615077</td>
      <td>-0.245800</td>
      <td>-0.633531</td>
      <td>-0.680635</td>
      <td>0.751265</td>
      <td>0.244356</td>
      <td>-0.244356</td>
      <td>-0.305732</td>
      <td>0.305732</td>
    </tr>
    <tr>
      <th>height</th>
      <td>-0.550160</td>
      <td>-0.373737</td>
      <td>0.590742</td>
      <td>0.492063</td>
      <td>0.306002</td>
      <td>1.000000</td>
      <td>0.307581</td>
      <td>0.074694</td>
      <td>-0.240217</td>
      <td>-0.060663</td>
      <td>0.259737</td>
      <td>-0.087027</td>
      <td>-0.309974</td>
      <td>-0.049800</td>
      <td>-0.104812</td>
      <td>0.135486</td>
      <td>0.281578</td>
      <td>-0.281578</td>
      <td>-0.090336</td>
      <td>0.090336</td>
    </tr>
    <tr>
      <th>curb-weight</th>
      <td>-0.233118</td>
      <td>0.099404</td>
      <td>0.782097</td>
      <td>0.880665</td>
      <td>0.866201</td>
      <td>0.307581</td>
      <td>1.000000</td>
      <td>0.849072</td>
      <td>-0.029485</td>
      <td>0.167438</td>
      <td>0.156433</td>
      <td>0.757976</td>
      <td>-0.279361</td>
      <td>-0.749543</td>
      <td>-0.794889</td>
      <td>0.834415</td>
      <td>0.221046</td>
      <td>-0.221046</td>
      <td>-0.321955</td>
      <td>0.321955</td>
    </tr>
    <tr>
      <th>engine-size</th>
      <td>-0.110581</td>
      <td>0.112360</td>
      <td>0.572027</td>
      <td>0.685025</td>
      <td>0.729436</td>
      <td>0.074694</td>
      <td>0.849072</td>
      <td>1.000000</td>
      <td>-0.177698</td>
      <td>0.205928</td>
      <td>0.028889</td>
      <td>0.822676</td>
      <td>-0.256733</td>
      <td>-0.650546</td>
      <td>-0.679571</td>
      <td>0.872335</td>
      <td>0.070779</td>
      <td>-0.070779</td>
      <td>-0.110040</td>
      <td>0.110040</td>
    </tr>
    <tr>
      <th>bore</th>
      <td>0.243521</td>
      <td>0.124511</td>
      <td>-0.074380</td>
      <td>-0.050463</td>
      <td>-0.004059</td>
      <td>-0.240217</td>
      <td>-0.029485</td>
      <td>-0.177698</td>
      <td>1.000000</td>
      <td>-0.001549</td>
      <td>-0.027237</td>
      <td>0.032443</td>
      <td>0.259276</td>
      <td>-0.196827</td>
      <td>-0.170635</td>
      <td>0.005399</td>
      <td>-0.046482</td>
      <td>0.046482</td>
      <td>0.062876</td>
      <td>-0.062876</td>
    </tr>
    <tr>
      <th>stroke</th>
      <td>-0.008153</td>
      <td>0.055045</td>
      <td>0.158018</td>
      <td>0.123952</td>
      <td>0.188822</td>
      <td>-0.060663</td>
      <td>0.167438</td>
      <td>0.205928</td>
      <td>-0.001549</td>
      <td>1.000000</td>
      <td>0.187871</td>
      <td>0.098267</td>
      <td>-0.063561</td>
      <td>-0.033956</td>
      <td>-0.034636</td>
      <td>0.082269</td>
      <td>0.241064</td>
      <td>-0.241064</td>
      <td>-0.218233</td>
      <td>0.218233</td>
    </tr>
    <tr>
      <th>compression-ratio</th>
      <td>-0.182196</td>
      <td>-0.114713</td>
      <td>0.250313</td>
      <td>0.159733</td>
      <td>0.189867</td>
      <td>0.259737</td>
      <td>0.156433</td>
      <td>0.028889</td>
      <td>-0.027237</td>
      <td>0.187871</td>
      <td>1.000000</td>
      <td>-0.214514</td>
      <td>-0.435780</td>
      <td>0.331425</td>
      <td>0.268465</td>
      <td>0.071107</td>
      <td>0.985231</td>
      <td>-0.985231</td>
      <td>-0.307522</td>
      <td>0.307522</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>0.075819</td>
      <td>0.217299</td>
      <td>0.371147</td>
      <td>0.579821</td>
      <td>0.615077</td>
      <td>-0.087027</td>
      <td>0.757976</td>
      <td>0.822676</td>
      <td>0.032443</td>
      <td>0.098267</td>
      <td>-0.214514</td>
      <td>1.000000</td>
      <td>0.107885</td>
      <td>-0.822214</td>
      <td>-0.804575</td>
      <td>0.809575</td>
      <td>-0.169053</td>
      <td>0.169053</td>
      <td>-0.251127</td>
      <td>0.251127</td>
    </tr>
    <tr>
      <th>peak-rpm</th>
      <td>0.279740</td>
      <td>0.239543</td>
      <td>-0.360305</td>
      <td>-0.285970</td>
      <td>-0.245800</td>
      <td>-0.309974</td>
      <td>-0.279361</td>
      <td>-0.256733</td>
      <td>0.259276</td>
      <td>-0.063561</td>
      <td>-0.435780</td>
      <td>0.107885</td>
      <td>1.000000</td>
      <td>-0.115413</td>
      <td>-0.058598</td>
      <td>-0.101616</td>
      <td>-0.475812</td>
      <td>0.475812</td>
      <td>0.190057</td>
      <td>-0.190057</td>
    </tr>
    <tr>
      <th>city-mpg</th>
      <td>-0.035527</td>
      <td>-0.225016</td>
      <td>-0.470606</td>
      <td>-0.665192</td>
      <td>-0.633531</td>
      <td>-0.049800</td>
      <td>-0.749543</td>
      <td>-0.650546</td>
      <td>-0.196827</td>
      <td>-0.033956</td>
      <td>0.331425</td>
      <td>-0.822214</td>
      <td>-0.115413</td>
      <td>1.000000</td>
      <td>0.972044</td>
      <td>-0.686571</td>
      <td>0.265676</td>
      <td>-0.265676</td>
      <td>0.189237</td>
      <td>-0.189237</td>
    </tr>
    <tr>
      <th>highway-mpg</th>
      <td>0.036233</td>
      <td>-0.181877</td>
      <td>-0.543304</td>
      <td>-0.698142</td>
      <td>-0.680635</td>
      <td>-0.104812</td>
      <td>-0.794889</td>
      <td>-0.679571</td>
      <td>-0.170635</td>
      <td>-0.034636</td>
      <td>0.268465</td>
      <td>-0.804575</td>
      <td>-0.058598</td>
      <td>0.972044</td>
      <td>1.000000</td>
      <td>-0.704692</td>
      <td>0.198690</td>
      <td>-0.198690</td>
      <td>0.241851</td>
      <td>-0.241851</td>
    </tr>
    <tr>
      <th>price</th>
      <td>-0.082391</td>
      <td>0.133999</td>
      <td>0.584642</td>
      <td>0.690628</td>
      <td>0.751265</td>
      <td>0.135486</td>
      <td>0.834415</td>
      <td>0.872335</td>
      <td>0.005399</td>
      <td>0.082269</td>
      <td>0.071107</td>
      <td>0.809575</td>
      <td>-0.101616</td>
      <td>-0.686571</td>
      <td>-0.704692</td>
      <td>1.000000</td>
      <td>0.110326</td>
      <td>-0.110326</td>
      <td>-0.179578</td>
      <td>0.179578</td>
    </tr>
    <tr>
      <th>diesel</th>
      <td>-0.196735</td>
      <td>-0.101546</td>
      <td>0.307237</td>
      <td>0.211187</td>
      <td>0.244356</td>
      <td>0.281578</td>
      <td>0.221046</td>
      <td>0.070779</td>
      <td>-0.046482</td>
      <td>0.241064</td>
      <td>0.985231</td>
      <td>-0.169053</td>
      <td>-0.475812</td>
      <td>0.265676</td>
      <td>0.198690</td>
      <td>0.110326</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>-0.408228</td>
      <td>0.408228</td>
    </tr>
    <tr>
      <th>gas</th>
      <td>0.196735</td>
      <td>0.101546</td>
      <td>-0.307237</td>
      <td>-0.211187</td>
      <td>-0.244356</td>
      <td>-0.281578</td>
      <td>-0.221046</td>
      <td>-0.070779</td>
      <td>0.046482</td>
      <td>-0.241064</td>
      <td>-0.985231</td>
      <td>0.169053</td>
      <td>0.475812</td>
      <td>-0.265676</td>
      <td>-0.198690</td>
      <td>-0.110326</td>
      <td>-1.000000</td>
      <td>1.000000</td>
      <td>0.408228</td>
      <td>-0.408228</td>
    </tr>
    <tr>
      <th>aspiration-std</th>
      <td>0.054615</td>
      <td>0.006911</td>
      <td>-0.256889</td>
      <td>-0.230085</td>
      <td>-0.305732</td>
      <td>-0.090336</td>
      <td>-0.321955</td>
      <td>-0.110040</td>
      <td>0.062876</td>
      <td>-0.218233</td>
      <td>-0.307522</td>
      <td>-0.251127</td>
      <td>0.190057</td>
      <td>0.189237</td>
      <td>0.241851</td>
      <td>-0.179578</td>
      <td>-0.408228</td>
      <td>0.408228</td>
      <td>1.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>aspiration-turbo</th>
      <td>-0.054615</td>
      <td>-0.006911</td>
      <td>0.256889</td>
      <td>0.230085</td>
      <td>0.305732</td>
      <td>0.090336</td>
      <td>0.321955</td>
      <td>0.110040</td>
      <td>-0.062876</td>
      <td>0.218233</td>
      <td>0.307522</td>
      <td>0.251127</td>
      <td>-0.190057</td>
      <td>-0.189237</td>
      <td>-0.241851</td>
      <td>0.179578</td>
      <td>0.408228</td>
      <td>-0.408228</td>
      <td>-1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['horsepower'].unique()
```




    array([111.        , 154.        , 102.        , 115.        ,
           110.        , 140.        , 101.        , 121.        ,
           182.        ,  48.        ,  70.        ,  68.        ,
            88.        , 145.        ,  58.        ,  76.        ,
            60.        ,  86.        , 100.        ,  78.        ,
            90.        , 176.        , 262.        , 135.        ,
            84.        ,  64.        , 120.        ,  72.        ,
           123.        , 155.        , 184.        , 175.        ,
           116.        ,  69.        ,  55.        ,  97.        ,
           152.        , 160.        , 200.        ,  95.        ,
           142.        , 143.        , 207.        , 104.25615764,
            73.        ,  82.        ,  94.        ,  62.        ,
            56.        , 112.        ,  92.        , 161.        ,
           156.        ,  52.        ,  85.        , 114.        ,
           162.        , 134.        , 106.        ])



To know the significance of the correlation estimate, we calculate the P-value.  
The P-value is the probability value that the correlation between these two variables is statistically significant. Normally, we choose a significance level of 0.05, which means that we are 95% confident that the correlation between the variables is significant.

By convention, when the  
*   p-value is < 0.001: we say there is strong evidence that the correlation is significant.
*   p-value is < 0.05: there is moderate evidence that the correlation is significant.
*   p-value is < 0.1: there is weak evidence that the correlation is significant.
*   p-value is > 0.1: there is no evidence that the correlation is significant.


```python
from scipy import stats
```


```python
# calcualte the Pearson coefficient and p-value of wheel base and price
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print('The Pearson Correlation Coefficient is ', pearson_coef, ' with a P-value of P=', p_value)
```

    The Pearson Correlation Coefficient is  0.5846418222655081  with a P-value of P= 8.076488270732989e-20
    

Since the p-value is < 0.001, the correlation between wheel-base and price is statistically significant, although the linear relationship isn't extremely strong (~0.585)


```python
# calcualte the Pearson coefficient and p-value of horsepower and price
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print('The Pearson Correlation Coefficient is ', pearson_coef, ' with a P-value of P=', p_value)
```

    The Pearson Correlation Coefficient is  0.809574567003656  with a P-value of P= 6.369057428259557e-48
    

Since the p-value is < 0.001, the correlation between horsepower and price is statistically significant, and the linear relationship is quite strong (~0.809, close to 1)


```python
# calcualte the Pearson coefficient and p-value of length and price
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print('The Pearson Correlation Coefficient is ', pearson_coef, ' with a P-value of P=', p_value)
```

    The Pearson Correlation Coefficient is  0.6906283804483642  with a P-value of P= 8.016477466158759e-30
    

Since the p-value is < 0.001, the correlation between length and price is statistically significant, and the linear relationship is moderately strong (~0.691).


```python
# calcualte the Pearson coefficient and p-value of width and price
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print('The Pearson Correlation Coefficient is ', pearson_coef, ' with a P-value of P=', p_value)
```

    The Pearson Correlation Coefficient is  0.7512653440522673  with a P-value of P= 9.200335510481646e-38
    

Since the p-value is < 0.001, the correlation between width and price is statistically significant, and the linear relationship is quite strong (~0.751).


```python
# calcualte the Pearson coefficient and p-value of curb weight and price
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print('The Pearson Correlation Coefficient is ', pearson_coef, ' with a P-value of P=', p_value)
```

    The Pearson Correlation Coefficient is  0.8344145257702846  with a P-value of P= 2.1895772388936914e-53
    

Since the p-value is < 0.001, the correlation between curb-weight and price is statistically significant, and the linear relationship is quite strong (~0.834).


```python
# calcualte the Pearson coefficient and p-value of engine size and price
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print('The Pearson Correlation Coefficient is ', pearson_coef, ' with a P-value of P=', p_value)
```

    The Pearson Correlation Coefficient is  0.8723351674455185  with a P-value of P= 9.265491622198389e-64
    

Since the p-value is < 0.001, the correlation between engine-size and price is statistically significant, and the linear relationship is very strong (~0.872).


```python
# calcualte the Pearson coefficient and p-value of bore and price
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print('The Pearson Correlation Coefficient is ', pearson_coef, ' with a P-value of P=', p_value)
```

    The Pearson Correlation Coefficient is  0.005399275177997414  with a P-value of P= 0.9393625495207799
    

Since the p-value is < 0.001, the correlation between bore and price is statistically significant, but the linear relationship is only moderate (~0.521).


```python
# calcualte the Pearson coefficient and p-value of city-mpg and price
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print('The Pearson Correlation Coefficient is ', pearson_coef, ' with a P-value of P=', p_value)
```

    The Pearson Correlation Coefficient is  -0.6865710067844677  with a P-value of P= 2.321132065567674e-29
    

Since the p-value is < 0.001, the correlation between city-mpg and price is statistically significant, and the coefficient of ~ -0.687 shows that the relationship is negative and moderately strong.


```python
# calcualte the Pearson coefficient and p-value of highway-mpg and price
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print('The Pearson Correlation Coefficient is ', pearson_coef, ' with a P-value of P=', p_value)
```

    The Pearson Correlation Coefficient is  -0.7046922650589529  with a P-value of P= 1.7495471144477352e-31
    

Since the p-value is < 0.001, the correlation between highway-mpg and price is statistically significant, and the coefficient of ~ -0.705 shows that the relationship is negative and moderately strong.

### ANOVA (Analyis of Variance)
The Analysis of Variance (ANOVA) is a statistical method used to test whether there are significant differences between the means of two or more groups. ANOVA returns two parameters:

**F-test score**: ANOVA assumes the means of all groups are the same, calculates how much the actual means deviate from the assumption, and reports it as the F-test score. A larger score means there is a larger difference between the means.

**P-value**: P-value tells how statistically significant is our calculated score value.

If our price variable is strongly correlated with the variable we are analyzing, expect ANOVA to return a sizeable F-test score and a small p-value.


Since ANOVA analyzes the difference between different groups of the same variable, the groupby function will come in handy. Because the ANOVA algorithm averages the data automatically, we do not need to take the average before hand.


```python
# check if different types of drive wheels impact price
# group the data
grouped_test2 = df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)
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
      <th>drive-wheels</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>rwd</td>
      <td>13495.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>rwd</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fwd</td>
      <td>13950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4wd</td>
      <td>17450.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>fwd</td>
      <td>15250.0</td>
    </tr>
    <tr>
      <th>136</th>
      <td>4wd</td>
      <td>7603.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_gptest
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
      <th>drive-wheels</th>
      <th>body-style</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>rwd</td>
      <td>convertible</td>
      <td>13495.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>rwd</td>
      <td>convertible</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>rwd</td>
      <td>hatchback</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fwd</td>
      <td>sedan</td>
      <td>13950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4wd</td>
      <td>sedan</td>
      <td>17450.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>196</th>
      <td>rwd</td>
      <td>sedan</td>
      <td>16845.0</td>
    </tr>
    <tr>
      <th>197</th>
      <td>rwd</td>
      <td>sedan</td>
      <td>19045.0</td>
    </tr>
    <tr>
      <th>198</th>
      <td>rwd</td>
      <td>sedan</td>
      <td>21485.0</td>
    </tr>
    <tr>
      <th>199</th>
      <td>rwd</td>
      <td>sedan</td>
      <td>22470.0</td>
    </tr>
    <tr>
      <th>200</th>
      <td>rwd</td>
      <td>sedan</td>
      <td>22625.0</td>
    </tr>
  </tbody>
</table>
<p>201 rows × 3 columns</p>
</div>




```python
# obtain the values of the method group using the method "get_group"
grouped_test2.get_group('4wd')['price']
```




    4      17450.0
    136     7603.0
    140     9233.0
    141    11259.0
    144     8013.0
    145    11694.0
    150     7898.0
    151     8778.0
    Name: price, dtype: float64




```python
# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val) 
```

    ANOVA results: F= 67.95406500780399 , P = 3.3945443577151245e-23
    

This is a great result, with a large F test score showing a strong correlation and a P value of almost 0 implying almost certain statistical significance. But does this mean all three tested groups are all this highly correlated?


```python
# separately fwd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
print( "ANOVA results: F=", f_val, ", P =", p_val )
```

    ANOVA results: F= 130.5533160959111 , P = 2.2355306355677845e-23
    


```python
# separately 4wd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
print( "ANOVA results: F=", f_val, ", P =", p_val) 
```

    ANOVA results: F= 8.580681368924756 , P = 0.004411492211225333
    


```python
# separately 4wd and fwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])   
print("ANOVA results: F=", f_val, ", P =", p_val)   
```

    ANOVA results: F= 0.665465750252303 , P = 0.41620116697845666
    

#### Conclusion: Important Variables

We now have a better idea of what our data looks like and which variables are important to take into account when predicting the car price. We have narrowed it down to the following variables:

Continuous numerical variables:

    Length
    Width
    Curb-weight
    Engine-size
    Horsepower
    City-mpg
    Highway-mpg
    Wheel-base
    Bore

Categorical variables:

    Drive-wheels


## Model Development


```python
# path of data 
df.head()
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
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>...</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
      <th>horsepower-binned</th>
      <th>diesel</th>
      <th>gas</th>
      <th>aspiration-std</th>
      <th>aspiration-turbo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>0.811148</td>
      <td>0.890278</td>
      <td>...</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21</td>
      <td>27</td>
      <td>13495.0</td>
      <td>low</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>0.811148</td>
      <td>0.890278</td>
      <td>...</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21</td>
      <td>27</td>
      <td>16500.0</td>
      <td>low</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>0.822681</td>
      <td>0.909722</td>
      <td>...</td>
      <td>154.0</td>
      <td>5000.0</td>
      <td>19</td>
      <td>26</td>
      <td>16500.0</td>
      <td>medium</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>0.848630</td>
      <td>0.919444</td>
      <td>...</td>
      <td>102.0</td>
      <td>5500.0</td>
      <td>24</td>
      <td>30</td>
      <td>13950.0</td>
      <td>low</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>0.848630</td>
      <td>0.922222</td>
      <td>...</td>
      <td>115.0</td>
      <td>5500.0</td>
      <td>18</td>
      <td>22</td>
      <td>17450.0</td>
      <td>low</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>



### Linear Regression and Multiple Linear Regression

### Simple Linear Regression
Simple Linear Regression is a method to help us understand the relationship between two variables:  
*   The predictor/independent variable (X) 
*   The response/dependent variable (that we want to predict)(Y)  

The result of Linear Regression is a linear function that predicts the response (dependent) variable as a function of the predictor (independent) variable.

Y: Response Variable  
X: Predictor Varaible

Linear function:
𝑌ℎ𝑎𝑡 = 𝑎 + 𝑏𝑋  
*   a refers to the intercept of the regression line, in other words: the value of Y when X is 0
*   b refers to the slope of the regression line, in other words: the value with which Y changes when X increases by 1 unit






```python
# load the module for linear regression
from sklearn.linear_model import LinearRegression
```

###### How can highway-mpg help predict the price?


```python
# create the linear regression object
lm = LinearRegression()
lm
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
X = df[['highway-mpg']]
Y = df['price']

# fit the linear model using highway-mpg
lm.fit(X,Y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# output a prediction
Yhat = lm.predict(X)
Yhat[0:5]
```




    array([16236.50464347, 16236.50464347, 17058.23802179, 13771.3045085 ,
           20345.17153508])




```python
# value of intercept a
lm.intercept_
```




    38423.305858157386




```python
# value of slope b
lm.coef_
```




    array([-821.73337832])



Final estimated linear model:  
price = 38423.31 - 821.73 * highway-mpg

###### How can engine size help predict the price?


```python
X = df[['engine-size']]
Y = df['price']

# fit the linear model using highway-mpg
lm.fit(X,Y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# output a prediction
Yhat = lm.predict(X)
Yhat[0:5]
```




    array([13728.4631336 , 13728.4631336 , 17399.38347881, 10224.40280408,
           14729.62322775])




```python
# value of intercept a
lm.intercept_
```




    -7963.338906281049




```python
# value of slope b
lm.coef_
```




    array([166.86001569])



Final estimated linear model:  
Price = -7963.34 + 166.86 * Engine-size

### Multiple Linear Regression

If we want to use more variables in our model to predict car price, we can use Multiple Linear Regression.  
This method is used to explain the relationship between one continuous response (dependent) variable and two or more predictor (independent) variables. Most of the real-world regression models involve multiple predictors.

*𝑌ℎ𝑎𝑡 = 𝑎 + 𝑏1𝑋1 + 𝑏2𝑋2 + 𝑏3𝑋3 + 𝑏4𝑋4*  

From the previous section we know that other good predictors of price could be:  

*   Horsepower
*   Curb-weight
*   Engine-size
*   Highway-mpg






```python
# develop a model using these variables as the predictor variables
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
```


```python
# fit the linear model using the above four variables
lm.fit(Z, df['price'])
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# value of the intercept
lm.intercept_
```




    -15806.624626329198




```python
# value of the coefficients (b1, b2, b3, b4)
lm.coef_
```




    array([53.49574423,  4.70770099, 81.53026382, 36.05748882])



Final estimated linear model:  
Price = -15678.74 + 52.65851272 * horsepower + 4.699 * curb-weight + 81.96 * engine-size + 33.58 * highway-mpg



```python
# use two other predictor variables
lm.fit(df[['normalized-losses', 'highway-mpg']], df['price'])
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# value of the intercept
lm.intercept_
```




    38201.31327245728




```python
# value of the coefficients (b1, b2)
lm.coef_
```




    array([   1.49789586, -820.45434016])



Final estimated linear model:  
Price = 38201.31 + 1.498 * normalized-losses - 820.45 * highway-mpg

## Model Evaluation using Visualization


```python
# import the visualization package: seaborn
import seaborn as sns
%matplotlib inline
```

### Regression Plot for Simple Linear Regression
This plot will show a combination of a scattered data points (a scatter plot), as well as the fitted linear regression line going through the data. This will give us a reasonable estimate of the relationship between the two variables, the strength of the correlation, as well as the direction (positive or negative correlation).


```python
# visualize highway-mpg as a potential predictor of price
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x='highway-mpg', y='price', data=df)
plt.ylim(0,)
```




    (0, 48163.464897503036)




![png](output_176_1.png)


We can see from this plot that price is negatively correlated to highway-mpg, since the regression slope is negative. One thing to keep in mind when looking at a regression plot is to pay attention to how scattered the data points are around the regression line. This will give you a good indication of the variance of the data, and whether a linear model would be the best fit or not. If the data is too far off from the line, this linear model might not be the best model for this data.


```python
# visualize peak-rpm as a potential predictor of price
plt.figure(figsize=(width, height))
sns.regplot(x='peak-rpm', y='price', data=df)
plt.ylim(0,)
```




    (0, 47414.10667770421)




![png](output_178_1.png)


Comparing the regression plot of "peak-rpm" and "highway-mpg" we see that the points for "highway-mpg" are much closer to the generated line and on the average decrease. The points for "peak-rpm" have more spread around the predicted line, and it is much harder to determine if the points are decreasing or increasing as the "highway-mpg" increases.


```python
# find whether peak-rpm or highway-mpg is more strongly correlated with price
df[['peak-rpm', 'highway-mpg', 'price']].corr()
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
      <th>peak-rpm</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>peak-rpm</th>
      <td>1.000000</td>
      <td>-0.058598</td>
      <td>-0.101616</td>
    </tr>
    <tr>
      <th>highway-mpg</th>
      <td>-0.058598</td>
      <td>1.000000</td>
      <td>-0.704692</td>
    </tr>
    <tr>
      <th>price</th>
      <td>-0.101616</td>
      <td>-0.704692</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



As we can see, highway-mpg is more strongly correlated with price as compared to peak-rpm.

##### Residual Plot to visualize variance of data

The difference between the observed value (y) and the predicted value (Yhat) is called the residual (e). When we look at a regression plot, the residual is the distance from the data point to the fitted regression line.

A residual plot is a graph that shows the residuals on the vertical y-axis and the independent variable on the horizontal x-axis.

We look at the spread of the residuals:  
- If the points in a residual plot are randomly spread out around the x-axis, then a linear model is appropriate for the data.  
- Randomly spread out residuals means that the variance is constant, and thus the linear model is a good fit for this data.


```python
# create a residal plot
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()
```


![png](output_183_0.png)


We can see from this residual plot that the residuals are not randomly spread around the x-axis, which leads us to believe that maybe a non-linear model is more appropriate for this data.

### Distribution Plot for Multiple Linear Regression
You cannot visualize Multiple Linear Regression with a regression or residual plot.  
One way to look at the fit of the model is by looking at the distribution plot. We can look at the distribution of the fitted values that result from the model and compare it to the distribution of the actual values.


```python
# develop a model using these variables as the predictor variables
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
```


```python
# fit the linear model using the above four variables
lm.fit(Z, df['price'])
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# make a prediction 
Y_hat = lm.predict(Z)
```


```python
plt.figure(figsize=(width, height))

ax1 = sns.distplot(df['price'], hist=False, color='r', label='Actual Value')
sns.distplot(Yhat, hist=False, color='b', label='Fitted Values', ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()
```


![png](output_189_0.png)


We can see that the fitted values are reasonably close to the actual values, since the two distributions overlap a bit. However, there is definitely some room for improvement.

## Polynomial Regression and Pipelines
Polynomial regression is a particular case of the general linear regression model or multiple linear regression models.  
We get non-linear relationships by squaring or setting higher-order terms of the predictor variables.

There are different orders of polynomial regression:  
- Quadratic - 2nd order  
𝑌ℎ𝑎𝑡=𝑎+𝑏1𝑋2+𝑏2𝑋2  
- Cubic - 3rd order  
𝑌ℎ𝑎𝑡=𝑎+𝑏1𝑋2+𝑏2𝑋2+𝑏3𝑋3  
- Higher order:  
𝑌=𝑎+𝑏1𝑋2+𝑏2𝑋2+𝑏3𝑋3....  

We saw earlier that a linear model did not provide the best fit while using highway-mpg as the predictor variable. Let's see if we can try fitting a polynomial model to the data instead.


```python
# plot the data
def PlotPolly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()
```


```python
# get the variables
x = df['highway-mpg']
y = df['price']
```


```python
# fit the polynomial using the polyfit function
# we use a polynomial of the 3rd order
f = np.polyfit(x, y, 3)

# use the poly1d function to display the polynomial function
p = np.poly1d(f)
print(p)
```

            3         2
    -1.557 x + 204.8 x - 8965 x + 1.379e+05
    


```python
# plot the function
PlotPolly(p, x, y, 'highway-mpg')
```


![png](output_195_0.png)



```python
np.polyfit(x, y, 3)
```




    array([-1.55663829e+00,  2.04754306e+02, -8.96543312e+03,  1.37923594e+05])



We can already see from plotting that this polynomial model performs better than the linear model. This is because the generated polynomial function "hits" more of the data points.


```python
# create an 11 order polynomial model with the same variables
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p1)
```

                11             10             9           8         7
    -1.243e-08 x  + 4.722e-06 x  - 0.0008028 x + 0.08056 x - 5.297 x
              6        5             4             3             2
     + 239.5 x - 7588 x + 1.684e+05 x - 2.565e+06 x + 2.551e+07 x - 1.491e+08 x + 3.879e+08
    


```python
PlotPolly(p1, x, y, 'highway-mpg')
```


![png](output_199_0.png)


We see that by using very high order polynomials, overfitting is observed.

### Multivariate Polynomial Function
The analytical expression for Multivariate Polynomial function gets complicated. For example, the expression for a second-order (degree=2)polynomial with two variables is given by:

𝑌ℎ𝑎𝑡=𝑎+𝑏1𝑋1+𝑏2𝑋2+𝑏3𝑋1𝑋2+𝑏4𝑋21+𝑏5𝑋22

We will now perform a polynomial transform on multiple features.


```python
# import the module
from sklearn.preprocessing import PolynomialFeatures
```


```python
# create a PolynomialFeatures object of degree 2
pr = PolynomialFeatures(degree=2)
pr
```




    PolynomialFeatures(degree=2, include_bias=True, interaction_only=False,
                       order='C')




```python
Z_pr = pr.fit_transform(Z)
```


```python
Z.shape
```




    (201, 4)




```python
Z_pr.shape
```




    (201, 15)



### Pipeline
Data Pipelines simplify the steps of processing the data.


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```


```python
# create the pipeline
# create a list of tuples inlcuding the name of the model/estimator and its corresponding constructor
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
```


```python
# input the list as an argument to the pipeline constructor
pipe = Pipeline(Input)
pipe
```




    Pipeline(memory=None,
             steps=[('scale',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('polynomial',
                     PolynomialFeatures(degree=2, include_bias=False,
                                        interaction_only=False, order='C')),
                    ('model',
                     LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
                                      normalize=False))],
             verbose=False)




```python
# we can normalize the data, perform a transform and fit the model simultaneously
pipe.fit(Z,y)
```




    Pipeline(memory=None,
             steps=[('scale',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('polynomial',
                     PolynomialFeatures(degree=2, include_bias=False,
                                        interaction_only=False, order='C')),
                    ('model',
                     LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
                                      normalize=False))],
             verbose=False)




```python
# we can normalize the data, perform a transform and produced a prediction simultaneously
ypipe = pipe.predict(Z)
ypipe[0:10]
```




    array([13102.74784201, 13102.74784201, 18225.54572197, 10390.29636555,
           16136.29619164, 13880.09787302, 15041.58694037, 15457.93465485,
           17974.49032347, 10510.56542385])



## Measures for In-Sample Evaluation
When evaluating our models, not only do we want to visualize the results, but we also want a quantitative measure to determine how accurate the model is.

Two very important measures that are often used in Statistics to determine the accuracy of a model are:
- R^2 / R-squared
- Mean Squared Error (MSE)

**R-squared:** R squared, also known as the coefficient of determination, is a measure to indicate how close the data is to the fitted regression line. The value of the R-squared is the percentage of variation of the response variable (y) that is explained by a linear model.

**Mean Squared Error (MSE):** The Mean Squared Error measures the average of the squares of errors, that is, the difference between actual value (y) and the estimated value (ŷ).

### Model 1: Simple Linear Regression


```python
X = df[['highway-mpg']]
Y = df['price']
# highway_mpg_fit
lm.fit(X,Y)
# calculate the R^2
print('The R-square is:', lm.score(X,Y))
```

    The R-square is: 0.4965911884339175
    

We can say that ~ 49.659% of the variation of the price is explained by this simple linear model "highway_mpg_fit".


```python
# predict the output
Yhat = lm.predict(X)
print('The output of the first four predicted values is', Yhat[0:4])
```

    The output of the first four predicted values is [16236.50464347 16236.50464347 17058.23802179 13771.3045085 ]
    


```python
# import the module
from sklearn.metrics import mean_squared_error
# calculate the MSE
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)
```

    The mean square error of price and predicted value is:  31635042.944639895
    

### Model 2: Multiple Linear Regression 


```python
# fit the model
lm.fit(Z, df['price'])
# find the R^2
print('The R-square value is: ', lm.score(Z, df['price']))
```

    The R-square value is:  0.8093562806577457
    

We can say that ~ 80.93 % of the variation of price is explained by this multiple linear regression "multi_fit".


```python
# produce a prediction
Y_predict_multifit = lm.predict(Z)
```


```python
# calcualte MSE
print('The mean square error of price and predicted value using multifit is: ', mean_squared_error(df['price'], Y_predict_multifit))
```

    The mean square error of price and predicted value using multifit is:  11980366.87072649
    

#### Model 3: Polynomial Fit


```python
# import the module
from sklearn.metrics import r2_score
```


```python
# calculate R^2
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)
```

    The R-square value is:  0.674194666390652
    

We can say that ~ 67.419 % of the variation of price is explained by this polynomial fit.


```python
# calculate MSE
mean_squared_error(df['price'], p(x))
```




    20474146.426361218



## Prediction and Decision Making
#### Prediction  
We trained the model using fit. Now we will use the method 'predict' to produce a prediction.


```python
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```


```python
# create a new input
new_input = np.arange(1, 100, 1).reshape(-1, 1)
```


```python
# fit the model
lm.fit(X,Y)
lm
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# produce a prediction
yhat = lm.predict(new_input)
yhat[0:5]
```




    array([37601.57247984, 36779.83910151, 35958.10572319, 35136.37234487,
           34314.63896655])




```python
# plot the data
plt.plot(new_input, yhat)
plt.show()
```


![png](output_234_0.png)


### Decision Making: Determing a Good Model Fit
Now that we have visualized the different models, and generated the R-squared and MSE values for the fits, how do we determine a good model fit?

What is a good R-squared value?
When comparing models, the model with the higher R-squared value is a better fit for the data.

What is a good MSE?
When comparing models, the model with the smallest MSE value is a better fit for the data.

#### Let's take a look at the values for the different models. 

Simple Linear Regression: Using Highway-mpg as a Predictor Variable of Price.
- R-squared: 0.49659118843391759
- MSE: 3.16 x10^7

Multiple Linear Regression: Using Horsepower, Curb-weight, Engine-size, and Highway-mpg as Predictor Variables of Price.  
- R-squared: 0.80896354913783497
- MSE: 1.2 x10^7

Polynomial Fit: Using Highway-mpg as a Predictor Variable of Price.  
- R-squared: 0.6741946663906514
- MSE: 2.05 x 10^7

#### Simple Linear Regression model (SLR) vs Multiple Linear Regression model (MLR)
Usually, the more variables you have, the better your model is at predicting, but this is not always true. Sometimes you may not have enough data, you may run into numerical problems, or many of the variables may not be useful and or even act as noise. As a result, you should always check the MSE and R^2.  
So to be able to compare the results of the MLR vs SLR models, we look at a combination of both the R-squared and MSE to make the best conclusion about the fit of the model.

- **MSE:** The MSE of SLR is 3.16x10^7 while MLR has an MSE of 1.2 x10^7. The MSE of MLR is much smaller.  
- **R-squared:** In this case, we can also see that there is a big difference between the R-squared of the SLR and the R-squared of the MLR. The R-squared for the SLR (0.497) is very small compared to the R-squared for the MLR (0.809).  
This R-squared in combination with the MSE show that MLR seems like the better model fit in this case, compared to SLR.

#### Simple Linear Model (SLR) vs Polynomial Fit
- **MSE:** We can see that Polynomial Fit brought down the MSE, since this MSE is smaller than the one from the SLR.
- **R-squared:** The R-squared for the Polyfit is larger than the R-squared for the SLR, so the Polynomial Fit also brought up the R-squared quite a bit.  
Since the Polynomial Fit resulted in a lower MSE and a higher R-squared, we can conclude that this was a better fit model than the simple linear regression for predicting Price with Highway-mpg as a predictor variable.

#### Multiple Linear Regression (MLR) vs Polynomial Fit
- **MSE:** The MSE for the MLR is smaller than the MSE for the Polynomial Fit.
- **R-squared:** The R-squared for the MLR is also much larger than for the Polynomial Fit.

#### Conclusion:
Comparing these three models, we conclude that **the MLR model is the best model** to be able to predict price from our dataset. This result makes sense, since we have 27 variables in total, and we know that more than one of those variables are potential predictors of the final car price.

## Model Evaluation and Refinement
We have built models and made predictions of vehicle prices. Now we will determine how accurate these predictions are.


```python
import pandas as pd
import numpy as np
```


```python
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/module_5_auto.csv'
df = pd.read_csv(path)
```


```python
# first let's only use numeric data
df = df._get_numeric_data()
df.head()
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
      <th>Unnamed: 0</th>
      <th>Unnamed: 0.1</th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>engine-size</th>
      <th>...</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
      <th>city-L/100km</th>
      <th>diesel</th>
      <th>gas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>122</td>
      <td>88.6</td>
      <td>0.811148</td>
      <td>0.890278</td>
      <td>48.8</td>
      <td>2548</td>
      <td>130</td>
      <td>...</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21</td>
      <td>27</td>
      <td>13495.0</td>
      <td>11.190476</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>122</td>
      <td>88.6</td>
      <td>0.811148</td>
      <td>0.890278</td>
      <td>48.8</td>
      <td>2548</td>
      <td>130</td>
      <td>...</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21</td>
      <td>27</td>
      <td>16500.0</td>
      <td>11.190476</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>122</td>
      <td>94.5</td>
      <td>0.822681</td>
      <td>0.909722</td>
      <td>52.4</td>
      <td>2823</td>
      <td>152</td>
      <td>...</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154.0</td>
      <td>5000.0</td>
      <td>19</td>
      <td>26</td>
      <td>16500.0</td>
      <td>12.368421</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>164</td>
      <td>99.8</td>
      <td>0.848630</td>
      <td>0.919444</td>
      <td>54.3</td>
      <td>2337</td>
      <td>109</td>
      <td>...</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102.0</td>
      <td>5500.0</td>
      <td>24</td>
      <td>30</td>
      <td>13950.0</td>
      <td>9.791667</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>164</td>
      <td>99.4</td>
      <td>0.848630</td>
      <td>0.922222</td>
      <td>54.3</td>
      <td>2824</td>
      <td>136</td>
      <td>...</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115.0</td>
      <td>5500.0</td>
      <td>18</td>
      <td>22</td>
      <td>17450.0</td>
      <td>13.055556</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# import the modules
from IPython.display import display
from IPython.html import widgets
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
```

    C:\Users\prana\Anaconda3\lib\site-packages\IPython\html.py:14: ShimWarning: The `IPython.html` package has been deprecated since IPython 4.0. You should import from `notebook` instead. `IPython.html.widgets` has moved to `ipywidgets`.
      "`IPython.html.widgets` has moved to `ipywidgets`.", ShimWarning)
    

#### Functions for Plotting


```python
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
  width = 12
  height = 10
  plt.figure(figsize=(width, height))

  ax1 = sns.distplot(RedFunction, hist=False, color='r', label=RedName)
  ax2 = sns.distplot(BlueFunction, hist=False, color='b', label=BlueName, ax=ax1)

  plt.title(Title)
  plt.xlabel('Price (in dollars)')
  plt.ylabel('Proportion of Cars')

  plt.show()
  plt.close()
```


```python
def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
  #training data 
  #testing data 
  # lr:linear regression object     #poly_transform:polynomial transformation object 
  width = 12
  height = 10    
  plt.figure(figsize=(width, height))
 
  xmax=max([xtrain.values.max(), xtest.values.max()])
  xmin=min([xtrain.values.min(), xtest.values.min()])
  x=np.arange(xmin, xmax, 0.1)

  plt.plot(xtrain, y_train, 'ro', label='Training Data')
  plt.plot(xtest, y_test, 'go', label='Test Data')
  plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
  plt.ylim([-10000, 60000])
  plt.ylabel('Price')
  plt.legend()
```

### Training and Testing


```python
# place target data 'price' in a seaparate dataframe y
y_data = df['price']
```


```python
# drop price data in x_data
x_data = df.drop('price', axis=1)
```


```python
# randomly split the data into training and testing data using the function train_test_split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=1)
# test_size setes the proportion of data that is split. The testing set is set to 10% of the total dataset.
# use the same random_state value throughout your code

print('number of test samples: ', x_test.shape[0])
print('number of training samples: ', x_train.shape[0])
```

    number of test samples:  21
    number of training samples:  180
    


```python
# import LinearRegression module
from sklearn.linear_model import LinearRegression

# create Linear Regression object
lre = LinearRegression()

# fit the model using the feature 'horsepower'
lre.fit(x_train[['horsepower']], y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# calculate R^2 on the test data
lre.score(x_test[['horsepower']], y_test)
```




    0.3635875575078824




```python
# calcuate R^2 on the training data
lre.score(x_train[['horsepower']], y_train)
```




    0.6619724197515103



We can see that the R^2 is much smaller using the test data.

### Cross-Validation Score


```python
# import the module
from sklearn.model_selection import cross_val_score
```


```python
# input the object(lre), the feature(horsepower), the target data(y_data), number of folds(cv)
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
Rcross
```




    array([0.7746232 , 0.51716687, 0.74785353, 0.04839605])



The default scoring is R^2. Each element in the array has the average R^2 value in the fold.


```python
# calculate the average and standard deviation of our estimate
print('The mean of the folds is', Rcross.mean(), 'and the standard deviation is', Rcross.std())
```

    The mean of the folds is 0.522009915042119 and the standard deviation is 0.2911839444756029
    

We can use negative squared error as a score by setting the parameter 'scoring' metric to 'neg_mean_squared_error'.


```python
-1 * cross_val_score(lre, x_data[['horsepower']], y_data, cv=4, scoring='neg_mean_squared_error')
```




    array([20254142.84026704, 43745493.2650517 , 12539630.34014931,
           17561927.72247591])



Use the function 'cross_val_predict' to predict the output. The function splits up the data into the specified number of folds, using one fold to get a prediction while the rest of the folds are used as test data. First import the function.


```python
from sklearn.model_selection import cross_val_predict
yhat= cross_val_predict(lre, x_data[['horsepower']], y_data, cv=4)
yhat[0:5]
```




    array([14141.63807508, 14141.63807508, 20814.29423473, 12745.03562306,
           14762.35027598])



### Overfitting, Underfitting and Model Selection
It turns out that the test data sometimes referred to as the out of sample data is a much better measure of how well your model performs in the real world. One reason for this is overfitting; let's go over some examples. It turns out these differences are more apparent in Multiple Linear Regression and Polynomial Regression so we will explore overfitting in that context.


```python
# create MLR objects and train the model
lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# prediction using training data
y_hat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
y_hat_train[0:5]
```




    array([ 7426.6731551 , 28323.75090803, 14213.38819709,  4052.34146983,
           34500.19124244])




```python
# prediction using test data
y_hat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
y_hat_test[0:5]
```




    array([11349.35089149,  5884.11059106, 11208.6928275 ,  6641.07786278,
           15565.79920282])



Lets perform some model evaluation using our training and testing data separately.


```python
# import seaborn and matplotlib libraries
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```


```python
# examine the distribution of the predicted values of the training data
Title = 'Distribution Plot of Predicted Value using Training Data vs Training Data distribution'
DistributionPlot(y_train, y_hat_train, 'Actual Values (Train)', 'Predicted Values (Train)', Title)
```


![png](output_268_0.png)


Figure 1: Plot of predicted values using the training data compared to the training data. 

So far the model seems to be doing well in learning from the training dataset. But what happens when the model encounters new data from the testing dataset?


```python
# examine the distribution of the predicted values of the test data
Title = 'Distribution Plot of Predicted Value using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test, y_hat_test, 'Actual Values (Test)', 'Predicted Values (Test)', Title)
```


![png](output_271_0.png)


Figure 2: Plot of predicted value using the test data compared to the test data. 

When the model generates new values from the test data, we see the distribution of the predicted values is much different from the actual target values.  
Comparing Figure 1 and Figure 2, it is evident the distribution of the training data in Figure 1 is much better at fitting the data. This difference in Figure 2 is apparent where the ranges are from 5000 to 15000. This is where the distribution shape is exceptionally different.

Let's see if polynomial regression also exhibits a drop in the prediction accuracy when analysing the test dataset.


```python
from sklearn.preprocessing import PolynomialFeatures
```

### Overfitting
Overfitting occurs when the model fits the noise, not the underlying process. Therefore when testing your model using the test-set, your model does not perform as well as it is modelling noise, not the underlying process that generated the relationship.


```python
# use 45% of the data for testing and the rest for training
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
```


```python
# perform a degree 5 polynomial transformation on the feature 'horsepower'
pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
pr
```




    PolynomialFeatures(degree=5, include_bias=True, interaction_only=False,
                       order='C')




```python
# create a linear regression model 
poly = LinearRegression()

# train the model
poly.fit(x_train_pr, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# see the output of the model using predict
y_hat = poly.predict(x_test_pr)
yhat[0:5]
```




    array([14141.63807508, 14141.63807508, 20814.29423473, 12745.03562306,
           14762.35027598])




```python
# take the first five predicted values and compare it to the actual targets
print('Predicted values:', yhat[0:4])
print('True values', y_test[0:4].values)
```

    Predicted values: [14141.63807508 14141.63807508 20814.29423473 12745.03562306]
    True values [ 6295. 10698. 13860. 13499.]
    


```python
# display the plot
PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly, pr)
```


![png](output_282_0.png)


Figure 3: A polynomial regression model; red dots represent training data, green dots represent test data, and the blue line represents the model prediction. 

We see that the estimated function appears to track the data but around 200 horsepower, the function begins to diverge from the data points.


```python
# R^2 of training data
poly.score(x_train_pr, y_train)
```




    0.5567716902635091




```python
# R^2 of test data
poly.score(x_test_pr, y_test)
```




    -29.87141885918752



We see the R^2 for the training data is 0.5567 while the R^2 on the test data was -29.87. The lower the R^2, the worse the model, a Negative R^2 is a sign of overfitting.

Let's see how the R^2 changes on the test data for different order polynomials and plot the results.


```python
Rsqu_test = []
order = [1, 2, 3, 4]
for n in order:
  pr = PolynomialFeatures(degree=n)
  x_train_pr = pr.fit_transform(x_train[['horsepower']])
  x_test_pr = pr.fit_transform(x_test[['horsepower']])    
  lr.fit(x_train_pr, y_train)
  Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')    
```




    Text(3, 0.75, 'Maximum R^2 ')




![png](output_289_1.png)


We see the R^2 gradually increases until an order three polynomial is used. Then the R^2 dramatically decreases at four.


```python
def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train,y_test, poly, pr)
```

The following interface allows you to experiment with different polynomial orders and different amounts of data. 


```python
interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))
```


    interactive(children=(IntSlider(value=3, description='order', max=6), FloatSlider(value=0.45, description='tes…





    <function __main__.f(order, test_data)>



### Ridge Regression
Review Ridge Regression, see how the parameter Alpha changes the model. Our test data will be used as validation data.


```python
# perform a degree 2 polynomial transformation on our data
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'normalized-losses', 'symboling']])
x_test_pr = pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
```


```python
# import Ridge from the module
from sklearn.linear_model import Ridge
```


```python
# Create a Ridge Regression object, setting the regularization parameter to 0.1
RidgeModel = Ridge(alpha=0.1)
```


```python
# fit the model
RidgeModel.fit(x_train_pr, y_train)
```

    C:\Users\prana\Anaconda3\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=1.02972e-16): result may not be accurate.
      overwrite_a=True).T
    




    Ridge(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='auto', tol=0.001)




```python
# get the prediction
yhat = RidgeModel.predict(x_test_pr)
```


```python
# Compare the first 5 predicted samples to our test set
print('predicted:', yhat[0:4])
print('test set:', y_test[0:4].values)
```

    predicted: [ 6567.83081933  9597.97151399 20836.22326843 19347.69543463]
    test set: [ 6295. 10698. 13860. 13499.]
    

Select the value of Alpha that minimizes the test error. For e.g., we can use a loop.


```python
Rsqu_test = []
Rsqu_train = []
dummy1 = []
ALPHA = 10 * np.array(range(0,1000))
for alfa in ALPHA:
    RigeModel = Ridge(alpha=alfa) 
    RigeModel.fit(x_train_pr, y_train)
    Rsqu_test.append(RigeModel.score(x_test_pr, y_test))
    Rsqu_train.append(RigeModel.score(x_train_pr, y_train))
```


```python
# Plot the value of R^2 for different Alphas
width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(ALPHA,Rsqu_test, label='validation data  ')
plt.plot(ALPHA,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1d19ca7f780>




![png](output_303_1.png)


Figure 6: The blue line represents the R^2 of the test data, and the red line represents the R^2 of the training data. The x-axis represents the different values of Alpha. 

### Grid Search
The term Alpha is a hyperparameter; sklearn has the class GridSearchCV to make the process of finding the best hyperparameter simpler.


```python
# import GridSearchCV from the module model_selection
from sklearn.model_selection import GridSearchCV
```


```python
# create a dictionary of parameter values
parameters1 = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000]}]
parameters1
```




    [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000]}]




```python
# create a Ridge regions object
RR = Ridge()
RR
```




    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='auto', tol=0.001)




```python
# create a ridge grid seacrch object
Grid1 = GridSearchCV(RR, parameters1, cv=4)
```


```python
# Fit the model
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
```




    GridSearchCV(cv=4, error_score=nan,
                 estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True,
                                 max_iter=None, normalize=False, random_state=None,
                                 solver='auto', tol=0.001),
                 iid='deprecated', n_jobs=None,
                 param_grid=[{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000,
                                        100000]}],
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)



The object finds the best parameter values on the validation data. We can obtain the estimator with the best parameters and assign it to the variable BestRR.


```python
BestRR = Grid1.best_estimator_
BestRR
```




    Ridge(alpha=10000, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='auto', tol=0.001)




```python
# test our model on the test data
BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)
```




    0.8411649831036149




```python
# Perform a grid search for the alpha parameter and the normalization parameter, then find the best values of the parameters 
parameters2= [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000],'normalize':[True,False]} ]
Grid2 = GridSearchCV(Ridge(), parameters2,cv=4)
Grid2.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_data)
Grid2.best_estimator_
```




    Ridge(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=None, normalize=True,
          random_state=None, solver='auto', tol=0.001)


