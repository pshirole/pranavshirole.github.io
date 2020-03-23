---
layout: post
title: Where should you live in London?
tags: [segmentation, clustering, k-means, web scraping]
---

Imagine you are moving to London, UK. It's a major metropolitan city, a financial hub, a famous tourist destination, and home to around 9 million people. But as with every big city, crime is a concern, and you would like to live in a neighborhood that is safe and also popular. In this blog, we'll use the London Crime data and the Foursquare API to select which neighborhood best fits our needs.  

The London Crime data consists of more than 13 million rows containing counts of criminal reports by month, LSOA (Lower Super Output Area) borough, and major/minor category. You can [download the data here](https://www.kaggle.com/jboysen/london-crime#london_crime_by_lsoa.csv).  
About the data:  
- lsoa_code: code for Lower Super Output Area in Greater London.
- borough: Common name for London borough.
- major_category: High level categorization of crime
- minor_category: Low level categorization of crime within major category.
- value: monthly reported count of categorical crime in given borough
- year: Year of reported counts, 2008-2016
- month: Month of reported counts, 1-12


```python
# import libraries
import pandas as pd # library for data analysis
import numpy as np # library to handle data in a vectorized manner
import random # library for random number generation
import requests # library to handle requests
from bs4 import BeautifulSoup # library for web scraping

#!conda install -c conda-forge geocoder --yes
import geocoder

#!conda install -c conda-forge geopy --yes 
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values

# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 
    
# transforming json file into a pandas dataframe library
from pandas.io.json import json_normalize

#!conda install -c conda-forge folium=0.5.0 --yes
import folium # plotting library

print('Libraries imported.')
```

    Libraries imported.
    

Define Foursquare credentials.


```python
CLIENT_ID = '**********'
CLIENT_SECRET = '**********'
VERSION = '20191912'

# limit the number of venues returned by the foursquare API
LIMIT = 50
```

### Preprocessing the data

Read the dataset into a pandas dataframe.


```python
df = pd.read_csv('london_crime_by_lsoa.csv')
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
      <th>lsoa_code</th>
      <th>borough</th>
      <th>major_category</th>
      <th>minor_category</th>
      <th>value</th>
      <th>year</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E01001116</td>
      <td>Croydon</td>
      <td>Burglary</td>
      <td>Burglary in Other Buildings</td>
      <td>0</td>
      <td>2016</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E01001646</td>
      <td>Greenwich</td>
      <td>Violence Against the Person</td>
      <td>Other violence</td>
      <td>0</td>
      <td>2016</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E01000677</td>
      <td>Bromley</td>
      <td>Violence Against the Person</td>
      <td>Other violence</td>
      <td>0</td>
      <td>2015</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E01003774</td>
      <td>Redbridge</td>
      <td>Burglary</td>
      <td>Burglary in Other Buildings</td>
      <td>0</td>
      <td>2016</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E01004563</td>
      <td>Wandsworth</td>
      <td>Robbery</td>
      <td>Personal Property</td>
      <td>0</td>
      <td>2008</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# dimensions of the dataframe
df.shape
```




    (13490604, 7)



### Preprocessing the data


```python
# remove all null value entries
df = df[df.value != 0]

# reset the index and drop the previous index
df = df.reset_index(drop=True)

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
      <th>lsoa_code</th>
      <th>borough</th>
      <th>major_category</th>
      <th>minor_category</th>
      <th>value</th>
      <th>year</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E01004177</td>
      <td>Sutton</td>
      <td>Theft and Handling</td>
      <td>Theft/Taking of Pedal Cycle</td>
      <td>1</td>
      <td>2016</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E01000086</td>
      <td>Barking and Dagenham</td>
      <td>Theft and Handling</td>
      <td>Other Theft Person</td>
      <td>1</td>
      <td>2009</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E01001301</td>
      <td>Ealing</td>
      <td>Theft and Handling</td>
      <td>Other Theft Person</td>
      <td>2</td>
      <td>2012</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E01001794</td>
      <td>Hackney</td>
      <td>Violence Against the Person</td>
      <td>Harassment</td>
      <td>1</td>
      <td>2013</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E01000733</td>
      <td>Bromley</td>
      <td>Criminal Damage</td>
      <td>Criminal Damage To Motor Vehicle</td>
      <td>1</td>
      <td>2016</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# new dimensions of the dataframe
df.shape
```




    (3419099, 7)



Change the column names.


```python
df.columns = ['LSOA_Code', 'Borough', 'Major_Category', 'Minor_Category', 'No_of_Crimes', 'Year', 'Month']
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
      <th>LSOA_Code</th>
      <th>Borough</th>
      <th>Major_Category</th>
      <th>Minor_Category</th>
      <th>No_of_Crimes</th>
      <th>Year</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E01004177</td>
      <td>Sutton</td>
      <td>Theft and Handling</td>
      <td>Theft/Taking of Pedal Cycle</td>
      <td>1</td>
      <td>2016</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E01000086</td>
      <td>Barking and Dagenham</td>
      <td>Theft and Handling</td>
      <td>Other Theft Person</td>
      <td>1</td>
      <td>2009</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# dataset information
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3419099 entries, 0 to 3419098
    Data columns (total 7 columns):
     #   Column          Dtype 
    ---  ------          ----- 
     0   LSOA_Code       object
     1   Borough         object
     2   Major_Category  object
     3   Minor_Category  object
     4   No_of_Crimes    int64 
     5   Year            int64 
     6   Month           int64 
    dtypes: int64(3), object(4)
    memory usage: 182.6+ MB
    

**What is the total number of crimes in each Borough?**


```python
df['Borough'].value_counts()
```




    Lambeth                   152784
    Croydon                   147203
    Southwark                 144362
    Ealing                    140006
    Newham                    137275
    Brent                     129925
    Lewisham                  128232
    Barnet                    127194
    Tower Hamlets             120099
    Wandsworth                118995
    Enfield                   117953
    Hackney                   116521
    Haringey                  116315
    Waltham Forest            114603
    Camden                    112029
    Islington                 111755
    Hillingdon                110614
    Westminster               110070
    Bromley                   109855
    Hounslow                  106561
    Redbridge                 105932
    Greenwich                 104654
    Hammersmith and Fulham     92084
    Barking and Dagenham       86849
    Havering                   82288
    Kensington and Chelsea     81295
    Harrow                     73993
    Bexley                     73948
    Merton                     73661
    Sutton                     62776
    Richmond upon Thames       61857
    Kingston upon Thames       46846
    City of London               565
    Name: Borough, dtype: int64



The Boroughs of Lambeth, Croydon, Southwark and Ealing have the highest number of crimes from the year 2008 to 2016.

**What is the total number of crimes per major category?**


```python
df['Major_Category'].value_counts()
```




    Theft and Handling             1136994
    Violence Against the Person     894859
    Criminal Damage                 466268
    Burglary                        441209
    Drugs                           231894
    Robbery                         163549
    Other Notifiable Offences        80569
    Fraud or Forgery                  2682
    Sexual Offences                   1075
    Name: Major_Category, dtype: int64



Pivot the table to view the number of crimes for each major category in each Borough.


```python
London_crime = pd.pivot_table(df, values=['No_of_Crimes'], 
                              index=['Borough'],
                             columns=['Major_Category'],
                             aggfunc=np.sum, fill_value=0)
London_crime.head()
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
      <th colspan="9" halign="left">No_of_Crimes</th>
    </tr>
    <tr>
      <th>Major_Category</th>
      <th>Burglary</th>
      <th>Criminal Damage</th>
      <th>Drugs</th>
      <th>Fraud or Forgery</th>
      <th>Other Notifiable Offences</th>
      <th>Robbery</th>
      <th>Sexual Offences</th>
      <th>Theft and Handling</th>
      <th>Violence Against the Person</th>
    </tr>
    <tr>
      <th>Borough</th>
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
      <th>Barking and Dagenham</th>
      <td>18103</td>
      <td>18888</td>
      <td>9188</td>
      <td>205</td>
      <td>2819</td>
      <td>6105</td>
      <td>49</td>
      <td>50999</td>
      <td>43091</td>
    </tr>
    <tr>
      <th>Barnet</th>
      <td>36981</td>
      <td>21024</td>
      <td>9796</td>
      <td>175</td>
      <td>2953</td>
      <td>7374</td>
      <td>38</td>
      <td>87285</td>
      <td>46565</td>
    </tr>
    <tr>
      <th>Bexley</th>
      <td>14973</td>
      <td>17244</td>
      <td>7346</td>
      <td>106</td>
      <td>1999</td>
      <td>2338</td>
      <td>22</td>
      <td>40071</td>
      <td>30037</td>
    </tr>
    <tr>
      <th>Brent</th>
      <td>28923</td>
      <td>20569</td>
      <td>25978</td>
      <td>157</td>
      <td>3711</td>
      <td>12473</td>
      <td>39</td>
      <td>72523</td>
      <td>63178</td>
    </tr>
    <tr>
      <th>Bromley</th>
      <td>27135</td>
      <td>24039</td>
      <td>8942</td>
      <td>196</td>
      <td>2637</td>
      <td>4868</td>
      <td>31</td>
      <td>69742</td>
      <td>46759</td>
    </tr>
  </tbody>
</table>
</div>




```python
# reset the index
London_crime.reset_index(inplace=True)
```


```python
# total crimes per Borough
London_crime['Total'] = London_crime.sum(axis=1)
London_crime.head()
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
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Borough</th>
      <th colspan="9" halign="left">No_of_Crimes</th>
      <th>Total</th>
    </tr>
    <tr>
      <th>Major_Category</th>
      <th></th>
      <th>Burglary</th>
      <th>Criminal Damage</th>
      <th>Drugs</th>
      <th>Fraud or Forgery</th>
      <th>Other Notifiable Offences</th>
      <th>Robbery</th>
      <th>Sexual Offences</th>
      <th>Theft and Handling</th>
      <th>Violence Against the Person</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Barking and Dagenham</td>
      <td>18103</td>
      <td>18888</td>
      <td>9188</td>
      <td>205</td>
      <td>2819</td>
      <td>6105</td>
      <td>49</td>
      <td>50999</td>
      <td>43091</td>
      <td>149447</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barnet</td>
      <td>36981</td>
      <td>21024</td>
      <td>9796</td>
      <td>175</td>
      <td>2953</td>
      <td>7374</td>
      <td>38</td>
      <td>87285</td>
      <td>46565</td>
      <td>212191</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bexley</td>
      <td>14973</td>
      <td>17244</td>
      <td>7346</td>
      <td>106</td>
      <td>1999</td>
      <td>2338</td>
      <td>22</td>
      <td>40071</td>
      <td>30037</td>
      <td>114136</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brent</td>
      <td>28923</td>
      <td>20569</td>
      <td>25978</td>
      <td>157</td>
      <td>3711</td>
      <td>12473</td>
      <td>39</td>
      <td>72523</td>
      <td>63178</td>
      <td>227551</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bromley</td>
      <td>27135</td>
      <td>24039</td>
      <td>8942</td>
      <td>196</td>
      <td>2637</td>
      <td>4868</td>
      <td>31</td>
      <td>69742</td>
      <td>46759</td>
      <td>184349</td>
    </tr>
  </tbody>
</table>
</div>



Remove the multi-index so that it will be easier to merge the columns.


```python
London_crime.columns = London_crime.columns.map(' '.join)
London_crime.head()
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
      <th>Borough</th>
      <th>No_of_Crimes Burglary</th>
      <th>No_of_Crimes Criminal Damage</th>
      <th>No_of_Crimes Drugs</th>
      <th>No_of_Crimes Fraud or Forgery</th>
      <th>No_of_Crimes Other Notifiable Offences</th>
      <th>No_of_Crimes Robbery</th>
      <th>No_of_Crimes Sexual Offences</th>
      <th>No_of_Crimes Theft and Handling</th>
      <th>No_of_Crimes Violence Against the Person</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Barking and Dagenham</td>
      <td>18103</td>
      <td>18888</td>
      <td>9188</td>
      <td>205</td>
      <td>2819</td>
      <td>6105</td>
      <td>49</td>
      <td>50999</td>
      <td>43091</td>
      <td>149447</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barnet</td>
      <td>36981</td>
      <td>21024</td>
      <td>9796</td>
      <td>175</td>
      <td>2953</td>
      <td>7374</td>
      <td>38</td>
      <td>87285</td>
      <td>46565</td>
      <td>212191</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bexley</td>
      <td>14973</td>
      <td>17244</td>
      <td>7346</td>
      <td>106</td>
      <td>1999</td>
      <td>2338</td>
      <td>22</td>
      <td>40071</td>
      <td>30037</td>
      <td>114136</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brent</td>
      <td>28923</td>
      <td>20569</td>
      <td>25978</td>
      <td>157</td>
      <td>3711</td>
      <td>12473</td>
      <td>39</td>
      <td>72523</td>
      <td>63178</td>
      <td>227551</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bromley</td>
      <td>27135</td>
      <td>24039</td>
      <td>8942</td>
      <td>196</td>
      <td>2637</td>
      <td>4868</td>
      <td>31</td>
      <td>69742</td>
      <td>46759</td>
      <td>184349</td>
    </tr>
  </tbody>
</table>
</div>



Let's rename the columns for better comprehensibility.


```python
London_crime.columns = ['Borough', 'Burglary', 'Criminal Damage', 'Drugs', 'Fraud or Forgery', 'Other Notifiable Offenses', 
                        'Robbery', 'Sexual Offences', 'Theft and Handling', 'Violence Against the Person', 'Total']
London_crime
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
      <th>Borough</th>
      <th>Burglary</th>
      <th>Criminal Damage</th>
      <th>Drugs</th>
      <th>Fraud or Forgery</th>
      <th>Other Notifiable Offenses</th>
      <th>Robbery</th>
      <th>Sexual Offences</th>
      <th>Theft and Handling</th>
      <th>Violence Against the Person</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Barking and Dagenham</td>
      <td>18103</td>
      <td>18888</td>
      <td>9188</td>
      <td>205</td>
      <td>2819</td>
      <td>6105</td>
      <td>49</td>
      <td>50999</td>
      <td>43091</td>
      <td>149447</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barnet</td>
      <td>36981</td>
      <td>21024</td>
      <td>9796</td>
      <td>175</td>
      <td>2953</td>
      <td>7374</td>
      <td>38</td>
      <td>87285</td>
      <td>46565</td>
      <td>212191</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bexley</td>
      <td>14973</td>
      <td>17244</td>
      <td>7346</td>
      <td>106</td>
      <td>1999</td>
      <td>2338</td>
      <td>22</td>
      <td>40071</td>
      <td>30037</td>
      <td>114136</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brent</td>
      <td>28923</td>
      <td>20569</td>
      <td>25978</td>
      <td>157</td>
      <td>3711</td>
      <td>12473</td>
      <td>39</td>
      <td>72523</td>
      <td>63178</td>
      <td>227551</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bromley</td>
      <td>27135</td>
      <td>24039</td>
      <td>8942</td>
      <td>196</td>
      <td>2637</td>
      <td>4868</td>
      <td>31</td>
      <td>69742</td>
      <td>46759</td>
      <td>184349</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Camden</td>
      <td>27939</td>
      <td>18482</td>
      <td>21816</td>
      <td>123</td>
      <td>3857</td>
      <td>9286</td>
      <td>36</td>
      <td>140596</td>
      <td>53012</td>
      <td>275147</td>
    </tr>
    <tr>
      <th>6</th>
      <td>City of London</td>
      <td>15</td>
      <td>16</td>
      <td>33</td>
      <td>0</td>
      <td>17</td>
      <td>24</td>
      <td>0</td>
      <td>561</td>
      <td>114</td>
      <td>780</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Croydon</td>
      <td>33376</td>
      <td>31218</td>
      <td>19162</td>
      <td>270</td>
      <td>4340</td>
      <td>12645</td>
      <td>55</td>
      <td>91437</td>
      <td>67791</td>
      <td>260294</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ealing</td>
      <td>30831</td>
      <td>25613</td>
      <td>18591</td>
      <td>175</td>
      <td>4406</td>
      <td>9568</td>
      <td>52</td>
      <td>93834</td>
      <td>68492</td>
      <td>251562</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Enfield</td>
      <td>30213</td>
      <td>22487</td>
      <td>13251</td>
      <td>132</td>
      <td>3293</td>
      <td>9059</td>
      <td>38</td>
      <td>70371</td>
      <td>45036</td>
      <td>193880</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Greenwich</td>
      <td>20966</td>
      <td>22755</td>
      <td>10836</td>
      <td>107</td>
      <td>3598</td>
      <td>5430</td>
      <td>56</td>
      <td>64923</td>
      <td>52897</td>
      <td>181568</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hackney</td>
      <td>21450</td>
      <td>17327</td>
      <td>18144</td>
      <td>143</td>
      <td>3332</td>
      <td>8975</td>
      <td>46</td>
      <td>91118</td>
      <td>56584</td>
      <td>217119</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Hammersmith and Fulham</td>
      <td>17010</td>
      <td>14595</td>
      <td>15492</td>
      <td>91</td>
      <td>3352</td>
      <td>5279</td>
      <td>45</td>
      <td>86381</td>
      <td>43014</td>
      <td>185259</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Haringey</td>
      <td>28213</td>
      <td>22272</td>
      <td>14563</td>
      <td>207</td>
      <td>2971</td>
      <td>10084</td>
      <td>40</td>
      <td>83979</td>
      <td>50943</td>
      <td>213272</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Harrow</td>
      <td>19630</td>
      <td>12724</td>
      <td>7122</td>
      <td>92</td>
      <td>1998</td>
      <td>4242</td>
      <td>27</td>
      <td>40800</td>
      <td>30213</td>
      <td>116848</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Havering</td>
      <td>21302</td>
      <td>17252</td>
      <td>8171</td>
      <td>179</td>
      <td>2358</td>
      <td>3089</td>
      <td>19</td>
      <td>52609</td>
      <td>33968</td>
      <td>138947</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Hillingdon</td>
      <td>26056</td>
      <td>24485</td>
      <td>11413</td>
      <td>223</td>
      <td>6504</td>
      <td>5663</td>
      <td>44</td>
      <td>80028</td>
      <td>55264</td>
      <td>209680</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Hounslow</td>
      <td>21026</td>
      <td>21407</td>
      <td>13722</td>
      <td>183</td>
      <td>3963</td>
      <td>4847</td>
      <td>40</td>
      <td>70180</td>
      <td>51404</td>
      <td>186772</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Islington</td>
      <td>22207</td>
      <td>18354</td>
      <td>16553</td>
      <td>85</td>
      <td>3675</td>
      <td>8736</td>
      <td>40</td>
      <td>107661</td>
      <td>52975</td>
      <td>230286</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Kensington and Chelsea</td>
      <td>14980</td>
      <td>9839</td>
      <td>14573</td>
      <td>85</td>
      <td>2203</td>
      <td>4744</td>
      <td>24</td>
      <td>95963</td>
      <td>29570</td>
      <td>171981</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Kingston upon Thames</td>
      <td>10131</td>
      <td>10610</td>
      <td>5682</td>
      <td>65</td>
      <td>1332</td>
      <td>1702</td>
      <td>18</td>
      <td>38226</td>
      <td>21540</td>
      <td>89306</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Lambeth</td>
      <td>30199</td>
      <td>26136</td>
      <td>25083</td>
      <td>137</td>
      <td>4520</td>
      <td>18408</td>
      <td>70</td>
      <td>114899</td>
      <td>72726</td>
      <td>292178</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Lewisham</td>
      <td>24871</td>
      <td>24810</td>
      <td>16825</td>
      <td>262</td>
      <td>3809</td>
      <td>10455</td>
      <td>71</td>
      <td>70382</td>
      <td>63652</td>
      <td>215137</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Merton</td>
      <td>16485</td>
      <td>14339</td>
      <td>6651</td>
      <td>111</td>
      <td>1571</td>
      <td>4021</td>
      <td>26</td>
      <td>44128</td>
      <td>28322</td>
      <td>115654</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Newham</td>
      <td>25356</td>
      <td>24177</td>
      <td>18389</td>
      <td>323</td>
      <td>4456</td>
      <td>16913</td>
      <td>43</td>
      <td>106146</td>
      <td>66221</td>
      <td>262024</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Redbridge</td>
      <td>26735</td>
      <td>17543</td>
      <td>15736</td>
      <td>284</td>
      <td>2619</td>
      <td>7688</td>
      <td>31</td>
      <td>71496</td>
      <td>41430</td>
      <td>183562</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Richmond upon Thames</td>
      <td>16097</td>
      <td>11722</td>
      <td>4707</td>
      <td>37</td>
      <td>1420</td>
      <td>1590</td>
      <td>26</td>
      <td>40858</td>
      <td>20314</td>
      <td>96771</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Southwark</td>
      <td>27980</td>
      <td>24450</td>
      <td>27381</td>
      <td>321</td>
      <td>4696</td>
      <td>16153</td>
      <td>40</td>
      <td>109432</td>
      <td>68356</td>
      <td>278809</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Sutton</td>
      <td>13207</td>
      <td>14474</td>
      <td>4586</td>
      <td>57</td>
      <td>1393</td>
      <td>2308</td>
      <td>20</td>
      <td>39533</td>
      <td>25409</td>
      <td>100987</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Tower Hamlets</td>
      <td>21510</td>
      <td>21593</td>
      <td>23408</td>
      <td>124</td>
      <td>4268</td>
      <td>10050</td>
      <td>47</td>
      <td>87620</td>
      <td>59993</td>
      <td>228613</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Waltham Forest</td>
      <td>25565</td>
      <td>20459</td>
      <td>14101</td>
      <td>236</td>
      <td>3040</td>
      <td>10606</td>
      <td>34</td>
      <td>77940</td>
      <td>51898</td>
      <td>203879</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Wandsworth</td>
      <td>25533</td>
      <td>19630</td>
      <td>9493</td>
      <td>161</td>
      <td>3091</td>
      <td>8398</td>
      <td>47</td>
      <td>92523</td>
      <td>45865</td>
      <td>204741</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Westminster</td>
      <td>29295</td>
      <td>20405</td>
      <td>34031</td>
      <td>273</td>
      <td>6148</td>
      <td>15752</td>
      <td>59</td>
      <td>277617</td>
      <td>71448</td>
      <td>455028</td>
    </tr>
  </tbody>
</table>
</div>



### Scraping data from the web
Let's scrape additional information about the different Boroughs in London from the ["List of London boroughs" Wikipedia page](https://en.wikipedia.org/wiki/List_of_London_boroughs).  
We'll use the **Beautiful Soup** library to scrape the latitude and longitude coordinates of the boroghs in London.


```python
# getting data from internet
wikipedia_link = 'https://en.wikipedia.org/wiki/List_of_London_boroughs'
raw_wikipedia_page = requests.get(wikipedia_link).text

# using beautiful soup to parse the HTML/XML codes.
soup = BeautifulSoup(raw_wikipedia_page,'xml')
print(soup.prettify())
```

*Note: I am not including the extracted data from the HTML page since it will take up too much space in this post.*
    

Extract the raw table inside the webpage.


```python
table = soup.find_all('table', {'class':'wikitable sortable'})
print(table)
```

*Note: I am not including the extracted data from the table since it will take up too much space in this post.*
    

Convert the table into a dataframe.


```python
London_table = pd.read_html(str(table[0]), index_col=None, header=0)[0]
London_table.head()
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
      <th>Borough</th>
      <th>Inner</th>
      <th>Status</th>
      <th>Local authority</th>
      <th>Political control</th>
      <th>Headquarters</th>
      <th>Area (sq mi)</th>
      <th>Population (2013 est)[1]</th>
      <th>Co-ordinates</th>
      <th>Nr. in map</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Barking and Dagenham [note 1]</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Barking and Dagenham London Borough Council</td>
      <td>Labour</td>
      <td>Town Hall, 1 Town Square</td>
      <td>13.93</td>
      <td>194352</td>
      <td>51°33′39″N 0°09′21″E﻿ / ﻿51.5607°N 0.1557°E</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barnet</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Barnet London Borough Council</td>
      <td>Conservative</td>
      <td>North London Business Park, Oakleigh Road South</td>
      <td>33.49</td>
      <td>369088</td>
      <td>51°37′31″N 0°09′06″W﻿ / ﻿51.6252°N 0.1517°W</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bexley</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Bexley London Borough Council</td>
      <td>Conservative</td>
      <td>Civic Offices, 2 Watling Street</td>
      <td>23.38</td>
      <td>236687</td>
      <td>51°27′18″N 0°09′02″E﻿ / ﻿51.4549°N 0.1505°E</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brent</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Brent London Borough Council</td>
      <td>Labour</td>
      <td>Brent Civic Centre, Engineers Way</td>
      <td>16.70</td>
      <td>317264</td>
      <td>51°33′32″N 0°16′54″W﻿ / ﻿51.5588°N 0.2817°W</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bromley</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Bromley London Borough Council</td>
      <td>Conservative</td>
      <td>Civic Centre, Stockwell Close</td>
      <td>57.97</td>
      <td>317899</td>
      <td>51°24′14″N 0°01′11″E﻿ / ﻿51.4039°N 0.0198°E</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



There is a second table on the webpage that contains the additional Borough - City of London.


```python
# read the second table
London_table1 = pd.read_html(str(table[1]), index_col=None, header=0)[0]

# rename the columns to match the previous table 
London_table1.columns = ['Borough', 'Inner', 'Status', 'Local authority', 'Political control', 'Headquarters', 
                         'Area (sq mi)', 'Population (2013 est)[1]', 'Co-ordinates', 'Nr. in map']

# view the table
London_table1
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
      <th>Borough</th>
      <th>Inner</th>
      <th>Status</th>
      <th>Local authority</th>
      <th>Political control</th>
      <th>Headquarters</th>
      <th>Area (sq mi)</th>
      <th>Population (2013 est)[1]</th>
      <th>Co-ordinates</th>
      <th>Nr. in map</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City of London</td>
      <td>([note 5]</td>
      <td>Sui generis;City;Ceremonial county</td>
      <td>Corporation of London;Inner Temple;Middle Temple</td>
      <td>?</td>
      <td>Guildhall</td>
      <td>1.12</td>
      <td>7000</td>
      <td>51°30′56″N 0°05′32″W﻿ / ﻿51.5155°N 0.0922°W</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Let's append the dataframes of 'London_table' and 'London_table1' together. A continuous index value will be maintained across the rows in the new appended dataframe.


```python
London_table = London_table.append(London_table1, ignore_index=True)

# check the last rows of the data set
London_table.tail()
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
      <th>Borough</th>
      <th>Inner</th>
      <th>Status</th>
      <th>Local authority</th>
      <th>Political control</th>
      <th>Headquarters</th>
      <th>Area (sq mi)</th>
      <th>Population (2013 est)[1]</th>
      <th>Co-ordinates</th>
      <th>Nr. in map</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>Tower Hamlets</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Tower Hamlets London Borough Council</td>
      <td>Labour</td>
      <td>Town Hall, Mulberry Place, 5 Clove Crescent</td>
      <td>7.63</td>
      <td>272890</td>
      <td>51°30′36″N 0°00′21″W﻿ / ﻿51.5099°N 0.0059°W</td>
      <td>8</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Waltham Forest</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Waltham Forest London Borough Council</td>
      <td>Labour</td>
      <td>Waltham Forest Town Hall, Forest Road</td>
      <td>14.99</td>
      <td>265797</td>
      <td>51°35′27″N 0°00′48″W﻿ / ﻿51.5908°N 0.0134°W</td>
      <td>28</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Wandsworth</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Wandsworth London Borough Council</td>
      <td>Conservative</td>
      <td>The Town Hall, Wandsworth High Street</td>
      <td>13.23</td>
      <td>310516</td>
      <td>51°27′24″N 0°11′28″W﻿ / ﻿51.4567°N 0.1910°W</td>
      <td>5</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Westminster</td>
      <td>NaN</td>
      <td>City</td>
      <td>Westminster City Council</td>
      <td>Conservative</td>
      <td>Westminster City Hall, 64 Victoria Street</td>
      <td>8.29</td>
      <td>226841</td>
      <td>51°29′50″N 0°08′14″W﻿ / ﻿51.4973°N 0.1372°W</td>
      <td>2</td>
    </tr>
    <tr>
      <th>32</th>
      <td>City of London</td>
      <td>([note 5]</td>
      <td>Sui generis;City;Ceremonial county</td>
      <td>Corporation of London;Inner Temple;Middle Temple</td>
      <td>?</td>
      <td>Guildhall</td>
      <td>1.12</td>
      <td>7000</td>
      <td>51°30′56″N 0°05′32″W﻿ / ﻿51.5155°N 0.0922°W</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We'll remove the unnecessary strings in the dataset.


```python
London_table = London_table.replace('note 1','', regex=True) 
London_table = London_table.replace('note 2','', regex=True) 
London_table = London_table.replace('note 3','', regex=True) 
London_table = London_table.replace('note 4','', regex=True) 
London_table = London_table.replace('note 5','', regex=True) 

London_table.head()
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
      <th>Borough</th>
      <th>Inner</th>
      <th>Status</th>
      <th>Local authority</th>
      <th>Political control</th>
      <th>Headquarters</th>
      <th>Area (sq mi)</th>
      <th>Population (2013 est)[1]</th>
      <th>Co-ordinates</th>
      <th>Nr. in map</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Barking and Dagenham []</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Barking and Dagenham London Borough Council</td>
      <td>Labour</td>
      <td>Town Hall, 1 Town Square</td>
      <td>13.93</td>
      <td>194352</td>
      <td>51°33′39″N 0°09′21″E﻿ / ﻿51.5607°N 0.1557°E</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barnet</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Barnet London Borough Council</td>
      <td>Conservative</td>
      <td>North London Business Park, Oakleigh Road South</td>
      <td>33.49</td>
      <td>369088</td>
      <td>51°37′31″N 0°09′06″W﻿ / ﻿51.6252°N 0.1517°W</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bexley</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Bexley London Borough Council</td>
      <td>Conservative</td>
      <td>Civic Offices, 2 Watling Street</td>
      <td>23.38</td>
      <td>236687</td>
      <td>51°27′18″N 0°09′02″E﻿ / ﻿51.4549°N 0.1505°E</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brent</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Brent London Borough Council</td>
      <td>Labour</td>
      <td>Brent Civic Centre, Engineers Way</td>
      <td>16.70</td>
      <td>317264</td>
      <td>51°33′32″N 0°16′54″W﻿ / ﻿51.5588°N 0.2817°W</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bromley</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Bromley London Borough Council</td>
      <td>Conservative</td>
      <td>Civic Centre, Stockwell Close</td>
      <td>57.97</td>
      <td>317899</td>
      <td>51°24′14″N 0°01′11″E﻿ / ﻿51.4039°N 0.0198°E</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>




```python
# type of the dataframe
type(London_table)
```




    pandas.core.frame.DataFrame




```python
# shape of the dataframe
London_table.shape
```




    (33, 10)



Check if the Borough in both the dataframes match.


```python
set(df.Borough) - set(London_table.Borough)
```




    {'Barking and Dagenham', 'Greenwich', 'Hammersmith and Fulham'}



These 3 Boroughs don't match because of the unnecessary symbols like '[ ]' present.  
Let's find the index of the 3 Boroughs that do not match.


```python
print("The index of first borough is",London_table.index[London_table['Borough'] == 'Barking and Dagenham []'].tolist())
print("The index of second borough is",London_table.index[London_table['Borough'] == 'Greenwich []'].tolist())
print("The index of third borough is",London_table.index[London_table['Borough'] == 'Hammersmith and Fulham []'].tolist())
```

    The index of first borough is [0]
    The index of second borough is [9]
    The index of third borough is [11]
    

Change the Borough names to match the other data frame.


```python
London_table.iloc[0,0] = 'Barking and Dagenham'
London_table.iloc[9,0] = 'Greenwich'
London_table.iloc[11,0] = 'Hammersmith and Fulham'
```


```python
set(df.Borough) - set(London_table.Borough)
```




    set()



The Borough names in both dataframes match.  
Now, we combine both the dataframes together.


```python
Ld_crime = pd.merge(London_crime, London_table, on='Borough')
Ld_crime.head()
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
      <th>Borough</th>
      <th>Burglary</th>
      <th>Criminal Damage</th>
      <th>Drugs</th>
      <th>Fraud or Forgery</th>
      <th>Other Notifiable Offenses</th>
      <th>Robbery</th>
      <th>Sexual Offences</th>
      <th>Theft and Handling</th>
      <th>Violence Against the Person</th>
      <th>Total</th>
      <th>Inner</th>
      <th>Status</th>
      <th>Local authority</th>
      <th>Political control</th>
      <th>Headquarters</th>
      <th>Area (sq mi)</th>
      <th>Population (2013 est)[1]</th>
      <th>Co-ordinates</th>
      <th>Nr. in map</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Barking and Dagenham</td>
      <td>18103</td>
      <td>18888</td>
      <td>9188</td>
      <td>205</td>
      <td>2819</td>
      <td>6105</td>
      <td>49</td>
      <td>50999</td>
      <td>43091</td>
      <td>149447</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Barking and Dagenham London Borough Council</td>
      <td>Labour</td>
      <td>Town Hall, 1 Town Square</td>
      <td>13.93</td>
      <td>194352</td>
      <td>51°33′39″N 0°09′21″E﻿ / ﻿51.5607°N 0.1557°E</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barnet</td>
      <td>36981</td>
      <td>21024</td>
      <td>9796</td>
      <td>175</td>
      <td>2953</td>
      <td>7374</td>
      <td>38</td>
      <td>87285</td>
      <td>46565</td>
      <td>212191</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Barnet London Borough Council</td>
      <td>Conservative</td>
      <td>North London Business Park, Oakleigh Road South</td>
      <td>33.49</td>
      <td>369088</td>
      <td>51°37′31″N 0°09′06″W﻿ / ﻿51.6252°N 0.1517°W</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bexley</td>
      <td>14973</td>
      <td>17244</td>
      <td>7346</td>
      <td>106</td>
      <td>1999</td>
      <td>2338</td>
      <td>22</td>
      <td>40071</td>
      <td>30037</td>
      <td>114136</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Bexley London Borough Council</td>
      <td>Conservative</td>
      <td>Civic Offices, 2 Watling Street</td>
      <td>23.38</td>
      <td>236687</td>
      <td>51°27′18″N 0°09′02″E﻿ / ﻿51.4549°N 0.1505°E</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brent</td>
      <td>28923</td>
      <td>20569</td>
      <td>25978</td>
      <td>157</td>
      <td>3711</td>
      <td>12473</td>
      <td>39</td>
      <td>72523</td>
      <td>63178</td>
      <td>227551</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Brent London Borough Council</td>
      <td>Labour</td>
      <td>Brent Civic Centre, Engineers Way</td>
      <td>16.70</td>
      <td>317264</td>
      <td>51°33′32″N 0°16′54″W﻿ / ﻿51.5588°N 0.2817°W</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bromley</td>
      <td>27135</td>
      <td>24039</td>
      <td>8942</td>
      <td>196</td>
      <td>2637</td>
      <td>4868</td>
      <td>31</td>
      <td>69742</td>
      <td>46759</td>
      <td>184349</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Bromley London Borough Council</td>
      <td>Conservative</td>
      <td>Civic Centre, Stockwell Close</td>
      <td>57.97</td>
      <td>317899</td>
      <td>51°24′14″N 0°01′11″E﻿ / ﻿51.4039°N 0.0198°E</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>




```python
# shape of the dataframe
Ld_crime.shape
```




    (33, 20)




```python
# check if the names of Boroughs in both the dataframes match
set(df.Borough) - set(Ld_crime.Borough)
```




    set()



Rearrange the Columns.


```python
# list the column names of the dataframe
list(Ld_crime)
```




    ['Borough',
     'Burglary',
     'Criminal Damage',
     'Drugs',
     'Fraud or Forgery',
     'Other Notifiable Offenses',
     'Robbery',
     'Sexual Offences',
     'Theft and Handling',
     'Violence Against the Person',
     'Total',
     'Inner',
     'Status',
     'Local authority',
     'Political control',
     'Headquarters',
     'Area (sq mi)',
     'Population (2013 est)[1]',
     'Co-ordinates',
     'Nr. in map']




```python
# rename the Population column
Ld_crime = Ld_crime.rename(columns = {'Population (2013 est)[1]':'Population'})
```


```python
columnsTitles = ['Borough', 'Local authority', 'Political control', 'Headquarters', 'Area (sq mi)', 'Population', 'Co-ordinates',
               'Burglary', 'Criminal Damage', 'Drugs', 'Fraud or Forgery', 'Other Notifiable Offenses', 'Robbery', 'Sexual Offenses', 
                'Theft and Handling', 'Violence Against the Person', 'Total']

Ld_crime = Ld_crime.reindex(columns=columnsTitles)

Ld_crime = Ld_crime[['Borough', 'Local authority', 'Political control', 'Headquarters', 'Area (sq mi)', 'Population', 'Co-ordinates',
               'Burglary', 'Criminal Damage', 'Drugs', 'Fraud or Forgery', 'Other Notifiable Offenses', 'Robbery', 'Sexual Offenses', 
                'Theft and Handling', 'Violence Against the Person', 'Total']]

Ld_crime
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
      <th>Borough</th>
      <th>Local authority</th>
      <th>Political control</th>
      <th>Headquarters</th>
      <th>Area (sq mi)</th>
      <th>Population</th>
      <th>Co-ordinates</th>
      <th>Burglary</th>
      <th>Criminal Damage</th>
      <th>Drugs</th>
      <th>Fraud or Forgery</th>
      <th>Other Notifiable Offenses</th>
      <th>Robbery</th>
      <th>Sexual Offenses</th>
      <th>Theft and Handling</th>
      <th>Violence Against the Person</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Barking and Dagenham</td>
      <td>Barking and Dagenham London Borough Council</td>
      <td>Labour</td>
      <td>Town Hall, 1 Town Square</td>
      <td>13.93</td>
      <td>194352</td>
      <td>51°33′39″N 0°09′21″E﻿ / ﻿51.5607°N 0.1557°E</td>
      <td>18103</td>
      <td>18888</td>
      <td>9188</td>
      <td>205</td>
      <td>2819</td>
      <td>6105</td>
      <td>NaN</td>
      <td>50999</td>
      <td>43091</td>
      <td>149447</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barnet</td>
      <td>Barnet London Borough Council</td>
      <td>Conservative</td>
      <td>North London Business Park, Oakleigh Road South</td>
      <td>33.49</td>
      <td>369088</td>
      <td>51°37′31″N 0°09′06″W﻿ / ﻿51.6252°N 0.1517°W</td>
      <td>36981</td>
      <td>21024</td>
      <td>9796</td>
      <td>175</td>
      <td>2953</td>
      <td>7374</td>
      <td>NaN</td>
      <td>87285</td>
      <td>46565</td>
      <td>212191</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bexley</td>
      <td>Bexley London Borough Council</td>
      <td>Conservative</td>
      <td>Civic Offices, 2 Watling Street</td>
      <td>23.38</td>
      <td>236687</td>
      <td>51°27′18″N 0°09′02″E﻿ / ﻿51.4549°N 0.1505°E</td>
      <td>14973</td>
      <td>17244</td>
      <td>7346</td>
      <td>106</td>
      <td>1999</td>
      <td>2338</td>
      <td>NaN</td>
      <td>40071</td>
      <td>30037</td>
      <td>114136</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brent</td>
      <td>Brent London Borough Council</td>
      <td>Labour</td>
      <td>Brent Civic Centre, Engineers Way</td>
      <td>16.70</td>
      <td>317264</td>
      <td>51°33′32″N 0°16′54″W﻿ / ﻿51.5588°N 0.2817°W</td>
      <td>28923</td>
      <td>20569</td>
      <td>25978</td>
      <td>157</td>
      <td>3711</td>
      <td>12473</td>
      <td>NaN</td>
      <td>72523</td>
      <td>63178</td>
      <td>227551</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bromley</td>
      <td>Bromley London Borough Council</td>
      <td>Conservative</td>
      <td>Civic Centre, Stockwell Close</td>
      <td>57.97</td>
      <td>317899</td>
      <td>51°24′14″N 0°01′11″E﻿ / ﻿51.4039°N 0.0198°E</td>
      <td>27135</td>
      <td>24039</td>
      <td>8942</td>
      <td>196</td>
      <td>2637</td>
      <td>4868</td>
      <td>NaN</td>
      <td>69742</td>
      <td>46759</td>
      <td>184349</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Camden</td>
      <td>Camden London Borough Council</td>
      <td>Labour</td>
      <td>Camden Town Hall, Judd Street</td>
      <td>8.40</td>
      <td>229719</td>
      <td>51°31′44″N 0°07′32″W﻿ / ﻿51.5290°N 0.1255°W</td>
      <td>27939</td>
      <td>18482</td>
      <td>21816</td>
      <td>123</td>
      <td>3857</td>
      <td>9286</td>
      <td>NaN</td>
      <td>140596</td>
      <td>53012</td>
      <td>275147</td>
    </tr>
    <tr>
      <th>6</th>
      <td>City of London</td>
      <td>Corporation of London;Inner Temple;Middle Temple</td>
      <td>?</td>
      <td>Guildhall</td>
      <td>1.12</td>
      <td>7000</td>
      <td>51°30′56″N 0°05′32″W﻿ / ﻿51.5155°N 0.0922°W</td>
      <td>15</td>
      <td>16</td>
      <td>33</td>
      <td>0</td>
      <td>17</td>
      <td>24</td>
      <td>NaN</td>
      <td>561</td>
      <td>114</td>
      <td>780</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Croydon</td>
      <td>Croydon London Borough Council</td>
      <td>Labour</td>
      <td>Bernard Weatherill House, Mint Walk</td>
      <td>33.41</td>
      <td>372752</td>
      <td>51°22′17″N 0°05′52″W﻿ / ﻿51.3714°N 0.0977°W</td>
      <td>33376</td>
      <td>31218</td>
      <td>19162</td>
      <td>270</td>
      <td>4340</td>
      <td>12645</td>
      <td>NaN</td>
      <td>91437</td>
      <td>67791</td>
      <td>260294</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ealing</td>
      <td>Ealing London Borough Council</td>
      <td>Labour</td>
      <td>Perceval House, 14-16 Uxbridge Road</td>
      <td>21.44</td>
      <td>342494</td>
      <td>51°30′47″N 0°18′32″W﻿ / ﻿51.5130°N 0.3089°W</td>
      <td>30831</td>
      <td>25613</td>
      <td>18591</td>
      <td>175</td>
      <td>4406</td>
      <td>9568</td>
      <td>NaN</td>
      <td>93834</td>
      <td>68492</td>
      <td>251562</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Enfield</td>
      <td>Enfield London Borough Council</td>
      <td>Labour</td>
      <td>Civic Centre, Silver Street</td>
      <td>31.74</td>
      <td>320524</td>
      <td>51°39′14″N 0°04′48″W﻿ / ﻿51.6538°N 0.0799°W</td>
      <td>30213</td>
      <td>22487</td>
      <td>13251</td>
      <td>132</td>
      <td>3293</td>
      <td>9059</td>
      <td>NaN</td>
      <td>70371</td>
      <td>45036</td>
      <td>193880</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Greenwich</td>
      <td>Greenwich London Borough Council</td>
      <td>Labour</td>
      <td>Woolwich Town Hall, Wellington Street</td>
      <td>18.28</td>
      <td>264008</td>
      <td>51°29′21″N 0°03′53″E﻿ / ﻿51.4892°N 0.0648°E</td>
      <td>20966</td>
      <td>22755</td>
      <td>10836</td>
      <td>107</td>
      <td>3598</td>
      <td>5430</td>
      <td>NaN</td>
      <td>64923</td>
      <td>52897</td>
      <td>181568</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hackney</td>
      <td>Hackney London Borough Council</td>
      <td>Labour</td>
      <td>Hackney Town Hall, Mare Street</td>
      <td>7.36</td>
      <td>257379</td>
      <td>51°32′42″N 0°03′19″W﻿ / ﻿51.5450°N 0.0553°W</td>
      <td>21450</td>
      <td>17327</td>
      <td>18144</td>
      <td>143</td>
      <td>3332</td>
      <td>8975</td>
      <td>NaN</td>
      <td>91118</td>
      <td>56584</td>
      <td>217119</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Hammersmith and Fulham</td>
      <td>Hammersmith and Fulham London Borough Council</td>
      <td>Labour</td>
      <td>Town Hall, King Street</td>
      <td>6.33</td>
      <td>178685</td>
      <td>51°29′34″N 0°14′02″W﻿ / ﻿51.4927°N 0.2339°W</td>
      <td>17010</td>
      <td>14595</td>
      <td>15492</td>
      <td>91</td>
      <td>3352</td>
      <td>5279</td>
      <td>NaN</td>
      <td>86381</td>
      <td>43014</td>
      <td>185259</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Haringey</td>
      <td>Haringey London Borough Council</td>
      <td>Labour</td>
      <td>Civic Centre, High Road</td>
      <td>11.42</td>
      <td>263386</td>
      <td>51°36′00″N 0°06′43″W﻿ / ﻿51.6000°N 0.1119°W</td>
      <td>28213</td>
      <td>22272</td>
      <td>14563</td>
      <td>207</td>
      <td>2971</td>
      <td>10084</td>
      <td>NaN</td>
      <td>83979</td>
      <td>50943</td>
      <td>213272</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Harrow</td>
      <td>Harrow London Borough Council</td>
      <td>Labour</td>
      <td>Civic Centre, Station Road</td>
      <td>19.49</td>
      <td>243372</td>
      <td>51°35′23″N 0°20′05″W﻿ / ﻿51.5898°N 0.3346°W</td>
      <td>19630</td>
      <td>12724</td>
      <td>7122</td>
      <td>92</td>
      <td>1998</td>
      <td>4242</td>
      <td>NaN</td>
      <td>40800</td>
      <td>30213</td>
      <td>116848</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Havering</td>
      <td>Havering London Borough Council</td>
      <td>Conservative (council NOC)</td>
      <td>Town Hall, Main Road</td>
      <td>43.35</td>
      <td>242080</td>
      <td>51°34′52″N 0°11′01″E﻿ / ﻿51.5812°N 0.1837°E</td>
      <td>21302</td>
      <td>17252</td>
      <td>8171</td>
      <td>179</td>
      <td>2358</td>
      <td>3089</td>
      <td>NaN</td>
      <td>52609</td>
      <td>33968</td>
      <td>138947</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Hillingdon</td>
      <td>Hillingdon London Borough Council</td>
      <td>Conservative</td>
      <td>Civic Centre, High Street</td>
      <td>44.67</td>
      <td>286806</td>
      <td>51°32′39″N 0°28′34″W﻿ / ﻿51.5441°N 0.4760°W</td>
      <td>26056</td>
      <td>24485</td>
      <td>11413</td>
      <td>223</td>
      <td>6504</td>
      <td>5663</td>
      <td>NaN</td>
      <td>80028</td>
      <td>55264</td>
      <td>209680</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Hounslow</td>
      <td>Hounslow London Borough Council</td>
      <td>Labour</td>
      <td>Hounslow House, 7 Bath Road</td>
      <td>21.61</td>
      <td>262407</td>
      <td>51°28′29″N 0°22′05″W﻿ / ﻿51.4746°N 0.3680°W</td>
      <td>21026</td>
      <td>21407</td>
      <td>13722</td>
      <td>183</td>
      <td>3963</td>
      <td>4847</td>
      <td>NaN</td>
      <td>70180</td>
      <td>51404</td>
      <td>186772</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Islington</td>
      <td>Islington London Borough Council</td>
      <td>Labour</td>
      <td>Municipal Offices, 222 Upper Street</td>
      <td>5.74</td>
      <td>215667</td>
      <td>51°32′30″N 0°06′08″W﻿ / ﻿51.5416°N 0.1022°W</td>
      <td>22207</td>
      <td>18354</td>
      <td>16553</td>
      <td>85</td>
      <td>3675</td>
      <td>8736</td>
      <td>NaN</td>
      <td>107661</td>
      <td>52975</td>
      <td>230286</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Kensington and Chelsea</td>
      <td>Kensington and Chelsea London Borough Council</td>
      <td>Conservative</td>
      <td>The Town Hall, Hornton Street</td>
      <td>4.68</td>
      <td>155594</td>
      <td>51°30′07″N 0°11′41″W﻿ / ﻿51.5020°N 0.1947°W</td>
      <td>14980</td>
      <td>9839</td>
      <td>14573</td>
      <td>85</td>
      <td>2203</td>
      <td>4744</td>
      <td>NaN</td>
      <td>95963</td>
      <td>29570</td>
      <td>171981</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Kingston upon Thames</td>
      <td>Kingston upon Thames London Borough Council</td>
      <td>Liberal Democrat</td>
      <td>Guildhall, High Street</td>
      <td>14.38</td>
      <td>166793</td>
      <td>51°24′31″N 0°18′23″W﻿ / ﻿51.4085°N 0.3064°W</td>
      <td>10131</td>
      <td>10610</td>
      <td>5682</td>
      <td>65</td>
      <td>1332</td>
      <td>1702</td>
      <td>NaN</td>
      <td>38226</td>
      <td>21540</td>
      <td>89306</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Lambeth</td>
      <td>Lambeth London Borough Council</td>
      <td>Labour</td>
      <td>Lambeth Town Hall, Brixton Hill</td>
      <td>10.36</td>
      <td>314242</td>
      <td>51°27′39″N 0°06′59″W﻿ / ﻿51.4607°N 0.1163°W</td>
      <td>30199</td>
      <td>26136</td>
      <td>25083</td>
      <td>137</td>
      <td>4520</td>
      <td>18408</td>
      <td>NaN</td>
      <td>114899</td>
      <td>72726</td>
      <td>292178</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Lewisham</td>
      <td>Lewisham London Borough Council</td>
      <td>Labour</td>
      <td>Town Hall, 1 Catford Road</td>
      <td>13.57</td>
      <td>286180</td>
      <td>51°26′43″N 0°01′15″W﻿ / ﻿51.4452°N 0.0209°W</td>
      <td>24871</td>
      <td>24810</td>
      <td>16825</td>
      <td>262</td>
      <td>3809</td>
      <td>10455</td>
      <td>NaN</td>
      <td>70382</td>
      <td>63652</td>
      <td>215137</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Merton</td>
      <td>Merton London Borough Council</td>
      <td>Labour</td>
      <td>Civic Centre, London Road</td>
      <td>14.52</td>
      <td>203223</td>
      <td>51°24′05″N 0°11′45″W﻿ / ﻿51.4014°N 0.1958°W</td>
      <td>16485</td>
      <td>14339</td>
      <td>6651</td>
      <td>111</td>
      <td>1571</td>
      <td>4021</td>
      <td>NaN</td>
      <td>44128</td>
      <td>28322</td>
      <td>115654</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Newham</td>
      <td>Newham London Borough Council</td>
      <td>Labour</td>
      <td>Newham Dockside, 1000 Dockside Road</td>
      <td>13.98</td>
      <td>318227</td>
      <td>51°30′28″N 0°02′49″E﻿ / ﻿51.5077°N 0.0469°E</td>
      <td>25356</td>
      <td>24177</td>
      <td>18389</td>
      <td>323</td>
      <td>4456</td>
      <td>16913</td>
      <td>NaN</td>
      <td>106146</td>
      <td>66221</td>
      <td>262024</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Redbridge</td>
      <td>Redbridge London Borough Council</td>
      <td>Labour</td>
      <td>Town Hall, 128-142 High Road</td>
      <td>21.78</td>
      <td>288272</td>
      <td>51°33′32″N 0°04′27″E﻿ / ﻿51.5590°N 0.0741°E</td>
      <td>26735</td>
      <td>17543</td>
      <td>15736</td>
      <td>284</td>
      <td>2619</td>
      <td>7688</td>
      <td>NaN</td>
      <td>71496</td>
      <td>41430</td>
      <td>183562</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Richmond upon Thames</td>
      <td>Richmond upon Thames London Borough Council</td>
      <td>Liberal Democrat</td>
      <td>Civic Centre, 44 York Street</td>
      <td>22.17</td>
      <td>191365</td>
      <td>51°26′52″N 0°19′34″W﻿ / ﻿51.4479°N 0.3260°W</td>
      <td>16097</td>
      <td>11722</td>
      <td>4707</td>
      <td>37</td>
      <td>1420</td>
      <td>1590</td>
      <td>NaN</td>
      <td>40858</td>
      <td>20314</td>
      <td>96771</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Southwark</td>
      <td>Southwark London Borough Council</td>
      <td>Labour</td>
      <td>160 Tooley Street</td>
      <td>11.14</td>
      <td>298464</td>
      <td>51°30′13″N 0°04′49″W﻿ / ﻿51.5035°N 0.0804°W</td>
      <td>27980</td>
      <td>24450</td>
      <td>27381</td>
      <td>321</td>
      <td>4696</td>
      <td>16153</td>
      <td>NaN</td>
      <td>109432</td>
      <td>68356</td>
      <td>278809</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Sutton</td>
      <td>Sutton London Borough Council</td>
      <td>Liberal Democrat</td>
      <td>Civic Offices, St Nicholas Way</td>
      <td>16.93</td>
      <td>195914</td>
      <td>51°21′42″N 0°11′40″W﻿ / ﻿51.3618°N 0.1945°W</td>
      <td>13207</td>
      <td>14474</td>
      <td>4586</td>
      <td>57</td>
      <td>1393</td>
      <td>2308</td>
      <td>NaN</td>
      <td>39533</td>
      <td>25409</td>
      <td>100987</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Tower Hamlets</td>
      <td>Tower Hamlets London Borough Council</td>
      <td>Labour</td>
      <td>Town Hall, Mulberry Place, 5 Clove Crescent</td>
      <td>7.63</td>
      <td>272890</td>
      <td>51°30′36″N 0°00′21″W﻿ / ﻿51.5099°N 0.0059°W</td>
      <td>21510</td>
      <td>21593</td>
      <td>23408</td>
      <td>124</td>
      <td>4268</td>
      <td>10050</td>
      <td>NaN</td>
      <td>87620</td>
      <td>59993</td>
      <td>228613</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Waltham Forest</td>
      <td>Waltham Forest London Borough Council</td>
      <td>Labour</td>
      <td>Waltham Forest Town Hall, Forest Road</td>
      <td>14.99</td>
      <td>265797</td>
      <td>51°35′27″N 0°00′48″W﻿ / ﻿51.5908°N 0.0134°W</td>
      <td>25565</td>
      <td>20459</td>
      <td>14101</td>
      <td>236</td>
      <td>3040</td>
      <td>10606</td>
      <td>NaN</td>
      <td>77940</td>
      <td>51898</td>
      <td>203879</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Wandsworth</td>
      <td>Wandsworth London Borough Council</td>
      <td>Conservative</td>
      <td>The Town Hall, Wandsworth High Street</td>
      <td>13.23</td>
      <td>310516</td>
      <td>51°27′24″N 0°11′28″W﻿ / ﻿51.4567°N 0.1910°W</td>
      <td>25533</td>
      <td>19630</td>
      <td>9493</td>
      <td>161</td>
      <td>3091</td>
      <td>8398</td>
      <td>NaN</td>
      <td>92523</td>
      <td>45865</td>
      <td>204741</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Westminster</td>
      <td>Westminster City Council</td>
      <td>Conservative</td>
      <td>Westminster City Hall, 64 Victoria Street</td>
      <td>8.29</td>
      <td>226841</td>
      <td>51°29′50″N 0°08′14″W﻿ / ﻿51.4973°N 0.1372°W</td>
      <td>29295</td>
      <td>20405</td>
      <td>34031</td>
      <td>273</td>
      <td>6148</td>
      <td>15752</td>
      <td>NaN</td>
      <td>277617</td>
      <td>71448</td>
      <td>455028</td>
    </tr>
  </tbody>
</table>
</div>




```python
# shape of the dataframe
Ld_crime.shape
```




    (33, 17)



### Exploratory Data Analysis


```python
# descriptive statistics of the data
London_crime.describe()
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
      <th>Burglary</th>
      <th>Criminal Damage</th>
      <th>Drugs</th>
      <th>Fraud or Forgery</th>
      <th>Other Notifiable Offenses</th>
      <th>Robbery</th>
      <th>Sexual Offences</th>
      <th>Theft and Handling</th>
      <th>Violence Against the Person</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>22857.363636</td>
      <td>19119.333333</td>
      <td>14265.606061</td>
      <td>161.363636</td>
      <td>3222.696970</td>
      <td>7844.636364</td>
      <td>38.575758</td>
      <td>80662.454545</td>
      <td>47214.575758</td>
      <td>195386.606061</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7452.366846</td>
      <td>5942.903618</td>
      <td>7544.259564</td>
      <td>81.603775</td>
      <td>1362.107294</td>
      <td>4677.643075</td>
      <td>15.139002</td>
      <td>45155.624776</td>
      <td>17226.165191</td>
      <td>79148.057551</td>
    </tr>
    <tr>
      <th>min</th>
      <td>15.000000</td>
      <td>16.000000</td>
      <td>33.000000</td>
      <td>0.000000</td>
      <td>17.000000</td>
      <td>24.000000</td>
      <td>0.000000</td>
      <td>561.000000</td>
      <td>114.000000</td>
      <td>780.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>18103.000000</td>
      <td>17244.000000</td>
      <td>8942.000000</td>
      <td>106.000000</td>
      <td>2358.000000</td>
      <td>4744.000000</td>
      <td>27.000000</td>
      <td>52609.000000</td>
      <td>33968.000000</td>
      <td>149447.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>24871.000000</td>
      <td>20405.000000</td>
      <td>14101.000000</td>
      <td>157.000000</td>
      <td>3293.000000</td>
      <td>7688.000000</td>
      <td>40.000000</td>
      <td>77940.000000</td>
      <td>50943.000000</td>
      <td>203879.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>27980.000000</td>
      <td>22755.000000</td>
      <td>18389.000000</td>
      <td>207.000000</td>
      <td>3963.000000</td>
      <td>10084.000000</td>
      <td>47.000000</td>
      <td>92523.000000</td>
      <td>59993.000000</td>
      <td>228613.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>36981.000000</td>
      <td>31218.000000</td>
      <td>34031.000000</td>
      <td>323.000000</td>
      <td>6504.000000</td>
      <td>18408.000000</td>
      <td>71.000000</td>
      <td>277617.000000</td>
      <td>72726.000000</td>
      <td>455028.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# import libraries for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
mpl.style.use('ggplot')
```

Check if the column names are strings.


```python
Ld_crime.columns = list(map(str, Ld_crime.columns))

# check the column labels type 
all(isinstance(column, str) for column in Ld_crime.columns)
```




    True



Let's sort the total crimes in descending order to see 5 boroughs with the highest number of crimes.


```python
Ld_crime.sort_values(['Total'], ascending=False, axis=0, inplace=True)

df_top5 = Ld_crime.head()
df_top5
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
      <th>Borough</th>
      <th>Local authority</th>
      <th>Political control</th>
      <th>Headquarters</th>
      <th>Area (sq mi)</th>
      <th>Population</th>
      <th>Co-ordinates</th>
      <th>Burglary</th>
      <th>Criminal Damage</th>
      <th>Drugs</th>
      <th>Fraud or Forgery</th>
      <th>Other Notifiable Offenses</th>
      <th>Robbery</th>
      <th>Sexual Offenses</th>
      <th>Theft and Handling</th>
      <th>Violence Against the Person</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32</th>
      <td>Westminster</td>
      <td>Westminster City Council</td>
      <td>Conservative</td>
      <td>Westminster City Hall, 64 Victoria Street</td>
      <td>8.29</td>
      <td>226841</td>
      <td>51°29′50″N 0°08′14″W﻿ / ﻿51.4973°N 0.1372°W</td>
      <td>29295</td>
      <td>20405</td>
      <td>34031</td>
      <td>273</td>
      <td>6148</td>
      <td>15752</td>
      <td>NaN</td>
      <td>277617</td>
      <td>71448</td>
      <td>455028</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Lambeth</td>
      <td>Lambeth London Borough Council</td>
      <td>Labour</td>
      <td>Lambeth Town Hall, Brixton Hill</td>
      <td>10.36</td>
      <td>314242</td>
      <td>51°27′39″N 0°06′59″W﻿ / ﻿51.4607°N 0.1163°W</td>
      <td>30199</td>
      <td>26136</td>
      <td>25083</td>
      <td>137</td>
      <td>4520</td>
      <td>18408</td>
      <td>NaN</td>
      <td>114899</td>
      <td>72726</td>
      <td>292178</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Southwark</td>
      <td>Southwark London Borough Council</td>
      <td>Labour</td>
      <td>160 Tooley Street</td>
      <td>11.14</td>
      <td>298464</td>
      <td>51°30′13″N 0°04′49″W﻿ / ﻿51.5035°N 0.0804°W</td>
      <td>27980</td>
      <td>24450</td>
      <td>27381</td>
      <td>321</td>
      <td>4696</td>
      <td>16153</td>
      <td>NaN</td>
      <td>109432</td>
      <td>68356</td>
      <td>278809</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Camden</td>
      <td>Camden London Borough Council</td>
      <td>Labour</td>
      <td>Camden Town Hall, Judd Street</td>
      <td>8.40</td>
      <td>229719</td>
      <td>51°31′44″N 0°07′32″W﻿ / ﻿51.5290°N 0.1255°W</td>
      <td>27939</td>
      <td>18482</td>
      <td>21816</td>
      <td>123</td>
      <td>3857</td>
      <td>9286</td>
      <td>NaN</td>
      <td>140596</td>
      <td>53012</td>
      <td>275147</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Newham</td>
      <td>Newham London Borough Council</td>
      <td>Labour</td>
      <td>Newham Dockside, 1000 Dockside Road</td>
      <td>13.98</td>
      <td>318227</td>
      <td>51°30′28″N 0°02′49″E﻿ / ﻿51.5077°N 0.0469°E</td>
      <td>25356</td>
      <td>24177</td>
      <td>18389</td>
      <td>323</td>
      <td>4456</td>
      <td>16913</td>
      <td>NaN</td>
      <td>106146</td>
      <td>66221</td>
      <td>262024</td>
    </tr>
  </tbody>
</table>
</div>



Let's visualize these 5 boroughs.


```python
df_tt = df_top5[['Borough','Total']]

df_tt.set_index('Borough',inplace = True)

ax = df_tt.plot(kind='bar', figsize=(10, 6), rot=0)

ax.set_ylabel('Number of Crimes')
ax.set_xlabel('Borough')
ax.set_title('London Boroughs with the Highest no. of crime')

# create a function to display the percentage.
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize = 14
               )

plt.show()
```


![png](output_65_0.png)


Okay. Now we know which places you need to stay away from.  

Now, let's sort the total crimes in ascending order to see 5 boroughs with the lowest number of crimes.


```python
Ld_crime.sort_values(['Total'], ascending=True, axis=0, inplace=True)

df_bot5 = Ld_crime.head()
df_bot5
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
      <th>Borough</th>
      <th>Local authority</th>
      <th>Political control</th>
      <th>Headquarters</th>
      <th>Area (sq mi)</th>
      <th>Population</th>
      <th>Co-ordinates</th>
      <th>Burglary</th>
      <th>Criminal Damage</th>
      <th>Drugs</th>
      <th>Fraud or Forgery</th>
      <th>Other Notifiable Offenses</th>
      <th>Robbery</th>
      <th>Sexual Offenses</th>
      <th>Theft and Handling</th>
      <th>Violence Against the Person</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>City of London</td>
      <td>Corporation of London;Inner Temple;Middle Temple</td>
      <td>?</td>
      <td>Guildhall</td>
      <td>1.12</td>
      <td>7000</td>
      <td>51°30′56″N 0°05′32″W﻿ / ﻿51.5155°N 0.0922°W</td>
      <td>15</td>
      <td>16</td>
      <td>33</td>
      <td>0</td>
      <td>17</td>
      <td>24</td>
      <td>NaN</td>
      <td>561</td>
      <td>114</td>
      <td>780</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Kingston upon Thames</td>
      <td>Kingston upon Thames London Borough Council</td>
      <td>Liberal Democrat</td>
      <td>Guildhall, High Street</td>
      <td>14.38</td>
      <td>166793</td>
      <td>51°24′31″N 0°18′23″W﻿ / ﻿51.4085°N 0.3064°W</td>
      <td>10131</td>
      <td>10610</td>
      <td>5682</td>
      <td>65</td>
      <td>1332</td>
      <td>1702</td>
      <td>NaN</td>
      <td>38226</td>
      <td>21540</td>
      <td>89306</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Richmond upon Thames</td>
      <td>Richmond upon Thames London Borough Council</td>
      <td>Liberal Democrat</td>
      <td>Civic Centre, 44 York Street</td>
      <td>22.17</td>
      <td>191365</td>
      <td>51°26′52″N 0°19′34″W﻿ / ﻿51.4479°N 0.3260°W</td>
      <td>16097</td>
      <td>11722</td>
      <td>4707</td>
      <td>37</td>
      <td>1420</td>
      <td>1590</td>
      <td>NaN</td>
      <td>40858</td>
      <td>20314</td>
      <td>96771</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Sutton</td>
      <td>Sutton London Borough Council</td>
      <td>Liberal Democrat</td>
      <td>Civic Offices, St Nicholas Way</td>
      <td>16.93</td>
      <td>195914</td>
      <td>51°21′42″N 0°11′40″W﻿ / ﻿51.3618°N 0.1945°W</td>
      <td>13207</td>
      <td>14474</td>
      <td>4586</td>
      <td>57</td>
      <td>1393</td>
      <td>2308</td>
      <td>NaN</td>
      <td>39533</td>
      <td>25409</td>
      <td>100987</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bexley</td>
      <td>Bexley London Borough Council</td>
      <td>Conservative</td>
      <td>Civic Offices, 2 Watling Street</td>
      <td>23.38</td>
      <td>236687</td>
      <td>51°27′18″N 0°09′02″E﻿ / ﻿51.4549°N 0.1505°E</td>
      <td>14973</td>
      <td>17244</td>
      <td>7346</td>
      <td>106</td>
      <td>1999</td>
      <td>2338</td>
      <td>NaN</td>
      <td>40071</td>
      <td>30037</td>
      <td>114136</td>
    </tr>
  </tbody>
</table>
</div>



Let's visualize these 5 boroughs.


```python
df_bt = df_bot5[['Borough','Total']]

df_bt.set_index('Borough',inplace = True)

ax = df_bt.plot(kind='bar', figsize=(10, 6), rot=0)

ax.set_ylabel('Number of Crimes') 
ax.set_xlabel('Borough') 
ax.set_title('London Boroughs with the least no. of crime')

# create a function to display the percentage.
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize = 14
               )

plt.show()
```


![png](output_69_0.png)


The borough **City of London** has the lowest crime recorded over the years. Let's look into its details.


```python
df_col = df_bot5[df_bot5['Borough'] == 'City of London']
df_col = df_col[['Borough','Total','Area (sq mi)','Population']]
df_col
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
      <th>Borough</th>
      <th>Total</th>
      <th>Area (sq mi)</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>City of London</td>
      <td>780</td>
      <td>1.12</td>
      <td>7000</td>
    </tr>
  </tbody>
</table>
</div>



According to the London Boroughs [Wikipedia page](https://en.wikipedia.org/wiki/List_of_London_boroughs), the City of London is the 33rd principal division of Greater London, but it is not a London borough. You also realise that living in this area would be very expensive and you're not looking to spend most of your income on rent.  
So let's focus on the next safest borough i.e. **Kingston upon Thames**, just to keep our options open.

Visualize different types of crimes in the borough 'Kingston upon Thames'.


```python
df_bc1 = df_bot5[df_bot5['Borough'] == 'Kingston upon Thames']

df_bc = df_bc1[['Borough', 'Burglary', 'Criminal Damage', 'Drugs', 'Fraud or Forgery', 'Other Notifiable Offenses', 
                'Robbery', 'Sexual Offenses', 'Theft and Handling', 'Violence Against the Person']]

df_bc.set_index('Borough', inplace=True)

ax = df_bc.plot(kind='bar', figsize=(10, 6), rot=0)

ax.set_ylabel('Number of Crimes') 
ax.set_xlabel('Borough') 
ax.set_title('Crimes in Kingston upon Thames')

# create a function to display the percentage.
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize = 14
               )

plt.show()
```


![png](output_73_0.png)


This borough is a great option for you to live in and is also extremely safe compared to the other boroughs.

### Dataset of the Neighborhood
The list of Neighborhoods in the Royal Borough of Kingston upon Thames can be [found here](https://en.wikipedia.org/wiki/List_of_districts_in_the_Royal_Borough_of_Kingston_upon_Thames).


```python
Neighborhood = ['Berrylands','Canbury','Chessington','Coombe','Kingston upon Thames','Kingston Vale',
                'Malden Rushett','Motspur Park','New Malden','Norbiton','Old Malden','Surbiton','Tolworth']

Borough = ['Kingston upon Thames','Kingston upon Thames','Kingston upon Thames','Kingston upon Thames','Kingston upon Thames',
          'Kingston upon Thames','Kingston upon Thames','Kingston upon Thames','Kingston upon Thames','Kingston upon Thames',
          'Kingston upon Thames','Kingston upon Thames','Kingston upon Thames']

Latitude = ['','','','','','','','','','','','','']
Longitude = ['','','','','','','','','','','','','']

df_neigh = {'Neighborhood':Neighborhood, 'Borough':Borough, 'Latitude':Latitude,  'Longitude':Longitude}
kut_neigh = pd.DataFrame(data=df_neigh, columns=['Neighborhood', 'Borough', 'Latitude', 'Longitude'], index=None)

kut_neigh
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
      <th>Neighborhood</th>
      <th>Borough</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Berrylands</td>
      <td>Kingston upon Thames</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>Canbury</td>
      <td>Kingston upon Thames</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chessington</td>
      <td>Kingston upon Thames</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>Coombe</td>
      <td>Kingston upon Thames</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kingston upon Thames</td>
      <td>Kingston upon Thames</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>5</th>
      <td>Kingston Vale</td>
      <td>Kingston upon Thames</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>6</th>
      <td>Malden Rushett</td>
      <td>Kingston upon Thames</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>Motspur Park</td>
      <td>Kingston upon Thames</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td>New Malden</td>
      <td>Kingston upon Thames</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>9</th>
      <td>Norbiton</td>
      <td>Kingston upon Thames</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>10</th>
      <td>Old Malden</td>
      <td>Kingston upon Thames</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>11</th>
      <td>Surbiton</td>
      <td>Kingston upon Thames</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>12</th>
      <td>Tolworth</td>
      <td>Kingston upon Thames</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



Find the co-ordinates of each neighborhood in the Kingston upon Thames borough.


```python
Latitude = []
Longitude = []

for i in range(len(Neighborhood)):
    address = '{}, London, United Kingdom'.format(Neighborhood[i])
    geolocator = Nominatim(user_agent='London_agent')
    location = geolocator.geocode(address)
    Latitude.append(location.latitude)
    Longitude.append(location.longitude)

print(Latitude, Longitude)
```

    [51.3937811, 51.41749865, 51.358336, 51.4194499, 51.4096275, 51.43185, 51.3410523, 51.3909852, 51.4053347, 51.4099994, 51.382484, 51.3937557, 51.3788758] [-0.2848024, -0.30555280504926163, -0.2986216, -0.2653985, -0.3062621, -0.2581379, -0.3190757, -0.2488979, -0.2634066, -0.2873963, -0.2590897, -0.3033105, -0.2828604]
    


```python
df_neigh = {'Neighborhood':Neighborhood, 'Borough':Borough, 'Latitude':Latitude,  'Longitude':Longitude}
kut_neigh = pd.DataFrame(data=df_neigh, columns=['Neighborhood', 'Borough', 'Latitude', 'Longitude'], index=None)

kut_neigh
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
      <th>Neighborhood</th>
      <th>Borough</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Berrylands</td>
      <td>Kingston upon Thames</td>
      <td>51.393781</td>
      <td>-0.284802</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Canbury</td>
      <td>Kingston upon Thames</td>
      <td>51.417499</td>
      <td>-0.305553</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chessington</td>
      <td>Kingston upon Thames</td>
      <td>51.358336</td>
      <td>-0.298622</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Coombe</td>
      <td>Kingston upon Thames</td>
      <td>51.419450</td>
      <td>-0.265398</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kingston upon Thames</td>
      <td>Kingston upon Thames</td>
      <td>51.409627</td>
      <td>-0.306262</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Kingston Vale</td>
      <td>Kingston upon Thames</td>
      <td>51.431850</td>
      <td>-0.258138</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Malden Rushett</td>
      <td>Kingston upon Thames</td>
      <td>51.341052</td>
      <td>-0.319076</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Motspur Park</td>
      <td>Kingston upon Thames</td>
      <td>51.390985</td>
      <td>-0.248898</td>
    </tr>
    <tr>
      <th>8</th>
      <td>New Malden</td>
      <td>Kingston upon Thames</td>
      <td>51.405335</td>
      <td>-0.263407</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Norbiton</td>
      <td>Kingston upon Thames</td>
      <td>51.409999</td>
      <td>-0.287396</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Old Malden</td>
      <td>Kingston upon Thames</td>
      <td>51.382484</td>
      <td>-0.259090</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Surbiton</td>
      <td>Kingston upon Thames</td>
      <td>51.393756</td>
      <td>-0.303310</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Tolworth</td>
      <td>Kingston upon Thames</td>
      <td>51.378876</td>
      <td>-0.282860</td>
    </tr>
  </tbody>
</table>
</div>



Let's get the co-ordinates of Berrylands, which is the center neighborhood of the Kingston upon Thames borough.


```python
address = 'Berrylands, London, United Kingdom'

geolocator = Nominatim(user_agent='ld_explorer')
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geographical co-ordinates of Berrylands, London are {}, {}.'.format(latitude, longitude))
```

    The geographical co-ordinates of Berrylands, London are 51.3937811, -0.2848024.
    

Let's visualize the neighborhood of Kingston upon Thames borough.


```python
# create map of London using latitude and longitude values
map_lon = folium.Map(location=[latitude, longitude], zoom_start=12)

# add markers to map
for lat, lng, borough, neighborhood in zip(kut_neigh['Latitude'], kut_neigh['Longitude'], 
                                           kut_neigh['Borough'], kut_neigh['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker([lat, lng], radius=5, popup=label, color='blue', fill=True, 
                        fill_color='#3186cc', fill_opacity=0.7, parse_html=False).add_to(map_lon)
    
map_lon
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfM2RmYWI0OTQ0ZWM5NDNiNWFhZGQ3OWI2YTdlOGY5MjEgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzNkZmFiNDk0NGVjOTQzYjVhYWRkNzliNmE3ZThmOTIxIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF8zZGZhYjQ5NDRlYzk0M2I1YWFkZDc5YjZhN2U4ZjkyMSA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF8zZGZhYjQ5NDRlYzk0M2I1YWFkZDc5YjZhN2U4ZjkyMScsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNTEuMzkzNzgxMSwtMC4yODQ4MDI0XSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHpvb206IDEyLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4Qm91bmRzOiBib3VuZHMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBsYXllcnM6IFtdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgd29ybGRDb3B5SnVtcDogZmFsc2UsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjcnM6IEwuQ1JTLkVQU0czODU3CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pOwogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgdGlsZV9sYXllcl9jOTA3ODNhNzcxODk0ZDMwYTRiZjk1NWVmNzZjZjhmNyA9IEwudGlsZUxheWVyKAogICAgICAgICAgICAgICAgJ2h0dHBzOi8ve3N9LnRpbGUub3BlbnN0cmVldG1hcC5vcmcve3p9L3t4fS97eX0ucG5nJywKICAgICAgICAgICAgICAgIHsKICAiYXR0cmlidXRpb24iOiBudWxsLAogICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwKICAibWF4Wm9vbSI6IDE4LAogICJtaW5ab29tIjogMSwKICAibm9XcmFwIjogZmFsc2UsCiAgInN1YmRvbWFpbnMiOiAiYWJjIgp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZGZhYjQ5NDRlYzk0M2I1YWFkZDc5YjZhN2U4ZjkyMSk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWYyZWE2ZmVmM2VmNGU5ZmFhMzYxMTdmOTM3Y2ZmMjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS4zOTM3ODExLC0wLjI4NDgwMjRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2RmYWI0OTQ0ZWM5NDNiNWFhZGQ3OWI2YTdlOGY5MjEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWQ4ZmVmODkwY2UxNDhkNDg4MWQ1MmJkNDZjN2VjMmYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDA5YjQzYjkwNDM5NGQ2NGFlZWIxOTMyODkyYzkwY2MgPSAkKCc8ZGl2IGlkPSJodG1sXzAwOWI0M2I5MDQzOTRkNjRhZWViMTkzMjg5MmM5MGNjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CZXJyeWxhbmRzLCBLaW5nc3RvbiB1cG9uIFRoYW1lczwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZWQ4ZmVmODkwY2UxNDhkNDg4MWQ1MmJkNDZjN2VjMmYuc2V0Q29udGVudChodG1sXzAwOWI0M2I5MDQzOTRkNjRhZWViMTkzMjg5MmM5MGNjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2VmMmVhNmZlZjNlZjRlOWZhYTM2MTE3ZjkzN2NmZjI0LmJpbmRQb3B1cChwb3B1cF9lZDhmZWY4OTBjZTE0OGQ0ODgxZDUyYmQ0NmM3ZWMyZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82YzZkODZhMTQ0YzA0MDU3OGQ3OTRiMzFiNTcyOTgxYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQxNzQ5ODY1LC0wLjMwNTU1MjgwNTA0OTI2MTYzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNkZmFiNDk0NGVjOTQzYjVhYWRkNzliNmE3ZThmOTIxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E2Yzk4NTBkMmQyZjRmNzZhZTVkNGRlNmFhMmQ1YzhkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRhNDMyYmI0ZTE2NTQ2NzM5NjY1YTkxZTYxZTRmYTI4ID0gJCgnPGRpdiBpZD0iaHRtbF80YTQzMmJiNGUxNjU0NjczOTY2NWE5MWU2MWU0ZmEyOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2FuYnVyeSwgS2luZ3N0b24gdXBvbiBUaGFtZXM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E2Yzk4NTBkMmQyZjRmNzZhZTVkNGRlNmFhMmQ1YzhkLnNldENvbnRlbnQoaHRtbF80YTQzMmJiNGUxNjU0NjczOTY2NWE5MWU2MWU0ZmEyOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82YzZkODZhMTQ0YzA0MDU3OGQ3OTRiMzFiNTcyOTgxYy5iaW5kUG9wdXAocG9wdXBfYTZjOTg1MGQyZDJmNGY3NmFlNWQ0ZGU2YWEyZDVjOGQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjk2NjBhZTU5YjJkNDkzM2FiZGI1NjliNWFjZGFhNmEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS4zNTgzMzYsLTAuMjk4NjIxNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZGZhYjQ5NDRlYzk0M2I1YWFkZDc5YjZhN2U4ZjkyMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83MjFjYzFlZmI2ZDQ0MDBiOGVmMDk2MGY1YzExZjQxYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84ZjBjYTc1NDEzYTQ0YmZlYWRlYzhmOWNkMzcyZDZmZiA9ICQoJzxkaXYgaWQ9Imh0bWxfOGYwY2E3NTQxM2E0NGJmZWFkZWM4ZjljZDM3MmQ2ZmYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNoZXNzaW5ndG9uLCBLaW5nc3RvbiB1cG9uIFRoYW1lczwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzIxY2MxZWZiNmQ0NDAwYjhlZjA5NjBmNWMxMWY0MWMuc2V0Q29udGVudChodG1sXzhmMGNhNzU0MTNhNDRiZmVhZGVjOGY5Y2QzNzJkNmZmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzI5NjYwYWU1OWIyZDQ5MzNhYmRiNTY5YjVhY2RhYTZhLmJpbmRQb3B1cChwb3B1cF83MjFjYzFlZmI2ZDQ0MDBiOGVmMDk2MGY1YzExZjQxYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lNDQxMDJlMGY4MmY0MGU4YTVlZTY4MzZmYWVlZmM0NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQxOTQ0OTksLTAuMjY1Mzk4NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZGZhYjQ5NDRlYzk0M2I1YWFkZDc5YjZhN2U4ZjkyMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83M2IyOWJkNjY1ZTQ0NjZhOTI1MTIwYWIzNDU5ZDAwMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81ZGRlZWZhMzA0MTY0YTQ5OTkzODA4Y2QyYmUwMzBlOSA9ICQoJzxkaXYgaWQ9Imh0bWxfNWRkZWVmYTMwNDE2NGE0OTk5MzgwOGNkMmJlMDMwZTkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNvb21iZSwgS2luZ3N0b24gdXBvbiBUaGFtZXM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzczYjI5YmQ2NjVlNDQ2NmE5MjUxMjBhYjM0NTlkMDAzLnNldENvbnRlbnQoaHRtbF81ZGRlZWZhMzA0MTY0YTQ5OTkzODA4Y2QyYmUwMzBlOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lNDQxMDJlMGY4MmY0MGU4YTVlZTY4MzZmYWVlZmM0Ni5iaW5kUG9wdXAocG9wdXBfNzNiMjliZDY2NWU0NDY2YTkyNTEyMGFiMzQ1OWQwMDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODcxNzdhNGQ4ODJlNGVkZDg5OThkYzk5YzhmOGY3ZGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40MDk2Mjc1LC0wLjMwNjI2MjFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2RmYWI0OTQ0ZWM5NDNiNWFhZGQ3OWI2YTdlOGY5MjEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfM2FjOTJkZDRmMmRmNDRiYWJmZjU1YTA0MDEyMmQ1YTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzYwOWUwMGQ0ZDVmNDQxM2E1MDllYTI5Yjk2ODU4ZGIgPSAkKCc8ZGl2IGlkPSJodG1sXzc2MDllMDBkNGQ1ZjQ0MTNhNTA5ZWEyOWI5Njg1OGRiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LaW5nc3RvbiB1cG9uIFRoYW1lcywgS2luZ3N0b24gdXBvbiBUaGFtZXM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzNhYzkyZGQ0ZjJkZjQ0YmFiZmY1NWEwNDAxMjJkNWE4LnNldENvbnRlbnQoaHRtbF83NjA5ZTAwZDRkNWY0NDEzYTUwOWVhMjliOTY4NThkYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84NzE3N2E0ZDg4MmU0ZWRkODk5OGRjOTljOGY4ZjdkZS5iaW5kUG9wdXAocG9wdXBfM2FjOTJkZDRmMmRmNDRiYWJmZjU1YTA0MDEyMmQ1YTgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzQwYWYwZDcxMjNjNDdlZjk5NTMxODIyOWEwNTQwY2MgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40MzE4NSwtMC4yNTgxMzc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNkZmFiNDk0NGVjOTQzYjVhYWRkNzliNmE3ZThmOTIxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y5NjdlYTE0ZTAwYjRiZmNiY2M3NzBjMzQ4OWVmOWUzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzliNjYzZmQwMzM5NTQ3YTRhZGZmZWMwNzdhN2M3OTQxID0gJCgnPGRpdiBpZD0iaHRtbF85YjY2M2ZkMDMzOTU0N2E0YWRmZmVjMDc3YTdjNzk0MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2luZ3N0b24gVmFsZSwgS2luZ3N0b24gdXBvbiBUaGFtZXM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y5NjdlYTE0ZTAwYjRiZmNiY2M3NzBjMzQ4OWVmOWUzLnNldENvbnRlbnQoaHRtbF85YjY2M2ZkMDMzOTU0N2E0YWRmZmVjMDc3YTdjNzk0MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83NDBhZjBkNzEyM2M0N2VmOTk1MzE4MjI5YTA1NDBjYy5iaW5kUG9wdXAocG9wdXBfZjk2N2VhMTRlMDBiNGJmY2JjYzc3MGMzNDg5ZWY5ZTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGMwNjA3ZmQ4MzQ1NDM3NGEyYjdjZTI1NzYyZWIzY2UgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS4zNDEwNTIzLC0wLjMxOTA3NTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2RmYWI0OTQ0ZWM5NDNiNWFhZGQ3OWI2YTdlOGY5MjEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWVjMzAwNGMxZWEyNDBjODliMDRhN2EzZThlNzdlYTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDZmNWM1M2Q2ODFmNDk1MWExNTlhZmMwOGUxODcyMTAgPSAkKCc8ZGl2IGlkPSJodG1sXzA2ZjVjNTNkNjgxZjQ5NTFhMTU5YWZjMDhlMTg3MjEwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NYWxkZW4gUnVzaGV0dCwgS2luZ3N0b24gdXBvbiBUaGFtZXM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFlYzMwMDRjMWVhMjQwYzg5YjA0YTdhM2U4ZTc3ZWE4LnNldENvbnRlbnQoaHRtbF8wNmY1YzUzZDY4MWY0OTUxYTE1OWFmYzA4ZTE4NzIxMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kYzA2MDdmZDgzNDU0Mzc0YTJiN2NlMjU3NjJlYjNjZS5iaW5kUG9wdXAocG9wdXBfMWVjMzAwNGMxZWEyNDBjODliMDRhN2EzZThlNzdlYTgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjkxMjYyODQ5YTc0NGU5NDkyMjJjOWExNGQwNmM4NzEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS4zOTA5ODUyLC0wLjI0ODg5NzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2RmYWI0OTQ0ZWM5NDNiNWFhZGQ3OWI2YTdlOGY5MjEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWY2N2IzYjM2ZjllNDZlNjhhZDMyZDFkNmM1NWZkMzcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDg1MWFiNDM4MzE3NDVlZmJkNTYzNmIwYmVjYjUyYzYgPSAkKCc8ZGl2IGlkPSJodG1sXzA4NTFhYjQzODMxNzQ1ZWZiZDU2MzZiMGJlY2I1MmM2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Nb3RzcHVyIFBhcmssIEtpbmdzdG9uIHVwb24gVGhhbWVzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lZjY3YjNiMzZmOWU0NmU2OGFkMzJkMWQ2YzU1ZmQzNy5zZXRDb250ZW50KGh0bWxfMDg1MWFiNDM4MzE3NDVlZmJkNTYzNmIwYmVjYjUyYzYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjkxMjYyODQ5YTc0NGU5NDkyMjJjOWExNGQwNmM4NzEuYmluZFBvcHVwKHBvcHVwX2VmNjdiM2IzNmY5ZTQ2ZTY4YWQzMmQxZDZjNTVmZDM3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzA2MDlhMzRjNmNiMDRiOGRiYzE3YzY2OWMxZjNkZjQ0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDA1MzM0NywtMC4yNjM0MDY2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNkZmFiNDk0NGVjOTQzYjVhYWRkNzliNmE3ZThmOTIxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNjY2Y5ZGVmNWY0ZDRlYzU4OGI0ZTM2YzUxOTEzNzEzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2EyNGNhNTBiODQxZDRiZjc5MzQxYjkzZjgyODNmZjA0ID0gJCgnPGRpdiBpZD0iaHRtbF9hMjRjYTUwYjg0MWQ0YmY3OTM0MWI5M2Y4MjgzZmYwNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TmV3IE1hbGRlbiwgS2luZ3N0b24gdXBvbiBUaGFtZXM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzNjY2Y5ZGVmNWY0ZDRlYzU4OGI0ZTM2YzUxOTEzNzEzLnNldENvbnRlbnQoaHRtbF9hMjRjYTUwYjg0MWQ0YmY3OTM0MWI5M2Y4MjgzZmYwNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wNjA5YTM0YzZjYjA0YjhkYmMxN2M2NjljMWYzZGY0NC5iaW5kUG9wdXAocG9wdXBfM2NjZjlkZWY1ZjRkNGVjNTg4YjRlMzZjNTE5MTM3MTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjZkZmY2YmYzYzIzNDZhNGJhYzViYzA1ZDk2ZmNhNGMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40MDk5OTk0LC0wLjI4NzM5NjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2RmYWI0OTQ0ZWM5NDNiNWFhZGQ3OWI2YTdlOGY5MjEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzEzY2E0YmY5ZDUwNGE1NDgwMmQwZGVhMjgxZDYwYjcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWQ0ZWZiOWRjYmFlNGNiZTk0ZjcwMWIxMTJmMzFiOTEgPSAkKCc8ZGl2IGlkPSJodG1sXzVkNGVmYjlkY2JhZTRjYmU5NGY3MDFiMTEyZjMxYjkxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3JiaXRvbiwgS2luZ3N0b24gdXBvbiBUaGFtZXM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMxM2NhNGJmOWQ1MDRhNTQ4MDJkMGRlYTI4MWQ2MGI3LnNldENvbnRlbnQoaHRtbF81ZDRlZmI5ZGNiYWU0Y2JlOTRmNzAxYjExMmYzMWI5MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mNmRmZjZiZjNjMjM0NmE0YmFjNWJjMDVkOTZmY2E0Yy5iaW5kUG9wdXAocG9wdXBfMzEzY2E0YmY5ZDUwNGE1NDgwMmQwZGVhMjgxZDYwYjcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWE4NDBkYTdhYzhiNDIxOWIzMWViYzJmNzQ1NzRiOTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS4zODI0ODQsLTAuMjU5MDg5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZGZhYjQ5NDRlYzk0M2I1YWFkZDc5YjZhN2U4ZjkyMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zY2Q0NzFiNDU3Y2Q0ODg0YmIzZWI0ZDU1YTk4OGU4YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yZjRmZjQzNjYxYWE0MjhmYjE1MDRlYzhhZmVlMDc0ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMmY0ZmY0MzY2MWFhNDI4ZmIxNTA0ZWM4YWZlZTA3NGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk9sZCBNYWxkZW4sIEtpbmdzdG9uIHVwb24gVGhhbWVzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zY2Q0NzFiNDU3Y2Q0ODg0YmIzZWI0ZDU1YTk4OGU4Yi5zZXRDb250ZW50KGh0bWxfMmY0ZmY0MzY2MWFhNDI4ZmIxNTA0ZWM4YWZlZTA3NGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNWE4NDBkYTdhYzhiNDIxOWIzMWViYzJmNzQ1NzRiOTguYmluZFBvcHVwKHBvcHVwXzNjZDQ3MWI0NTdjZDQ4ODRiYjNlYjRkNTVhOTg4ZThiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U1OGQzYzQ5MDU1MDQ1NjE4M2ZjOTZjNmQxMDQ0MmYwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuMzkzNzU1NywtMC4zMDMzMTA1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNkZmFiNDk0NGVjOTQzYjVhYWRkNzliNmE3ZThmOTIxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y5ZjkxOTBhODZiYjQwNGE4Nzg0YjgxMzIxODVhMWMxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2YwYWE0Zjk3MTY2MTQzM2E4MzI5NTgzMDU4MGZmMTA5ID0gJCgnPGRpdiBpZD0iaHRtbF9mMGFhNGY5NzE2NjE0MzNhODMyOTU4MzA1ODBmZjEwOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3VyYml0b24sIEtpbmdzdG9uIHVwb24gVGhhbWVzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mOWY5MTkwYTg2YmI0MDRhODc4NGI4MTMyMTg1YTFjMS5zZXRDb250ZW50KGh0bWxfZjBhYTRmOTcxNjYxNDMzYTgzMjk1ODMwNTgwZmYxMDkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTU4ZDNjNDkwNTUwNDU2MTgzZmM5NmM2ZDEwNDQyZjAuYmluZFBvcHVwKHBvcHVwX2Y5ZjkxOTBhODZiYjQwNGE4Nzg0YjgxMzIxODVhMWMxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2EwNTA3YWQ3Y2ZhYzQ0YmY5MzhlNjE0OWNiM2YxYmIxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuMzc4ODc1OCwtMC4yODI4NjA0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNkZmFiNDk0NGVjOTQzYjVhYWRkNzliNmE3ZThmOTIxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZkMmE2MWQwOTliMjRhMmI5YzNmNTA2NjRiNzI1NWI5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2MyZjJkZWQ2N2NlMDRiNTM5MTU2MTY3NTY4NDk2N2I5ID0gJCgnPGRpdiBpZD0iaHRtbF9jMmYyZGVkNjdjZTA0YjUzOTE1NjE2NzU2ODQ5NjdiOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VG9sd29ydGgsIEtpbmdzdG9uIHVwb24gVGhhbWVzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mZDJhNjFkMDk5YjI0YTJiOWMzZjUwNjY0YjcyNTViOS5zZXRDb250ZW50KGh0bWxfYzJmMmRlZDY3Y2UwNGI1MzkxNTYxNjc1Njg0OTY3YjkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTA1MDdhZDdjZmFjNDRiZjkzOGU2MTQ5Y2IzZjFiYjEuYmluZFBvcHVwKHBvcHVwX2ZkMmE2MWQwOTliMjRhMmI5YzNmNTA2NjRiNzI1NWI5KTsKCiAgICAgICAgICAgIAogICAgICAgIAo8L3NjcmlwdD4= onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### Modeling
- Find all the venues within a 500 meter radius of each neighborhood.
- Perform one hot encoding on the venues data.
- Group the venues by the neighborhood and calculate their mean.
- Perform a k-means clustering.

#### Create a function to extract the venues from each Neighborhood


```python
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()['response']['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
```


```python
kut_venues= getNearbyVenues(names=kut_neigh['Neighborhood'], 
                            latitudes=kut_neigh['Latitude'], 
                            longitudes=kut_neigh['Longitude'])
```

    Berrylands
    Canbury
    Chessington
    Coombe
    Kingston upon Thames
    Kingston Vale
    Malden Rushett
    Motspur Park
    New Malden
    Norbiton
    Old Malden
    Surbiton
    Tolworth
    


```python
print(kut_venues.shape)
kut_venues.head()
```

    (171, 7)
    




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
      <th>Neighborhood</th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Berrylands</td>
      <td>51.393781</td>
      <td>-0.284802</td>
      <td>Surbiton Racket &amp; Fitness Club</td>
      <td>51.392676</td>
      <td>-0.290224</td>
      <td>Gym / Fitness Center</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Berrylands</td>
      <td>51.393781</td>
      <td>-0.284802</td>
      <td>Alexandra Park</td>
      <td>51.394230</td>
      <td>-0.281206</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Berrylands</td>
      <td>51.393781</td>
      <td>-0.284802</td>
      <td>K2 Bus Stop</td>
      <td>51.392302</td>
      <td>-0.281534</td>
      <td>Bus Stop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Canbury</td>
      <td>51.417499</td>
      <td>-0.305553</td>
      <td>Canbury Gardens</td>
      <td>51.417409</td>
      <td>-0.305300</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Canbury</td>
      <td>51.417499</td>
      <td>-0.305553</td>
      <td>The Grey Horse</td>
      <td>51.414192</td>
      <td>-0.300759</td>
      <td>Pub</td>
    </tr>
  </tbody>
</table>
</div>




```python
kut_venues.groupby('Neighborhood').count()
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
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
    <tr>
      <th>Neighborhood</th>
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
      <th>Berrylands</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Canbury</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>Coombe</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Kingston Vale</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Kingston upon Thames</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Malden Rushett</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Motspur Park</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>New Malden</th>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Norbiton</th>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Old Malden</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Surbiton</th>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
    </tr>
    <tr>
      <th>Tolworth</th>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('There are {} uniques categories.'.format(len(kut_venues['Venue Category'].unique())))
```

    There are 72 uniques categories.
    

#### One hot encoding


```python
# one hot encoding
kut_onehot = pd.get_dummies(kut_venues[['Venue Category']], prefix='', prefix_sep='')

# add neighborhood column back to the dataframe
kut_onehot['Neighborhood'] = kut_venues['Neighborhood']

# move neighborhood column to the first column
fixed_columns = [kut_onehot.columns[-1]] + list(kut_onehot.columns[:-1])
kut_onehot = kut_onehot[fixed_columns]

kut_onehot.head()
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
      <th>Neighborhood</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>Auto Garage</th>
      <th>Bakery</th>
      <th>Bar</th>
      <th>Beer Bar</th>
      <th>Bistro</th>
      <th>Bookstore</th>
      <th>Bowling Alley</th>
      <th>...</th>
      <th>Spa</th>
      <th>Stationery Store</th>
      <th>Supermarket</th>
      <th>Sushi Restaurant</th>
      <th>Tea Room</th>
      <th>Thai Restaurant</th>
      <th>Theater</th>
      <th>Train Station</th>
      <th>Turkish Restaurant</th>
      <th>Wine Shop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Berrylands</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Berrylands</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Berrylands</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Canbury</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Canbury</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 73 columns</p>
</div>



Group the rows by neighborhood and take the mean of the frequency of coocurence of each category.


```python
kut_grouped = kut_onehot.groupby('Neighborhood').mean().reset_index()
kut_grouped
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
      <th>Neighborhood</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>Auto Garage</th>
      <th>Bakery</th>
      <th>Bar</th>
      <th>Beer Bar</th>
      <th>Bistro</th>
      <th>Bookstore</th>
      <th>Bowling Alley</th>
      <th>...</th>
      <th>Spa</th>
      <th>Stationery Store</th>
      <th>Supermarket</th>
      <th>Sushi Restaurant</th>
      <th>Tea Room</th>
      <th>Thai Restaurant</th>
      <th>Theater</th>
      <th>Train Station</th>
      <th>Turkish Restaurant</th>
      <th>Wine Shop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Berrylands</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Canbury</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.071429</td>
      <td>0.00</td>
      <td>0.071429</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Coombe</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kingston Vale</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kingston upon Thames</td>
      <td>0.02</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.02</td>
      <td>0.000000</td>
      <td>0.02</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.02</td>
      <td>0.020000</td>
      <td>0.040</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.02</td>
      <td>0.000000</td>
      <td>0.02</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Malden Rushett</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Motspur Park</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>New Malden</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.125000</td>
      <td>0.125</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Norbiton</td>
      <td>0.00</td>
      <td>0.035714</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.035714</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.035714</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Old Malden</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.333333</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Surbiton</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030303</td>
      <td>0.030303</td>
      <td>0.00</td>
      <td>0.030303</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.030303</td>
      <td>0.000</td>
      <td>0.030303</td>
      <td>0.030303</td>
      <td>0.00</td>
      <td>0.030303</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Tolworth</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.052632</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.00</td>
      <td>0.052632</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>12 rows × 73 columns</p>
</div>




```python
# dimensions of the dataframe
kut_grouped.shape
```




    (12, 73)




```python
num_top_venues = 5

for hood in kut_grouped['Neighborhood']:
    print('----'+hood+'----')
    temp = kut_grouped[kut_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue', 'freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')
```

    ----Berrylands----
                       venue  freq
    0   Gym / Fitness Center  0.33
    1                   Park  0.33
    2               Bus Stop  0.33
    3  Portuguese Restaurant  0.00
    4                  Plaza  0.00
    
    
    ----Canbury----
                   venue  freq
    0                Pub  0.29
    1              Plaza  0.07
    2               Park  0.07
    3              Hotel  0.07
    4  Indian Restaurant  0.07
    
    
    ----Coombe----
                  venue  freq
    0          Tea Room   1.0
    1  Asian Restaurant   0.0
    2            Market   0.0
    3          Platform   0.0
    4       Pizza Place   0.0
    
    
    ----Kingston Vale----
                  venue  freq
    0     Grocery Store  0.25
    1               Bar  0.25
    2    Sandwich Place  0.25
    3      Soccer Field  0.25
    4  Asian Restaurant  0.00
    
    
    ----Kingston upon Thames----
                  venue  freq
    0       Coffee Shop  0.12
    1              Café  0.08
    2  Department Store  0.06
    3   Thai Restaurant  0.04
    4    Clothing Store  0.04
    
    
    ----Malden Rushett----
                   venue  freq
    0  Convenience Store  0.25
    1         Restaurant  0.25
    2      Garden Center  0.25
    3                Pub  0.25
    4               Park  0.00
    
    
    ----Motspur Park----
                    venue  freq
    0                 Gym  0.25
    1          Restaurant  0.25
    2                Park  0.25
    3        Soccer Field  0.25
    4  Mexican Restaurant  0.00
    
    
    ----New Malden----
                   venue  freq
    0                Gym  0.12
    1  Indian Restaurant  0.12
    2                Bar  0.12
    3          Gastropub  0.12
    4  Korean Restaurant  0.12
    
    
    ----Norbiton----
                    venue  freq
    0   Indian Restaurant  0.11
    1  Italian Restaurant  0.07
    2                Food  0.07
    3                 Pub  0.07
    4           Wine Shop  0.04
    
    
    ----Old Malden----
               venue  freq
    0            Pub  0.33
    1           Food  0.33
    2  Train Station  0.33
    3       Platform  0.00
    4    Pizza Place  0.00
    
    
    ----Surbiton----
                    venue  freq
    0         Coffee Shop  0.18
    1                 Pub  0.12
    2  Italian Restaurant  0.06
    3            Pharmacy  0.06
    4       Grocery Store  0.06
    
    
    ----Tolworth----
                   venue  freq
    0      Grocery Store  0.16
    1         Restaurant  0.11
    2  Indian Restaurant  0.05
    3           Bus Stop  0.05
    4     Discount Store  0.05
    
    
    

#### Create a dataframe of the venues.

First, create a function to sort the venues in descending order.


```python
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
```

Create the new dataframe and display yhe top 10 venues for each neighborhood.


```python
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to tthe number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))
        
# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = kut_grouped['Neighborhood']

for ind in np.arange(kut_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(kut_grouped.iloc[ind, :], num_top_venues)
    
neighborhoods_venues_sorted.head()
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
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Berrylands</td>
      <td>Gym / Fitness Center</td>
      <td>Park</td>
      <td>Bus Stop</td>
      <td>Wine Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Department Store</td>
      <td>Discount Store</td>
      <td>Electronics Store</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Canbury</td>
      <td>Pub</td>
      <td>Shop &amp; Service</td>
      <td>Spa</td>
      <td>Plaza</td>
      <td>Café</td>
      <td>Indian Restaurant</td>
      <td>Hotel</td>
      <td>Park</td>
      <td>Supermarket</td>
      <td>Gym / Fitness Center</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Coombe</td>
      <td>Tea Room</td>
      <td>Wine Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Cosmetics Shop</td>
      <td>Deli / Bodega</td>
      <td>Department Store</td>
      <td>Discount Store</td>
      <td>Electronics Store</td>
      <td>Farmers Market</td>
      <td>Fish &amp; Chips Shop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kingston Vale</td>
      <td>Grocery Store</td>
      <td>Bar</td>
      <td>Sandwich Place</td>
      <td>Soccer Field</td>
      <td>Furniture / Home Store</td>
      <td>Garden Center</td>
      <td>Fried Chicken Joint</td>
      <td>French Restaurant</td>
      <td>Food</td>
      <td>Fish &amp; Chips Shop</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kingston upon Thames</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Department Store</td>
      <td>Thai Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Burger Joint</td>
      <td>Pub</td>
      <td>Clothing Store</td>
      <td>Italian Restaurant</td>
      <td>Asian Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



#### Cluster similar neighborhoods together using k-means clustering


```python
# import k-means 
from sklearn.cluster import KMeans

# set the number of clusters
kclusters = 5

kut_grouped_clustering = kut_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(kut_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]
```




    array([3, 1, 0, 2, 1, 1, 2, 1, 1, 4])




```python
# add clustering labels
neighborhoods_venues_sorted.insert(0,'Cluster Labels', kmeans.labels_)

kut_merged = kut_neigh

# merge kut_grouped with kut_neigh to add latitude/longitude for each neighborhood
kut_merged = kut_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

kut_merged.head()
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
      <th>Neighborhood</th>
      <th>Borough</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Berrylands</td>
      <td>Kingston upon Thames</td>
      <td>51.393781</td>
      <td>-0.284802</td>
      <td>3.0</td>
      <td>Gym / Fitness Center</td>
      <td>Park</td>
      <td>Bus Stop</td>
      <td>Wine Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Department Store</td>
      <td>Discount Store</td>
      <td>Electronics Store</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Canbury</td>
      <td>Kingston upon Thames</td>
      <td>51.417499</td>
      <td>-0.305553</td>
      <td>1.0</td>
      <td>Pub</td>
      <td>Shop &amp; Service</td>
      <td>Spa</td>
      <td>Plaza</td>
      <td>Café</td>
      <td>Indian Restaurant</td>
      <td>Hotel</td>
      <td>Park</td>
      <td>Supermarket</td>
      <td>Gym / Fitness Center</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chessington</td>
      <td>Kingston upon Thames</td>
      <td>51.358336</td>
      <td>-0.298622</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Coombe</td>
      <td>Kingston upon Thames</td>
      <td>51.419450</td>
      <td>-0.265398</td>
      <td>0.0</td>
      <td>Tea Room</td>
      <td>Wine Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Cosmetics Shop</td>
      <td>Deli / Bodega</td>
      <td>Department Store</td>
      <td>Discount Store</td>
      <td>Electronics Store</td>
      <td>Farmers Market</td>
      <td>Fish &amp; Chips Shop</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kingston upon Thames</td>
      <td>Kingston upon Thames</td>
      <td>51.409627</td>
      <td>-0.306262</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Department Store</td>
      <td>Thai Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Burger Joint</td>
      <td>Pub</td>
      <td>Clothing Store</td>
      <td>Italian Restaurant</td>
      <td>Asian Restaurant</td>
    </tr>
  </tbody>
</table>
</div>




```python
kut_merged.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 13 entries, 0 to 12
    Data columns (total 15 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Neighborhood            13 non-null     object 
     1   Borough                 13 non-null     object 
     2   Latitude                13 non-null     float64
     3   Longitude               13 non-null     float64
     4   Cluster Labels          12 non-null     float64
     5   1st Most Common Venue   12 non-null     object 
     6   2nd Most Common Venue   12 non-null     object 
     7   3rd Most Common Venue   12 non-null     object 
     8   4th Most Common Venue   12 non-null     object 
     9   5th Most Common Venue   12 non-null     object 
     10  6th Most Common Venue   12 non-null     object 
     11  7th Most Common Venue   12 non-null     object 
     12  8th Most Common Venue   12 non-null     object 
     13  9th Most Common Venue   12 non-null     object 
     14  10th Most Common Venue  12 non-null     object 
    dtypes: float64(3), object(12)
    memory usage: 1.6+ KB
    


```python
# drop the rows with NaN value
kut_merged.dropna(inplace=True)
```


```python
kut_merged.shape
```




    (12, 15)




```python
kut_merged['Cluster Labels'] = kut_merged['Cluster Labels'].astype(int)

kut_merged.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 12 entries, 0 to 12
    Data columns (total 15 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Neighborhood            12 non-null     object 
     1   Borough                 12 non-null     object 
     2   Latitude                12 non-null     float64
     3   Longitude               12 non-null     float64
     4   Cluster Labels          12 non-null     int32  
     5   1st Most Common Venue   12 non-null     object 
     6   2nd Most Common Venue   12 non-null     object 
     7   3rd Most Common Venue   12 non-null     object 
     8   4th Most Common Venue   12 non-null     object 
     9   5th Most Common Venue   12 non-null     object 
     10  6th Most Common Venue   12 non-null     object 
     11  7th Most Common Venue   12 non-null     object 
     12  8th Most Common Venue   12 non-null     object 
     13  9th Most Common Venue   12 non-null     object 
     14  10th Most Common Venue  12 non-null     object 
    dtypes: float64(2), int32(1), object(12)
    memory usage: 1.5+ KB
    

#### Visualize the clusters


```python
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11.5)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(kut_merged['Latitude'], kut_merged['Longitude'], kut_merged['Neighborhood'], kut_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=8,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.5).add_to(map_clusters)
       
map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfZGY1Y2Q5YTk0MjZjNDAyNzliYTA1YTY0ODJjMjE4MTAgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2RmNWNkOWE5NDI2YzQwMjc5YmEwNWE2NDgyYzIxODEwIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9kZjVjZDlhOTQyNmM0MDI3OWJhMDVhNjQ4MmMyMTgxMCA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9kZjVjZDlhOTQyNmM0MDI3OWJhMDVhNjQ4MmMyMTgxMCcsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNTEuMzkzNzgxMSwtMC4yODQ4MDI0XSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHpvb206IDExLjUsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYXhCb3VuZHM6IGJvdW5kcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxheWVyczogW10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3b3JsZENvcHlKdW1wOiBmYWxzZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSk7CiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyXzUyZDk1N2I5NWUwOTQyOTA4YmJmOWVlZDdiZDVjMTYxID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgICAgICAgICAgewogICJhdHRyaWJ1dGlvbiI6IG51bGwsCiAgImRldGVjdFJldGluYSI6IGZhbHNlLAogICJtYXhab29tIjogMTgsCiAgIm1pblpvb20iOiAxLAogICJub1dyYXAiOiBmYWxzZSwKICAic3ViZG9tYWlucyI6ICJhYmMiCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2RmNWNkOWE5NDI2YzQwMjc5YmEwNWE2NDgyYzIxODEwKTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80ZGQ4YWU0NTI1ZTk0ZGZlODVmYWMwZTVhNTVhMjlmNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjM5Mzc4MTEsLTAuMjg0ODAyNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC41LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogOCwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjVjZDlhOTQyNmM0MDI3OWJhMDVhNjQ4MmMyMTgxMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85YmNkYjExNWZmNGI0YmVjYjM4NDdhNTA3NGI4NDJiYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80ZDhlYjE3YjFkMzU0NjZiOTNkOWZjMzU3YjdkZGM3YSA9ICQoJzxkaXYgaWQ9Imh0bWxfNGQ4ZWIxN2IxZDM1NDY2YjkzZDlmYzM1N2I3ZGRjN2EiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJlcnJ5bGFuZHMgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85YmNkYjExNWZmNGI0YmVjYjM4NDdhNTA3NGI4NDJiYS5zZXRDb250ZW50KGh0bWxfNGQ4ZWIxN2IxZDM1NDY2YjkzZDlmYzM1N2I3ZGRjN2EpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGRkOGFlNDUyNWU5NGRmZTg1ZmFjMGU1YTU1YTI5ZjYuYmluZFBvcHVwKHBvcHVwXzliY2RiMTE1ZmY0YjRiZWNiMzg0N2E1MDc0Yjg0MmJhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQyYTYwYjJkOWNhYTRkY2I4NjdkNjcwNGUyZWQ2MmU0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDE3NDk4NjUsLTAuMzA1NTUyODA1MDQ5MjYxNjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNSwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDgsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGY1Y2Q5YTk0MjZjNDAyNzliYTA1YTY0ODJjMjE4MTApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzZjNGZjZGQxMDE4NGFiYmExODA2NmE5MjA1YTkxOWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjc2NmU1YjZiZjdlNDkzZmI3MWMxZjI5MGRkMDRhYzQgPSAkKCc8ZGl2IGlkPSJodG1sX2Y3NjZlNWI2YmY3ZTQ5M2ZiNzFjMWYyOTBkZDA0YWM0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DYW5idXJ5IENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzZjNGZjZGQxMDE4NGFiYmExODA2NmE5MjA1YTkxOWYuc2V0Q29udGVudChodG1sX2Y3NjZlNWI2YmY3ZTQ5M2ZiNzFjMWYyOTBkZDA0YWM0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQyYTYwYjJkOWNhYTRkY2I4NjdkNjcwNGUyZWQ2MmU0LmJpbmRQb3B1cChwb3B1cF83NmM0ZmNkZDEwMTg0YWJiYTE4MDY2YTkyMDVhOTE5Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jN2YyYjcxY2ViMzA0NmE2YWZlZmYyY2NkMGNiODhiMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQxOTQ0OTksLTAuMjY1Mzk4NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC41LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogOCwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjVjZDlhOTQyNmM0MDI3OWJhMDVhNjQ4MmMyMTgxMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yMTZkMGY5MDE1ZDM0M2U5OGQ3MDZhMzMwNTNlM2Q3ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ODAxMzcxY2JmZjk0ODhjODY2ZjQ4MzE0M2ZiZDFkMiA9ICQoJzxkaXYgaWQ9Imh0bWxfOTgwMTM3MWNiZmY5NDg4Yzg2NmY0ODMxNDNmYmQxZDIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNvb21iZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzIxNmQwZjkwMTVkMzQzZTk4ZDcwNmEzMzA1M2UzZDdmLnNldENvbnRlbnQoaHRtbF85ODAxMzcxY2JmZjk0ODhjODY2ZjQ4MzE0M2ZiZDFkMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jN2YyYjcxY2ViMzA0NmE2YWZlZmYyY2NkMGNiODhiMy5iaW5kUG9wdXAocG9wdXBfMjE2ZDBmOTAxNWQzNDNlOThkNzA2YTMzMDUzZTNkN2YpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWFjZTU0N2U0NjZmNDJlMGFkYzZjZDVmZjUwMTkzZTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40MDk2Mjc1LC0wLjMwNjI2MjFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNSwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDgsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGY1Y2Q5YTk0MjZjNDAyNzliYTA1YTY0ODJjMjE4MTApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjhiNjQxODNiZjNhNDZkYzgwODVlZGQ4YzNlY2IwM2UgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDI1NzJlODY0ZmY3NDY5ZTk1MmVjNmNkMzRjYTNiN2QgPSAkKCc8ZGl2IGlkPSJodG1sXzQyNTcyZTg2NGZmNzQ2OWU5NTJlYzZjZDM0Y2EzYjdkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LaW5nc3RvbiB1cG9uIFRoYW1lcyBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I4YjY0MTgzYmYzYTQ2ZGM4MDg1ZWRkOGMzZWNiMDNlLnNldENvbnRlbnQoaHRtbF80MjU3MmU4NjRmZjc0NjllOTUyZWM2Y2QzNGNhM2I3ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lYWNlNTQ3ZTQ2NmY0MmUwYWRjNmNkNWZmNTAxOTNlOC5iaW5kUG9wdXAocG9wdXBfYjhiNjQxODNiZjNhNDZkYzgwODVlZGQ4YzNlY2IwM2UpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjA4NzFkMzA4OTQyNDU3MmIzZTIyMTQxYWYxOTM5MTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40MzE4NSwtMC4yNTgxMzc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjUsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA4LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2RmNWNkOWE5NDI2YzQwMjc5YmEwNWE2NDgyYzIxODEwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E2NmQ2YzJiZDA2MzQ0ZDRiMTg5NDQ2OTJmOGU3MDVjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzU2ZDI3ZmFkODBjMjQ1NzdhNjhjOTFiZTk0YjcyOTZiID0gJCgnPGRpdiBpZD0iaHRtbF81NmQyN2ZhZDgwYzI0NTc3YTY4YzkxYmU5NGI3Mjk2YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2luZ3N0b24gVmFsZSBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E2NmQ2YzJiZDA2MzQ0ZDRiMTg5NDQ2OTJmOGU3MDVjLnNldENvbnRlbnQoaHRtbF81NmQyN2ZhZDgwYzI0NTc3YTY4YzkxYmU5NGI3Mjk2Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMDg3MWQzMDg5NDI0NTcyYjNlMjIxNDFhZjE5MzkxNS5iaW5kUG9wdXAocG9wdXBfYTY2ZDZjMmJkMDYzNDRkNGIxODk0NDY5MmY4ZTcwNWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzE0YTlhMzBhNGUwNDcyOTg2M2ExZWU5YWQ5OGUxNzUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS4zNDEwNTIzLC0wLjMxOTA3NTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNSwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDgsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGY1Y2Q5YTk0MjZjNDAyNzliYTA1YTY0ODJjMjE4MTApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjEwZGQ2MmIyZDVhNDliMTgzN2M2NzAyYWJlODZlYjUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzIyNmJhNTY5MGJiNDc5NGE3ZWUxYjg2YTc4MTk3NDMgPSAkKCc8ZGl2IGlkPSJodG1sX2MyMjZiYTU2OTBiYjQ3OTRhN2VlMWI4NmE3ODE5NzQzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NYWxkZW4gUnVzaGV0dCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2YxMGRkNjJiMmQ1YTQ5YjE4MzdjNjcwMmFiZTg2ZWI1LnNldENvbnRlbnQoaHRtbF9jMjI2YmE1NjkwYmI0Nzk0YTdlZTFiODZhNzgxOTc0Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zMTRhOWEzMGE0ZTA0NzI5ODYzYTFlZTlhZDk4ZTE3NS5iaW5kUG9wdXAocG9wdXBfZjEwZGQ2MmIyZDVhNDliMTgzN2M2NzAyYWJlODZlYjUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTJhMmNkMmQ3ODUwNGM0YmI5MjJkNGY1ZGVmYjdjNDAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS4zOTA5ODUyLC0wLjI0ODg5NzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzAwYjVlYiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMwMGI1ZWIiLAogICJmaWxsT3BhY2l0eSI6IDAuNSwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDgsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGY1Y2Q5YTk0MjZjNDAyNzliYTA1YTY0ODJjMjE4MTApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjU3ZjI1MTBhMWQyNGVmN2JkYjFiMjQzYzkwYmZiMjEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDFlM2FlOTliYjQyNGY5MWJhZDhjMjdlMTVmZDQ2NjggPSAkKCc8ZGl2IGlkPSJodG1sXzQxZTNhZTk5YmI0MjRmOTFiYWQ4YzI3ZTE1ZmQ0NjY4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Nb3RzcHVyIFBhcmsgQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yNTdmMjUxMGExZDI0ZWY3YmRiMWIyNDNjOTBiZmIyMS5zZXRDb250ZW50KGh0bWxfNDFlM2FlOTliYjQyNGY5MWJhZDhjMjdlMTVmZDQ2NjgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTJhMmNkMmQ3ODUwNGM0YmI5MjJkNGY1ZGVmYjdjNDAuYmluZFBvcHVwKHBvcHVwXzI1N2YyNTEwYTFkMjRlZjdiZGIxYjI0M2M5MGJmYjIxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2YzNTgzYzljZjUzNTQyOWVhZTg1NTk2Mzg5OTgyMjYwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDA1MzM0NywtMC4yNjM0MDY2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjUsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA4LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2RmNWNkOWE5NDI2YzQwMjc5YmEwNWE2NDgyYzIxODEwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzllOTdhMDRiZTY5NzQ5NWU4ZGZlYjEzMGYzNDU2Y2FjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzM2N2QxYzkzZWMwODRjZmQ4OGRjYTczYmRlMWYzOWU3ID0gJCgnPGRpdiBpZD0iaHRtbF8zNjdkMWM5M2VjMDg0Y2ZkODhkY2E3M2JkZTFmMzllNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TmV3IE1hbGRlbiBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzllOTdhMDRiZTY5NzQ5NWU4ZGZlYjEzMGYzNDU2Y2FjLnNldENvbnRlbnQoaHRtbF8zNjdkMWM5M2VjMDg0Y2ZkODhkY2E3M2JkZTFmMzllNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMzU4M2M5Y2Y1MzU0MjllYWU4NTU5NjM4OTk4MjI2MC5iaW5kUG9wdXAocG9wdXBfOWU5N2EwNGJlNjk3NDk1ZThkZmViMTMwZjM0NTZjYWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmUzOThhZjA4MjhmNDAyY2FlMWU5M2ExOWMxYTBjNWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40MDk5OTk0LC0wLjI4NzM5NjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNSwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDgsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGY1Y2Q5YTk0MjZjNDAyNzliYTA1YTY0ODJjMjE4MTApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjllY2M4ZDk2ZjMwNGVkOWJhM2U0NGIwZTZiZDRmZTAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDAzMDUzODMzZTdjNDhjN2JjODczZmM5NTg1ZmY5NzEgPSAkKCc8ZGl2IGlkPSJodG1sXzAwMzA1MzgzM2U3YzQ4YzdiYzg3M2ZjOTU4NWZmOTcxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3JiaXRvbiBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY5ZWNjOGQ5NmYzMDRlZDliYTNlNDRiMGU2YmQ0ZmUwLnNldENvbnRlbnQoaHRtbF8wMDMwNTM4MzNlN2M0OGM3YmM4NzNmYzk1ODVmZjk3MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mZTM5OGFmMDgyOGY0MDJjYWUxZTkzYTE5YzFhMGM1Yy5iaW5kUG9wdXAocG9wdXBfNjllY2M4ZDk2ZjMwNGVkOWJhM2U0NGIwZTZiZDRmZTApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTU4NmRmNGM4MmJkNDM0NWE1OWUyZTI5NDZmNDNlMmEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS4zODI0ODQsLTAuMjU5MDg5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC41LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogOCwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9kZjVjZDlhOTQyNmM0MDI3OWJhMDVhNjQ4MmMyMTgxMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMDY2ZjVmYmI0MDU0MzJkYmYxOGQxYzI4NzljODhhMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wYzg5YThjODc3ZTI0OTNmODdjZTA0Y2U0YzVhOWY1ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMGM4OWE4Yzg3N2UyNDkzZjg3Y2UwNGNlNGM1YTlmNWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk9sZCBNYWxkZW4gQ2x1c3RlciA0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xMDY2ZjVmYmI0MDU0MzJkYmYxOGQxYzI4NzljODhhMi5zZXRDb250ZW50KGh0bWxfMGM4OWE4Yzg3N2UyNDkzZjg3Y2UwNGNlNGM1YTlmNWQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTU4NmRmNGM4MmJkNDM0NWE1OWUyZTI5NDZmNDNlMmEuYmluZFBvcHVwKHBvcHVwXzEwNjZmNWZiYjQwNTQzMmRiZjE4ZDFjMjg3OWM4OGEyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2UwOTBiMjI2OTNiMjRkYzQ5MjEzMWFhZDJkYjViOGU0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuMzkzNzU1NywtMC4zMDMzMTA1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjUsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA4LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2RmNWNkOWE5NDI2YzQwMjc5YmEwNWE2NDgyYzIxODEwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzMzZTE3MjM0ZjcwYTRjZmJhNmVkMThhNTZhMDZlMGJiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRlMWUyOGJhOGFjOTRlMjhhN2EwNzkxNDc0NjcwZWE0ID0gJCgnPGRpdiBpZD0iaHRtbF80ZTFlMjhiYThhYzk0ZTI4YTdhMDc5MTQ3NDY3MGVhNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3VyYml0b24gQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zM2UxNzIzNGY3MGE0Y2ZiYTZlZDE4YTU2YTA2ZTBiYi5zZXRDb250ZW50KGh0bWxfNGUxZTI4YmE4YWM5NGUyOGE3YTA3OTE0NzQ2NzBlYTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTA5MGIyMjY5M2IyNGRjNDkyMTMxYWFkMmRiNWI4ZTQuYmluZFBvcHVwKHBvcHVwXzMzZTE3MjM0ZjcwYTRjZmJhNmVkMThhNTZhMDZlMGJiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNiMTQ5OWJlZDVhMDQ4NjY5MGExODcyNTA0YjE0NzIyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuMzc4ODc1OCwtMC4yODI4NjA0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjUsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA4LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2RmNWNkOWE5NDI2YzQwMjc5YmEwNWE2NDgyYzIxODEwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2UxNDU2MzIwOTdjNDQyNGNiODAzMDc3ZTI3MzQxMmIzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2JmNDJiMGE3YWE3MjQxM2Q4OWJkMDYzYTFlNDZmZDZkID0gJCgnPGRpdiBpZD0iaHRtbF9iZjQyYjBhN2FhNzI0MTNkODliZDA2M2ExZTQ2ZmQ2ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VG9sd29ydGggQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lMTQ1NjMyMDk3YzQ0MjRjYjgwMzA3N2UyNzM0MTJiMy5zZXRDb250ZW50KGh0bWxfYmY0MmIwYTdhYTcyNDEzZDg5YmQwNjNhMWU0NmZkNmQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2IxNDk5YmVkNWEwNDg2NjkwYTE4NzI1MDRiMTQ3MjIuYmluZFBvcHVwKHBvcHVwX2UxNDU2MzIwOTdjNDQyNGNiODAzMDc3ZTI3MzQxMmIzKTsKCiAgICAgICAgICAgIAogICAgICAgIAo8L3NjcmlwdD4= onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



Each cluster is color coded for the ease of presentation. We can see that the  majority of the neighborhoods fall in the purple cluster, which is Cluster 1. Three neighborhoods have their own cluster, which are Red, Green and Yellow, i.e. Cluster 0, 3 and 4 respectively. The Blue cluster, which is Cluster 2, consists of three neighborhoods.

### Analysis
Analyze each of the clusters to identify the characteristics of each cluster and the neighborhoods in them.

Examine the first cluster.


```python
kut_merged[kut_merged['Cluster Labels'] == 0]
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
      <th>Neighborhood</th>
      <th>Borough</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Coombe</td>
      <td>Kingston upon Thames</td>
      <td>51.41945</td>
      <td>-0.265398</td>
      <td>0</td>
      <td>Tea Room</td>
      <td>Wine Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Cosmetics Shop</td>
      <td>Deli / Bodega</td>
      <td>Department Store</td>
      <td>Discount Store</td>
      <td>Electronics Store</td>
      <td>Farmers Market</td>
      <td>Fish &amp; Chips Shop</td>
    </tr>
  </tbody>
</table>
</div>



Cluster 0 has only one neighborhood in it. The most common venues are Tea Rooms, Wine Shops, and Fast Food Restaurants.

Examine the second cluster.


```python
kut_merged[kut_merged['Cluster Labels'] == 1]
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
      <th>Neighborhood</th>
      <th>Borough</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Canbury</td>
      <td>Kingston upon Thames</td>
      <td>51.417499</td>
      <td>-0.305553</td>
      <td>1</td>
      <td>Pub</td>
      <td>Shop &amp; Service</td>
      <td>Spa</td>
      <td>Plaza</td>
      <td>Café</td>
      <td>Indian Restaurant</td>
      <td>Hotel</td>
      <td>Park</td>
      <td>Supermarket</td>
      <td>Gym / Fitness Center</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kingston upon Thames</td>
      <td>Kingston upon Thames</td>
      <td>51.409627</td>
      <td>-0.306262</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Department Store</td>
      <td>Thai Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Burger Joint</td>
      <td>Pub</td>
      <td>Clothing Store</td>
      <td>Italian Restaurant</td>
      <td>Asian Restaurant</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Malden Rushett</td>
      <td>Kingston upon Thames</td>
      <td>51.341052</td>
      <td>-0.319076</td>
      <td>1</td>
      <td>Convenience Store</td>
      <td>Pub</td>
      <td>Garden Center</td>
      <td>Restaurant</td>
      <td>Farmers Market</td>
      <td>Cosmetics Shop</td>
      <td>Deli / Bodega</td>
      <td>Department Store</td>
      <td>Discount Store</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>8</th>
      <td>New Malden</td>
      <td>Kingston upon Thames</td>
      <td>51.405335</td>
      <td>-0.263407</td>
      <td>1</td>
      <td>Indian Restaurant</td>
      <td>Korean Restaurant</td>
      <td>Gastropub</td>
      <td>Gym</td>
      <td>Bar</td>
      <td>Sushi Restaurant</td>
      <td>Supermarket</td>
      <td>Chinese Restaurant</td>
      <td>Department Store</td>
      <td>Discount Store</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Norbiton</td>
      <td>Kingston upon Thames</td>
      <td>51.409999</td>
      <td>-0.287396</td>
      <td>1</td>
      <td>Indian Restaurant</td>
      <td>Pub</td>
      <td>Italian Restaurant</td>
      <td>Food</td>
      <td>Hardware Store</td>
      <td>Pizza Place</td>
      <td>Pharmacy</td>
      <td>Japanese Restaurant</td>
      <td>Hotel</td>
      <td>Wine Shop</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Surbiton</td>
      <td>Kingston upon Thames</td>
      <td>51.393756</td>
      <td>-0.303310</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Pub</td>
      <td>Grocery Store</td>
      <td>Italian Restaurant</td>
      <td>Pharmacy</td>
      <td>Breakfast Spot</td>
      <td>Gastropub</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Gym / Fitness Center</td>
    </tr>
  </tbody>
</table>
</div>



Cluster 1 has six neighborhods, the highest number of neighborhoods, in it. After examining these neighborhoods, we can see that the most common venues are Restaurants, Coffee shops, Cafes, Convenience Stores, Department Stores, Grocery Stores, Pubs, Shops & Services, and Spas. There are also Gyms, Spas and other Stores around. This seems to be a great cluster to live in.

Examine the third cluster.


```python
kut_merged[kut_merged['Cluster Labels'] == 2]
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
      <th>Neighborhood</th>
      <th>Borough</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Kingston Vale</td>
      <td>Kingston upon Thames</td>
      <td>51.431850</td>
      <td>-0.258138</td>
      <td>2</td>
      <td>Grocery Store</td>
      <td>Bar</td>
      <td>Sandwich Place</td>
      <td>Soccer Field</td>
      <td>Furniture / Home Store</td>
      <td>Garden Center</td>
      <td>Fried Chicken Joint</td>
      <td>French Restaurant</td>
      <td>Food</td>
      <td>Fish &amp; Chips Shop</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Motspur Park</td>
      <td>Kingston upon Thames</td>
      <td>51.390985</td>
      <td>-0.248898</td>
      <td>2</td>
      <td>Soccer Field</td>
      <td>Gym</td>
      <td>Park</td>
      <td>Restaurant</td>
      <td>Farmers Market</td>
      <td>Cosmetics Shop</td>
      <td>Deli / Bodega</td>
      <td>Department Store</td>
      <td>Discount Store</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Tolworth</td>
      <td>Kingston upon Thames</td>
      <td>51.378876</td>
      <td>-0.282860</td>
      <td>2</td>
      <td>Grocery Store</td>
      <td>Restaurant</td>
      <td>Discount Store</td>
      <td>Pharmacy</td>
      <td>Pizza Place</td>
      <td>Furniture / Home Store</td>
      <td>Italian Restaurant</td>
      <td>Bus Stop</td>
      <td>Indian Restaurant</td>
      <td>Hotel</td>
    </tr>
  </tbody>
</table>
</div>



Cluster 2 has three bneighborhoods in it. The most common venues are Grocery Stores, Soccer Fields, Bars, Restaurants, Gyms, and Parks.

Examine the fourth cluster.


```python
kut_merged[kut_merged['Cluster Labels'] == 3]
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
      <th>Neighborhood</th>
      <th>Borough</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Berrylands</td>
      <td>Kingston upon Thames</td>
      <td>51.393781</td>
      <td>-0.284802</td>
      <td>3</td>
      <td>Gym / Fitness Center</td>
      <td>Park</td>
      <td>Bus Stop</td>
      <td>Wine Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Department Store</td>
      <td>Discount Store</td>
      <td>Electronics Store</td>
      <td>Farmers Market</td>
    </tr>
  </tbody>
</table>
</div>



Cluster 3 has only one neighborhood in it. The most common venues are Gyms, Parks, and Bus stops.

Examine the fifth cluster.


```python
kut_merged[kut_merged['Cluster Labels'] == 4]
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
      <th>Neighborhood</th>
      <th>Borough</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>Old Malden</td>
      <td>Kingston upon Thames</td>
      <td>51.382484</td>
      <td>-0.25909</td>
      <td>4</td>
      <td>Train Station</td>
      <td>Pub</td>
      <td>Food</td>
      <td>Wine Shop</td>
      <td>Farmers Market</td>
      <td>Cosmetics Shop</td>
      <td>Deli / Bodega</td>
      <td>Department Store</td>
      <td>Discount Store</td>
      <td>Electronics Store</td>
    </tr>
  </tbody>
</table>
</div>



Cluster 4 has only one neighborhood in it. The most common venues are Train Stations, Pubs, and Food Joints.

### Results
The aim of this project is to help people who want to relocate to the safest borough in London. Expats can chose the neighborhoods to which they want to relocate based on the most common venues in it. For example, if a person is looking for a neighborhood with good connectivity and public transportation we can see that Clusters 3 and 4 have Bus Stops and Train Stations respectively, as the most common venues. If a person is looking for a neighborhood with stores and restaurants in a close proximity, then the neighborhoods in the Cluster 1 is suitable. For a family, I feel that the neighborhoods in Cluster 2 are more suitable due to the common venues such as Parks, Gym/Fitness centers, Bus Stops, Restaurants, Grocery Stores and Soccer Fields, which is ideal for a family.

### Conclusion
This project helps a person get a better understanding of the neighborhoods with respect to the most common venues in that neighborhood. It is always helpful to make use of technology to stay one step ahead i.e. finding out more about places before moving into a neighborhood. We have just taken safety as a primary concern to shortlist the borough of London. The future of this project includes taking other factors such as cost of living in the areas into consideration to shortlist the boroughs based on safety and a predefined budget.
