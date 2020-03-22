# Where should you live in London?
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
CLIENT_ID = 'B50RYBOWHJ3ZMRVEFTLHJCXGOYNCXGI13VDT5FMYTUQSUTQC'
CLIENT_SECRET = 'ST2IN4ZGOQ3BXEWJ2HW1LKUML2BBGK1JX1QD2WKEPET31W4W'
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

    <?xml version="1.0" encoding="utf-8"?>
    <!DOCTYPE html>
    <html class="client-nojs" dir="ltr" lang="en">
     <head>
      <meta charset="UTF-8"/>
      <title>
       List of London boroughs - Wikipedia
      </title>
      <script>
       document.documentElement.className="client-js";RLCONF={"wgBreakFrames":!1,"wgSeparatorTransformTable":["",""],"wgDigitTransformTable":["",""],"wgDefaultDateFormat":"dmy","wgMonthNames":["","January","February","March","April","May","June","July","August","September","October","November","December"],"wgMonthNamesShort":["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],"wgRequestId":"XnGIbgpAMNIAAU@V7W8AAACD","wgCSPNonce":!1,"wgCanonicalNamespace":"","wgCanonicalSpecialPageName":!1,"wgNamespaceNumber":0,"wgPageName":"List_of_London_boroughs","wgTitle":"List of London boroughs","wgCurRevisionId":943613985,"wgRevisionId":943613985,"wgArticleId":28092685,"wgIsArticle":!0,"wgIsRedirect":!1,"wgAction":"view","wgUserName":null,"wgUserGroups":["*"],"wgCategories":["Use dmy dates from August 2015","Use British English from August 2015","Lists of coordinates","Geographic coordinate lists","Articles with Geo","London boroughs","Lists of places in London"],
    "wgPageContentLanguage":"en","wgPageContentModel":"wikitext","wgRelevantPageName":"List_of_London_boroughs","wgRelevantArticleId":28092685,"wgIsProbablyEditable":!0,"wgRelevantPageIsProbablyEditable":!0,"wgRestrictionEdit":[],"wgRestrictionMove":[],"wgMediaViewerOnClick":!0,"wgMediaViewerEnabledByDefault":!0,"wgPopupsReferencePreviews":!1,"wgPopupsConflictsWithNavPopupGadget":!1,"wgVisualEditor":{"pageLanguageCode":"en","pageLanguageDir":"ltr","pageVariantFallbacks":"en"},"wgMFDisplayWikibaseDescriptions":{"search":!0,"nearby":!0,"watchlist":!0,"tagline":!1},"wgWMESchemaEditAttemptStepOversample":!1,"wgULSCurrentAutonym":"English","wgNoticeProject":"wikipedia","wgWikibaseItemId":"Q6577004","wgCentralAuthMobileDomain":!1,"wgEditSubmitButtonLabelPublish":!0};RLSTATE={"ext.globalCssJs.user.styles":"ready","site.styles":"ready","noscript":"ready","user.styles":"ready","ext.globalCssJs.user":"ready","user":"ready","user.options":"ready","user.tokens":"loading"
    ,"ext.cite.styles":"ready","mediawiki.legacy.shared":"ready","mediawiki.legacy.commonPrint":"ready","jquery.tablesorter.styles":"ready","jquery.makeCollapsible.styles":"ready","mediawiki.toc.styles":"ready","skins.vector.styles":"ready","wikibase.client.init":"ready","ext.visualEditor.desktopArticleTarget.noscript":"ready","ext.uls.interlanguage":"ready","ext.wikimediaBadges":"ready"};RLPAGEMODULES=["ext.cite.ux-enhancements","site","mediawiki.page.startup","skins.vector.js","mediawiki.page.ready","jquery.tablesorter","jquery.makeCollapsible","mediawiki.toc","ext.gadget.ReferenceTooltips","ext.gadget.charinsert","ext.gadget.refToolbar","ext.gadget.extra-toolbar-buttons","ext.gadget.switcher","ext.centralauth.centralautologin","mmv.head","mmv.bootstrap.autostart","ext.popups","ext.visualEditor.desktopArticleTarget.init","ext.visualEditor.targetLoader","ext.eventLogging","ext.wikimediaEvents","ext.navigationTiming","ext.uls.compactlinks","ext.uls.interface",
    "ext.cx.eventlogging.campaigns","ext.quicksurveys.init","ext.centralNotice.geoIP","ext.centralNotice.startUp"];
      </script>
      <script>
       (RLQ=window.RLQ||[]).push(function(){mw.loader.implement("user.tokens@tffin",function($,jQuery,require,module){/*@nomin*/mw.user.tokens.set({"patrolToken":"+\\","watchToken":"+\\","csrfToken":"+\\"});
    });});
      </script>
      <link href="/w/load.php?lang=en&amp;modules=ext.cite.styles%7Cext.uls.interlanguage%7Cext.visualEditor.desktopArticleTarget.noscript%7Cext.wikimediaBadges%7Cjquery.makeCollapsible.styles%7Cjquery.tablesorter.styles%7Cmediawiki.legacy.commonPrint%2Cshared%7Cmediawiki.toc.styles%7Cskins.vector.styles%7Cwikibase.client.init&amp;only=styles&amp;skin=vector" rel="stylesheet"/>
      <script async="" src="/w/load.php?lang=en&amp;modules=startup&amp;only=scripts&amp;raw=1&amp;skin=vector"/>
      <meta content="" name="ResourceLoaderDynamicStyles"/>
      <link href="/w/load.php?lang=en&amp;modules=site.styles&amp;only=styles&amp;skin=vector" rel="stylesheet"/>
      <meta content="MediaWiki 1.35.0-wmf.22" name="generator"/>
      <meta content="origin" name="referrer"/>
      <meta content="origin-when-crossorigin" name="referrer"/>
      <meta content="origin-when-cross-origin" name="referrer"/>
      <link href="/w/index.php?title=List_of_London_boroughs&amp;action=edit" rel="alternate" title="Edit this page" type="application/x-wiki"/>
      <link href="/w/index.php?title=List_of_London_boroughs&amp;action=edit" rel="edit" title="Edit this page"/>
      <link href="/static/apple-touch/wikipedia.png" rel="apple-touch-icon"/>
      <link href="/static/favicon/wikipedia.ico" rel="shortcut icon"/>
      <link href="/w/opensearch_desc.php" rel="search" title="Wikipedia (en)" type="application/opensearchdescription+xml"/>
      <link href="//en.wikipedia.org/w/api.php?action=rsd" rel="EditURI" type="application/rsd+xml"/>
      <link href="//creativecommons.org/licenses/by-sa/3.0/" rel="license"/>
      <link href="/w/index.php?title=Special:RecentChanges&amp;feed=atom" rel="alternate" title="Wikipedia Atom feed" type="application/atom+xml"/>
      <link href="https://en.wikipedia.org/wiki/List_of_London_boroughs" rel="canonical"/>
      <link href="//login.wikimedia.org" rel="dns-prefetch"/>
      <link href="//meta.wikimedia.org" rel="dns-prefetch"/>
      <!--[if lt IE 9]><script src="/w/resources/lib/html5shiv/html5shiv.js"></script><![endif]-->
     </head>
     <body class="mediawiki ltr sitedir-ltr mw-hide-empty-elt ns-0 ns-subject mw-editable page-List_of_London_boroughs rootpage-List_of_London_boroughs skin-vector action-view">
      <div class="noprint" id="mw-page-base"/>
      <div class="noprint" id="mw-head-base"/>
      <div class="mw-body" id="content" role="main">
       <a id="top"/>
       <div class="mw-body-content" id="siteNotice">
        <!-- CentralNotice -->
       </div>
       <div class="mw-indicators mw-body-content">
       </div>
       <h1 class="firstHeading" id="firstHeading" lang="en">
        List of London boroughs
       </h1>
       <div class="mw-body-content" id="bodyContent">
        <div class="noprint" id="siteSub">
         From Wikipedia, the free encyclopedia
        </div>
        <div id="contentSub"/>
        <div id="jump-to-nav"/>
        <a class="mw-jump-link" href="#mw-head">
         Jump to navigation
        </a>
        <a class="mw-jump-link" href="#p-search">
         Jump to search
        </a>
        <div class="mw-content-ltr" dir="ltr" id="mw-content-text" lang="en">
         <div class="mw-parser-output">
          <p class="mw-empty-elt">
          </p>
          <div class="thumb tright">
           <div class="thumbinner" style="width:302px;">
            <a class="image" href="/wiki/File:London-boroughs.svg">
             <img alt="" class="thumbimage" data-file-height="386" data-file-width="489" decoding="async" height="237" src="//upload.wikimedia.org/wikipedia/commons/thumb/2/29/London-boroughs.svg/300px-London-boroughs.svg.png" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/2/29/London-boroughs.svg/450px-London-boroughs.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/2/29/London-boroughs.svg/600px-London-boroughs.svg.png 2x" width="300"/>
            </a>
            <div class="thumbcaption">
             <div class="magnify">
              <a class="internal" href="/wiki/File:London-boroughs.svg" title="Enlarge"/>
             </div>
             Map of the 32 London boroughs and the City of London.
            </div>
           </div>
          </div>
          <p>
           This is a list of
           <a href="/wiki/Districts_of_England" title="Districts of England">
            local authority districts
           </a>
           within
           <a href="/wiki/Greater_London" title="Greater London">
            Greater London
           </a>
           , including 32
           <a href="/wiki/London_boroughs" title="London boroughs">
            London boroughs
           </a>
           and the
           <a href="/wiki/City_of_London" title="City of London">
            City of London
           </a>
           . The London boroughs were all created on 1 April 1965. Upon creation, twelve were designated
           <a href="/wiki/Inner_London" title="Inner London">
            Inner London
           </a>
           boroughs and the remaining twenty were designated
           <a href="/wiki/Outer_London" title="Outer London">
            Outer London
           </a>
           boroughs. The
           <a class="mw-redirect" href="/wiki/Office_for_National_Statistics" title="Office for National Statistics">
            Office for National Statistics
           </a>
           has amended the designations of three boroughs for statistics purposes only. Three boroughs have been granted the designation
           <a class="mw-redirect" href="/wiki/Royal_borough" title="Royal borough">
            royal borough
           </a>
           and one has
           <a href="/wiki/City_status_in_the_United_Kingdom" title="City status in the United Kingdom">
            city status
           </a>
           . For planning purposes, in addition to the boroughs and City there are also two active development corporations, the
           <a href="/wiki/London_Legacy_Development_Corporation" title="London Legacy Development Corporation">
            London Legacy Development Corporation
           </a>
           and
           <a href="/wiki/Old_Oak_and_Park_Royal_Development_Corporation" title="Old Oak and Park Royal Development Corporation">
            Old Oak and Park Royal Development Corporation
           </a>
           .
          </p>
          <div aria-labelledby="mw-toc-heading" class="toc" id="toc" role="navigation">
           <input class="toctogglecheckbox" id="toctogglecheckbox" role="button" style="display:none" type="checkbox"/>
           <div class="toctitle" dir="ltr" lang="en">
            <h2 id="mw-toc-heading">
             Contents
            </h2>
            <span class="toctogglespan">
             <label class="toctogglelabel" for="toctogglecheckbox"/>
            </span>
           </div>
           <ul>
            <li class="toclevel-1 tocsection-1">
             <a href="#List_of_boroughs_and_local_authorities">
              <span class="tocnumber">
               1
              </span>
              <span class="toctext">
               List of boroughs and local authorities
              </span>
             </a>
            </li>
            <li class="toclevel-1 tocsection-2">
             <a href="#City_of_London">
              <span class="tocnumber">
               2
              </span>
              <span class="toctext">
               City of London
              </span>
             </a>
            </li>
            <li class="toclevel-1 tocsection-3">
             <a href="#See_also">
              <span class="tocnumber">
               3
              </span>
              <span class="toctext">
               See also
              </span>
             </a>
            </li>
            <li class="toclevel-1 tocsection-4">
             <a href="#Notes">
              <span class="tocnumber">
               4
              </span>
              <span class="toctext">
               Notes
              </span>
             </a>
            </li>
            <li class="toclevel-1 tocsection-5">
             <a href="#References">
              <span class="tocnumber">
               5
              </span>
              <span class="toctext">
               References
              </span>
             </a>
            </li>
            <li class="toclevel-1 tocsection-6">
             <a href="#External_links">
              <span class="tocnumber">
               6
              </span>
              <span class="toctext">
               External links
              </span>
             </a>
            </li>
           </ul>
          </div>
          <h2>
           <span class="mw-headline" id="List_of_boroughs_and_local_authorities">
            List of boroughs and local authorities
           </span>
           <span class="mw-editsection">
            <span class="mw-editsection-bracket">
             [
            </span>
            <a href="/w/index.php?title=List_of_London_boroughs&amp;action=edit&amp;section=1" title="Edit section: List of boroughs and local authorities">
             edit
            </a>
            <span class="mw-editsection-bracket">
             ]
            </span>
           </span>
          </h2>
          <table class="wikitable sortable" style="font-size:100%" width="100%">
           <tbody>
            <tr>
             <th>
              Borough
             </th>
             <th>
              Inner
             </th>
             <th>
              Status
             </th>
             <th>
              Local authority
             </th>
             <th>
              Political control
             </th>
             <th>
              Headquarters
             </th>
             <th>
              Area (sq mi)
             </th>
             <th>
              Population (2013 est)
              <sup class="reference" id="cite_ref-1">
               <a href="#cite_note-1">
                [1]
               </a>
              </sup>
             </th>
             <th>
              Co-ordinates
             </th>
             <th>
              <span style="background:#67BCD3">
               Nr. in map
              </span>
             </th>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Barking_and_Dagenham" title="London Borough of Barking and Dagenham">
               Barking and Dagenham
              </a>
              <sup class="reference" id="cite_ref-2">
               <a href="#cite_note-2">
                [note 1]
               </a>
              </sup>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Barking_and_Dagenham_London_Borough_Council" title="Barking and Dagenham London Borough Council">
               Barking and Dagenham London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Barking_Town_Hall&amp;action=edit&amp;redlink=1" title="Barking Town Hall (page does not exist)">
               Town Hall
              </a>
              , 1 Town Square
             </td>
             <td>
              13.93
             </td>
             <td>
              194,352
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5607_N_0.1557_E_region:GB_type:city&amp;title=Barking+and+Dagenham">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°33′39″N
                  </span>
                  <span class="longitude">
                   0°09′21″E
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.5607°N 0.1557°E
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.5607; 0.1557
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Barking and Dagenham
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              25
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Barnet" title="London Borough of Barnet">
               Barnet
              </a>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Barnet_London_Borough_Council" title="Barnet London Borough Council">
               Barnet London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Conservative_Party_(UK)" title="Conservative Party (UK)">
               Conservative
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=North_London_Business_Park&amp;action=edit&amp;redlink=1" title="North London Business Park (page does not exist)">
               North London Business Park
              </a>
              , Oakleigh Road South
             </td>
             <td>
              33.49
             </td>
             <td>
              369,088
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.6252_N_0.1517_W_region:GB_type:city&amp;title=Barnet">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°37′31″N
                  </span>
                  <span class="longitude">
                   0°09′06″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.6252°N 0.1517°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.6252; -0.1517
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Barnet
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              31
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Bexley" title="London Borough of Bexley">
               Bexley
              </a>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Bexley_London_Borough_Council" title="Bexley London Borough Council">
               Bexley London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Conservative_Party_(UK)" title="Conservative Party (UK)">
               Conservative
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Civic_Offices&amp;action=edit&amp;redlink=1" title="Civic Offices (page does not exist)">
               Civic Offices
              </a>
              , 2 Watling Street
             </td>
             <td>
              23.38
             </td>
             <td>
              236,687
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4549_N_0.1505_E_region:GB_type:city&amp;title=Bexley">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°27′18″N
                  </span>
                  <span class="longitude">
                   0°09′02″E
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.4549°N 0.1505°E
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.4549; 0.1505
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Bexley
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              23
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Brent" title="London Borough of Brent">
               Brent
              </a>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Brent_London_Borough_Council" title="Brent London Borough Council">
               Brent London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a href="/wiki/Brent_Civic_Centre" title="Brent Civic Centre">
               Brent Civic Centre
              </a>
              , Engineers Way
             </td>
             <td>
              16.70
             </td>
             <td>
              317,264
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5588_N_0.2817_W_region:GB_type:city&amp;title=Brent">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°33′32″N
                  </span>
                  <span class="longitude">
                   0°16′54″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.5588°N 0.2817°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.5588; -0.2817
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Brent
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              12
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Bromley" title="London Borough of Bromley">
               Bromley
              </a>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Bromley_London_Borough_Council" title="Bromley London Borough Council">
               Bromley London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Conservative_Party_(UK)" title="Conservative Party (UK)">
               Conservative
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Bromley_Civic_Centre&amp;action=edit&amp;redlink=1" title="Bromley Civic Centre (page does not exist)">
               Civic Centre
              </a>
              , Stockwell Close
             </td>
             <td>
              57.97
             </td>
             <td>
              317,899
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4039_N_0.0198_E_region:GB_type:city&amp;title=Bromley">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°24′14″N
                  </span>
                  <span class="longitude">
                   0°01′11″E
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.4039°N 0.0198°E
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.4039; 0.0198
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Bromley
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              20
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Camden" title="London Borough of Camden">
               Camden
              </a>
             </td>
             <td>
              <img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/>
              <span style="display:none">
               Y
              </span>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Camden_London_Borough_Council" title="Camden London Borough Council">
               Camden London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a href="/wiki/Camden_Town_Hall" title="Camden Town Hall">
               Camden Town Hall
              </a>
              , Judd Street
             </td>
             <td>
              8.40
             </td>
             <td>
              229,719
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.529_N_0.1255_W_region:GB_type:city&amp;title=Camden">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°31′44″N
                  </span>
                  <span class="longitude">
                   0°07′32″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.5290°N 0.1255°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.5290; -0.1255
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Camden
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              11
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Croydon" title="London Borough of Croydon">
               Croydon
              </a>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Croydon_London_Borough_Council" title="Croydon London Borough Council">
               Croydon London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Bernard_Weatherill_House&amp;action=edit&amp;redlink=1" title="Bernard Weatherill House (page does not exist)">
               Bernard Weatherill House
              </a>
              , Mint Walk
             </td>
             <td>
              33.41
             </td>
             <td>
              372,752
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.3714_N_0.0977_W_region:GB_type:city&amp;title=Croydon">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°22′17″N
                  </span>
                  <span class="longitude">
                   0°05′52″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.3714°N 0.0977°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.3714; -0.0977
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Croydon
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              19
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Ealing" title="London Borough of Ealing">
               Ealing
              </a>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Ealing_London_Borough_Council" title="Ealing London Borough Council">
               Ealing London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Perceval_House,_Ealing&amp;action=edit&amp;redlink=1" title="Perceval House, Ealing (page does not exist)">
               Perceval House
              </a>
              , 14-16 Uxbridge Road
             </td>
             <td>
              21.44
             </td>
             <td>
              342,494
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.513_N_0.3089_W_region:GB_type:city&amp;title=Ealing">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°30′47″N
                  </span>
                  <span class="longitude">
                   0°18′32″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.5130°N 0.3089°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.5130; -0.3089
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Ealing
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              13
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Enfield" title="London Borough of Enfield">
               Enfield
              </a>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Enfield_London_Borough_Council" title="Enfield London Borough Council">
               Enfield London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Enfield_Civic_Centre&amp;action=edit&amp;redlink=1" title="Enfield Civic Centre (page does not exist)">
               Civic Centre
              </a>
              , Silver Street
             </td>
             <td>
              31.74
             </td>
             <td>
              320,524
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.6538_N_0.0799_W_region:GB_type:city&amp;title=Enfield">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°39′14″N
                  </span>
                  <span class="longitude">
                   0°04′48″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.6538°N 0.0799°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.6538; -0.0799
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Enfield
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              30
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/Royal_Borough_of_Greenwich" title="Royal Borough of Greenwich">
               Greenwich
              </a>
              <sup class="reference" id="cite_ref-3">
               <a href="#cite_note-3">
                [note 2]
               </a>
              </sup>
             </td>
             <td>
              <img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/>
              <span style="display:none">
               Y
              </span>
              <sup class="reference" id="cite_ref-note2_4-0">
               <a href="#cite_note-note2-4">
                [note 3]
               </a>
              </sup>
             </td>
             <td>
              <a class="mw-redirect" href="/wiki/Royal_borough" title="Royal borough">
               Royal
              </a>
             </td>
             <td>
              <a href="/wiki/Greenwich_London_Borough_Council" title="Greenwich London Borough Council">
               Greenwich London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a href="/wiki/Woolwich_Town_Hall" title="Woolwich Town Hall">
               Woolwich Town Hall
              </a>
              , Wellington Street
             </td>
             <td>
              18.28
             </td>
             <td>
              264,008
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4892_N_0.0648_E_region:GB_type:city&amp;title=Greenwich">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°29′21″N
                  </span>
                  <span class="longitude">
                   0°03′53″E
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.4892°N 0.0648°E
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.4892; 0.0648
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Greenwich
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              22
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Hackney" title="London Borough of Hackney">
               Hackney
              </a>
             </td>
             <td>
              <img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/>
              <span style="display:none">
               Y
              </span>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Hackney_London_Borough_Council" title="Hackney London Borough Council">
               Hackney London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Hackney_Town_Hall&amp;action=edit&amp;redlink=1" title="Hackney Town Hall (page does not exist)">
               Hackney Town Hall
              </a>
              , Mare Street
             </td>
             <td>
              7.36
             </td>
             <td>
              257,379
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.545_N_0.0553_W_region:GB_type:city&amp;title=Hackney">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°32′42″N
                  </span>
                  <span class="longitude">
                   0°03′19″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.5450°N 0.0553°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.5450; -0.0553
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Hackney
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              9
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Hammersmith_and_Fulham" title="London Borough of Hammersmith and Fulham">
               Hammersmith and Fulham
              </a>
              <sup class="reference" id="cite_ref-5">
               <a href="#cite_note-5">
                [note 4]
               </a>
              </sup>
             </td>
             <td>
              <img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/>
              <span style="display:none">
               Y
              </span>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Hammersmith_and_Fulham_London_Borough_Council" title="Hammersmith and Fulham London Borough Council">
               Hammersmith and Fulham London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Hammersmith_and_Fulham_Town_Hall&amp;action=edit&amp;redlink=1" title="Hammersmith and Fulham Town Hall (page does not exist)">
               Town Hall
              </a>
              , King Street
             </td>
             <td>
              6.33
             </td>
             <td>
              178,685
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4927_N_0.2339_W_region:GB_type:city&amp;title=Hammersmith+and+Fulham">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°29′34″N
                  </span>
                  <span class="longitude">
                   0°14′02″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.4927°N 0.2339°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.4927; -0.2339
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Hammersmith and Fulham
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              4
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Haringey" title="London Borough of Haringey">
               Haringey
              </a>
             </td>
             <td>
              <sup class="reference" id="cite_ref-note2_4-1">
               <a href="#cite_note-note2-4">
                [note 3]
               </a>
              </sup>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Haringey_London_Borough_Council" title="Haringey London Borough Council">
               Haringey London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Haringey_Civic_Centre&amp;action=edit&amp;redlink=1" title="Haringey Civic Centre (page does not exist)">
               Civic Centre
              </a>
              , High Road
             </td>
             <td>
              11.42
             </td>
             <td>
              263,386
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.6_N_0.1119_W_region:GB_type:city&amp;title=Haringey">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°36′00″N
                  </span>
                  <span class="longitude">
                   0°06′43″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.6000°N 0.1119°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.6000; -0.1119
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Haringey
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              29
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Harrow" title="London Borough of Harrow">
               Harrow
              </a>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Harrow_London_Borough_Council" title="Harrow London Borough Council">
               Harrow London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Harrow_Civic_Centre&amp;action=edit&amp;redlink=1" title="Harrow Civic Centre (page does not exist)">
               Civic Centre
              </a>
              , Station Road
             </td>
             <td>
              19.49
             </td>
             <td>
              243,372
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5898_N_0.3346_W_region:GB_type:city&amp;title=Harrow">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°35′23″N
                  </span>
                  <span class="longitude">
                   0°20′05″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.5898°N 0.3346°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.5898; -0.3346
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Harrow
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              32
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Havering" title="London Borough of Havering">
               Havering
              </a>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Havering_London_Borough_Council" title="Havering London Borough Council">
               Havering London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Conservative_Party_(UK)" title="Conservative Party (UK)">
               Conservative
              </a>
              (council
              <a href="/wiki/No_overall_control" title="No overall control">
               NOC
              </a>
              )
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Havering_Town_Hall&amp;action=edit&amp;redlink=1" title="Havering Town Hall (page does not exist)">
               Town Hall
              </a>
              , Main Road
             </td>
             <td>
              43.35
             </td>
             <td>
              242,080
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5812_N_0.1837_E_region:GB_type:city&amp;title=Havering">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°34′52″N
                  </span>
                  <span class="longitude">
                   0°11′01″E
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.5812°N 0.1837°E
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.5812; 0.1837
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Havering
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              24
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Hillingdon" title="London Borough of Hillingdon">
               Hillingdon
              </a>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Hillingdon_London_Borough_Council" title="Hillingdon London Borough Council">
               Hillingdon London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Conservative_Party_(UK)" title="Conservative Party (UK)">
               Conservative
              </a>
             </td>
             <td>
              <a href="/wiki/Hillingdon_Civic_Centre" title="Hillingdon Civic Centre">
               Civic Centre
              </a>
              , High Street
             </td>
             <td>
              44.67
             </td>
             <td>
              286,806
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5441_N_0.476_W_region:GB_type:city&amp;title=Hillingdon">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°32′39″N
                  </span>
                  <span class="longitude">
                   0°28′34″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.5441°N 0.4760°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.5441; -0.4760
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Hillingdon
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              33
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Hounslow" title="London Borough of Hounslow">
               Hounslow
              </a>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Hounslow_London_Borough_Council" title="Hounslow London Borough Council">
               Hounslow London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              Hounslow House, 7 Bath Road
             </td>
             <td>
              21.61
             </td>
             <td>
              262,407
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4746_N_0.368_W_region:GB_type:city&amp;title=Hounslow">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°28′29″N
                  </span>
                  <span class="longitude">
                   0°22′05″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.4746°N 0.3680°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.4746; -0.3680
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Hounslow
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              14
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Islington" title="London Borough of Islington">
               Islington
              </a>
             </td>
             <td>
              <img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/>
              <span style="display:none">
               Y
              </span>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Islington_London_Borough_Council" title="Islington London Borough Council">
               Islington London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Islington_Municipal_Offices&amp;action=edit&amp;redlink=1" title="Islington Municipal Offices (page does not exist)">
               Municipal Offices
              </a>
              , 222 Upper Street
             </td>
             <td>
              5.74
             </td>
             <td>
              215,667
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5416_N_0.1022_W_region:GB_type:city&amp;title=Islington">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°32′30″N
                  </span>
                  <span class="longitude">
                   0°06′08″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.5416°N 0.1022°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.5416; -0.1022
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Islington
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              10
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/Royal_Borough_of_Kensington_and_Chelsea" title="Royal Borough of Kensington and Chelsea">
               Kensington and Chelsea
              </a>
             </td>
             <td>
              <img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/>
              <span style="display:none">
               Y
              </span>
             </td>
             <td>
              <a class="mw-redirect" href="/wiki/Royal_borough" title="Royal borough">
               Royal
              </a>
             </td>
             <td>
              <a href="/wiki/Kensington_and_Chelsea_London_Borough_Council" title="Kensington and Chelsea London Borough Council">
               Kensington and Chelsea London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Conservative_Party_(UK)" title="Conservative Party (UK)">
               Conservative
              </a>
             </td>
             <td>
              <a href="/wiki/Kensington_Town_Hall,_London" title="Kensington Town Hall, London">
               The Town Hall
              </a>
              ,
              <a href="/wiki/Hornton_Street" title="Hornton Street">
               Hornton Street
              </a>
             </td>
             <td>
              4.68
             </td>
             <td>
              155,594
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.502_N_0.1947_W_region:GB_type:city&amp;title=Kensington+and+Chelsea">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°30′07″N
                  </span>
                  <span class="longitude">
                   0°11′41″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.5020°N 0.1947°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.5020; -0.1947
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Kensington and Chelsea
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              3
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/Royal_Borough_of_Kingston_upon_Thames" title="Royal Borough of Kingston upon Thames">
               Kingston upon Thames
              </a>
             </td>
             <td>
             </td>
             <td>
              <a class="mw-redirect" href="/wiki/Royal_borough" title="Royal borough">
               Royal
              </a>
             </td>
             <td>
              <a href="/wiki/Kingston_upon_Thames_London_Borough_Council" title="Kingston upon Thames London Borough Council">
               Kingston upon Thames London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Liberal_Democrats_(UK)" title="Liberal Democrats (UK)">
               Liberal Democrat
              </a>
             </td>
             <td>
              <a href="/wiki/Kingston_upon_Thames_Guildhall" title="Kingston upon Thames Guildhall">
               Guildhall
              </a>
              , High Street
             </td>
             <td>
              14.38
             </td>
             <td>
              166,793
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4085_N_0.3064_W_region:GB_type:city&amp;title=Kingston+upon+Thames">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°24′31″N
                  </span>
                  <span class="longitude">
                   0°18′23″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.4085°N 0.3064°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.4085; -0.3064
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Kingston upon Thames
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              16
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Lambeth" title="London Borough of Lambeth">
               Lambeth
              </a>
             </td>
             <td>
              <img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/>
              <span style="display:none">
               Y
              </span>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Lambeth_London_Borough_Council" title="Lambeth London Borough Council">
               Lambeth London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a href="/wiki/Lambeth_Town_Hall" title="Lambeth Town Hall">
               Lambeth Town Hall
              </a>
              , Brixton Hill
             </td>
             <td>
              10.36
             </td>
             <td>
              314,242
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4607_N_0.1163_W_region:GB_type:city&amp;title=Lambeth">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°27′39″N
                  </span>
                  <span class="longitude">
                   0°06′59″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.4607°N 0.1163°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.4607; -0.1163
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Lambeth
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              6
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Lewisham" title="London Borough of Lewisham">
               Lewisham
              </a>
             </td>
             <td>
              <img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/>
              <span style="display:none">
               Y
              </span>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Lewisham_London_Borough_Council" title="Lewisham London Borough Council">
               Lewisham London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Lewisham_Town_Hall&amp;action=edit&amp;redlink=1" title="Lewisham Town Hall (page does not exist)">
               Town Hall
              </a>
              , 1 Catford Road
             </td>
             <td>
              13.57
             </td>
             <td>
              286,180
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4452_N_0.0209_W_region:GB_type:city&amp;title=Lewisham">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°26′43″N
                  </span>
                  <span class="longitude">
                   0°01′15″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.4452°N 0.0209°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.4452; -0.0209
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Lewisham
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              21
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Merton" title="London Borough of Merton">
               Merton
              </a>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Merton_London_Borough_Council" title="Merton London Borough Council">
               Merton London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Merton_Civic_Centre&amp;action=edit&amp;redlink=1" title="Merton Civic Centre (page does not exist)">
               Civic Centre
              </a>
              , London Road
             </td>
             <td>
              14.52
             </td>
             <td>
              203,223
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4014_N_0.1958_W_region:GB_type:city&amp;title=Merton">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°24′05″N
                  </span>
                  <span class="longitude">
                   0°11′45″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.4014°N 0.1958°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.4014; -0.1958
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Merton
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              17
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Newham" title="London Borough of Newham">
               Newham
              </a>
             </td>
             <td>
              <sup class="reference" id="cite_ref-note2_4-2">
               <a href="#cite_note-note2-4">
                [note 3]
               </a>
              </sup>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Newham_London_Borough_Council" title="Newham London Borough Council">
               Newham London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Newham_Dockside&amp;action=edit&amp;redlink=1" title="Newham Dockside (page does not exist)">
               Newham Dockside
              </a>
              , 1000 Dockside Road
             </td>
             <td>
              13.98
             </td>
             <td>
              318,227
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5077_N_0.0469_E_region:GB_type:city&amp;title=Newham">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°30′28″N
                  </span>
                  <span class="longitude">
                   0°02′49″E
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.5077°N 0.0469°E
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.5077; 0.0469
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Newham
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              27
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Redbridge" title="London Borough of Redbridge">
               Redbridge
              </a>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Redbridge_London_Borough_Council" title="Redbridge London Borough Council">
               Redbridge London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Redbridge_Town_Hall&amp;action=edit&amp;redlink=1" title="Redbridge Town Hall (page does not exist)">
               Town Hall
              </a>
              , 128-142 High Road
             </td>
             <td>
              21.78
             </td>
             <td>
              288,272
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.559_N_0.0741_E_region:GB_type:city&amp;title=Redbridge">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°33′32″N
                  </span>
                  <span class="longitude">
                   0°04′27″E
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.5590°N 0.0741°E
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.5590; 0.0741
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Redbridge
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              26
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Richmond_upon_Thames" title="London Borough of Richmond upon Thames">
               Richmond upon Thames
              </a>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Richmond_upon_Thames_London_Borough_Council" title="Richmond upon Thames London Borough Council">
               Richmond upon Thames London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Liberal_Democrats_(UK)" title="Liberal Democrats (UK)">
               Liberal Democrat
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Richmond_upon_Thames_Civic_Centre&amp;action=edit&amp;redlink=1" title="Richmond upon Thames Civic Centre (page does not exist)">
               Civic Centre
              </a>
              , 44 York Street
             </td>
             <td>
              22.17
             </td>
             <td>
              191,365
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4479_N_0.326_W_region:GB_type:city&amp;title=Richmond+upon+Thames">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°26′52″N
                  </span>
                  <span class="longitude">
                   0°19′34″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.4479°N 0.3260°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.4479; -0.3260
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Richmond upon Thames
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              15
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Southwark" title="London Borough of Southwark">
               Southwark
              </a>
             </td>
             <td>
              <img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/>
              <span style="display:none">
               Y
              </span>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Southwark_London_Borough_Council" title="Southwark London Borough Council">
               Southwark London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=160_Tooley_Street&amp;action=edit&amp;redlink=1" title="160 Tooley Street (page does not exist)">
               160 Tooley Street
              </a>
             </td>
             <td>
              11.14
             </td>
             <td>
              298,464
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5035_N_0.0804_W_region:GB_type:city&amp;title=Southwark">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°30′13″N
                  </span>
                  <span class="longitude">
                   0°04′49″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.5035°N 0.0804°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.5035; -0.0804
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Southwark
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              7
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Sutton" title="London Borough of Sutton">
               Sutton
              </a>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Sutton_London_Borough_Council" title="Sutton London Borough Council">
               Sutton London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Liberal_Democrats_(UK)" title="Liberal Democrats (UK)">
               Liberal Democrat
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Sutton_Civic_Offices&amp;action=edit&amp;redlink=1" title="Sutton Civic Offices (page does not exist)">
               Civic Offices
              </a>
              , St Nicholas Way
             </td>
             <td>
              16.93
             </td>
             <td>
              195,914
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.3618_N_0.1945_W_region:GB_type:city&amp;title=Sutton">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°21′42″N
                  </span>
                  <span class="longitude">
                   0°11′40″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.3618°N 0.1945°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.3618; -0.1945
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Sutton
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              18
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Tower_Hamlets" title="London Borough of Tower Hamlets">
               Tower Hamlets
              </a>
             </td>
             <td>
              <img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/>
              <span style="display:none">
               Y
              </span>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Tower_Hamlets_London_Borough_Council" title="Tower Hamlets London Borough Council">
               Tower Hamlets London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Tower_Hamlets_Town_Hall&amp;action=edit&amp;redlink=1" title="Tower Hamlets Town Hall (page does not exist)">
               Town Hall
              </a>
              , Mulberry Place, 5 Clove Crescent
             </td>
             <td>
              7.63
             </td>
             <td>
              272,890
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5099_N_0.0059_W_region:GB_type:city&amp;title=Tower+Hamlets">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°30′36″N
                  </span>
                  <span class="longitude">
                   0°00′21″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.5099°N 0.0059°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.5099; -0.0059
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Tower Hamlets
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              8
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Waltham_Forest" title="London Borough of Waltham Forest">
               Waltham Forest
              </a>
             </td>
             <td>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Waltham_Forest_London_Borough_Council" title="Waltham Forest London Borough Council">
               Waltham Forest London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">
               Labour
              </a>
             </td>
             <td>
              <a href="/wiki/Waltham_Forest_Town_Hall" title="Waltham Forest Town Hall">
               Waltham Forest Town Hall
              </a>
              , Forest Road
             </td>
             <td>
              14.99
             </td>
             <td>
              265,797
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5908_N_0.0134_W_region:GB_type:city&amp;title=Waltham+Forest">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°35′27″N
                  </span>
                  <span class="longitude">
                   0°00′48″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.5908°N 0.0134°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.5908; -0.0134
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Waltham Forest
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              28
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/London_Borough_of_Wandsworth" title="London Borough of Wandsworth">
               Wandsworth
              </a>
             </td>
             <td>
              <img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/>
              <span style="display:none">
               Y
              </span>
             </td>
             <td>
             </td>
             <td>
              <a href="/wiki/Wandsworth_London_Borough_Council" title="Wandsworth London Borough Council">
               Wandsworth London Borough Council
              </a>
             </td>
             <td>
              <a href="/wiki/Conservative_Party_(UK)" title="Conservative Party (UK)">
               Conservative
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Wandsworth_Town_Hall&amp;action=edit&amp;redlink=1" title="Wandsworth Town Hall (page does not exist)">
               The Town Hall
              </a>
              ,
              <a href="/wiki/Wandsworth_High_Street" title="Wandsworth High Street">
               Wandsworth High Street
              </a>
             </td>
             <td>
              13.23
             </td>
             <td>
              310,516
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4567_N_0.191_W_region:GB_type:city&amp;title=Wandsworth">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°27′24″N
                  </span>
                  <span class="longitude">
                   0°11′28″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.4567°N 0.1910°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.4567; -0.1910
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Wandsworth
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              5
             </td>
            </tr>
            <tr>
             <td>
              <a href="/wiki/City_of_Westminster" title="City of Westminster">
               Westminster
              </a>
             </td>
             <td>
              <img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/>
              <span style="display:none">
               Y
              </span>
             </td>
             <td>
              <a href="/wiki/City_status_in_the_United_Kingdom" title="City status in the United Kingdom">
               City
              </a>
             </td>
             <td>
              <a href="/wiki/Westminster_City_Council" title="Westminster City Council">
               Westminster City Council
              </a>
             </td>
             <td>
              <a href="/wiki/Conservative_Party_(UK)" title="Conservative Party (UK)">
               Conservative
              </a>
             </td>
             <td>
              <a class="new" href="/w/index.php?title=Westminster_City_Hall&amp;action=edit&amp;redlink=1" title="Westminster City Hall (page does not exist)">
               Westminster City Hall
              </a>
              , 64 Victoria Street
             </td>
             <td>
              8.29
             </td>
             <td>
              226,841
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4973_N_0.1372_W_region:GB_type:city&amp;title=Westminster">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°29′50″N
                  </span>
                  <span class="longitude">
                   0°08′14″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.4973°N 0.1372°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.4973; -0.1372
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    Westminster
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              2
             </td>
            </tr>
           </tbody>
          </table>
          <h2>
           <span class="mw-headline" id="City_of_London">
            City of London
           </span>
           <span class="mw-editsection">
            <span class="mw-editsection-bracket">
             [
            </span>
            <a href="/w/index.php?title=List_of_London_boroughs&amp;action=edit&amp;section=2" title="Edit section: City of London">
             edit
            </a>
            <span class="mw-editsection-bracket">
             ]
            </span>
           </span>
          </h2>
          <p>
           The
           <a href="/wiki/City_of_London" title="City of London">
            City of London
           </a>
           is the 33rd principal division of Greater London but it is not a London borough.
          </p>
          <table class="wikitable sortable" style="font-size:95%" width="100%">
           <tbody>
            <tr>
             <th width="100px">
              Borough
             </th>
             <th>
              Inner
             </th>
             <th width="100px">
              Status
             </th>
             <th>
              Local authority
             </th>
             <th>
              Political control
             </th>
             <th width="120px">
              Headquarters
             </th>
             <th>
              Area (sq mi)
             </th>
             <th>
              Population
              <br/>
              (2011 est)
             </th>
             <th width="20px">
              Co-ordinates
             </th>
             <th>
              <span style="background:#67BCD3">
               Nr. in
               <br/>
               map
              </span>
             </th>
            </tr>
            <tr>
             <td>
              <a href="/wiki/City_of_London" title="City of London">
               City of London
              </a>
             </td>
             <td>
              (
              <img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/>
              <span style="display:none">
               Y
              </span>
              )
              <br/>
              <sup class="reference" id="cite_ref-6">
               <a href="#cite_note-6">
                [note 5]
               </a>
              </sup>
             </td>
             <td>
              <i>
               <a href="/wiki/Sui_generis" title="Sui generis">
                Sui generis
               </a>
              </i>
              ;
              <br/>
              <a href="/wiki/City_status_in_the_United_Kingdom" title="City status in the United Kingdom">
               City
              </a>
              ;
              <br/>
              <a href="/wiki/Ceremonial_counties_of_England" title="Ceremonial counties of England">
               Ceremonial county
              </a>
             </td>
             <td>
              <a class="mw-redirect" href="/wiki/Corporation_of_London" title="Corporation of London">
               Corporation of London
              </a>
              ;
              <br/>
              <a href="/wiki/Inner_Temple" title="Inner Temple">
               Inner Temple
              </a>
              ;
              <br/>
              <a href="/wiki/Middle_Temple" title="Middle Temple">
               Middle Temple
              </a>
             </td>
             <td>
              ?
             </td>
             <td>
              <a href="/wiki/Guildhall,_London" title="Guildhall, London">
               Guildhall
              </a>
             </td>
             <td>
              1.12
             </td>
             <td>
              7,000
             </td>
             <td>
              <span class="plainlinks nourlexpansion">
               <a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5155_N_0.0922_W_region:GB_type:city&amp;title=City+of+London">
                <span class="geo-nondefault">
                 <span class="geo-dms" title="Maps, aerial photos, and other data for this location">
                  <span class="latitude">
                   51°30′56″N
                  </span>
                  <span class="longitude">
                   0°05′32″W
                  </span>
                 </span>
                </span>
                <span class="geo-multi-punct">
                 ﻿ / ﻿
                </span>
                <span class="geo-default">
                 <span class="vcard">
                  <span class="geo-dec" title="Maps, aerial photos, and other data for this location">
                   51.5155°N 0.0922°W
                  </span>
                  <span style="display:none">
                   ﻿ /
                   <span class="geo">
                    51.5155; -0.0922
                   </span>
                  </span>
                  <span style="display:none">
                   ﻿ (
                   <span class="fn org">
                    City of London
                   </span>
                   )
                  </span>
                 </span>
                </span>
               </a>
              </span>
             </td>
             <td>
              1
             </td>
            </tr>
           </tbody>
          </table>
          <table class="noprint infobox" id="GeoGroup" style="width: 23em; font-size: 88%; line-height: 1.5em">
           <tbody>
            <tr>
             <td>
              <b>
               Map all coordinates using:
              </b>
              <a class="external text" href="//tools.wmflabs.org/osm4wiki/cgi-bin/wiki/wiki-osm.pl?project=en&amp;article=List_of_London_boroughs">
               OpenStreetMap
              </a>
             </td>
            </tr>
            <tr>
             <td>
              <b>
               Download coordinates as:
              </b>
              <a class="external text" href="//tools.wmflabs.org/kmlexport?article=List_of_London_boroughs">
               KML
              </a>
              <b>
               ·
              </b>
              <a class="external text" href="http://tripgang.com/kml2gpx/http%3A%2F%2Ftools.wmflabs.org%2Fkmlexport%3Farticle%3DList_of_London_boroughs?gpx=1" rel="nofollow">
               GPX
              </a>
             </td>
            </tr>
           </tbody>
          </table>
          <h2>
           <span class="mw-headline" id="See_also">
            See also
           </span>
           <span class="mw-editsection">
            <span class="mw-editsection-bracket">
             [
            </span>
            <a href="/w/index.php?title=List_of_London_boroughs&amp;action=edit&amp;section=3" title="Edit section: See also">
             edit
            </a>
            <span class="mw-editsection-bracket">
             ]
            </span>
           </span>
          </h2>
          <ul>
           <li>
            <a class="mw-redirect" href="/wiki/Political_make-up_of_London_borough_councils" title="Political make-up of London borough councils">
             Political make-up of London borough councils
            </a>
           </li>
           <li>
            <a href="/wiki/List_of_areas_of_London" title="List of areas of London">
             List of areas of London
            </a>
           </li>
           <li>
            <a href="/wiki/Subdivisions_of_England" title="Subdivisions of England">
             Subdivisions of England
            </a>
           </li>
          </ul>
          <h2>
           <span class="mw-headline" id="Notes">
            Notes
           </span>
           <span class="mw-editsection">
            <span class="mw-editsection-bracket">
             [
            </span>
            <a href="/w/index.php?title=List_of_London_boroughs&amp;action=edit&amp;section=4" title="Edit section: Notes">
             edit
            </a>
            <span class="mw-editsection-bracket">
             ]
            </span>
           </span>
          </h2>
          <div class="reflist" style="list-style-type: decimal;">
           <div class="mw-references-wrap">
            <ol class="references">
             <li id="cite_note-2">
              <span class="mw-cite-backlink">
               <b>
                <a href="#cite_ref-2">
                 ^
                </a>
               </b>
              </span>
              <span class="reference-text">
               Renamed from London Borough of Barking 1 January 1980.
               <cite class="citation magazine" id="CITEREFGazette48021">
                <a class="external text" href="https://www.thegazette.co.uk/London/issue/48021/page/15280" rel="nofollow">
                 "No. 48021"
                </a>
                .
                <i>
                 <a href="/wiki/The_London_Gazette" title="The London Gazette">
                  The London Gazette
                 </a>
                </i>
                . 4 December 1979. p. 15280.
               </cite>
               <span class="Z3988" title="ctx_ver=Z39.88-2004&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Ajournal&amp;rft.genre=article&amp;rft.jtitle=The+London+Gazette&amp;rft.atitle=No.+48021&amp;rft.pages=15280&amp;rft.date=1979-12-04&amp;rft_id=https%3A%2F%2Fwww.thegazette.co.uk%2FLondon%2Fissue%2F48021%2Fpage%2F15280&amp;rfr_id=info%3Asid%2Fen.wikipedia.org%3AList+of+London+boroughs"/>
               <style data-mw-deduplicate="TemplateStyles:r935243608">
                .mw-parser-output cite.citation{font-style:inherit}.mw-parser-output .citation q{quotes:"\"""\"""'""'"}.mw-parser-output .id-lock-free a,.mw-parser-output .citation .cs1-lock-free a{background:url("//upload.wikimedia.org/wikipedia/commons/thumb/6/65/Lock-green.svg/9px-Lock-green.svg.png")no-repeat;background-position:right .1em center}.mw-parser-output .id-lock-limited a,.mw-parser-output .id-lock-registration a,.mw-parser-output .citation .cs1-lock-limited a,.mw-parser-output .citation .cs1-lock-registration a{background:url("//upload.wikimedia.org/wikipedia/commons/thumb/d/d6/Lock-gray-alt-2.svg/9px-Lock-gray-alt-2.svg.png")no-repeat;background-position:right .1em center}.mw-parser-output .id-lock-subscription a,.mw-parser-output .citation .cs1-lock-subscription a{background:url("//upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Lock-red-alt-2.svg/9px-Lock-red-alt-2.svg.png")no-repeat;background-position:right .1em center}.mw-parser-output .cs1-subscription,.mw-parser-output .cs1-registration{color:#555}.mw-parser-output .cs1-subscription span,.mw-parser-output .cs1-registration span{border-bottom:1px dotted;cursor:help}.mw-parser-output .cs1-ws-icon a{background:url("//upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Wikisource-logo.svg/12px-Wikisource-logo.svg.png")no-repeat;background-position:right .1em center}.mw-parser-output code.cs1-code{color:inherit;background:inherit;border:inherit;padding:inherit}.mw-parser-output .cs1-hidden-error{display:none;font-size:100%}.mw-parser-output .cs1-visible-error{font-size:100%}.mw-parser-output .cs1-maint{display:none;color:#33aa33;margin-left:0.3em}.mw-parser-output .cs1-subscription,.mw-parser-output .cs1-registration,.mw-parser-output .cs1-format{font-size:95%}.mw-parser-output .cs1-kern-left,.mw-parser-output .cs1-kern-wl-left{padding-left:0.2em}.mw-parser-output .cs1-kern-right,.mw-parser-output .cs1-kern-wl-right{padding-right:0.2em}
               </style>
              </span>
             </li>
             <li id="cite_note-3">
              <span class="mw-cite-backlink">
               <b>
                <a href="#cite_ref-3">
                 ^
                </a>
               </b>
              </span>
              <span class="reference-text">
               Royal borough from 2012
              </span>
             </li>
             <li id="cite_note-note2-4">
              <span class="mw-cite-backlink">
               ^
               <a href="#cite_ref-note2_4-0">
                <sup>
                 <i>
                  <b>
                   a
                  </b>
                 </i>
                </sup>
               </a>
               <a href="#cite_ref-note2_4-1">
                <sup>
                 <i>
                  <b>
                   b
                  </b>
                 </i>
                </sup>
               </a>
               <a href="#cite_ref-note2_4-2">
                <sup>
                 <i>
                  <b>
                   c
                  </b>
                 </i>
                </sup>
               </a>
              </span>
              <span class="reference-text">
               Haringey and Newham are Inner London for statistics; Greenwich is Outer London for statistics
              </span>
             </li>
             <li id="cite_note-5">
              <span class="mw-cite-backlink">
               <b>
                <a href="#cite_ref-5">
                 ^
                </a>
               </b>
              </span>
              <span class="reference-text">
               Renamed from London Borough of Hammersmith 1 April 1979.
               <cite class="citation magazine" id="CITEREFGazette47771">
                <a class="external text" href="https://www.thegazette.co.uk/London/issue/47771/page/2095" rel="nofollow">
                 "No. 47771"
                </a>
                .
                <i>
                 <a href="/wiki/The_London_Gazette" title="The London Gazette">
                  The London Gazette
                 </a>
                </i>
                . 13 February 1979. p. 2095.
               </cite>
               <span class="Z3988" title="ctx_ver=Z39.88-2004&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Ajournal&amp;rft.genre=article&amp;rft.jtitle=The+London+Gazette&amp;rft.atitle=No.+47771&amp;rft.pages=2095&amp;rft.date=1979-02-13&amp;rft_id=https%3A%2F%2Fwww.thegazette.co.uk%2FLondon%2Fissue%2F47771%2Fpage%2F2095&amp;rfr_id=info%3Asid%2Fen.wikipedia.org%3AList+of+London+boroughs"/>
               <link href="mw-data:TemplateStyles:r935243608" rel="mw-deduplicated-inline-style"/>
              </span>
             </li>
             <li id="cite_note-6">
              <span class="mw-cite-backlink">
               <b>
                <a href="#cite_ref-6">
                 ^
                </a>
               </b>
              </span>
              <span class="reference-text">
               The City of London was not part of the
               <a href="/wiki/County_of_London" title="County of London">
                County of London
               </a>
               and is not a London Borough but can be counted to
               <a href="/wiki/Inner_London" title="Inner London">
                Inner London
               </a>
               .
              </span>
             </li>
            </ol>
           </div>
          </div>
          <h2>
           <span class="mw-headline" id="References">
            References
           </span>
           <span class="mw-editsection">
            <span class="mw-editsection-bracket">
             [
            </span>
            <a href="/w/index.php?title=List_of_London_boroughs&amp;action=edit&amp;section=5" title="Edit section: References">
             edit
            </a>
            <span class="mw-editsection-bracket">
             ]
            </span>
           </span>
          </h2>
          <div class="reflist" style="list-style-type: decimal;">
           <div class="mw-references-wrap">
            <ol class="references">
             <li id="cite_note-1">
              <span class="mw-cite-backlink">
               <b>
                <a href="#cite_ref-1">
                 ^
                </a>
               </b>
              </span>
              <span class="reference-text">
               <cite class="citation web">
                ONS (2 July 2010).
                <a class="external text" href="https://webarchive.nationalarchives.gov.uk/20160107070948/http://www.ons.gov.uk/ons/publications/re-reference-tables.html" rel="nofollow">
                 "Release Edition Reference Tables"
                </a>
                .
                <i>
                 Webarchive.nationalarchives.gov.uk
                </i>
                <span class="reference-accessdate">
                 . Retrieved
                 <span class="nowrap">
                  5 February
                 </span>
                 2019
                </span>
                .
               </cite>
               <span class="Z3988" title="ctx_ver=Z39.88-2004&amp;rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Ajournal&amp;rft.genre=unknown&amp;rft.jtitle=Webarchive.nationalarchives.gov.uk&amp;rft.atitle=Release+Edition+Reference+Tables&amp;rft.date=2010-07-02&amp;rft.au=ONS&amp;rft_id=https%3A%2F%2Fwebarchive.nationalarchives.gov.uk%2F20160107070948%2Fhttp%3A%2F%2Fwww.ons.gov.uk%2Fons%2Fpublications%2Fre-reference-tables.html&amp;rfr_id=info%3Asid%2Fen.wikipedia.org%3AList+of+London+boroughs"/>
               <link href="mw-data:TemplateStyles:r935243608" rel="mw-deduplicated-inline-style"/>
              </span>
             </li>
            </ol>
           </div>
          </div>
          <h2>
           <span class="mw-headline" id="External_links">
            External links
           </span>
           <span class="mw-editsection">
            <span class="mw-editsection-bracket">
             [
            </span>
            <a href="/w/index.php?title=List_of_London_boroughs&amp;action=edit&amp;section=6" title="Edit section: External links">
             edit
            </a>
            <span class="mw-editsection-bracket">
             ]
            </span>
           </span>
          </h2>
          <ul>
           <li>
            <a class="external text" href="https://web.archive.org/web/20101010011530/http://londoncouncils.gov.uk/londonlocalgovernment/londonboroughs.htm" rel="nofollow">
             London Councils: List of inner/outer London boroughs
            </a>
           </li>
           <li>
            <a class="external text" href="http://londonboroughsmap.co.uk/" rel="nofollow">
             London Boroughs Map
            </a>
           </li>
          </ul>
          <div aria-labelledby="Governance_of_Greater_London" class="navbox" role="navigation" style="padding:3px">
           <table class="nowraplinks hlist mw-collapsible mw-collapsed navbox-inner" style="border-spacing:0;background:transparent;color:inherit">
            <tbody>
             <tr>
              <th class="navbox-title" colspan="2" scope="col">
               <div class="plainlinks hlist navbar mini">
                <ul>
                 <li class="nv-view">
                  <a href="/wiki/Template:Governance_of_Greater_London" title="Template:Governance of Greater London">
                   <abbr style=";;background:none transparent;border:none;-moz-box-shadow:none;-webkit-box-shadow:none;box-shadow:none; padding:0;" title="View this template">
                    v
                   </abbr>
                  </a>
                 </li>
                 <li class="nv-talk">
                  <a href="/wiki/Template_talk:Governance_of_Greater_London" title="Template talk:Governance of Greater London">
                   <abbr style=";;background:none transparent;border:none;-moz-box-shadow:none;-webkit-box-shadow:none;box-shadow:none; padding:0;" title="Discuss this template">
                    t
                   </abbr>
                  </a>
                 </li>
                 <li class="nv-edit">
                  <a class="external text" href="https://en.wikipedia.org/w/index.php?title=Template:Governance_of_Greater_London&amp;action=edit">
                   <abbr style=";;background:none transparent;border:none;-moz-box-shadow:none;-webkit-box-shadow:none;box-shadow:none; padding:0;" title="Edit this template">
                    e
                   </abbr>
                  </a>
                 </li>
                </ul>
               </div>
               <div id="Governance_of_Greater_London" style="font-size:114%;margin:0 4em">
                Governance of
                <a href="/wiki/Greater_London" title="Greater London">
                 Greater London
                </a>
               </div>
              </th>
             </tr>
             <tr>
              <td class="navbox-abovebelow" colspan="2">
               <div id="*_City_of_London&amp;#10;*_London">
                <ul>
                 <li>
                  <a href="/wiki/City_of_London" title="City of London">
                   City of London
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London" title="London">
                   London
                  </a>
                 </li>
                </ul>
               </div>
              </td>
             </tr>
             <tr>
              <th class="navbox-group" scope="row" style="width:1%">
               Regional
              </th>
              <td class="navbox-list navbox-odd" style="text-align:left;border-left-width:2px;border-left-style:solid;width:100%;padding:0px">
               <div style="padding:0em 0.25em">
                <ul>
                 <li>
                  <b>
                   <a href="/wiki/Greater_London_Authority" title="Greater London Authority">
                    Greater London Authority
                   </a>
                   :
                  </b>
                  <a href="/wiki/London_Assembly" title="London Assembly">
                   London Assembly
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/Mayor_of_London" title="Mayor of London">
                   Mayor of London
                  </a>
                 </li>
                </ul>
               </div>
              </td>
             </tr>
             <tr>
              <th class="navbox-group" scope="row" style="width:1%">
               <a href="/wiki/London_boroughs" title="London boroughs">
                Boroughs
               </a>
               <br/>
               (
               <a class="mw-selflink selflink">
                list
               </a>
               )
              </th>
              <td class="navbox-list navbox-even" style="text-align:left;border-left-width:2px;border-left-style:solid;width:100%;padding:0px">
               <div style="padding:0em 0.25em">
                <ul>
                 <li>
                  <b>
                   <a href="/wiki/London_Councils" title="London Councils">
                    London Councils
                   </a>
                   :
                  </b>
                  <a href="/wiki/London_Borough_of_Barking_and_Dagenham" title="London Borough of Barking and Dagenham">
                   Barking and Dagenham
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Barnet" title="London Borough of Barnet">
                   Barnet
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Bexley" title="London Borough of Bexley">
                   Bexley
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Brent" title="London Borough of Brent">
                   Brent
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Bromley" title="London Borough of Bromley">
                   Bromley
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Camden" title="London Borough of Camden">
                   Camden
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Croydon" title="London Borough of Croydon">
                   Croydon
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Ealing" title="London Borough of Ealing">
                   Ealing
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Enfield" title="London Borough of Enfield">
                   Enfield
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/Royal_Borough_of_Greenwich" title="Royal Borough of Greenwich">
                   Greenwich
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Hackney" title="London Borough of Hackney">
                   Hackney
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Hammersmith_and_Fulham" title="London Borough of Hammersmith and Fulham">
                   Hammersmith and Fulham
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Haringey" title="London Borough of Haringey">
                   Haringey
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Harrow" title="London Borough of Harrow">
                   Harrow
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Havering" title="London Borough of Havering">
                   Havering
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Hillingdon" title="London Borough of Hillingdon">
                   Hillingdon
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Hounslow" title="London Borough of Hounslow">
                   Hounslow
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Islington" title="London Borough of Islington">
                   Islington
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/Royal_Borough_of_Kensington_and_Chelsea" title="Royal Borough of Kensington and Chelsea">
                   Kensington and Chelsea
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/Royal_Borough_of_Kingston_upon_Thames" title="Royal Borough of Kingston upon Thames">
                   Kingston upon Thames
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Lambeth" title="London Borough of Lambeth">
                   Lambeth
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Lewisham" title="London Borough of Lewisham">
                   Lewisham
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Merton" title="London Borough of Merton">
                   Merton
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Newham" title="London Borough of Newham">
                   Newham
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Redbridge" title="London Borough of Redbridge">
                   Redbridge
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Richmond_upon_Thames" title="London Borough of Richmond upon Thames">
                   Richmond upon Thames
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Southwark" title="London Borough of Southwark">
                   Southwark
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Sutton" title="London Borough of Sutton">
                   Sutton
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Tower_Hamlets" title="London Borough of Tower Hamlets">
                   Tower Hamlets
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Waltham_Forest" title="London Borough of Waltham Forest">
                   Waltham Forest
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/London_Borough_of_Wandsworth" title="London Borough of Wandsworth">
                   Wandsworth
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/City_of_Westminster" title="City of Westminster">
                   Westminster
                  </a>
                 </li>
                </ul>
               </div>
              </td>
             </tr>
             <tr>
              <th class="navbox-group" scope="row" style="width:1%">
               Ceremonial
              </th>
              <td class="navbox-list navbox-odd" style="text-align:left;border-left-width:2px;border-left-style:solid;width:100%;padding:0px">
               <div style="padding:0em 0.25em">
                <ul>
                 <li>
                  <a href="/wiki/Lord_Mayor_of_London" title="Lord Mayor of London">
                   Lord Mayor of the City of London
                  </a>
                 </li>
                 <li>
                  <a class="mw-redirect" href="/wiki/Lord_Lieutenant_of_Greater_London" title="Lord Lieutenant of Greater London">
                   Lord Lieutenant of Greater London
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/High_Sheriff_of_Greater_London" title="High Sheriff of Greater London">
                   High Sheriff of Greater London
                  </a>
                 </li>
                </ul>
               </div>
              </td>
             </tr>
             <tr>
              <th class="navbox-group" scope="row" style="width:1%">
               <a href="/wiki/History_of_local_government_in_London" title="History of local government in London">
                Historical
               </a>
              </th>
              <td class="navbox-list navbox-even" style="text-align:left;border-left-width:2px;border-left-style:solid;width:100%;padding:0px">
               <div style="padding:0em 0.25em">
                <ul>
                 <li>
                  <a href="/wiki/Metropolitan_Board_of_Works" title="Metropolitan Board of Works">
                   Metropolitan Board of Works
                  </a>
                  <span style="font-size:85%;">
                   (MBW) 1855–1889
                  </span>
                 </li>
                 <li>
                  <a href="/wiki/London_County_Council" title="London County Council">
                   London County Council
                  </a>
                  <span style="font-size:85%;">
                   (LCC) 1889–1965
                  </span>
                 </li>
                 <li>
                  <a href="/wiki/Greater_London_Council" title="Greater London Council">
                   Greater London Council
                  </a>
                  <span style="font-size:85%;">
                   (GLC) 1965–1986
                  </span>
                 </li>
                 <li>
                  <a href="/wiki/List_of_heads_of_London_government" title="List of heads of London government">
                   Leaders
                  </a>
                 </li>
                 <li>
                  <a href="/wiki/Sheriffs_of_the_City_of_London" title="Sheriffs of the City of London">
                   Sheriffs of the City of London
                  </a>
                 </li>
                </ul>
               </div>
              </td>
             </tr>
            </tbody>
           </table>
          </div>
          <!-- 
    NewPP limit report
    Parsed by mw1330
    Cached time: 20200311193737
    Cache expiry: 2592000
    Dynamic content: false
    Complications: [vary‐revision‐sha1]
    CPU time usage: 0.356 seconds
    Real time usage: 0.442 seconds
    Preprocessor visited node count: 5140/1000000
    Post‐expand include size: 79370/2097152 bytes
    Template argument size: 1081/2097152 bytes
    Highest expansion depth: 13/40
    Expensive parser function count: 2/500
    Unstrip recursion depth: 1/20
    Unstrip post‐expand size: 10566/5000000 bytes
    Number of Wikibase entities loaded: 0/400
    Lua time usage: 0.109/10.000 seconds
    Lua memory usage: 3.11 MB/50 MB
    -->
          <!--
    Transclusion expansion time report (%,ms,calls,template)
    100.00%  301.212      1 -total
     38.94%  117.279      2 Template:Reflist
     32.47%   97.807      2 Template:London_Gazette
     30.53%   91.950      2 Template:Cite_magazine
     19.10%   57.526     33 Template:Coord
     12.97%   39.078      1 Template:Use_dmy_dates
     10.87%   32.731     33 Template:English_district_control
      7.42%   22.349      1 Template:London
      5.55%   16.731      2 Template:DMCA
      5.23%   15.756      1 Template:Navbox
    -->
          <!-- Saved in parser cache with key enwiki:pcache:idhash:28092685-0!canonical and timestamp 20200311193736 and revision id 943613985
     -->
         </div>
         <noscript>
          <img alt="" height="1" src="//en.wikipedia.org/wiki/Special:CentralAutoLogin/start?type=1x1" style="border: none; position: absolute;" title="" width="1"/>
         </noscript>
        </div>
        <div class="printfooter">
         Retrieved from "
         <a dir="ltr" href="https://en.wikipedia.org/w/index.php?title=List_of_London_boroughs&amp;oldid=943613985">
          https://en.wikipedia.org/w/index.php?title=List_of_London_boroughs&amp;oldid=943613985
         </a>
         "
        </div>
        <div class="catlinks" data-mw="interface" id="catlinks">
         <div class="mw-normal-catlinks" id="mw-normal-catlinks">
          <a href="/wiki/Help:Category" title="Help:Category">
           Categories
          </a>
          :
          <ul>
           <li>
            <a href="/wiki/Category:London_boroughs" title="Category:London boroughs">
             London boroughs
            </a>
           </li>
           <li>
            <a href="/wiki/Category:Lists_of_places_in_London" title="Category:Lists of places in London">
             Lists of places in London
            </a>
           </li>
          </ul>
         </div>
         <div class="mw-hidden-catlinks mw-hidden-cats-hidden" id="mw-hidden-catlinks">
          Hidden categories:
          <ul>
           <li>
            <a href="/wiki/Category:Use_dmy_dates_from_August_2015" title="Category:Use dmy dates from August 2015">
             Use dmy dates from August 2015
            </a>
           </li>
           <li>
            <a href="/wiki/Category:Use_British_English_from_August_2015" title="Category:Use British English from August 2015">
             Use British English from August 2015
            </a>
           </li>
           <li>
            <a href="/wiki/Category:Lists_of_coordinates" title="Category:Lists of coordinates">
             Lists of coordinates
            </a>
           </li>
           <li>
            <a href="/wiki/Category:Geographic_coordinate_lists" title="Category:Geographic coordinate lists">
             Geographic coordinate lists
            </a>
           </li>
           <li>
            <a href="/wiki/Category:Articles_with_Geo" title="Category:Articles with Geo">
             Articles with Geo
            </a>
           </li>
          </ul>
         </div>
        </div>
        <div class="visualClear"/>
       </div>
      </div>
      <div id="mw-data-after-content">
       <div class="read-more-container"/>
      </div>
      <div id="mw-navigation">
       <h2>
        Navigation menu
       </h2>
       <div id="mw-head">
        <div aria-labelledby="p-personal-label" class="" id="p-personal" role="navigation">
         <h3 id="p-personal-label">
          Personal tools
         </h3>
         <ul>
          <li id="pt-anonuserpage">
           Not logged in
          </li>
          <li id="pt-anontalk">
           <a accesskey="n" href="/wiki/Special:MyTalk" title="Discussion about edits from this IP address [n]">
            Talk
           </a>
          </li>
          <li id="pt-anoncontribs">
           <a accesskey="y" href="/wiki/Special:MyContributions" title="A list of edits made from this IP address [y]">
            Contributions
           </a>
          </li>
          <li id="pt-createaccount">
           <a href="/w/index.php?title=Special:CreateAccount&amp;returnto=List+of+London+boroughs" title="You are encouraged to create an account and log in; however, it is not mandatory">
            Create account
           </a>
          </li>
          <li id="pt-login">
           <a accesskey="o" href="/w/index.php?title=Special:UserLogin&amp;returnto=List+of+London+boroughs" title="You're encouraged to log in; however, it's not mandatory. [o]">
            Log in
           </a>
          </li>
         </ul>
        </div>
        <div id="left-navigation">
         <div aria-labelledby="p-namespaces-label" class="vectorTabs " id="p-namespaces" role="navigation">
          <h3 id="p-namespaces-label">
           Namespaces
          </h3>
          <ul>
           <li class="selected" id="ca-nstab-main">
            <a accesskey="c" href="/wiki/List_of_London_boroughs" title="View the content page [c]">
             Article
            </a>
           </li>
           <li id="ca-talk">
            <a accesskey="t" href="/wiki/Talk:List_of_London_boroughs" rel="discussion" title="Discussion about the content page [t]">
             Talk
            </a>
           </li>
          </ul>
         </div>
         <div aria-labelledby="p-variants-label" class="vectorMenu emptyPortlet" id="p-variants" role="navigation">
          <input aria-labelledby="p-variants-label" class="vectorMenuCheckbox" type="checkbox"/>
          <h3 id="p-variants-label">
           <span>
            Variants
           </span>
          </h3>
          <ul class="menu">
          </ul>
         </div>
        </div>
        <div id="right-navigation">
         <div aria-labelledby="p-views-label" class="vectorTabs " id="p-views" role="navigation">
          <h3 id="p-views-label">
           Views
          </h3>
          <ul>
           <li class="collapsible selected" id="ca-view">
            <a href="/wiki/List_of_London_boroughs">
             Read
            </a>
           </li>
           <li class="collapsible" id="ca-edit">
            <a accesskey="e" href="/w/index.php?title=List_of_London_boroughs&amp;action=edit" title="Edit this page [e]">
             Edit
            </a>
           </li>
           <li class="collapsible" id="ca-history">
            <a accesskey="h" href="/w/index.php?title=List_of_London_boroughs&amp;action=history" title="Past revisions of this page [h]">
             View history
            </a>
           </li>
          </ul>
         </div>
         <div aria-labelledby="p-cactions-label" class="vectorMenu emptyPortlet" id="p-cactions" role="navigation">
          <input aria-labelledby="p-cactions-label" class="vectorMenuCheckbox" type="checkbox"/>
          <h3 id="p-cactions-label">
           <span>
            More
           </span>
          </h3>
          <ul class="menu">
          </ul>
         </div>
         <div id="p-search" role="search">
          <h3>
           <label for="searchInput">
            Search
           </label>
          </h3>
          <form action="/w/index.php" id="searchform">
           <div id="simpleSearch">
            <input accesskey="f" id="searchInput" name="search" placeholder="Search Wikipedia" title="Search Wikipedia [f]" type="search"/>
            <input name="title" type="hidden" value="Special:Search"/>
            <input class="searchButton mw-fallbackSearchButton" id="mw-searchButton" name="fulltext" title="Search Wikipedia for this text" type="submit" value="Search"/>
            <input class="searchButton" id="searchButton" name="go" title="Go to a page with this exact name if it exists" type="submit" value="Go"/>
           </div>
          </form>
         </div>
        </div>
       </div>
       <div id="mw-panel">
        <div id="p-logo" role="banner">
         <a class="mw-wiki-logo" href="/wiki/Main_Page" title="Visit the main page"/>
        </div>
        <div aria-labelledby="p-navigation-label" class="portal" id="p-navigation" role="navigation">
         <h3 id="p-navigation-label">
          Navigation
         </h3>
         <div class="body">
          <ul>
           <li id="n-mainpage-description">
            <a accesskey="z" href="/wiki/Main_Page" title="Visit the main page [z]">
             Main page
            </a>
           </li>
           <li id="n-contents">
            <a href="/wiki/Wikipedia:Contents" title="Guides to browsing Wikipedia">
             Contents
            </a>
           </li>
           <li id="n-featuredcontent">
            <a href="/wiki/Wikipedia:Featured_content" title="Featured content – the best of Wikipedia">
             Featured content
            </a>
           </li>
           <li id="n-currentevents">
            <a href="/wiki/Portal:Current_events" title="Find background information on current events">
             Current events
            </a>
           </li>
           <li id="n-randompage">
            <a accesskey="x" href="/wiki/Special:Random" title="Load a random article [x]">
             Random article
            </a>
           </li>
           <li id="n-sitesupport">
            <a href="https://donate.wikimedia.org/wiki/Special:FundraiserRedirector?utm_source=donate&amp;utm_medium=sidebar&amp;utm_campaign=C13_en.wikipedia.org&amp;uselang=en" title="Support us">
             Donate to Wikipedia
            </a>
           </li>
           <li id="n-shoplink">
            <a href="//shop.wikimedia.org" title="Visit the Wikipedia store">
             Wikipedia store
            </a>
           </li>
          </ul>
         </div>
        </div>
        <div aria-labelledby="p-interaction-label" class="portal" id="p-interaction" role="navigation">
         <h3 id="p-interaction-label">
          Interaction
         </h3>
         <div class="body">
          <ul>
           <li id="n-help">
            <a href="/wiki/Help:Contents" title="Guidance on how to use and edit Wikipedia">
             Help
            </a>
           </li>
           <li id="n-aboutsite">
            <a href="/wiki/Wikipedia:About" title="Find out about Wikipedia">
             About Wikipedia
            </a>
           </li>
           <li id="n-portal">
            <a href="/wiki/Wikipedia:Community_portal" title="About the project, what you can do, where to find things">
             Community portal
            </a>
           </li>
           <li id="n-recentchanges">
            <a accesskey="r" href="/wiki/Special:RecentChanges" title="A list of recent changes in the wiki [r]">
             Recent changes
            </a>
           </li>
           <li id="n-contactpage">
            <a href="//en.wikipedia.org/wiki/Wikipedia:Contact_us" title="How to contact Wikipedia">
             Contact page
            </a>
           </li>
          </ul>
         </div>
        </div>
        <div aria-labelledby="p-tb-label" class="portal" id="p-tb" role="navigation">
         <h3 id="p-tb-label">
          Tools
         </h3>
         <div class="body">
          <ul>
           <li id="t-whatlinkshere">
            <a accesskey="j" href="/wiki/Special:WhatLinksHere/List_of_London_boroughs" title="List of all English Wikipedia pages containing links to this page [j]">
             What links here
            </a>
           </li>
           <li id="t-recentchangeslinked">
            <a accesskey="k" href="/wiki/Special:RecentChangesLinked/List_of_London_boroughs" rel="nofollow" title="Recent changes in pages linked from this page [k]">
             Related changes
            </a>
           </li>
           <li id="t-upload">
            <a accesskey="u" href="/wiki/Wikipedia:File_Upload_Wizard" title="Upload files [u]">
             Upload file
            </a>
           </li>
           <li id="t-specialpages">
            <a accesskey="q" href="/wiki/Special:SpecialPages" title="A list of all special pages [q]">
             Special pages
            </a>
           </li>
           <li id="t-permalink">
            <a href="/w/index.php?title=List_of_London_boroughs&amp;oldid=943613985" title="Permanent link to this revision of the page">
             Permanent link
            </a>
           </li>
           <li id="t-info">
            <a href="/w/index.php?title=List_of_London_boroughs&amp;action=info" title="More information about this page">
             Page information
            </a>
           </li>
           <li id="t-wikibase">
            <a accesskey="g" href="https://www.wikidata.org/wiki/Special:EntityPage/Q6577004" title="Link to connected data repository item [g]">
             Wikidata item
            </a>
           </li>
           <li id="t-cite">
            <a href="/w/index.php?title=Special:CiteThisPage&amp;page=List_of_London_boroughs&amp;id=943613985" title="Information on how to cite this page">
             Cite this page
            </a>
           </li>
          </ul>
         </div>
        </div>
        <div aria-labelledby="p-coll-print_export-label" class="portal" id="p-coll-print_export" role="navigation">
         <h3 id="p-coll-print_export-label">
          Print/export
         </h3>
         <div class="body">
          <ul>
           <li id="coll-create_a_book">
            <a href="/w/index.php?title=Special:Book&amp;bookcmd=book_creator&amp;referer=List+of+London+boroughs">
             Create a book
            </a>
           </li>
           <li id="coll-download-as-rl">
            <a href="/w/index.php?title=Special:ElectronPdf&amp;page=List+of+London+boroughs&amp;action=show-download-screen">
             Download as PDF
            </a>
           </li>
           <li id="t-print">
            <a accesskey="p" href="/w/index.php?title=List_of_London_boroughs&amp;printable=yes" title="Printable version of this page [p]">
             Printable version
            </a>
           </li>
          </ul>
         </div>
        </div>
        <div aria-labelledby="p-lang-label" class="portal" id="p-lang" role="navigation">
         <h3 id="p-lang-label">
          Languages
         </h3>
         <div class="body">
          <ul>
           <li class="interlanguage-link interwiki-ru">
            <a class="interlanguage-link-target" href="https://ru.wikipedia.org/wiki/%D0%A1%D0%BF%D0%B8%D1%81%D0%BE%D0%BA_%D0%BB%D0%BE%D0%BD%D0%B4%D0%BE%D0%BD%D1%81%D0%BA%D0%B8%D1%85_%D0%B1%D0%BE%D1%80%D0%BE" hreflang="ru" lang="ru" title="Список лондонских боро – Russian">
             Русский
            </a>
           </li>
          </ul>
          <div class="after-portlet after-portlet-lang">
           <span class="wb-langlinks-edit wb-langlinks-link">
            <a class="wbc-editpage" href="https://www.wikidata.org/wiki/Special:EntityPage/Q6577004#sitelinks-wikipedia" title="Edit interlanguage links">
             Edit links
            </a>
           </span>
          </div>
         </div>
        </div>
       </div>
      </div>
      <div id="footer" role="contentinfo">
       <ul class="" id="footer-info">
        <li id="footer-info-lastmod">
         This page was last edited on 2 March 2020, at 22:20
         <span class="anonymous-show">
          (UTC)
         </span>
         .
        </li>
        <li id="footer-info-copyright">
         Text is available under the
         <a href="//en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License" rel="license">
          Creative Commons Attribution-ShareAlike License
         </a>
         <a href="//creativecommons.org/licenses/by-sa/3.0/" rel="license" style="display:none;"/>
         ;
    additional terms may apply.  By using this site, you agree to the
         <a href="//foundation.wikimedia.org/wiki/Terms_of_Use">
          Terms of Use
         </a>
         and
         <a href="//foundation.wikimedia.org/wiki/Privacy_policy">
          Privacy Policy
         </a>
         . Wikipedia® is a registered trademark of the
         <a href="//www.wikimediafoundation.org/">
          Wikimedia Foundation, Inc.
         </a>
         , a non-profit organization.
        </li>
       </ul>
       <ul class="" id="footer-places">
        <li id="footer-places-privacy">
         <a class="extiw" href="https://foundation.wikimedia.org/wiki/Privacy_policy" title="wmf:Privacy policy">
          Privacy policy
         </a>
        </li>
        <li id="footer-places-about">
         <a href="/wiki/Wikipedia:About" title="Wikipedia:About">
          About Wikipedia
         </a>
        </li>
        <li id="footer-places-disclaimer">
         <a href="/wiki/Wikipedia:General_disclaimer" title="Wikipedia:General disclaimer">
          Disclaimers
         </a>
        </li>
        <li id="footer-places-contact">
         <a href="//en.wikipedia.org/wiki/Wikipedia:Contact_us">
          Contact Wikipedia
         </a>
        </li>
        <li id="footer-places-developers">
         <a href="https://www.mediawiki.org/wiki/Special:MyLanguage/How_to_contribute">
          Developers
         </a>
        </li>
        <li id="footer-places-statslink">
         <a href="https://stats.wikimedia.org/#/en.wikipedia.org">
          Statistics
         </a>
        </li>
        <li id="footer-places-cookiestatement">
         <a href="https://foundation.wikimedia.org/wiki/Cookie_statement">
          Cookie statement
         </a>
        </li>
        <li id="footer-places-mobileview">
         <a class="noprint stopMobileRedirectToggle" href="//en.m.wikipedia.org/w/index.php?title=List_of_London_boroughs&amp;mobileaction=toggle_view_mobile">
          Mobile view
         </a>
        </li>
       </ul>
       <ul class="noprint" id="footer-icons">
        <li id="footer-copyrightico">
         <a href="https://wikimediafoundation.org/">
          <img alt="Wikimedia Foundation" height="31" src="/static/images/wikimedia-button.png" srcset="/static/images/wikimedia-button-1.5x.png 1.5x, /static/images/wikimedia-button-2x.png 2x" width="88"/>
         </a>
        </li>
        <li id="footer-poweredbyico">
         <a href="https://www.mediawiki.org/">
          <img alt="Powered by MediaWiki" height="31" src="/static/images/poweredby_mediawiki_88x31.png" srcset="/static/images/poweredby_mediawiki_132x47.png 1.5x, /static/images/poweredby_mediawiki_176x62.png 2x" width="88"/>
         </a>
        </li>
       </ul>
       <div style="clear: both;"/>
      </div>
      <script>
       (RLQ=window.RLQ||[]).push(function(){mw.config.set({"wgPageParseReport":{"limitreport":{"cputime":"0.356","walltime":"0.442","ppvisitednodes":{"value":5140,"limit":1000000},"postexpandincludesize":{"value":79370,"limit":2097152},"templateargumentsize":{"value":1081,"limit":2097152},"expansiondepth":{"value":13,"limit":40},"expensivefunctioncount":{"value":2,"limit":500},"unstrip-depth":{"value":1,"limit":20},"unstrip-size":{"value":10566,"limit":5000000},"entityaccesscount":{"value":0,"limit":400},"timingprofile":["100.00%  301.212      1 -total"," 38.94%  117.279      2 Template:Reflist"," 32.47%   97.807      2 Template:London_Gazette"," 30.53%   91.950      2 Template:Cite_magazine"," 19.10%   57.526     33 Template:Coord"," 12.97%   39.078      1 Template:Use_dmy_dates"," 10.87%   32.731     33 Template:English_district_control","  7.42%   22.349      1 Template:London","  5.55%   16.731      2 Template:DMCA","  5.23%   15.756      1 Template:Navbox"]},"scribunto":{"limitreport-timeusage":{"value":"0.109","limit":"10.000"},"limitreport-memusage":{"value":3259903,"limit":52428800}},"cachereport":{"origin":"mw1330","timestamp":"20200311193737","ttl":2592000,"transientcontent":false}}});});
      </script>
      <script type="application/ld+json">
       {"@context":"https:\/\/schema.org","@type":"Article","name":"List of London boroughs","url":"https:\/\/en.wikipedia.org\/wiki\/List_of_London_boroughs","sameAs":"http:\/\/www.wikidata.org\/entity\/Q6577004","mainEntity":"http:\/\/www.wikidata.org\/entity\/Q6577004","author":{"@type":"Organization","name":"Contributors to Wikimedia projects"},"publisher":{"@type":"Organization","name":"Wikimedia Foundation, Inc.","logo":{"@type":"ImageObject","url":"https:\/\/www.wikimedia.org\/static\/images\/wmf-hor-googpub.png"}},"datePublished":"2010-07-20T07:28:35Z","dateModified":"2020-03-02T22:20:38Z","headline":"Wikimedia list article"}
      </script>
      <script>
       (RLQ=window.RLQ||[]).push(function(){mw.config.set({"wgBackendResponseTime":100,"wgHostname":"mw1368"});});
      </script>
     </body>
    </html>
    

Extract the raw table inside the webpage.


```python
table = soup.find_all('table', {'class':'wikitable sortable'})
print(table)
```

    [<table class="wikitable sortable" style="font-size:100%" width="100%">
    <tbody><tr>
    <th>Borough
    </th>
    <th>Inner
    </th>
    <th>Status
    </th>
    <th>Local authority
    </th>
    <th>Political control
    </th>
    <th>Headquarters
    </th>
    <th>Area (sq mi)
    </th>
    <th>Population (2013 est)<sup class="reference" id="cite_ref-1"><a href="#cite_note-1">[1]</a></sup>
    </th>
    <th>Co-ordinates
    </th>
    <th><span style="background:#67BCD3"> Nr. in map </span>
    </th></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Barking_and_Dagenham" title="London Borough of Barking and Dagenham">Barking and Dagenham</a> <sup class="reference" id="cite_ref-2"><a href="#cite_note-2">[note 1]</a></sup>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Barking_and_Dagenham_London_Borough_Council" title="Barking and Dagenham London Borough Council">Barking and Dagenham London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Barking_Town_Hall&amp;action=edit&amp;redlink=1" title="Barking Town Hall (page does not exist)">Town Hall</a>, 1 Town Square
    </td>
    <td>13.93
    </td>
    <td>194,352
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5607_N_0.1557_E_region:GB_type:city&amp;title=Barking+and+Dagenham"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°33′39″N</span> <span class="longitude">0°09′21″E</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.5607°N 0.1557°E</span><span style="display:none">﻿ / <span class="geo">51.5607; 0.1557</span></span><span style="display:none">﻿ (<span class="fn org">Barking and Dagenham</span>)</span></span></span></a></span>
    </td>
    <td>25
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Barnet" title="London Borough of Barnet">Barnet</a>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Barnet_London_Borough_Council" title="Barnet London Borough Council">Barnet London Borough Council</a>
    </td>
    <td><a href="/wiki/Conservative_Party_(UK)" title="Conservative Party (UK)">Conservative</a>
    </td>
    <td><a class="new" href="/w/index.php?title=North_London_Business_Park&amp;action=edit&amp;redlink=1" title="North London Business Park (page does not exist)">North London Business Park</a>, Oakleigh Road South
    </td>
    <td>33.49
    </td>
    <td>369,088
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.6252_N_0.1517_W_region:GB_type:city&amp;title=Barnet"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°37′31″N</span> <span class="longitude">0°09′06″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.6252°N 0.1517°W</span><span style="display:none">﻿ / <span class="geo">51.6252; -0.1517</span></span><span style="display:none">﻿ (<span class="fn org">Barnet</span>)</span></span></span></a></span>
    </td>
    <td>31
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Bexley" title="London Borough of Bexley">Bexley</a>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Bexley_London_Borough_Council" title="Bexley London Borough Council">Bexley London Borough Council</a>
    </td>
    <td><a href="/wiki/Conservative_Party_(UK)" title="Conservative Party (UK)">Conservative</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Civic_Offices&amp;action=edit&amp;redlink=1" title="Civic Offices (page does not exist)">Civic Offices</a>, 2 Watling Street
    </td>
    <td>23.38
    </td>
    <td>236,687
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4549_N_0.1505_E_region:GB_type:city&amp;title=Bexley"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°27′18″N</span> <span class="longitude">0°09′02″E</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.4549°N 0.1505°E</span><span style="display:none">﻿ / <span class="geo">51.4549; 0.1505</span></span><span style="display:none">﻿ (<span class="fn org">Bexley</span>)</span></span></span></a></span>
    </td>
    <td>23
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Brent" title="London Borough of Brent">Brent</a>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Brent_London_Borough_Council" title="Brent London Borough Council">Brent London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a href="/wiki/Brent_Civic_Centre" title="Brent Civic Centre">Brent Civic Centre</a>, Engineers Way
    </td>
    <td>16.70
    </td>
    <td>317,264
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5588_N_0.2817_W_region:GB_type:city&amp;title=Brent"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°33′32″N</span> <span class="longitude">0°16′54″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.5588°N 0.2817°W</span><span style="display:none">﻿ / <span class="geo">51.5588; -0.2817</span></span><span style="display:none">﻿ (<span class="fn org">Brent</span>)</span></span></span></a></span>
    </td>
    <td>12
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Bromley" title="London Borough of Bromley">Bromley</a>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Bromley_London_Borough_Council" title="Bromley London Borough Council">Bromley London Borough Council</a>
    </td>
    <td><a href="/wiki/Conservative_Party_(UK)" title="Conservative Party (UK)">Conservative</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Bromley_Civic_Centre&amp;action=edit&amp;redlink=1" title="Bromley Civic Centre (page does not exist)">Civic Centre</a>, Stockwell Close
    </td>
    <td>57.97
    </td>
    <td>317,899
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4039_N_0.0198_E_region:GB_type:city&amp;title=Bromley"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°24′14″N</span> <span class="longitude">0°01′11″E</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.4039°N 0.0198°E</span><span style="display:none">﻿ / <span class="geo">51.4039; 0.0198</span></span><span style="display:none">﻿ (<span class="fn org">Bromley</span>)</span></span></span></a></span>
    </td>
    <td>20
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Camden" title="London Borough of Camden">Camden</a>
    </td>
    <td><img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/><span style="display:none">Y</span>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Camden_London_Borough_Council" title="Camden London Borough Council">Camden London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a href="/wiki/Camden_Town_Hall" title="Camden Town Hall">Camden Town Hall</a>, Judd Street
    </td>
    <td>8.40
    </td>
    <td>229,719
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.529_N_0.1255_W_region:GB_type:city&amp;title=Camden"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°31′44″N</span> <span class="longitude">0°07′32″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.5290°N 0.1255°W</span><span style="display:none">﻿ / <span class="geo">51.5290; -0.1255</span></span><span style="display:none">﻿ (<span class="fn org">Camden</span>)</span></span></span></a></span>
    </td>
    <td>11
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Croydon" title="London Borough of Croydon">Croydon</a>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Croydon_London_Borough_Council" title="Croydon London Borough Council">Croydon London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Bernard_Weatherill_House&amp;action=edit&amp;redlink=1" title="Bernard Weatherill House (page does not exist)">Bernard Weatherill House</a>, Mint Walk
    </td>
    <td>33.41
    </td>
    <td>372,752
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.3714_N_0.0977_W_region:GB_type:city&amp;title=Croydon"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°22′17″N</span> <span class="longitude">0°05′52″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.3714°N 0.0977°W</span><span style="display:none">﻿ / <span class="geo">51.3714; -0.0977</span></span><span style="display:none">﻿ (<span class="fn org">Croydon</span>)</span></span></span></a></span>
    </td>
    <td>19
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Ealing" title="London Borough of Ealing">Ealing</a>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Ealing_London_Borough_Council" title="Ealing London Borough Council">Ealing London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Perceval_House,_Ealing&amp;action=edit&amp;redlink=1" title="Perceval House, Ealing (page does not exist)">Perceval House</a>, 14-16 Uxbridge Road
    </td>
    <td>21.44
    </td>
    <td>342,494
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.513_N_0.3089_W_region:GB_type:city&amp;title=Ealing"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°30′47″N</span> <span class="longitude">0°18′32″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.5130°N 0.3089°W</span><span style="display:none">﻿ / <span class="geo">51.5130; -0.3089</span></span><span style="display:none">﻿ (<span class="fn org">Ealing</span>)</span></span></span></a></span>
    </td>
    <td>13
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Enfield" title="London Borough of Enfield">Enfield</a>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Enfield_London_Borough_Council" title="Enfield London Borough Council">Enfield London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Enfield_Civic_Centre&amp;action=edit&amp;redlink=1" title="Enfield Civic Centre (page does not exist)">Civic Centre</a>, Silver Street
    </td>
    <td>31.74
    </td>
    <td>320,524
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.6538_N_0.0799_W_region:GB_type:city&amp;title=Enfield"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°39′14″N</span> <span class="longitude">0°04′48″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.6538°N 0.0799°W</span><span style="display:none">﻿ / <span class="geo">51.6538; -0.0799</span></span><span style="display:none">﻿ (<span class="fn org">Enfield</span>)</span></span></span></a></span>
    </td>
    <td>30
    </td></tr>
    <tr>
    <td><a href="/wiki/Royal_Borough_of_Greenwich" title="Royal Borough of Greenwich">Greenwich</a> <sup class="reference" id="cite_ref-3"><a href="#cite_note-3">[note 2]</a></sup>
    </td>
    <td><img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/><span style="display:none">Y</span> <sup class="reference" id="cite_ref-note2_4-0"><a href="#cite_note-note2-4">[note 3]</a></sup>
    </td>
    <td><a class="mw-redirect" href="/wiki/Royal_borough" title="Royal borough">Royal</a>
    </td>
    <td><a href="/wiki/Greenwich_London_Borough_Council" title="Greenwich London Borough Council">Greenwich London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a href="/wiki/Woolwich_Town_Hall" title="Woolwich Town Hall">Woolwich Town Hall</a>, Wellington Street
    </td>
    <td>18.28
    </td>
    <td>264,008
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4892_N_0.0648_E_region:GB_type:city&amp;title=Greenwich"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°29′21″N</span> <span class="longitude">0°03′53″E</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.4892°N 0.0648°E</span><span style="display:none">﻿ / <span class="geo">51.4892; 0.0648</span></span><span style="display:none">﻿ (<span class="fn org">Greenwich</span>)</span></span></span></a></span>
    </td>
    <td>22
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Hackney" title="London Borough of Hackney">Hackney</a>
    </td>
    <td><img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/><span style="display:none">Y</span>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Hackney_London_Borough_Council" title="Hackney London Borough Council">Hackney London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Hackney_Town_Hall&amp;action=edit&amp;redlink=1" title="Hackney Town Hall (page does not exist)">Hackney Town Hall</a>, Mare Street
    </td>
    <td>7.36
    </td>
    <td>257,379
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.545_N_0.0553_W_region:GB_type:city&amp;title=Hackney"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°32′42″N</span> <span class="longitude">0°03′19″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.5450°N 0.0553°W</span><span style="display:none">﻿ / <span class="geo">51.5450; -0.0553</span></span><span style="display:none">﻿ (<span class="fn org">Hackney</span>)</span></span></span></a></span>
    </td>
    <td>9
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Hammersmith_and_Fulham" title="London Borough of Hammersmith and Fulham">Hammersmith and Fulham</a> <sup class="reference" id="cite_ref-5"><a href="#cite_note-5">[note 4]</a></sup>
    </td>
    <td><img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/><span style="display:none">Y</span>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Hammersmith_and_Fulham_London_Borough_Council" title="Hammersmith and Fulham London Borough Council">Hammersmith and Fulham London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Hammersmith_and_Fulham_Town_Hall&amp;action=edit&amp;redlink=1" title="Hammersmith and Fulham Town Hall (page does not exist)">Town Hall</a>, King Street
    </td>
    <td>6.33
    </td>
    <td>178,685
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4927_N_0.2339_W_region:GB_type:city&amp;title=Hammersmith+and+Fulham"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°29′34″N</span> <span class="longitude">0°14′02″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.4927°N 0.2339°W</span><span style="display:none">﻿ / <span class="geo">51.4927; -0.2339</span></span><span style="display:none">﻿ (<span class="fn org">Hammersmith and Fulham</span>)</span></span></span></a></span>
    </td>
    <td>4
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Haringey" title="London Borough of Haringey">Haringey</a>
    </td>
    <td><sup class="reference" id="cite_ref-note2_4-1"><a href="#cite_note-note2-4">[note 3]</a></sup>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Haringey_London_Borough_Council" title="Haringey London Borough Council">Haringey London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Haringey_Civic_Centre&amp;action=edit&amp;redlink=1" title="Haringey Civic Centre (page does not exist)">Civic Centre</a>, High Road
    </td>
    <td>11.42
    </td>
    <td>263,386
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.6_N_0.1119_W_region:GB_type:city&amp;title=Haringey"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°36′00″N</span> <span class="longitude">0°06′43″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.6000°N 0.1119°W</span><span style="display:none">﻿ / <span class="geo">51.6000; -0.1119</span></span><span style="display:none">﻿ (<span class="fn org">Haringey</span>)</span></span></span></a></span>
    </td>
    <td>29
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Harrow" title="London Borough of Harrow">Harrow</a>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Harrow_London_Borough_Council" title="Harrow London Borough Council">Harrow London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Harrow_Civic_Centre&amp;action=edit&amp;redlink=1" title="Harrow Civic Centre (page does not exist)">Civic Centre</a>, Station Road
    </td>
    <td>19.49
    </td>
    <td>243,372
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5898_N_0.3346_W_region:GB_type:city&amp;title=Harrow"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°35′23″N</span> <span class="longitude">0°20′05″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.5898°N 0.3346°W</span><span style="display:none">﻿ / <span class="geo">51.5898; -0.3346</span></span><span style="display:none">﻿ (<span class="fn org">Harrow</span>)</span></span></span></a></span>
    </td>
    <td>32
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Havering" title="London Borough of Havering">Havering</a>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Havering_London_Borough_Council" title="Havering London Borough Council">Havering London Borough Council</a>
    </td>
    <td><a href="/wiki/Conservative_Party_(UK)" title="Conservative Party (UK)">Conservative</a> (council <a href="/wiki/No_overall_control" title="No overall control">NOC</a>)
    </td>
    <td><a class="new" href="/w/index.php?title=Havering_Town_Hall&amp;action=edit&amp;redlink=1" title="Havering Town Hall (page does not exist)">Town Hall</a>, Main Road
    </td>
    <td>43.35
    </td>
    <td>242,080
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5812_N_0.1837_E_region:GB_type:city&amp;title=Havering"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°34′52″N</span> <span class="longitude">0°11′01″E</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.5812°N 0.1837°E</span><span style="display:none">﻿ / <span class="geo">51.5812; 0.1837</span></span><span style="display:none">﻿ (<span class="fn org">Havering</span>)</span></span></span></a></span>
    </td>
    <td>24
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Hillingdon" title="London Borough of Hillingdon">Hillingdon</a>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Hillingdon_London_Borough_Council" title="Hillingdon London Borough Council">Hillingdon London Borough Council</a>
    </td>
    <td><a href="/wiki/Conservative_Party_(UK)" title="Conservative Party (UK)">Conservative</a>
    </td>
    <td><a href="/wiki/Hillingdon_Civic_Centre" title="Hillingdon Civic Centre">Civic Centre</a>, High Street
    </td>
    <td>44.67
    </td>
    <td>286,806
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5441_N_0.476_W_region:GB_type:city&amp;title=Hillingdon"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°32′39″N</span> <span class="longitude">0°28′34″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.5441°N 0.4760°W</span><span style="display:none">﻿ / <span class="geo">51.5441; -0.4760</span></span><span style="display:none">﻿ (<span class="fn org">Hillingdon</span>)</span></span></span></a></span>
    </td>
    <td>33
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Hounslow" title="London Borough of Hounslow">Hounslow</a>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Hounslow_London_Borough_Council" title="Hounslow London Borough Council">Hounslow London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td>Hounslow House, 7 Bath Road
    </td>
    <td>21.61
    </td>
    <td>262,407
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4746_N_0.368_W_region:GB_type:city&amp;title=Hounslow"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°28′29″N</span> <span class="longitude">0°22′05″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.4746°N 0.3680°W</span><span style="display:none">﻿ / <span class="geo">51.4746; -0.3680</span></span><span style="display:none">﻿ (<span class="fn org">Hounslow</span>)</span></span></span></a></span>
    </td>
    <td>14
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Islington" title="London Borough of Islington">Islington</a>
    </td>
    <td><img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/><span style="display:none">Y</span>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Islington_London_Borough_Council" title="Islington London Borough Council">Islington London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Islington_Municipal_Offices&amp;action=edit&amp;redlink=1" title="Islington Municipal Offices (page does not exist)">Municipal Offices</a>, 222 Upper Street
    </td>
    <td>5.74
    </td>
    <td>215,667
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5416_N_0.1022_W_region:GB_type:city&amp;title=Islington"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°32′30″N</span> <span class="longitude">0°06′08″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.5416°N 0.1022°W</span><span style="display:none">﻿ / <span class="geo">51.5416; -0.1022</span></span><span style="display:none">﻿ (<span class="fn org">Islington</span>)</span></span></span></a></span>
    </td>
    <td>10
    </td></tr>
    <tr>
    <td><a href="/wiki/Royal_Borough_of_Kensington_and_Chelsea" title="Royal Borough of Kensington and Chelsea">Kensington and Chelsea</a>
    </td>
    <td><img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/><span style="display:none">Y</span>
    </td>
    <td><a class="mw-redirect" href="/wiki/Royal_borough" title="Royal borough">Royal</a>
    </td>
    <td><a href="/wiki/Kensington_and_Chelsea_London_Borough_Council" title="Kensington and Chelsea London Borough Council">Kensington and Chelsea London Borough Council</a>
    </td>
    <td><a href="/wiki/Conservative_Party_(UK)" title="Conservative Party (UK)">Conservative</a>
    </td>
    <td><a href="/wiki/Kensington_Town_Hall,_London" title="Kensington Town Hall, London">The Town Hall</a>, <a href="/wiki/Hornton_Street" title="Hornton Street">Hornton Street</a>
    </td>
    <td>4.68
    </td>
    <td>155,594
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.502_N_0.1947_W_region:GB_type:city&amp;title=Kensington+and+Chelsea"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°30′07″N</span> <span class="longitude">0°11′41″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.5020°N 0.1947°W</span><span style="display:none">﻿ / <span class="geo">51.5020; -0.1947</span></span><span style="display:none">﻿ (<span class="fn org">Kensington and Chelsea</span>)</span></span></span></a></span>
    </td>
    <td>3
    </td></tr>
    <tr>
    <td><a href="/wiki/Royal_Borough_of_Kingston_upon_Thames" title="Royal Borough of Kingston upon Thames">Kingston upon Thames</a>
    </td>
    <td>
    </td>
    <td><a class="mw-redirect" href="/wiki/Royal_borough" title="Royal borough">Royal</a>
    </td>
    <td><a href="/wiki/Kingston_upon_Thames_London_Borough_Council" title="Kingston upon Thames London Borough Council">Kingston upon Thames London Borough Council</a>
    </td>
    <td><a href="/wiki/Liberal_Democrats_(UK)" title="Liberal Democrats (UK)">Liberal Democrat</a>
    </td>
    <td><a href="/wiki/Kingston_upon_Thames_Guildhall" title="Kingston upon Thames Guildhall">Guildhall</a>, High Street
    </td>
    <td>14.38
    </td>
    <td>166,793
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4085_N_0.3064_W_region:GB_type:city&amp;title=Kingston+upon+Thames"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°24′31″N</span> <span class="longitude">0°18′23″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.4085°N 0.3064°W</span><span style="display:none">﻿ / <span class="geo">51.4085; -0.3064</span></span><span style="display:none">﻿ (<span class="fn org">Kingston upon Thames</span>)</span></span></span></a></span>
    </td>
    <td>16
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Lambeth" title="London Borough of Lambeth">Lambeth</a>
    </td>
    <td><img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/><span style="display:none">Y</span>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Lambeth_London_Borough_Council" title="Lambeth London Borough Council">Lambeth London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a href="/wiki/Lambeth_Town_Hall" title="Lambeth Town Hall">Lambeth Town Hall</a>, Brixton Hill
    </td>
    <td>10.36
    </td>
    <td>314,242
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4607_N_0.1163_W_region:GB_type:city&amp;title=Lambeth"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°27′39″N</span> <span class="longitude">0°06′59″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.4607°N 0.1163°W</span><span style="display:none">﻿ / <span class="geo">51.4607; -0.1163</span></span><span style="display:none">﻿ (<span class="fn org">Lambeth</span>)</span></span></span></a></span>
    </td>
    <td>6
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Lewisham" title="London Borough of Lewisham">Lewisham</a>
    </td>
    <td><img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/><span style="display:none">Y</span>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Lewisham_London_Borough_Council" title="Lewisham London Borough Council">Lewisham London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Lewisham_Town_Hall&amp;action=edit&amp;redlink=1" title="Lewisham Town Hall (page does not exist)">Town Hall</a>, 1 Catford Road
    </td>
    <td>13.57
    </td>
    <td>286,180
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4452_N_0.0209_W_region:GB_type:city&amp;title=Lewisham"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°26′43″N</span> <span class="longitude">0°01′15″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.4452°N 0.0209°W</span><span style="display:none">﻿ / <span class="geo">51.4452; -0.0209</span></span><span style="display:none">﻿ (<span class="fn org">Lewisham</span>)</span></span></span></a></span>
    </td>
    <td>21
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Merton" title="London Borough of Merton">Merton</a>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Merton_London_Borough_Council" title="Merton London Borough Council">Merton London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Merton_Civic_Centre&amp;action=edit&amp;redlink=1" title="Merton Civic Centre (page does not exist)">Civic Centre</a>, London Road
    </td>
    <td>14.52
    </td>
    <td>203,223
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4014_N_0.1958_W_region:GB_type:city&amp;title=Merton"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°24′05″N</span> <span class="longitude">0°11′45″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.4014°N 0.1958°W</span><span style="display:none">﻿ / <span class="geo">51.4014; -0.1958</span></span><span style="display:none">﻿ (<span class="fn org">Merton</span>)</span></span></span></a></span>
    </td>
    <td>17
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Newham" title="London Borough of Newham">Newham</a>
    </td>
    <td><sup class="reference" id="cite_ref-note2_4-2"><a href="#cite_note-note2-4">[note 3]</a></sup>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Newham_London_Borough_Council" title="Newham London Borough Council">Newham London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Newham_Dockside&amp;action=edit&amp;redlink=1" title="Newham Dockside (page does not exist)">Newham Dockside</a>, 1000 Dockside Road
    </td>
    <td>13.98
    </td>
    <td>318,227
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5077_N_0.0469_E_region:GB_type:city&amp;title=Newham"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°30′28″N</span> <span class="longitude">0°02′49″E</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.5077°N 0.0469°E</span><span style="display:none">﻿ / <span class="geo">51.5077; 0.0469</span></span><span style="display:none">﻿ (<span class="fn org">Newham</span>)</span></span></span></a></span>
    </td>
    <td>27
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Redbridge" title="London Borough of Redbridge">Redbridge</a>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Redbridge_London_Borough_Council" title="Redbridge London Borough Council">Redbridge London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Redbridge_Town_Hall&amp;action=edit&amp;redlink=1" title="Redbridge Town Hall (page does not exist)">Town Hall</a>, 128-142 High Road
    </td>
    <td>21.78
    </td>
    <td>288,272
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.559_N_0.0741_E_region:GB_type:city&amp;title=Redbridge"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°33′32″N</span> <span class="longitude">0°04′27″E</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.5590°N 0.0741°E</span><span style="display:none">﻿ / <span class="geo">51.5590; 0.0741</span></span><span style="display:none">﻿ (<span class="fn org">Redbridge</span>)</span></span></span></a></span>
    </td>
    <td>26
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Richmond_upon_Thames" title="London Borough of Richmond upon Thames">Richmond upon Thames</a>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Richmond_upon_Thames_London_Borough_Council" title="Richmond upon Thames London Borough Council">Richmond upon Thames London Borough Council</a>
    </td>
    <td><a href="/wiki/Liberal_Democrats_(UK)" title="Liberal Democrats (UK)">Liberal Democrat</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Richmond_upon_Thames_Civic_Centre&amp;action=edit&amp;redlink=1" title="Richmond upon Thames Civic Centre (page does not exist)">Civic Centre</a>, 44 York Street
    </td>
    <td>22.17
    </td>
    <td>191,365
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4479_N_0.326_W_region:GB_type:city&amp;title=Richmond+upon+Thames"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°26′52″N</span> <span class="longitude">0°19′34″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.4479°N 0.3260°W</span><span style="display:none">﻿ / <span class="geo">51.4479; -0.3260</span></span><span style="display:none">﻿ (<span class="fn org">Richmond upon Thames</span>)</span></span></span></a></span>
    </td>
    <td>15
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Southwark" title="London Borough of Southwark">Southwark</a>
    </td>
    <td><img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/><span style="display:none">Y</span>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Southwark_London_Borough_Council" title="Southwark London Borough Council">Southwark London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a class="new" href="/w/index.php?title=160_Tooley_Street&amp;action=edit&amp;redlink=1" title="160 Tooley Street (page does not exist)">160 Tooley Street</a>
    </td>
    <td>11.14
    </td>
    <td>298,464
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5035_N_0.0804_W_region:GB_type:city&amp;title=Southwark"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°30′13″N</span> <span class="longitude">0°04′49″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.5035°N 0.0804°W</span><span style="display:none">﻿ / <span class="geo">51.5035; -0.0804</span></span><span style="display:none">﻿ (<span class="fn org">Southwark</span>)</span></span></span></a></span>
    </td>
    <td>7
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Sutton" title="London Borough of Sutton">Sutton</a>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Sutton_London_Borough_Council" title="Sutton London Borough Council">Sutton London Borough Council</a>
    </td>
    <td><a href="/wiki/Liberal_Democrats_(UK)" title="Liberal Democrats (UK)">Liberal Democrat</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Sutton_Civic_Offices&amp;action=edit&amp;redlink=1" title="Sutton Civic Offices (page does not exist)">Civic Offices</a>, St Nicholas Way
    </td>
    <td>16.93
    </td>
    <td>195,914
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.3618_N_0.1945_W_region:GB_type:city&amp;title=Sutton"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°21′42″N</span> <span class="longitude">0°11′40″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.3618°N 0.1945°W</span><span style="display:none">﻿ / <span class="geo">51.3618; -0.1945</span></span><span style="display:none">﻿ (<span class="fn org">Sutton</span>)</span></span></span></a></span>
    </td>
    <td>18
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Tower_Hamlets" title="London Borough of Tower Hamlets">Tower Hamlets</a>
    </td>
    <td><img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/><span style="display:none">Y</span>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Tower_Hamlets_London_Borough_Council" title="Tower Hamlets London Borough Council">Tower Hamlets London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Tower_Hamlets_Town_Hall&amp;action=edit&amp;redlink=1" title="Tower Hamlets Town Hall (page does not exist)">Town Hall</a>, Mulberry Place, 5 Clove Crescent
    </td>
    <td>7.63
    </td>
    <td>272,890
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5099_N_0.0059_W_region:GB_type:city&amp;title=Tower+Hamlets"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°30′36″N</span> <span class="longitude">0°00′21″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.5099°N 0.0059°W</span><span style="display:none">﻿ / <span class="geo">51.5099; -0.0059</span></span><span style="display:none">﻿ (<span class="fn org">Tower Hamlets</span>)</span></span></span></a></span>
    </td>
    <td>8
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Waltham_Forest" title="London Borough of Waltham Forest">Waltham Forest</a>
    </td>
    <td>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Waltham_Forest_London_Borough_Council" title="Waltham Forest London Borough Council">Waltham Forest London Borough Council</a>
    </td>
    <td><a href="/wiki/Labour_Party_(UK)" title="Labour Party (UK)">Labour</a>
    </td>
    <td><a href="/wiki/Waltham_Forest_Town_Hall" title="Waltham Forest Town Hall">Waltham Forest Town Hall</a>, Forest Road
    </td>
    <td>14.99
    </td>
    <td>265,797
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5908_N_0.0134_W_region:GB_type:city&amp;title=Waltham+Forest"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°35′27″N</span> <span class="longitude">0°00′48″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.5908°N 0.0134°W</span><span style="display:none">﻿ / <span class="geo">51.5908; -0.0134</span></span><span style="display:none">﻿ (<span class="fn org">Waltham Forest</span>)</span></span></span></a></span>
    </td>
    <td>28
    </td></tr>
    <tr>
    <td><a href="/wiki/London_Borough_of_Wandsworth" title="London Borough of Wandsworth">Wandsworth</a>
    </td>
    <td><img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/><span style="display:none">Y</span>
    </td>
    <td>
    </td>
    <td><a href="/wiki/Wandsworth_London_Borough_Council" title="Wandsworth London Borough Council">Wandsworth London Borough Council</a>
    </td>
    <td><a href="/wiki/Conservative_Party_(UK)" title="Conservative Party (UK)">Conservative</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Wandsworth_Town_Hall&amp;action=edit&amp;redlink=1" title="Wandsworth Town Hall (page does not exist)">The Town Hall</a>, <a href="/wiki/Wandsworth_High_Street" title="Wandsworth High Street">Wandsworth High Street</a>
    </td>
    <td>13.23
    </td>
    <td>310,516
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4567_N_0.191_W_region:GB_type:city&amp;title=Wandsworth"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°27′24″N</span> <span class="longitude">0°11′28″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.4567°N 0.1910°W</span><span style="display:none">﻿ / <span class="geo">51.4567; -0.1910</span></span><span style="display:none">﻿ (<span class="fn org">Wandsworth</span>)</span></span></span></a></span>
    </td>
    <td>5
    </td></tr>
    <tr>
    <td><a href="/wiki/City_of_Westminster" title="City of Westminster">Westminster</a>
    </td>
    <td><img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/><span style="display:none">Y</span>
    </td>
    <td><a href="/wiki/City_status_in_the_United_Kingdom" title="City status in the United Kingdom">City</a>
    </td>
    <td><a href="/wiki/Westminster_City_Council" title="Westminster City Council">Westminster City Council</a>
    </td>
    <td><a href="/wiki/Conservative_Party_(UK)" title="Conservative Party (UK)">Conservative</a>
    </td>
    <td><a class="new" href="/w/index.php?title=Westminster_City_Hall&amp;action=edit&amp;redlink=1" title="Westminster City Hall (page does not exist)">Westminster City Hall</a>, 64 Victoria Street
    </td>
    <td>8.29
    </td>
    <td>226,841
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.4973_N_0.1372_W_region:GB_type:city&amp;title=Westminster"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°29′50″N</span> <span class="longitude">0°08′14″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.4973°N 0.1372°W</span><span style="display:none">﻿ / <span class="geo">51.4973; -0.1372</span></span><span style="display:none">﻿ (<span class="fn org">Westminster</span>)</span></span></span></a></span>
    </td>
    <td>2
    </td></tr></tbody></table>, <table class="wikitable sortable" style="font-size:95%" width="100%">
    <tbody><tr>
    <th width="100px">Borough
    </th>
    <th>Inner
    </th>
    <th width="100px">Status
    </th>
    <th>Local authority
    </th>
    <th>Political control
    </th>
    <th width="120px">Headquarters
    </th>
    <th>Area (sq mi)
    </th>
    <th>Population<br/>(2011 est)
    </th>
    <th width="20px">Co-ordinates
    </th>
    <th><span style="background:#67BCD3"> Nr. in<br/>map </span>
    </th></tr>
    <tr>
    <td><a href="/wiki/City_of_London" title="City of London">City of London</a>
    </td>
    <td>(<img alt="☑" data-file-height="600" data-file-width="600" decoding="async" height="20" src="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/20px-Yes_check.svg.png" srcset="//upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/30px-Yes_check.svg.png 1.5x, //upload.wikimedia.org/wikipedia/en/thumb/f/fb/Yes_check.svg/40px-Yes_check.svg.png 2x" width="20"/><span style="display:none">Y</span>)<br/><sup class="reference" id="cite_ref-6"><a href="#cite_note-6">[note 5]</a></sup>
    </td>
    <td><i><a href="/wiki/Sui_generis" title="Sui generis">Sui generis</a></i>;<br/><a href="/wiki/City_status_in_the_United_Kingdom" title="City status in the United Kingdom">City</a>;<br/><a href="/wiki/Ceremonial_counties_of_England" title="Ceremonial counties of England">Ceremonial county</a>
    </td>
    <td><a class="mw-redirect" href="/wiki/Corporation_of_London" title="Corporation of London">Corporation of London</a>;<br/><a href="/wiki/Inner_Temple" title="Inner Temple">Inner Temple</a>;<br/><a href="/wiki/Middle_Temple" title="Middle Temple">Middle Temple</a>
    </td>
    <td>? 
    </td>
    <td><a href="/wiki/Guildhall,_London" title="Guildhall, London">Guildhall</a>
    </td>
    <td>1.12
    </td>
    <td>7,000
    </td>
    <td><span class="plainlinks nourlexpansion"><a class="external text" href="//tools.wmflabs.org/geohack/geohack.php?pagename=List_of_London_boroughs&amp;params=51.5155_N_0.0922_W_region:GB_type:city&amp;title=City+of+London"><span class="geo-nondefault"><span class="geo-dms" title="Maps, aerial photos, and other data for this location"><span class="latitude">51°30′56″N</span> <span class="longitude">0°05′32″W</span></span></span><span class="geo-multi-punct">﻿ / ﻿</span><span class="geo-default"><span class="vcard"><span class="geo-dec" title="Maps, aerial photos, and other data for this location">51.5155°N 0.0922°W</span><span style="display:none">﻿ / <span class="geo">51.5155; -0.0922</span></span><span style="display:none">﻿ (<span class="fn org">City of London</span>)</span></span></span></a></span>
    </td>
    <td>1
    </td></tr></tbody></table>]
    

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
