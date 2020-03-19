---
layout: post
title: Studying crime in San Francisco using Folium
tags: [maps]
---

In this blog, we will be looking at the crime data in the city of San Francisco. The data we will be using contains all crimes in San Francisco from the year 2018 to 2020. You can download the data [here](https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783). Since this dataset is very large (more than 330,000 crimes), we will be considering only a small part of the data for this post.


```python
# import libraries
import numpy as np
import pandas as pd
import folium
```


```python
# read the data into a pandas dataframe
df = pd.read_csv('SF_Crime_data.csv')
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
      <th>Incident Datetime</th>
      <th>Incident Date</th>
      <th>Incident Time</th>
      <th>Incident Year</th>
      <th>Incident Day of Week</th>
      <th>Report Datetime</th>
      <th>Row ID</th>
      <th>Incident ID</th>
      <th>Incident Number</th>
      <th>CAD Number</th>
      <th>...</th>
      <th>Current Supervisor Districts</th>
      <th>Analysis Neighborhoods</th>
      <th>HSOC Zones as of 2018-06-05</th>
      <th>OWED Public Spaces</th>
      <th>Central Market/Tenderloin Boundary Polygon - Updated</th>
      <th>Parks Alliance CPSI (27+TL sites)</th>
      <th>ESNCAG - Boundary File</th>
      <th>Areas of Vulnerability, 2016</th>
      <th>Unnamed: 36</th>
      <th>Unnamed: 37</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2/3/2020 14:45</td>
      <td>2/3/2020</td>
      <td>14:45</td>
      <td>2020</td>
      <td>Monday</td>
      <td>2/3/2020 17:50</td>
      <td>89881675000</td>
      <td>898816</td>
      <td>200085557</td>
      <td>200342870.0</td>
      <td>...</td>
      <td>8.0</td>
      <td>16.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2/3/2020 3:45</td>
      <td>2/3/2020</td>
      <td>3:45</td>
      <td>2020</td>
      <td>Monday</td>
      <td>2/3/2020 3:45</td>
      <td>89860711012</td>
      <td>898607</td>
      <td>200083749</td>
      <td>200340316.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2/3/2020 10:00</td>
      <td>2/3/2020</td>
      <td>10:00</td>
      <td>2020</td>
      <td>Monday</td>
      <td>2/3/2020 10:06</td>
      <td>89867264015</td>
      <td>898672</td>
      <td>200084060</td>
      <td>200340808.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>35.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1/19/2020 17:12</td>
      <td>1/19/2020</td>
      <td>17:12</td>
      <td>2020</td>
      <td>Sunday</td>
      <td>2/1/2020 13:01</td>
      <td>89863571000</td>
      <td>898635</td>
      <td>206024187</td>
      <td>NaN</td>
      <td>...</td>
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
      <th>4</th>
      <td>1/5/2020 0:00</td>
      <td>1/5/2020</td>
      <td>0:00</td>
      <td>2020</td>
      <td>Sunday</td>
      <td>2/3/2020 16:09</td>
      <td>89877368020</td>
      <td>898773</td>
      <td>200085193</td>
      <td>200342341.0</td>
      <td>...</td>
      <td>6.0</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
# dimensions of the data
df.shape
```




    (330054, 38)



There have been more than 330,000 crimes in San Francisco in the past two years.


```python
df.columns
```




    Index(['Incident Datetime', 'Incident Date', 'Incident Time', 'Incident Year',
           'Incident Day of Week', 'Report Datetime', 'Row ID', 'Incident ID',
           'Incident Number', 'CAD Number', 'Report Type Code',
           'Report Type Description', 'Filed Online', 'Incident Code',
           'Incident Category', 'Incident Subcategory', 'Incident Description',
           'Resolution', 'Intersection', 'CNN', 'Police District',
           'Analysis Neighborhood', 'Supervisor District', 'Latitude', 'Longitude',
           'point', 'SF Find Neighborhoods', 'Current Police Districts',
           'Current Supervisor Districts', 'Analysis Neighborhoods',
           'HSOC Zones as of 2018-06-05', 'OWED Public Spaces',
           'Central Market/Tenderloin Boundary Polygon - Updated',
           'Parks Alliance CPSI (27+TL sites)', 'ESNCAG - Boundary File',
           'Areas of Vulnerability, 2016', 'Unnamed: 36', 'Unnamed: 37'],
          dtype='object')



We do not need all these columns for our analysis. So we will consider only the necessary columns.


```python
df = df[['Incident Datetime', 'Incident Day of Week', 'Incident Number', 'Incident Category', 'Incident Description', 
         'Police District', 'Analysis Neighborhood', 'Resolution', 'Latitude', 'Longitude', 'point']]
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
      <th>Incident Datetime</th>
      <th>Incident Day of Week</th>
      <th>Incident Number</th>
      <th>Incident Category</th>
      <th>Incident Description</th>
      <th>Police District</th>
      <th>Analysis Neighborhood</th>
      <th>Resolution</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>point</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2/3/2020 14:45</td>
      <td>Monday</td>
      <td>200085557</td>
      <td>Missing Person</td>
      <td>Found Person</td>
      <td>Taraval</td>
      <td>Lakeshore</td>
      <td>Open or Active</td>
      <td>37.726950</td>
      <td>-122.476039</td>
      <td>(37.72694991292525, -122.47603947349434)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2/3/2020 3:45</td>
      <td>Monday</td>
      <td>200083749</td>
      <td>Stolen Property</td>
      <td>Stolen Property, Possession with Knowledge, Re...</td>
      <td>Mission</td>
      <td>Mission</td>
      <td>Cite or Arrest Adult</td>
      <td>37.752440</td>
      <td>-122.415172</td>
      <td>(37.752439644389675, -122.41517229045435)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2/3/2020 10:00</td>
      <td>Monday</td>
      <td>200084060</td>
      <td>Non-Criminal</td>
      <td>Aided Case, Injured or Sick Person</td>
      <td>Tenderloin</td>
      <td>Financial District/South Beach</td>
      <td>Open or Active</td>
      <td>37.784560</td>
      <td>-122.407337</td>
      <td>(37.784560141211806, -122.40733704162238)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1/19/2020 17:12</td>
      <td>Sunday</td>
      <td>206024187</td>
      <td>Lost Property</td>
      <td>Lost Property</td>
      <td>Taraval</td>
      <td>NaN</td>
      <td>Open or Active</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/5/2020 0:00</td>
      <td>Sunday</td>
      <td>200085193</td>
      <td>Miscellaneous Investigation</td>
      <td>Miscellaneous Investigation</td>
      <td>Richmond</td>
      <td>Pacific Heights</td>
      <td>Open or Active</td>
      <td>37.787112</td>
      <td>-122.440250</td>
      <td>(37.78711245591735, -122.44024995765258)</td>
    </tr>
  </tbody>
</table>
</div>



Now, each row consists of the following 11 features:
- **Incident Datetime:** The date and time when the incident occurred
- **Incident Day of Week:** The day of week on which the incident occurred
- **Incident Number:** The incident or crime number
- **Incident Category:** The category of the incident or crime
- **Incident Desccription:** The description of the incident or crime
- **Police:** The police department district
- **Resolution:** The resolution of the crime in terms of whether the perpertrator was arrested or not
- **Analysis Neighborhoods:** The neighborhood where the incident took place
- **Latitude:** The latitude value of the crime location
- **Longitude:** The longitude value of the crime location
- **point:** A tuple of the latitude and logitude values

Let's drop the missing values from the Latitude and Longitude columns as they will result in an error when creating a map. 


```python
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
```

Rename the Incident Category column for the sake of simplicity.


```python
df.rename(columns={'Incident Category':'Category'}, inplace=True)
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
      <th>Incident Datetime</th>
      <th>Incident Day of Week</th>
      <th>Incident Number</th>
      <th>Category</th>
      <th>Incident Description</th>
      <th>Police District</th>
      <th>Analysis Neighborhood</th>
      <th>Resolution</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>point</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2/3/2020 14:45</td>
      <td>Monday</td>
      <td>200085557</td>
      <td>Missing Person</td>
      <td>Found Person</td>
      <td>Taraval</td>
      <td>Lakeshore</td>
      <td>Open or Active</td>
      <td>37.726950</td>
      <td>-122.476039</td>
      <td>(37.72694991292525, -122.47603947349434)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2/3/2020 3:45</td>
      <td>Monday</td>
      <td>200083749</td>
      <td>Stolen Property</td>
      <td>Stolen Property, Possession with Knowledge, Re...</td>
      <td>Mission</td>
      <td>Mission</td>
      <td>Cite or Arrest Adult</td>
      <td>37.752440</td>
      <td>-122.415172</td>
      <td>(37.752439644389675, -122.41517229045435)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2/3/2020 10:00</td>
      <td>Monday</td>
      <td>200084060</td>
      <td>Non-Criminal</td>
      <td>Aided Case, Injured or Sick Person</td>
      <td>Tenderloin</td>
      <td>Financial District/South Beach</td>
      <td>Open or Active</td>
      <td>37.784560</td>
      <td>-122.407337</td>
      <td>(37.784560141211806, -122.40733704162238)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/5/2020 0:00</td>
      <td>Sunday</td>
      <td>200085193</td>
      <td>Miscellaneous Investigation</td>
      <td>Miscellaneous Investigation</td>
      <td>Richmond</td>
      <td>Pacific Heights</td>
      <td>Open or Active</td>
      <td>37.787112</td>
      <td>-122.440250</td>
      <td>(37.78711245591735, -122.44024995765258)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2/3/2020 8:36</td>
      <td>Monday</td>
      <td>200083909</td>
      <td>Miscellaneous Investigation</td>
      <td>Miscellaneous Investigation</td>
      <td>Central</td>
      <td>Financial District/South Beach</td>
      <td>Open or Active</td>
      <td>37.796926</td>
      <td>-122.399507</td>
      <td>(37.796926429317054, -122.39950750040278)</td>
    </tr>
  </tbody>
</table>
</div>




```python
limit = 100
df = df.iloc[0:limit, :]
```


```python
# new dimensions of the data
df.shape
```




    (100, 11)



### Visualization
Let's visualize where these crimes took place in the city of San Francisco.


```python
# San Francisco latitude and longitude values
latitude = 37.7749
longitude = -122.4194
```


```python
# create a map
sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)
sanfran_map
```


<img src= "assets/img/sf_crime/map1.JPG">


Let's create clusters of neighborhoods. The number of crimes per clusters is denoted on the cluster circle. In a Jupyter notebook, you can interact with the map - click on a cluster to zoom in, in on a marker to check the category of the crime.


```python
from folium import plugins

# let's start again with a clean copy of the map of San Francisco
sanfran_map = folium.Map(location = [latitude, longitude], zoom_start = 12)

# instantiate a mark cluster object for the incidents in the dataframe
incidents = plugins.MarkerCluster().add_to(sanfran_map)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(df.Latitude, df.Longitude, df.Category):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(incidents)

# display map
sanfran_map
```

<img src= "assets/img/sf_crime/map2.JPG">


```python

```
