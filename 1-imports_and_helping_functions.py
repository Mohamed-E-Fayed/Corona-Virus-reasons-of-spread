"""
This file contains imports and any helping function that is needed in our data cleaning and analysis.
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import requests
from bs4 import BeautifulSoup
import time 
import functools 
import sqlite3 
import math
import datetime
#pandas should print all columns
pd.set_option('display.max_columns', None)
#Fit all columns in one line
pd.set_option('display.width', None)

#helping functions 
def create_dict_of_codes_for_str_columns(df):
    """
    This function creates a dictionary containing numeric codes for string values. 
    """
    dict={}
    numeric_types = [np.int64, np.float64] 
    for column in df.columns:
        if type(df[column].loc[1]) in numeric_types :
            continue 
        tmp={}
        lst = list(df[column].unique())
        for i in range(len(lst)):
            tmp[lst[i]]=i
        dict[column]=tmp
    print(dict)
    return dict

def create_str_int_pair_dict(df):
    """
    Another naming, hoping it would be easier to remember, for create_dict_of_codes_for_str_columns(df).
    """
    return create_dict_of_codes_for_str_columns(df)

def create_codes_dictionary(df):
    """
    Simpler naming
    """
    return create_dict_of_codes_for_str_columns(df)

def convert_str_columns_into_int(df, dict):
    """
    This function aims to convert all columns in a pandas dataframe , that contains string values into integer values
    """
    
    tmp={} #to be optimized 
    for column in list(dict.keys()):
        tmp[column]=df[column].map(dict[column])
        df[column]=tmp[column]
        
    return df
def convert_into_numeric(df, dict):
    """
    Simpler naming for the function convert_str_columns_into_int(df, dict)
    """
    return convert_str_columns_into_int(df, dict)

def get_indices_of_ll(latitude, longitude, error, latitudes=[], longitudes=[]):
    """
    this function returns the index of tuple  containing longitudes and latitudes within a specific error. 
    """
    indices=[]
    for i in range(len(latitudes)):
        splitted_line=ls_info[i].split()
        lat, lon=float(splitted_line[1]), float(splitted_line[2])
        if (latitude-error <=lat and lat <= latitude+error) and (longitude-error<=lon and lon <= longitude+error):
            indices.append(i)
    return indices 

def run_command(cmd): 
    """
    runs the gvien terminal command 
    """
    print('Running command:\n', cmd)
    os.system(cmd)


def get_country_of(city):
    """returns the country in which the given city exists
    """
    city_countries={'Berlin':'DEU', 'Munich':'DEU'}
    city_countries['London'], city_countries['Edinburgh']='UK', 'UK'
    city_countries['Rome'], city_countries['Tuscany'], city_countries['Napoli']= 'ITA', 'ITA', 'ITA'
    return city_countries[city]

def get_country_containing(city):
    """
    meaningful  naming for get_country_of(city)
    """
    return get_country_of(city) 

def get_cities_in(country):
    """
    function that returns a list of cities inside a given country 
    """
    country_cities={'DEU':['Berlin', 'Munich']}
    return country_cities[country]
def get_nearest_coordinates_index(ls_info, indices, lat, long):
    """
    This function returns the index nearest to given latitude and longitude
    """
    distances=[]
    for i in indices: 
        distances.append(math.sqrt((float(ls_info[i].split()[1])-lat)**2 + (float(ls_info[i].split()[2])-long)**2))
    return indices[distances.index(min(list(distances)))]

def check_date_format(lst):
    """
    This function checks whether the list of dates given have the required format.
    The format is 'yyyy-mm-dd'
    """
    for item in lst:
        try:
            datetime.datetime.strptime(item, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")

#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.optimize import curve_fit
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import Ridge


# In[2]:


def get_education_level(location):
    data_uploaded = pd.read_csv(location)
    data = np.array(data_uploaded)
    education_data = []
    country_edu_data = []
    for i in range(0,np.shape(data)[0]):
        if data[i][0] =="Germany":
            country_edu_data.append(data[i][0])
            education_data.append(data[i][1])
        elif data[i][0] =="United Kingdom":
            country_edu_data.append(data[i][0])
            education_data.append(data[i][1])
        elif data[i][0] =="Italy":
            country_edu_data.append(data[i][0])
            education_data.append(data[i][1])
    return country_edu_data,education_data


# In[3]:


def get_health_level(location):
    data_uploaded = pd.read_csv(location)
    data = np.array(data_uploaded)
    health_data = []
    country_health_data = []
    for i in range(0,np.shape(data)[0]):
        if data[i][2] =="Germany":
            country_health_data.append(data[i][2])
            education_data.append(data[i][3])
        elif data[i][2] =="United Kingdom":
            country_health_data.append(data[i][2])
            education_data.append(data[i][3])
        elif data[i][2] =="Italy":
            country_health_data.append(data[i][2])
            education_data.append(data[i][3])
    return health_data,country_health_data


# In[4]:


def get_population_lifeexpectancy(location):
    data_uploaded = pd.read_csv(location)
    data = np.array(data_uploaded)
    life_expectancy_data = []
    total_pop_data = []
    pop_density_data = []
    country_data = []
    for i in range(0,np.shape(data)[0]):
        if data[i][0] =="DEU":
            country_data.append(data[i][0])
            total_pop_data.append(data[i][1])
            life_expectancy_data.append(data[i][2])
            pop_density_data.append(data[i][4])
        elif data[i][0] =="GBR":
            country_data.append(data[i][0])
            total_pop_data.append(data[i][1])
            life_expectancy_data.append(data[i][2])
            pop_density_data.append(data[i][4])
        elif data[i][0] =="ITA":
            country_data.append(data[i][0])
            total_pop_data.append(data[i][1])
            life_expectancy_data.append(data[i][2])
            pop_density_data.append(data[i][4])
    return country_data,pop_density_data,total_pop_data,life_expectancy_data


# In[5]:


def get_country_data(location,country):
    data_uploaded = pd.read_csv(location)
    data = np.array(data_uploaded)
    country_data = []
    found = False
    for i in range(0,np.shape(data)[0]):
        if data[i][0] == country:
            country_data = data[i]
            found = True
    if not found:
        print("not found")
    headers = data_uploaded.columns
    return country_data[4:,],headers[4:,]


# In[6]:


def get_tempreture(location):
    data_uploaded = pd.read_csv(location)
    data = np.array(data_uploaded)
    tempreture_data = data[30594:30594+80]
    t_max = []
    t_min = []
    for i in range(0,np.shape(tempreture_data)[0]):
        t_max.append(tempreture_data[i][10])
        t_min.append(tempreture_data[i][12])
    return t_max,t_min


# In[7]:


def get_death_recovered(location,country):
    data_uploaded = pd.read_csv(location)
    data = np.array(data_uploaded)
    i=0
    found = False
    deaths = []
    recovered = []
    while i < np.shape(data)[0]:
        if data[i][1] == country:
            deaths.append(data[i][4])
            recovered.append(data[i][5])
            found = True
        i+=5
    if not found:
        print("not found")
    return deaths,recovered


# In[8]:


def decumiltate(data):
    decumiltative_data = []
    cumil = 0
    for i in range(0,len(data)):
        decumiltative_data.append(data[i])
    for i in range(1,len(decumiltative_data)):
        decumiltative_data[i]=decumiltative_data[i] - decumiltative_data[i-1] - cumil
        cumil += decumiltative_data[i-1]
    return np.array(decumiltative_data)


# In[9]:


def get_country_totalcases(data_uploaded):
    data = np.array(data_uploaded)
    country_data = []
    country_list = []
    for i in range(0,np.shape(data)[0]):
        country_list.append(data[i][0])
        country_data.append( data[i][len(data[i])-1])
    return country_list,country_data
    


# In[10]:


def get_model_for_Class(parameters):# add paramters
    model = Sequential()
    model.add(Dense(8, input_dim=parameters, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


# In[11]:


def get_model_for_Class2(parameters):# add paramters
    model = Sequential()
    model.add(Dense(2, input_dim=parameters, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


