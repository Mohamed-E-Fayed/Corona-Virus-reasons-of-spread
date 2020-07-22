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

def decumiltate(data):
    decumiltative_data = []
    cumil = 0
    for i in range(0,len(data)):
        decumiltative_data.append(data[i])
    for i in range(1,len(decumiltative_data)):
        decumiltative_data[i]=decumiltative_data[i] - decumiltative_data[i-1] - cumil
        cumil += decumiltative_data[i-1]
    return np.array(decumiltative_data)

def get_country_totalcases(data_uploaded):
    data = np.array(data_uploaded)
    country_data = []
    country_list = []
    for i in range(0,np.shape(data)[0]):
        country_list.append(data[i][0])
        country_data.append( data[i][len(data[i])-1])
    return country_list,country_data

def get_model_for_Class(parameters):# add paramters
    model = Sequential()
    model.add(Dense(8, input_dim=parameters, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def get_model_for_Class2(parameters):# add paramters
    model = Sequential()
    model.add(Dense(2, input_dim=parameters, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model




# health data 
df_hq=df_hq.drop(['age group', 'flag codes', 'gen', 'ind', 'country'], axis=1) # I left COU instead of country since values in countries are not easily guessable. 
hq_corr=df_hq.corr()

print(separator)
print('•  Health quality correlation matrix:')
print(hq_corr)
print(separator)


print('weather data columns :')
print(df_w.columns)

print('date column example:')
print(df_w['date'])

for file in files:
    tmp = pd.read_csv('data/weather/'+file)
    print('number of different stations in file{}={}'.format(file,  len(tmp['STATION'].unique())))

#check the differences between different stations for same day in the same city 
print('temperatures from different station in same city:')
query='date=="2019-09-09" and City=="Berlin"'
tmp=df_w.query(query)
print(tmp[['STATION', 'TMAX', 'TMIN']])

# get data from Dec 2019 
df_w= df_w[df_w['date']>='2020-01-22']
df_w=df_w.replace({'cou':'UK'}, 'GBR')
# remove attributes columns 
attributes_cols=['station', 'name', 'prcp_attributes', 'snwd_attributes', 'tmax_attributes', 'tmin_attributes', 'tavg_attributes']
for acol in attributes_cols:
    if acol in df_w.columns.tolist():
        df_w=df_w.drop(acol, axis=1)

#print(df_w.columns)

# If tmin and tmin exists and tavg is none, 
# calculate tavg to avoid null columns.
ctr=0 #counter for number of nulls after getting average.
print('tavg is {}'.format('tavg' in df_w.columns.tolist()))
print('tmin is {}'.format('tmin' in df_w.columns.tolist()))
for i in df_w.index:
    if df_w.at[i, 'tavg'] is np.NaN:
        if df_w.at[i, 'tmin'] is np.NaN or df_w.at[i, 'tmax'] is np.NaN:
            ctr+=1
            continue
        else:
            df_w.at[i, 'tavg']=(df_w[i]['tmin']+df_w.loc[i]['tmax'])/2

print('there is still {} nulls'.format(ctr))

# for given countries, get average of columns of temperature
os.system('rm data/weather/weather.db')
conn=sqlite3.connect('data/weather/weather.db')
df_w.to_sql('weather', conn)
sql_query='select cou, date, avg(snwd) as snwd, avg(prcp) as prcp, '
sql_query=sql_query+'avg(tmax) as tmax, avg(tmin) as tmin '
sql_query=sql_query+'from weather GROUP BY cou, date'
df_w=pd.read_sql(sql_query, conn)


# The data should have the category as its columns names + cou 
# there should be a row for each country.

#health quality 
df_hq = df_hq.drop('country', axis=1)

# health resources 
"""
#print rows for a given variable to check its quality.
for var in df_hr['variable'].unique().tolist():
    print('variable=', var)
    print(df_hr[df_hr['variable']==var])
"""

df_tmp=pd.DataFrame({})

# make sure that all countries in same variable are ordered.
for col in df_hr['variable'].unique().tolist():
    cou_lst=df_hr[df_hr['variable']==col]['cou'].tolist()
    #print(' checking variable =', col)
    while len(cou_lst)<3:
        cou_lst.append(np.NaN)

    if cou_lst[0]=='DEU' and cou_lst[1]=='ITA' and cou_lst[2]=='GBR':
        continue 
    else:
        print('variable =', col, ' has different order')

# add the list of same variable for each column 
df_tmp['cou']=['DEU', 'ITA', 'GBR']
for col in df_hr['variable'].unique().tolist():
    query= 'variable=="{}"'.format(col)
    #print('column=', col)
    #print(df_hr.query(query))
    #print('assigned value=',df_hr.query(query)['value'].tolist())
    df_tmp[col]=df_hr.query(query)['value'].tolist()

df_hr=df_tmp 

#print the data after cleaning.
print('Health data after cleaning:')
print('• Health resources:')
print(df_hr)
print separator)

print('• Health quality:')
print(df_hq)
print(separator)


# select data for countries of interest. 
# corona virus data 
query='country in {}'.format(list(countries_codes.keys()))
df_cvs=df_cvs.query(query)
df_cvd=df_cvd.query(query)
df_cvr=df_cvr.query(query)

# add country codes to allow for join operation 
codes=[]
for code in list(df_cvs['country']):
    codes.append(countries_codes[code])
df_cvs['cou']=codes 
df_cvd['cou']=codes
df_cvr['cou']=codes 

# education data 
query='country in {}'.format(list(countries_codes.keys()))
df_er=df_er.query(query)
df_mec=df_mec.query(query)
df_mec= df_mec.drop('pop2020', axis=1)

codes=[]
for code  in list(df_er['country']):
    codes.append(countries_codes[code])
df_er['cou']=codes

codes=[]
for code in list(df_mec['country']):
    codes.append(countries_codes[code])
df_mec['cou']=codes

# remove country column 
df_er=df_er.drop('country', axis=1)
df_mec=df_mec.drop('country', axis=1)

# demography data and temperature are already in selected countries.


# join the data 
df_lst=[df_cv, df_er, df_mec, df_d, df_w]
df=df_lst[0]
df=pd.merge(df, df_er, on='cou')
df=pd.merge(df, df_mec, on='cou')
df=pd.merge(df, df_d, on='cou')
df=pd.merge(df, df_hr, on='cou')
df=pd.merge(df, df_hq, on='cou')
#print(df_hq.columns)
#print(df_hr.columns)
# store data without weather 
df.to_csv('data/data_without_weather.csv')

df=pd.merge(df, df_w, on=['cou','date'])

#store the data into a csv file
df.to_csv('data/data.csv') 

#print the data 
print('The final data:')
print(df.head()) 


#Generate integer codes for all different string values 
#we need to convert string values into numeric values to be able to  calculate the correlation factor to  remove unnecessary columns. 

countries_numeric_codes={'CHN':0, 'DEU':1, 'ITA':2, 'RUS':3, 'GBR':4, 'UK':4}
dict_df=dict_data=create_codes_dictionary(df)
df_numeric=convert_into_numeric(df, dict_df)

df_corr=df_numeric.corr()
#print('string integer pairs:') 
#print(dict_df)
#print(separator)

df_numeric.to_csv('data/data_numeric.csv')
print('numeric data is stored on data/data_numeric.csv')

#print('correlation matrix:')
#print(df_corr)

#print required columns 
print(df_corr[['confirmed', 'deaths', 'recovered']])

print('The correlation between confirmed cases and snwd is strange!')
print('the percentage of nulls =', df['snwd'].isnull().sum()/len(df['snwd']))
print('that explains the strange values of correlation')



# Read corona virus spread data 
df_cvs=df_corona_virus_spread=pd.read_csv('data/covid19-daily-data-worldometer/time_series_19-covid-Confirmed.csv')
df_cvd=df_corona_virus_death=pd.read_csv('data/covid19-daily-data-worldometer/time_series_19-covid-Deaths.csv')
df_cvr=df_corona_virus_recovered=pd.read_csv('data/covid19-daily-data-worldometer/time_series_19-covid-Recovered.csv')

# country 3-letter codes
#This dictionary is used reversably. 
# If given code, it will return corresponding country name.
# If given country name, it will return the code.
countries_codes={'Germany':'DEU', 'China':'CHN', 'Italy':'ITA', 'Russia':'RUS', 'United Kingdom':'GBR', 'UK':'GBR'}
countries_3_letters_codes=['DEU', 'CHN', 'ITA', 'RUS', 'GBR']
for column in list(countries_codes.keys()):
    countries_codes[countries_codes[column]]=column

#read health data 
df_hq= df_health_quality = pd.read_csv('data/health/HEALTH_HCQI.csv')
df_hr= df_health_resources=pd.read_csv('data/health/HEALTH_REAC.csv')

# read weather data 
#In order to cknow which data to download
# we need to get IDs for specific cities or near to it.

error=1
required_coordinates=[[52.52, 13.41], [48.33, 11.566]] #Berlin and Munich
required_coordinates.append([55.953251, -3.188267]) # Edinburgh, UK
required_coordinates.append([51.509865, -0.118092]) #London  
required_coordinates.append([41.8919300, 12.5113300]) # Rome
required_coordinates.append([43.5671, 10.9807]) # Tuscany, Italy
required_coordinates.append([40.8517746, 14.2681244]) # Napoli, Italy

cities_info={'Berlin':required_coordinates[0], 'Munich':required_coordinates[1]} #Germany 
cities_info['London'], cities_info['Edinburgh']= required_coordinates[3], required_coordinates[2] #UK
cities_info['Rome'], cities_info['Tuscany'], cities_info['Napoli']=required_coordinates[4], required_coordinates[5], required_coordinates[6] #Italy

cities_indices={}
cities_ids={}
ls_info=landstations_info=''
latitudes, longitudes, indices, indices_lst=[], [], [], []
with open('data/weather/ghcnd-stations.txt', 'r') as f:
    ls_info=landstations_info=str(f.read()).split('\n')

for i in range(len(ls_info)):
    splitted_line = ls_info[i].split()
    latitudes.append(float(splitted_line[1]))
    longitudes.append(float( splitted_line[2] ))

for city in cities_info:
    indices=get_indices_of_ll(cities_info[city][0], cities_info[city][1], error, latitudes, longitudes)
    print(indices)
    if not indices:
        print('{} does not have data'.format(city))
        continue 
    if type(indices)==list :
        indices=get_nearest_coordinates_index(ls_info, indices, cities_info[city][0], cities_info[city][1])
    cities_indices[city]=indices
    cities_ids[city]=ls_info[cities_indices[city]].split()[0]

#print data to check its correctness 
print('cities_ids=', cities_ids)
print('cities_info=', cities_info)

#download corresponding files
wget='wget '
url='https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/'
ext='.csv'
directory='data/weather/'
"""
#Download the required files. 
for city in cities_ids.keys():
    print('city name=', city)
    if '-' not in cities_ids[city]:
        run_command(str(wget+' -O '+directory+city+'-'+cities_ids[city]+ext + ' ' + url+cities_ids[city]+ext))
        continue 
    elif '-' in cities_ids[city]:
        print('cities_ids[city]=\n', cities_ids[city])
        for id in cities_ids[city].split('-'):
            print('file id=', id)
            cmd=wget+' -O '+directory+city+'-'+id+ext+ ' ' +url+id+ext
            run_command(cmd)
            time.sleep(1)
    else:
        print('Error in type of cities_ids[city]\n type is ', type(cities_ids[city]))
"""
#read the downloaded data
df_w = df_weather = pd.DataFrame({})
df_tmp=pd.DataFrame({})
files=[ f for f in os.listdir(directory) if ext in f]
files.sort()
for file in files:
    # read file
    df_tmp=pd.read_csv(directory+file)
    # get city name 
    city_name = file[:file.find('-')]
    #add a city name and country to all rows 
    df_tmp['City Name']=[city_name]*len(df_tmp['STATION']) # create a column of longest column based on exploration.
    print(city_name)
    df_tmp['COU']=[get_country_of(city_name)]*len(df_tmp['STATION'])
    # merge it with other weather data.
    df_w=pd.concat([df_w, df_tmp])

# read education data 
df_er= df_education_rankings=pd.read_csv('data/education/education_rankings.csv')
df_mec=df_most_educated_countries=pd.read_csv('data/education/most_educated_countries.csv')

#read population demography data 
df_pd=df_population_demography=pd.read_csv('data/demography/worldometer_demography.csv')
df_d=df_demography=df_pd 

# make all columns names lower cased 
# corona virus data 
columns=list(df_cvs.columns)
columns[0], columns[1]='province', 'country'
columns[2], columns[3]='long', 'lat'
df_cvs.columns=columns 

columns=list(df_cvd.columns) 
columns[0], columns[1]='province', 'country'
columns[2], columns[3]='long', 'lat'
df_cvd.columns=columns 

columns=list(df_cvr.columns)
columns[0], columns[1]='province', 'country'
columns[2], columns[3]='long', 'lat'
df_cvr.columns = columns 

# health data 
columns=[]
for column in df_hq.columns:
    columns.append(column.lower())
df_hq.columns=columns

columns=[]
for column in df_hr.columns:
    columns.append(column.lower())
df_hr.columns=columns

#weather data 
columns=[]
for column in df_w.columns:
    columns.append(column.lower())
df_w.columns=columns

#education data
columns=[]
for column in df_er.columns:
    columns.append(column.lower())
df_er.columns=columns

# demography data 
columns=[]
for column in df_d.columns:
    columns.append(column.lower())
df_d.columns=columns



"""
In this cell (or file), we explore the data. In every possibly useful way. 
"""

separator='-----------'
#print first 5 rows to discover the data

print('Covid-19 total number of cases:')
print(df_cvs.head())
print(separator)

print('Covid-19 total number of death:')
print(df_cvd.head())
print(separator)

print('Covid-19 total number of recovered cases:')
print(df_cvr.head())
print(separator)

print('Demography data:')
print(df_d.head())
print(separator)

print('Education data:')
print('• Education rankings:')
print(df_er.head())
print(separator)

print('• Most Educated Countries:')
print(df_mec.head())
print(separator)

print('Health data:')
print('• Health care resources:')
print(df_hr.head())
print(separator)

print('• Health care quality indicators:')
print(df_hq.head())
print(separator)

print('Weather data:')
print(df_w.head())
print(separator)


#print columns with their types 
print('Health quality data')
print(df_health_quality.info())
print(df_health_quality.describe())
print(separator)

#print some rows 
print(df_health_quality.head(10))

#Get correlation among columns to determine which columns to remove 
corr = df_health_quality.corr() 
print(corr)


"""
#print  columns with their info 
print('Health resources data')
print(df_health_resources.info())
print(df_health_resources.describe())
print(separator)
"""

#convert data into a proper format to re-explore the data.

#change the names of columns in health data 
#I've made them manually in a duplicate file, which we will use in the upcoming analysis.
#I've converted 'VAL' -> 'V_T',
#'VALUE -> Value_type (the first column, that contained string value in the main data.
#The second was essential since it makes confusion in the code.

# health data 
# convert all its strings values into numeric.
df_health_quality=convert_into_numeric(df_health_quality, dict_health_quality)
df_health_resources=convert_into_numeric(df_health_resources, dict_health_resources)


# education data 
#I've converted the names of column "name" manually 
# into "country" to be easily joined.


#find correlation among columns 
hq_corr= df_health_quality.corr() #hq=health_quatliy 
hr_corr= df_health_resources.corr() # hr=health_resources.

print(separator)

print('Correlation matrix:')
print('• Health quality data:')
print(hq_corr)
print(separator)
"""
print('• Health resources data:')
print(hr_corr)
print(separator)
"""


print('number of nulls in the corona virus data =',df_cv.isnull().sum().sum())

# prepare corona virus data for join 
non_date_columns=['country', 'province', 'long', 'lat', 'cou']
df_cvs_numbers=df_cvs.drop(['province', 'country', 'long', 'lat'], axis=1)
df_cvd_numbers=df_cvd.drop(['province', 'country', 'long', 'lat'], axis=1)
df_cvr_numbers=df_cvr.drop(['province', 'country', 'long', 'lat'], axis=1)

date=df_cvs.drop(non_date_columns, axis=1).columns.tolist()
# update format of date to match that in weather.
for i in range(len(date)):
    month, day, year=date[i].split('/')
    year='20'+year
    if int(month) < 10:
        month = '0'+month
    if int(day) <10:
        day='0'+day
    date[i]=year+'-'+month+'-'+day
df_cvs_numbers_t=df_cvs_numbers.T
df_cvd_numbers_t=df_cvd_numbers.T
df_cvr_numbers_t=df_cvr_numbers.T

columns =[]
for i in  df_cvs_numbers_t.columns.tolist():
    columns.append(df_cvs_numbers_t[i]['cou'])
# I've added the other data assuming same order of countries and same indices.
df_cvs_numbers_t.columns=columns 
df_cvd_numbers_t.columns=columns 
df_cvr_numbers_t.columns=columns 

# add date column 
df_cvs_numbers_t=df_cvs_numbers_t.drop('cou', axis=0)
df_cvd_numbers_t=df_cvd_numbers_t.drop('cou', axis=0)
df_cvr_numbers_t=df_cvr_numbers_t.drop('cou', axis=0)

df_cvs_numbers_t['date']=date
df_cvd_numbers_t['date']=date
df_cvr_numbers_t['date']=date

# make data frame containing 
#['cou', 'date', 'confirmed', 'death', ', 'recovered']
# 1. create column cou including codes for each country 
# 2. list all data for a country in other columns in same rows of that country.
codes=columns # same variable used to name columns in df_cvx_numbers_t
cou_column=[]
all_dates=[]
all_confirmed, all_deaths, all_recovered=[], [], []
for code in codes:
    for i in range(len(df_cvs_numbers_t['date'])):
        cou_column.append(code)
    for item in df_cvs_numbers_t['date']:
        all_dates.append(item)
    for item in df_cvs_numbers_t[code]:
        all_confirmed.append(item)
    for item in df_cvd_numbers_t[code]:
        all_deaths.append(item)
    for item in df_cvr_numbers_t[code]:
        all_recovered.append(item)

df_cv=df_corona_virus=pd.DataFrame({})

df_cv['cou']=cou_column 
df_cv['date']=all_dates
df_cv['confirmed']=all_confirmed 
df_cv['deaths']=all_deaths 
df_cv['recovered']=all_recovered


