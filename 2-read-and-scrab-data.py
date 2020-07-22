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

