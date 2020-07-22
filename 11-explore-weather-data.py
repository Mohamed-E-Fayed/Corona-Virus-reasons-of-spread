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
