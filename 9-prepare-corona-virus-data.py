# prepare corona virus data for join 
non_date_columns=['country', 'province', 'long', 'lat', 'cou']
df_cvs_numbers=df_cvs.drop(['province', 'country', 'long', 'lat'], axis=1)
df_cvd_numbers=df_cvd.drop(['province', 'country', 'long', 'lat'], axis=1)
df_cvr_numbers=df_cvr.drop(['province', 'country', 'long', 'lat'], axis=1)
df_tmp=df_cvs.copy()
for col in non_date_columns:
    if col in df_cvs.columns.tolist():
        df_tmp=df_tmp.drop(col, axis=1)
date=df_tmp.copy()
del df_tmp

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

print('number of nulls in the corona virus data =',df_cv.isnull().sum().sum())

