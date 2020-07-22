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

