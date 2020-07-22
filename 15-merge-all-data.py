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

