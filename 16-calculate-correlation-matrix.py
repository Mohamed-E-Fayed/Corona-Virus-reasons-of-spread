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


