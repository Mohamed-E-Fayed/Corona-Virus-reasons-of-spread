# health data 
df_hq=df_hq.drop(['age group', 'flag codes', 'gen', 'ind', 'country'], axis=1) # I left COU instead of country since values in countries are not easily guessable. 
hq_corr=df_hq.corr()

print(separator)
print('•  Health quality correlation matrix:')
print(hq_corr)
print(separator)

