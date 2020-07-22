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

