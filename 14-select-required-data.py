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

