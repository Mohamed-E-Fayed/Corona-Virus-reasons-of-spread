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
