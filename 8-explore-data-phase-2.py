#find correlation among columns 
hq_corr= df_health_quality.corr() #hq=health_quatliy 
hr_corr= df_health_resources.corr() # hr=health_resources.

print(separator)

print('Correlation matrix:')
print('• Health quality data:')
print(hq_corr)
print(separator)
"""
print('• Health resources data:')
print(hr_corr)
print(separator)
"""

