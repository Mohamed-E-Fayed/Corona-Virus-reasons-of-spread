# # Class 1

# In[12]:


country_data,date = get_country_data("D:\\optimization project\\time_series_19-covid-Confirmed.csv","Germany")
decumiltative_country_data = decumiltate(country_data)
time_axis = np.arange(len(date))


# # checking graph for class 1

# In[13]:


plt.plot(time_axis,country_data)


# In[14]:


plt.plot(time_axis[30:66],decumiltative_country_data[30:66])


# In[15]:


plt.plot(time_axis,decumiltative_country_data)


# In[16]:


plt.plot(time_axis[30:57],decumiltative_country_data[30:57])


# # fit data

# In[17]:


#fit time axis
def func(x, a, b, c, d):
    return a*x**3 + b*x**2 +c*x + d
y = decumiltative_country_data[30:57]
x = time_axis[30:57]
yn = y + 0.2*np.random.normal(size=len(x))
popt, pcov = curve_fit(func, np.array(x), np.array(yn))


# # fitted expo curve

# In[19]:


#fit on training
plt.figure()
plt.plot(x, yn, 'ko', label="Original Noised Data")
plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
plt.legend()
plt.show()


# # use fitted curve from above to predict increase in cases per day

# In[20]:


#test
y_test = decumiltative_country_data[30:66]
x_test = time_axis[30:66]
yn_test = y_test + 0.2*np.random.normal(size=len(x_test))
plt.figure()
plt.plot(x_test, yn_test, 'ko', label="Original Noised Data")
plt.plot(x_test, func(x_test, *popt), 'r-', label="Fitted Curve")
plt.legend()
#x_min = 30
#x_max = 66
#plt.axis([x_min, x_max, 0, 7000])
plt.show()


# In[50]:


#data preperation
location = "D:\\optimization project\\berlin.csv"
data_uploaded = pd.read_csv(location)
data = np.array(data_uploaded)
tempreture_data = data[30594:30594+80]
deaths,recovered = get_death_recovered("D:\\optimization project\\data.csv",'DEU')
t_max,t_min = get_tempreture("D:\\optimization project\\berlin.csv")

deaths,recovered = get_death_recovered("D:\\optimization project\\data.csv",'DEU')
deaths = deaths[30:66]
recovered = recovered[30:66]
t_max,t_min = get_tempreture("D:\\optimization project\\berlin.csv")
t_max = t_max[30:66]
t_min = t_min[30:66]
day_yesterday_1 = decumiltative_country_data[29:65]
day_yesterday_2 = decumiltative_country_data[28:64]
day_yesterday_3 = decumiltative_country_data[27:63]
day_yesterday_4 = decumiltative_country_data[26:62]
day_yesterday_5 = decumiltative_country_data[25:61]
day_yesterday_6 = decumiltative_country_data[24:60]
day_yesterday_7 = decumiltative_country_data[23:59]
day_yesterday_8 = decumiltative_country_data[22:58]
day_yesterday_9 = decumiltative_country_data[21:57]
day_yesterday_10 = decumiltative_country_data[20:56]
day_yesterday_11 = decumiltative_country_data[19:55]
day_yesterday_12 = decumiltative_country_data[18:54]
day_yesterday_13 = decumiltative_country_data[17:53]
day_yesterday_14 = decumiltative_country_data[16:52]
x_nn = np.transpose(np.vstack((t_max,t_min,deaths,recovered,day_yesterday_1,day_yesterday_2,day_yesterday_3,day_yesterday_4,day_yesterday_5,day_yesterday_6,day_yesterday_7,day_yesterday_8,day_yesterday_9,day_yesterday_10,day_yesterday_11,day_yesterday_12,day_yesterday_13,day_yesterday_14)))
y_nn = decumiltative_country_data[30:66]


# In[56]:


#linear regression no regularization
reg = linear_model.Ridge(alpha=0)
reg = reg.fit(x_nn, y_nn)
y_predicted = reg.predict(x_nn)
plt.figure()
plt.plot(time_axis[30:66],y_predicted,'r-')
plt.plot(x_test, yn_test, 'ko', label="Original Noised Data")
plt.show()


# In[ ]:


#linear regression using regularization alpha = 10000000
clf = Ridge()
clf.set_params(alpha=100000000)
regression = clf.fit(x_nn, y_nn)
y_predicted_2 = regression.predict(x_nn)
plt.figure()
plt.plot(time_axis[30:66],y_predicted_2,'r-')
plt.plot(x_test, yn_test, 'ko', label="Original Noised Data")
plt.show()


# # Neural networks

# # Class 1

# In[53]:


model_class1 = get_model_for_Class(18)
model_class1.fit(x=x_nn, y=y_nn, epochs=100, batch_size=1)


# In[54]:


y_nn_predicted = model_class1.predict(x=x_nn)
plt.figure()
plt.plot(time_axis[30:66],y_nn_predicted,'r-')
plt.plot(x_test, yn_test, 'ko', label="Original Noised Data")
plt.show()


# # Class 2

# In[206]:


#germany, italy and united kingdom
country_edu_data,education_data = get_education_level("D:\\optimization project\\education_rankings (1).csv")
health_data = [3594338,2680279,1332924.02]
country_data_demo,pop_density_data,total_pop_data,life_expectancy_data = get_population_lifeexpectancy("D:\\optimization project\\worldometer_demography.csv")
country_data_demo[1],country_data_demo[2] = country_data_demo[2],country_data_demo[1]
pop_density_data[1],pop_density_data[2] = pop_density_data[2],pop_density_data[1]
total_pop_data[1],total_pop_data[2] = total_pop_data[2],total_pop_data[1]
life_expectancy_data[1],life_expectancy_data[2] = life_expectancy_data[2],life_expectancy_data[1]


# In[207]:


education_data = np.array(education_data)/(2000)
health_data = np.array(health_data)/(4*10^6)
pop_density_data = np.array(pop_density_data)/(3*10^2)
total_pop_data = np.array(total_pop_data)/(10^8)
life_expectancy_data = np.array(life_expectancy_data) / 100


# In[210]:


X_2 =np.transpose(np.vstack((education_data,health_data,pop_density_data,total_pop_data,life_expectancy_data)))
Y_2 = [177583,249534,226675]
model_class2 = get_model_for_Class2(5)
model_class2.fit(x=X_2, y=Y_2, epochs=1000, batch_size=1)
y_predict_class2 = model_class2.predict(x =X_2 )
print(y_predict_class2)


# # plot bar chart

# In[212]:


y_predict_class2 = y_predict_class2.squeeze()
labels = ['GER','UK','ITA']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, y_predict_class2 , width, label='predicted')
rects2 = ax.bar(x + width/2, Y_2, width, label='Actual')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()

