#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import plotly.express as px
import streamlit as st
import os


# In[6]:


os.chdir('C:\\Users\\adima\\OneDrive\\Documents\\Cohort 128_ML_ Day 43\\Stremlit Deployment and PBI Dashboard\\')


# In[7]:


car_data=pd.read_csv("car_data.csv")
display(car_data)


# In[8]:


np.unique(car_data['Trim'].fillna('0'))
len(np.unique(car_data['Trim'].fillna('0')))


# In[9]:


len(np.unique(car_data['Make'].fillna('0')))


# In[10]:


for column in car_data:
    unique_vals=np.unique(car_data[column].fillna('0'))
    num_vals=len(unique_vals)
    if num_vals < 12:
        print('the unique values of column {}:{}--{}'.format(column,num_vals,unique_vals))
    else:
        print('the unique values of column {}:{}'.format(column,num_vals))


# In[11]:


print(car_data.isnull().sum())


# In[12]:


car_data.shape


# In[13]:


car_data=car_data.drop(['Invoice Price','Cylinders','Highway Fuel Economy'],axis=1)
display(car_data)


# In[14]:


car_data["Horsepower_No"]=car_data["Horsepower"].str[0:3].astype(float)
display(car_data["Horsepower_No"])


# In[15]:


display(car_data[car_data["Horsepower_No"].isna()])


# In[16]:


mean_HP=car_data["Horsepower_No"][car_data["Make"]=="Ford"].mean()
print(mean_HP)


# In[17]:


car_data["Horsepower_No"]=car_data["Horsepower_No"].fillna(mean_HP)
car_data["Horsepower"]=car_data["Horsepower"].fillna(mean_HP)
display(car_data[car_data["Horsepower"].isna()])


# In[18]:


car_data["Torque_No"]=car_data["Torque"].str[0:3].astype(float)
display(car_data)


# In[19]:


mean_tor=car_data['Torque_No'][car_data['Make']=='Ford'].mean()
car_data['Torque_No']=car_data['Torque_No'].fillna(mean_tor)
car_data['Torque']=car_data['Torque'].fillna(mean_tor)


# In[20]:


display(car_data[car_data['Torque_No'].isna()])


# In[21]:


display(car_data.isna().sum())


# In[22]:


print(car_data.dtypes)


# In[23]:


display(car_data)
car_data['MSRP']=car_data['MSRP'].str.replace('$','')
car_data['MSRP']=car_data['MSRP'].str.replace(',','').astype(float)


# In[24]:


car_data['Used/New Price']=car_data['Used/New Price'].str.replace('$','')
car_data['Used/New Price']=car_data['Used/New Price'].str.replace(',','').astype(float)
display(car_data)


# In[25]:


sns.pairplot(car_data)
plt.show()


# In[26]:


sns.pairplot(car_data[['MSRP','Horsepower_No','Torque_No','Engine Aspiration']],hue='Engine Aspiration',height=5)
plt.show()


# In[27]:


sns.pairplot(car_data[['MSRP','Horsepower_No','Torque_No','Make']],hue='Make',height=5)
plt.show()


# In[28]:


categories=['Make','Body Size','Body Style', 'Engine Aspiration', 'Drivetrain','Transmission']


# In[29]:


sns.set(rc={'figure.figsize':(14,5)})


# In[30]:


for c in categories:
    ax=sns.barplot(x=c,y="MSRP",data=car_data,errorbar=('ci',False))
    for container in ax.containers:
        ax.bar_label(container)
    plt.title(c)
    plt.show()


# In[31]:


n_categories=['MSRP','Used/New Price','Horsepower_No','Torque_No']


# In[32]:


sns.set(rc={'figure.figsize':(10,5)})


# In[33]:


for n in n_categories:
    x=car_data[n].values
    sns.displot(x,color='blue')
    mean=car_data[n].mean()
    plt.axvline(x=mean,ymin=0,ymax=1,color='red')
    plt.title(n)
    plt.show()


# In[34]:


sns.set(rc={'figure.figsize':(8,5)})
for c in n_categories:
    x=car_data[c].values
    sns.boxplot(x,color='k')
    print('the median of ',c,'is: ',car_data[c].median())
    plt.title(c)
    plt.show()


# In[35]:


categories = ['Make','Body Size','Body Style', 'Engine Aspiration', 'Drivetrain','Transmission']


# In[38]:


import warnings
warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize':(8,5)})
for c in categories:
    sns.boxplot(x=c,y='MSRP',data=car_data,color='k')
    sns.swarmplot(x=c,y='MSRP',data=car_data,color='r')
    plt.title(c)
    plt.show()


# In[40]:


new_car_data=car_data.drop(['index','Model','Year', 'Trim', 'Used/New Price', 'Horsepower', 'Torque'], axis=1)


# In[41]:


new_car_data=pd.get_dummies(new_car_data,columns=['Make','Body Size','Body Style', 'Engine Aspiration', 'Drivetrain','Transmission'],dtype=int)
display(new_car_data)


# In[43]:


n_variables=['MSRP','Horsepower_No','Torque_No']
pc=new_car_data[n_variables].corr(method='pearson')
print(pc)


# In[44]:


sns.set(rc={'figure.figsize':(12,4)})
cols=n_variables
ax=sns.heatmap(pc,annot=True,yticklabels=cols,xticklabels=cols,annot_kws={'size':10},cmap='Blues')


# In[45]:


x=new_car_data.drop(['MSRP'],axis=1).values
print(x.shape)


# In[47]:


y=new_car_data['MSRP'].astype(int)
print(y.shape)


# In[48]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=15)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[51]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,explained_variance_score,mean_absolute_error,mean_squared_error
lr=LinearRegression(fit_intercept=True)
model=lr.fit(x_train,y_train)
y_pred=lr.predict(x_train)
y_final_predict=lr.predict(x)
print(y_final_predict)


# In[53]:


accuracy=r2_score(y_train,y_pred)
final_accuracy=r2_score(y,y_final_predict)
print(accuracy,final_accuracy)


# In[54]:


print(lr.score(x_train,y_train))


# In[55]:


print(lr.score(x_test,y_test))


# In[57]:


from math import sqrt
print(sqrt(mean_squared_error(y_train,y_pred)))


# In[59]:


print(sqrt(mean_squared_error(y_test,lr.predict(x_test))))
print(mean_absolute_error(y_train,y_pred))
print(mean_absolute_error(y_test,lr.predict(x_test)))


# In[60]:


print("coeff",lr.coef_)
print("intercept",lr.intercept_)


# In[64]:


#pickle file deployment
pickle.dump(lr,open('linear_model.pkl','wb'))
model=pickle.load(open('linear_model.pkl','rb'))
print(model.predict(x_test))


# In[ ]:





# In[ ]:




