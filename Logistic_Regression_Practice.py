#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:03:13 2018

@author: thileepan

Using Logistic Regression on direct marketing phone calls dataset of a Portugese 
bank to predict if a  customer will subscribe to a term deposit. 

The dataset can be downloaded from UCI Macine Learning Repository 
link to dataset: archive.ics.uci.edu/ml/index.php 

***Note: THIS IS NOT MY OWN WORK. THIS WORK IS PART OF MY MACHINE LEARNING PRACTICE AND 
IS A REPLICATION FROM THE FOLLOWING LINK
https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
****
----------------------------------------------------------
WORKFLOW

1. IMPORTING AND DATA LOADING
2. EXPLORING DATA AND VISUALIZING DEPENDENCIES
3. FEATURE ENGINEERING AND FEATURE SELECTION
4. MODEL FITTING AND EVALUATION
----------------------------------------------------------

"""


#---------------  IMPORTING NECESSARY MODULES ----------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


#--------------- READING THE DATA------------------------------------

data = pd.read_csv('bank-additional-full.csv', header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))

        #--------PRINTING THE DATA TYPE OF EACH FEATURES------------
for i in list(data.columns):
    print("data type of " + i + " is {}".format (data[i].dtype) )
    print (data[i]. unique())


#-------------- EXPLORING DATA --------------------------

""" From the previous print statement, we notice that, 'education' 
feature has some similar categories like 'basic.4y' 'basic.6y' 'basic.9y' 

data type of education is object
['basic.4y' 'high.school' 'basic.6y' 'basic.9y' 'professional.course'
 'unknown' 'university.degree' 'illiterate']

So, we are going to combine these three categories into a single category
'basic'

"""

data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])


print (data['education']. unique()) #now we have only one category 'basic'

data['y'].value_counts() # finding how many yes and no categories we have in the output variable

#sns gets the value of 'y' from 'data' and plots a histogram
sns.countplot(x='y', data=data, palette='hls')
plt.show()

#Let's group the data by 'y' and see the mean of each variable

data.groupby('y').mean()
data.groupby('job').mean()
data.groupby('marital').mean()
data.groupby('education').mean()

# Visualizing the purchase frequency for each feature
        #Purchase frequecy for each job category#
pd.crosstab(data.job, data.y). plot(kind='bar')
plt.title('Purchase Frequency for job title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')

        #purchase frequency for marital status#
pd.crosstab(data.marital, data.y).plot(kind='bar')
plt.title('Purchase Frequency for marital status')        
plt.xlabel('Marital Status')
plt.ylabel('Frequency of Purchase')

        #purchase frequency vs education
pd.crosstab(data.education, data.y).plot(kind='bar')
plt.xlabel('Education')
plt.ylabel('Purchase Frequency')
plt.title('Purchase Frequency according to Education')

        #Purchase Frequency for day of the week
pd.crosstab(data.day_of_week, data.y).plot(kind='bar')
plt.title('Purchase Frequency for each day')
plt.xlabel('Day of the week')
plt.ylabel('Purchase Frequency')

        #Purchase frequency by month
pd.crosstab(data.month, data.y).plot(kind='bar')
plt.title('Purchase Frequecy for each month')
plt.xlabel('Month')
plt.ylabel('Purchase Frequency')


         #How many people have credit default
sns.countplot(x='default', data=data)
plt.show()

        #How many people have a housing loan
sns.countplot(x='housing', data=data)
plt.show()

        #How many people have a personal loan
        
sns.countplot(x='loan', data=data)
plt.show()

        #What was the outcome of previous marketing campaing
sns.countplot(x='poutcome', data=data)
plt.show()

#------------FEATURE ENGINEERING AND SELECTION--------------

        #encoding the features with categorical variables with dummy variables
        
cat_vars=['job','marital','education','default','housing','loan','contact',
          'month','day_of_week','poutcome'] # List of categorical variables

for var in cat_vars:
    #cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
    
        
        #This is how the final data will look
data_final=data[to_keep]
print("Features in the final data set is: {}".format( data_final.columns.summary()))
       
        #Seperating 'X' and 'y' from the final dataset
data_final_features= data_final.columns.values.tolist()
y=['y']
X = [i for i in data_final_features if i not in y]

        #finding correlation between features
sns.heatmap(data_final.corr())


        #Feature Selection using RFE
from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

rfe= RFE(logreg, 18)
rfe= rfe.fit(data_final[X], data_final[y])
print(rfe.support_)
print(rfe.ranking_)

        #Choosing only the features that were selected by rfe
cols=["previous", "euribor3m", "job_blue-collar", "job_retired", "job_services", "job_student", "default_no", 
   "month_aug", "month_dec", "month_jul", "month_nov", "month_oct", "month_sep", "day_of_week_fri", "day_of_week_wed", 
  "poutcome_failure", "poutcome_nonexistent", "poutcome_success"] 
#cols = ["poutcome_success"]
X=data_final[cols]
y=data_final['y']

#------------MODEL FITTING AND EVALUATION-------------------------------------
        
        #Splitting the data into traiing and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg.fit(X_train, y_train)
print("Accuracy of Logistic Regression Classifier on test_set: {}".
      format(logreg.score(X_test, y_test)))


        #Cross Validation

kfold= model_selection.KFold(n_splits=10, random_state= 7)
results = model_selection.cross_val_score(LogisticRegression(), X_train, y_train, 
                                          cv= kfold, scoring='accuracy')
print("10-fold Cross Validation Accuracy is: {}". format(results.mean()))

        #Evaluation 
y_pred= logreg.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print(classification_report(y_test, y_pred))

