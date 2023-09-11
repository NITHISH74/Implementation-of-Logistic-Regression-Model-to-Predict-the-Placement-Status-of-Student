# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm:
### Step 1:
Import pandas library to read CSV or excel files.
### Step 2:

Import LabelEncoder using sklearn.preprocessing library.
### Step 3:

Transform the data's using LabelEncoder.
### Step 4:

Import Logistic regression module from sklearn.linear_model library to predict the values.
### Step 5:

Find accuracy, confusion matrix ,and classification report using sklearn.metrics library.
### Step 6:

Predict for the new given values. End of the program.

## Program:
```python
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NITHISHWAR S
RegisterNumber:  212221230071
*/

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data.isnull().sum()
data1.head()
data.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(a1["gender"])
data1["ssc_b"]=le.fit_transform(a1["ssc_b"])
data1["hsc_b"]=le.fit_transform(a1["hsc_b"])
data1["hsc_s"]=le.fit_transform(a1["hsc_s"])
data1["degree_t"]=le.fit_transform(a1["degree_t"])
data1["workex"]=le.fit_transform(a1["workex"])
data1["specialisation"]=le.fit_transform(a1["specialisation"])
data1["status"]=le.fit_transform(a1["status"])
print (data1)
x=data1.iloc[:,:-1]
y=data1["status"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
c=LogisticRegression(solver="liblinear")
c.fit(x_train,y_train)
y_pred=c.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
acur=accuracy_score(y_test,y_pred)
acur
from sklearn.metrics import confusion_matrix
con=confusion_matrix(y_test,y_pred)
print(con)
from sklearn.metrics import classification_report
class_report=classification_report(y_test,y_pred)
print(class_report)
print(c.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))
```

## OUTPUT:

## Data.head():
![image](https://user-images.githubusercontent.com/94164665/167389544-42034a61-4769-4b25-8c3b-284d1f0e922b.png)

## ACCURACY:
![image](https://user-images.githubusercontent.com/94164665/167389578-b639ffce-4efb-424d-8293-8edcfe67aba6.png)

## CONFUSION MATRIX:
![image](https://user-images.githubusercontent.com/94164665/167389627-4f98462f-89c4-42b1-b4b3-e662c27db685.png)

## CLASS REPORT:
![image](https://user-images.githubusercontent.com/94164665/167389683-92491fd0-5798-45b9-a9d1-b74fbe3fed8f.png)

## PREDICTION(Y):
![image](https://user-images.githubusercontent.com/94164665/167389747-1e105eea-add5-47e4-8266-4168e544f8ad.png)

## FINAL PREDICTION:
![image](https://user-images.githubusercontent.com/94164665/167389964-f683b40c-f2bb-4d92-9098-72e88e1a264f.png)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
