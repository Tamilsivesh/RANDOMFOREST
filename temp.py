import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv('E:/Student/Iris.csv')
data.head()
x=data.drop("Species",axis=1)
y=data["Species"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
cs=RandomForestClassifier()
cs.fit(x_train,y_train)
y_pred=cs.predict(x_test)
from sklearn import metrics
print("acuracy:", metrics.accuracy_score(y_test,y_pred))





