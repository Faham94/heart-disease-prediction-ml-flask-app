import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('heart (1).csv')
print(df.head())
#eda

print(df.info())
print(df.describe())
print(df.duplicated().sum())
print(df.isnull().sum())

# def plotting(var,num):
#     plt.subplot(2,2,num)
#     sns.histplot(df[var],kde=True)

# plotting('Age',1)
# plotting('RestingBP',2)
# plotting('Cholesterol',3)
# plotting('MaxHR',4)
# plt.tight_layout()
# plt.show()

#ab graph main cholestrol aur resting bp glt tha google pe dekha glt plit tha to sahi krenge

ch_mean=df.loc[df['Cholesterol']!=0,'Cholesterol'].mean()
df['Cholesterol']=df['Cholesterol'].replace(0,ch_mean)
df['Cholesterol']=df['Cholesterol'].round(2)

resting_bp_mean=df.loc[df['RestingBP']!=0,'RestingBP'].mean()
df['RestingBP']=df['RestingBP'].replace(0,resting_bp_mean)
df['RestingBP']=df['RestingBP'].round(2)

def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(df[var],kde=True)

plotting('Age',1)
plotting('RestingBP',2)
plotting('Cholesterol',3)
plotting('MaxHR',4)
plt.tight_layout()
plt.show()


sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()

#data prepocsiing
#convert kra h vals ko 
df_encode=pd.get_dummies(df,drop_first=True)
print(df_encode)

#now importing models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

x=df_encode.drop('HeartDisease',axis=1)
y=df_encode['HeartDisease']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#kuch models m krna prta h standardscalar tbhi lagaua h
scalar = StandardScaler()
x_train_scaled = scalar.fit_transform(x_train)
x_test_scaled = scalar.transform(x_test)        

#ab saray models one go main chahye train krne hain

models={
    "LogisticRegression":LogisticRegression(),
    "KNN":KNeighborsClassifier(),
    "NaiveBayes":GaussianNB(),
    "DecisionTrees":DecisionTreeClassifier(),
    "StandardVectorMachine":SVC()
}

result=[]
for name,model in models.items():
    model.fit(x_train_scaled,y_train)
    y_pred=model.predict(x_test_scaled)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    
    result.append({
        'model':name,
        'accuracy':round(acc,4),
        'f1 score':round(f1,4)
    })

print(result)

#svm ka sbse acha aya h usi ko use krenge ab ese krty
import joblib
joblib.dump(models['StandardVectorMachine'],'SVM_heart.pkl')
joblib.dump(scalar,'scalar.pkl')
joblib.dump(x.columns.tolist(),'columns.pkl')
