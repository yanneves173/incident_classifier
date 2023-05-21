import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

###### This algoritm has the purpose of classify early warnings, which is news' description
###### that it's supposed to be defined as incident or not manually by a Team.
###### The main idea is using Machine learning to determine automatically if an alert is considered
###### critical incident or not based on the data already gathered manually by the Early Warning and Monitoring team.


#Function 1 - encoding data in 0 and 1
def convert_status(status):
    if status == "Triage/Closed":
        return 0
    if status == "Incident Created":
        return 1
    else:
        return -1
    
        ### Extracting data with pandas features

#Load the data and get desired columns on data --- Incident Report

data_incident = pd.read_csv(r'incident_report1.csv', 
                            encoding="ISO-8859-1")
df_inc = pd.DataFrame(data_incident)
df_inc = df_inc.dropna(subset =["Description"])
df_inc["is_incident"]= 1
df_inc = df_inc[['Description', 'is_incident']]


#Load the data and get desired columns --- Early Warning Report

data_ew = pd.read_csv(r'earlywarning_report1.csv', 
                      encoding = "UTF-8")
df_ew = pd.DataFrame(data_ew)
df_ew = df_ew.dropna(subset =["Description"])
df_ew["is_incident"] = df_ew["Action status"].apply(convert_status)
df_ew = df_ew[['Description','is_incident']]

df= pd.concat([df_inc,df_ew],ignore_index=True)

incident_created = df[df['is_incident'] == 1 ]
triage_closed = df[df['is_incident'] == 0 ]

print("incident porcentage:\n", (len(incident_created)/len(df))*100, '%')
print("Not incident porcentage:\n", (len(triage_closed)/len(df))*100, '%')

df_x =df["Description"]
df_y = df["is_incident"]

#spliting data in  - 80% for training &&&&& 20% test size

x_train,x_test,y_train,y_test = train_test_split(df_x,df_y, train_size = 0.8, test_size = 0.2, random_state=4)

#Using TFIDF VECTORIZER to preprocess the data (feature extraction, conversion to lower case and removal of stop words) 

tfvec = TfidfVectorizer(min_df=1, stop_words='english', lowercase= True)
x_trainFeat = tfvec.fit_transform(x_train)
x_testFeat = tfvec.transform(x_test)

#SVM is used to model

y_trainSvm = y_train.astype('int')
classifierModel = LinearSVC()
classifierModel.fit(x_trainFeat, y_trainSvm)
predResult = classifierModel.predict(x_testFeat)

# GNB is used to model
y_trainGnb = y_train.astype('int')
classifierModel2 = MultinomialNB()
classifierModel2.fit(x_trainFeat, y_trainGnb)
predResult2 = classifierModel2.predict(x_testFeat)

# Calc accuracy,converting to int - solves - cant handle mix of unknown and binary
y_test = y_test.astype('int')
actual_Y = y_test.values

print("~~~~~~~~~~SVM RESULTS~~~~~~~~~~")

#Accuracy score using SVM
print("Accuracy Score using SVM: {0:.4f}".format(accuracy_score(actual_Y, predResult)*100))
#FScore MACRO using SVM
print("F Score using SVM: {0: .4f}".format(f1_score(actual_Y, predResult, average='macro')*100))
cmSVM=confusion_matrix(actual_Y, predResult)
#"[True negative  False Positive\nFalse Negative True Positive]"
print("Confusion matrix using SVM:")
print(cmSVM)

print("~~~~~~~~~~MNB RESULTS~~~~~~~~~~")
#Accuracy score using MNB
print("Accuracy Score using MNB: {0:.4f}".format(accuracy_score(actual_Y, predResult2)*100))
#FScore MACRO using MNB
print("F Score using MNB:{0: .4f}".format(f1_score(actual_Y, predResult2, average='macro')*100))
cmMNb=confusion_matrix(actual_Y, predResult2)
#"[True negative  False Positive\nFalse Negative True Positive]"
print("Confusion matrix using MNB:")
print(cmMNb)



print("Final preprocessing")
