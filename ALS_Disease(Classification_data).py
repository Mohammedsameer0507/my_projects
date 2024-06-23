#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\data science\internship\Minsk2020_ALS_dataset.csv")
df.head(5)


# In[3]:


df.shape


# In[4]:


df['Diagnosis (ALS)'].value_counts()


# In[5]:


df.columns.unique()


# In[6]:


df["Sex"].value_counts()


# In[7]:


df.isnull().sum().max()


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'])
df


# In[11]:


# Ensure that all values in x are non-negative
x=df.iloc[:,:-1].clip(lower=0)
y=df.iloc[:,-1]

from sklearn.feature_selection import SelectKBest,chi2,f_classif

chi_best=SelectKBest(chi2,k=10)

k_best=chi_best.fit(x,y)

# Continue with the rest of your feature selection and modeling process


# In[12]:


np.set_printoptions(precision=3)


# In[13]:


print(k_best.scores_)


# In[14]:


k_best.get_feature_names_out()


# In[15]:


features=df[['ID','Age','Sex', 'CCa(6)', 'CCa(8)', 'CCa(9)', 'CCi(2)', 'CCi(3)', 'CCi(6)',
       'CCi(8)', 'F2_i', 'F2_{conv}']]


# In[16]:


features.head()


# In[17]:


target=df['Diagnosis (ALS)']


# In[18]:


target.tail()


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=10)


# In[21]:


corr=features.corr()


# In[22]:


sns.heatmap(corr,annot=True,cmap='coolwarm',)
plt.show()


# In[23]:


n_columns = ['ID','Sex','Age', 'CCa(6)', 'CCa(8)', 'CCa(9)', 'CCi(2)', 'CCi(3)', 'CCi(6)','CCi(8)', 'F2_i', 'F2_{conv}']
plt.figure(figsize=(8,10))
for i, col in enumerate(n_columns):
    plt.subplot(4,4,i+1)
    sns.boxplot(x=n_columns[i], data=df)
    plt.title(f' {col}')
plt.tight_layout()
plt.show()
#  'Sex' as it doesn't make sense in a boxplot


# In[24]:


n_columns = ['ID','Age','Sex', 'CCa(6)', 'CCa(8)', 'CCa(9)', 'CCi(2)', 'CCi(3)', 'CCi(6)','CCi(8)', 'F2_i', 'F2_{conv}']
df[n_columns].boxplot(figsize=(15,10))
plt.tight_layout()
plt.show()


# In[25]:


sns.countplot(y=df['Sex'], order=df['Sex'].value_counts().index)
plt.title('Count of Sex')
plt.tight_layout()
plt.show()


# In[26]:


sns.countplot(x=df['Diagnosis (ALS)'],order=df['Diagnosis (ALS)'].value_counts().index,hue=df['Diagnosis (ALS)'])
plt.title('Count of Diagnosis (ALS)')
plt.tight_layout()
plt.show()


# In[27]:


plt.figure(figsize=(10,10))
plt.hist(df['Age'])
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[28]:


numerical_columns = ['ID','Age','Sex', 'CCa(6)', 'CCa(8)', 'CCa(9)', 'CCi(2)', 'CCi(3)', 'CCi(6)','CCi(8)', 'F2_i', 'F2_{conv}']
df[numerical_columns].hist(figsize=(15, 10), bins=20, edgecolor='black')
plt.suptitle('Histogram of Numerical Columns')
plt.tight_layout()
plt.show()
#In the sex columns there only two values 0 and 1,we get only count of males and females graph


# In[29]:


selected_columns =['ID','Age','Sex', 'CCa(6)', 'CCa(8)', 'CCa(9)', 'CCi(2)', 'CCi(3)', 'CCi(6)','CCi(8)', 'F2_i', 'F2_{conv}','Diagnosis (ALS)']
selected_data = df[selected_columns]

# Create pairplot
#sns.pairplot(selected_data, hue='Diagnosis (ALS)', diag_kind='kde')

# Show plot
plt.show()


# In[30]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score,ShuffleSplit,StratifiedKFold,LeaveOneOut


# In[31]:


model= [DecisionTreeClassifier(criterion='gini',max_depth=3,min_samples_leaf=10,random_state=100),
        RandomForestClassifier(),BaggingClassifier(),LogisticRegression(),SVC(),KNeighborsClassifier(),GaussianNB()]


# In[32]:


import warnings
warnings.filterwarnings('ignore')


# In[33]:


for i in model:
  kfold = KFold(n_splits=10, shuffle=True, random_state=100)
  score= cross_val_score(i,X=features,y=target,cv=kfold)
  print(i)
  print("Accuracy",score.mean())
  print("min :",score.min())
  print("max :",score.max())
  print("----"*10)


# In[35]:


for i in model:
  shuffle_split = ShuffleSplit(n_splits=10, test_size=0.2, random_state=100)
  score= cross_val_score(i,X=features,y=target,cv=shuffle_split)
  print(i)
  print("Accuracy :",score.mean())
  print("Min :",score.min())
  print("max :",score.max())
  print("----"*10)


# In[36]:


for i in model:
  stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=100)
  score= cross_val_score(i,X=features,y=target,cv=stratified_kfold)
  print(i)
  print("Accuracy :",score.mean())
  print("min :",score.min())
  print("max :",score.max())
  print("----"*10)


# In[37]:


for i in model:
  leave_one_out = LeaveOneOut()
  score= cross_val_score(i,X=features,y=target,cv=leave_one_out)
  print(i)
  print("Accuracy :",score.mean())
  print("min :",score.min())
  print("max :",score.max())
  print("----"*10)


# In[38]:


# using the grid search to get the best hyperparameters
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


# In[39]:


#DecisionTreeClassifier
model_1=DecisionTreeClassifier()
grid={
    'criterion':['gini','entropy'],
    'max_depth':[2,4,6,8,10],
    'min_samples_split':[1,5,10],
    'max_features':['sqrt','log2'],
    'random_state':[5,10,15,20]
}
gscv=GridSearchCV(estimator=model_1,param_grid=grid,cv=5)
gscv.fit(x_train,y_train)


# In[40]:


print("Decisiontreeclassifier best score :-",gscv.best_score_)
gscv.best_params_


# In[41]:


rscv=RandomizedSearchCV(estimator=model_1,param_distributions=grid,cv=5)
rscv.fit(x_train,y_train)


# In[42]:


print("RandomizedSearchCV best score :-",rscv.best_score_)
rscv.best_params_


# In[43]:


#RandomForestClassifier
model_2=RandomForestClassifier()
gscv=GridSearchCV(estimator=model_2,param_grid=grid,cv=5)
gscv.fit(x_train,y_train)


# In[44]:


print("RandomForestClassifier best score :-",gscv.best_score_)
gscv.best_params_


# In[45]:


rscv=RandomizedSearchCV(estimator=model_2,param_distributions=grid,cv=5)
rscv.fit(x_train,y_train)


# In[46]:


print("RandomizedSearchCV best score :-",rscv.best_score_)
rscv.best_params_


# In[47]:


#Bagging classifier
model_3=BaggingClassifier()
grid3={
    'n_estimators':[10,20,30,40,50],
    'max_samples':[0.5,0.7,0.9,1.0],
    'max_features':[0.5,0.7,0.9,1.0],
    'random_state':[5,10,15,20]
}


# In[48]:


gscv=GridSearchCV(estimator=model_3,param_grid=grid3,cv=5)
gscv.fit(x_train,y_train)


# In[49]:


print("baggingClassifier best score",gscv.best_score_)
gscv.best_params_


# In[50]:


rscv=RandomizedSearchCV(estimator=model_3,param_distributions=grid3,cv=5)
rscv.fit(x_train,y_train)


# In[51]:


print("RandomizedSearchCV best score :-",rscv.best_score_)
rscv.best_params_


# In[52]:


#logistic Regression
model_4=LogisticRegression()
grid4={
    'C':[1,2,3,4,5],
    'max_iter':[50,100,150,200]
}


# In[53]:


gscv=GridSearchCV(estimator=model_4,param_grid=grid4,cv=5)
gscv.fit(x_train,y_train)


# In[54]:


print("LogisticRegression best score",gscv.best_score_)
gscv.best_params_


# In[55]:


rscv=RandomizedSearchCV(estimator=model_4,param_distributions=grid4,cv=5)
rscv.fit(x_train,y_train)


# In[56]:


print("RandomizedSearchCV best score :-",rscv.best_score_)
rscv.best_params_


# In[57]:


#svc
model_5=SVC()
grid5={
    'C':[1,2,3,4,5],
    'max_iter':[50,100,150,200],
    'random_state':[5,10,15],
    'kernel':['linear','poly','sigmoid']
}


# In[58]:


gscv=GridSearchCV(estimator=model_5,param_grid=grid5,cv=5)
gscv.fit(x_train,y_train)


# In[59]:


print("SVC best score",gscv.best_score_)
gscv.best_params_


# In[60]:


rscv=RandomizedSearchCV(estimator=model_5,param_distributions=grid5,cv=5)
rscv.fit(x_train,y_train)


# In[61]:


print("RandomizedSearchCV best score :-",rscv.best_score_)
rscv.best_params_


# In[62]:


#GaussianNB
model_6=GaussianNB()
grid6={
    'var_smoothing':[1e-9,1e-8,1e-7,1e-6]
}


# In[63]:


gscv=GridSearchCV(estimator=model_6,param_grid=grid6,cv=5)
gscv.fit(x_train,y_train)


# In[64]:


print("GaussianNB best score",gscv.best_score_)
gscv.best_params_


# In[65]:


rscv=RandomizedSearchCV(estimator=model_6,param_distributions=grid6,cv=5)
rscv.fit(x_train,y_train)


# In[66]:


print("GaussianNB best score",rscv.best_score_)
rscv.best_params_


# In[ ]:





# In[67]:


from sklearn.metrics import confusion_matrix


# In[68]:


#using the best hyperparameter
model_b= [DecisionTreeClassifier(criterion='gini',max_depth=2,min_samples_split=5,random_state=15),
          RandomForestClassifier(criterion='entropy',max_depth=8,min_samples_split=5,random_state=5),
          BaggingClassifier(max_features=0.9,max_samples=0.9,n_estimators=10,random_state=5),LogisticRegression(max_iter=50,C=4),SVC(random_state=5,max_iter=100,kernel='poly',C=2),KNeighborsClassifier(),GaussianNB()]


# In[69]:


for i in model_b:
    i.fit(x_train,y_train)
    y_pred=i.predict(x_test)
    print(i)
    print(classification_report(y_test,y_pred))
    cm=confusion_matrix(y_test,y_pred)
    print(cm)
    sns.heatmap(cm,annot=True,fmt='d').figsize=(4,5)
    plt.show()
    


# In[70]:


rfc=RandomForestClassifier(criterion='entropy',max_depth=8,min_samples_split=5,random_state=5)


# In[71]:


rfc.fit(x_train,y_train)


# In[72]:


y_pred=rfc.predict(x_test)


# In[73]:


r_probs=[0 for _ in range (len(y_test))]


# In[74]:


rf_probs=rfc.predict_proba(x_test)


# In[75]:


rf_probs=rf_probs[:,1]


# In[76]:


from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve,PrecisionRecallDisplay,precision_recall_fscore_support


# In[77]:


rf_auc=roc_auc_score(y_test,rf_probs)


# In[78]:


print('Randomforest :AUROC= % 0.3f' % (rf_auc))


# In[79]:


rf_fpr,rf_tpr,_=roc_curve(y_test,rf_probs)


# In[80]:


plt.plot(rf_fpr,rf_tpr)
plt.show()


# In[81]:


from sklearn.preprocessing import StandardScaler


# In[82]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

# Train a classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

# Predict probabilities
y_probs = clf.predict_proba(X_test_scaled)[:, 1]

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.title('Precision-Recall Curve')
plt.show()


# In[83]:


clf = LogisticRegression(random_state=42)
clf.fit(X_train_scaled, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test_scaled)

# Compute precision, recall, and F1 scores for each class
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1])

# Class labels
class_labels = ['Non-ALS', 'ALS']

# Compute macro-averaged precision, recall, and F1 scores
precision_macro_avg = np.mean(precision)
recall_macro_avg = np.mean(recall)
f1_macro_avg = np.mean(f1_score)

# Plotting the precision, recall, and F1 scores for each class
x = np.arange(len(class_labels))

plt.figure(figsize=(10, 6))
plt.plot(x, precision, label='Precision', marker='o', linestyle='-')
plt.plot(x, recall, label='Recall', marker='o', linestyle='-')
plt.plot(x, f1_score, label='F1 Score', marker='o', linestyle='-')
plt.axhline(y=precision_macro_avg, color='r', linestyle='--', label='Precision Macro Avg')
plt.axhline(y=recall_macro_avg, color='g', linestyle='--', label='Recall Macro Avg')
plt.axhline(y=f1_macro_avg, color='b', linestyle='--', label='F1 Macro Avg')
plt.xticks(x, class_labels)
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Logistic Regression\nPrecision, Recall, and F1 Score by Class')
plt.legend()
plt.show()


# In[84]:


#precision,recall,f1 scores for DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test_scaled)

# Compute precision, recall, and F1 scores for each class
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1])

# Class labels
class_labels = ['Non-ALS', 'ALS']

# Compute macro-averaged precision, recall, and F1 scores
precision_macro_avg = np.mean(precision)
recall_macro_avg = np.mean(recall)
f1_macro_avg = np.mean(f1_score)

# Plotting the precision, recall, and F1 scores for each class
x = np.arange(len(class_labels))

plt.figure(figsize=(10, 6))
plt.plot(x, precision, label='Precision', marker='o', linestyle='-')
plt.plot(x, recall, label='Recall', marker='o', linestyle='-')
plt.plot(x, f1_score, label='F1 Score', marker='o', linestyle='-')
plt.axhline(y=precision_macro_avg, color='r', linestyle='--', label='Precision Macro Avg')
plt.axhline(y=recall_macro_avg, color='g', linestyle='--', label='Recall Macro Avg')
plt.axhline(y=f1_macro_avg, color='b', linestyle='--', label='F1 Macro Avg')
plt.xticks(x, class_labels)
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('DecisionTreeClassifier\nPrecision, Recall, and F1 Score by Class')
plt.legend()
plt.show()


# In[85]:


#precision,recall,f1 scores for RandomForestClassifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test_scaled)

# Compute precision, recall, and F1 scores for each class
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1])

# Class labels
class_labels = ['Non-ALS', 'ALS']

# Compute macro-averaged precision, recall, and F1 scores
precision_macro_avg = np.mean(precision)
recall_macro_avg = np.mean(recall)
f1_macro_avg = np.mean(f1_score)

# Plotting the precision, recall, and F1 scores for each class
x = np.arange(len(class_labels))

plt.figure(figsize=(10, 6))
plt.plot(x, precision, label='Precision', marker='o', linestyle='-')
plt.plot(x, recall, label='Recall', marker='o', linestyle='-')
plt.plot(x, f1_score, label='F1 Score', marker='o', linestyle='-')
plt.axhline(y=precision_macro_avg, color='r', linestyle='--', label='Precision Macro Avg')
plt.axhline(y=recall_macro_avg, color='g', linestyle='--', label='Recall Macro Avg')
plt.axhline(y=f1_macro_avg, color='b', linestyle='--', label='F1 Macro Avg')
plt.xticks(x, class_labels)
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('RandomForestClassifier\nPrecision, Recall, and F1 Score by Class')
plt.legend()
plt.show()


# In[86]:


#precision,recall,f1 scores for GaussianNB
clf = GaussianNB()
clf.fit(X_train_scaled, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test_scaled)

# Compute precision, recall, and F1 scores for each class
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1])

# Class labels
class_labels = ['Non-ALS', 'ALS']

# Compute macro-averaged precision, recall, and F1 scores
precision_macro_avg = np.mean(precision)
recall_macro_avg = np.mean(recall)
f1_macro_avg = np.mean(f1_score)

# Plotting the precision, recall, and F1 scores for each class
x = np.arange(len(class_labels))

plt.figure(figsize=(10, 6))
plt.plot(x, precision, label='Precision', marker='o', linestyle='-')
plt.plot(x, recall, label='Recall', marker='o', linestyle='-')
plt.plot(x, f1_score, label='F1 Score', marker='o', linestyle='-')
plt.axhline(y=precision_macro_avg, color='r', linestyle='--', label='Precision Macro Avg')
plt.axhline(y=recall_macro_avg, color='g', linestyle='--', label='Recall Macro Avg')
plt.axhline(y=f1_macro_avg, color='b', linestyle='--', label='F1 Macro Avg')
plt.xticks(x, class_labels)
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('GaussianNB\nPrecision, Recall, and F1 Score by Class')
plt.legend()
plt.show()


# In[87]:


for i in model_b:
  kfold = KFold()
  score= cross_val_score(i,X=features,y=target,cv=kfold)
  print(i)
  print(score.mean())
  print(score.min())
  print(score.max())
  print("----"*10)


# In[89]:


for i in model_b:
  shuffle_split = ShuffleSplit()
  score= cross_val_score(i,X=features,y=target,cv=shuffle_split)
  print(i)
  print(score.mean())
  print(score.min())
  print(score.max())
  print("----"*10)


# In[90]:


for i in model_b:
  stratified_kfold = StratifiedKFold()
  score= cross_val_score(i,X=features,y=target,cv=stratified_kfold)
  print(i)
  print(score.mean())
  print(score.min())
  print(score.max())
  print("----"*10)


# In[91]:


for i in model_b:
  leave_one_out = LeaveOneOut()
  score= cross_val_score(i,X=features,y=target,cv=leave_one_out)
  print(i)
  print(score.mean())
  print(score.min())
  print(score.max())
  print("----"*10)


# In[ ]:


# from the above results we can classify that logisticRegression and GaussianNB having best accuracy and best models 


# In[ ]:


#To increase more accuracy we use neural network 


# In[ ]:




