#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import classification_report
from sklearn.datasets import load_breast_cancer 


# In[2]:


diabset = pd.read_csv(r'C:\Users\homer\OneDrive\Documents\School Folder\diabetes.csv')


# In[3]:


diabset.head()


# In[4]:


diabset.shape


# In[5]:


diab_index = diabset.index.values
diab_index.shape


# In[6]:


diab_labels = np.reshape(diab_index,(768,1))


# In[7]:


diab_data = np.concatenate([diabset,diab_labels],axis=1)


# In[8]:


diab_data.shape


# In[9]:


diab_dataset = pd.DataFrame(diab_data)


# In[10]:


diab_dataset.head()


# In[11]:


diab_X = diab_dataset.values[:, 9]
diab_Y = diab_dataset.values[:, 8]


# In[12]:


diab_X_train, diab_X_test, diab_Y_train, diab_Y_test = train_test_split(diab_X, diab_Y, test_size=0.2, random_state=42)


# In[13]:


sc = StandardScaler()
diab_X_reshape = diab_X_train.reshape(-1, 1)
diab_X_std = sc.fit_transform(diab_X_reshape)
diab_Xtest_reshape = diab_X_test.reshape(-1, 1)
diab_Xtest_std = sc.transform(diab_Xtest_reshape)


# In[14]:


diab_logreg = LogisticRegression(solver ='liblinear', random_state=0)
diab_logreg.fit(diab_X_std, diab_Y_train)


# In[15]:


diab_Y_pred = diab_logreg.predict(diab_Xtest_std)


# In[16]:


diab_cnf_matrix = confusion_matrix(diab_Y_test, diab_Y_pred)
diab_cnf_matrix


# In[17]:


print("Accuracy:",metrics.accuracy_score(diab_Y_test, diab_Y_pred))
print("Precision:",metrics.precision_score(diab_Y_test, diab_Y_pred))
print("Recall:",metrics.recall_score(diab_Y_test, diab_Y_pred))


# In[18]:


class_names = [0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(diab_cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[19]:


diab_kfold = KFold(n_splits=5, random_state=42, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, diab_X.reshape(-1, 1), diab_Y, cv=diab_kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[20]:


diab_kfold = KFold(n_splits=10, random_state=42, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, diab_X.reshape(-1, 1), diab_Y, cv=diab_kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[21]:


cancer = load_breast_cancer()


# In[22]:


cancer_data = cancer.data
cancer_data.shape


# In[23]:


cancer_input = pd.DataFrame(cancer_data)
cancer_input.head()


# In[24]:


cancer_labels = cancer.target


# In[25]:


cancer_labels.shape


# In[26]:


labels = np.reshape(cancer_labels, (569,1))


# In[27]:


final_cancer_data = np.concatenate([cancer_data,labels],axis=1)


# In[28]:


final_cancer_data.shape


# In[29]:


cancer_dataset = pd.DataFrame(final_cancer_data)


# In[30]:


features = cancer.feature_names
features


# In[31]:


features_labels = np.append(features, 'label')


# In[32]:


cancer_dataset.columns = features_labels


# In[33]:


cancer_dataset.head()


# In[34]:


cancer_dataset['label'].replace(0, 'Benign', inplace=True)
cancer_dataset['label'].replace(1, 'Malignant', inplace=True)


# In[35]:


cancer_dataset.tail()


# In[36]:


cancer_X = cancer_dataset.iloc[:,0:29].values
cancer_Y = cancer_dataset.iloc[:,30].values


# In[37]:


cancer_X_train, cancer_X_test, cancer_Y_train, cancer_Y_test = train_test_split(cancer_X, cancer_Y, test_size=0.2, random_state=42)


# In[38]:


sc_X = StandardScaler()
cancer_X_trainstd = sc_X.fit_transform(cancer_X_train)
cancer_X_teststd = sc_X.transform(cancer_X_test)


# In[39]:


cancerClass = LogisticRegression(random_state=42)
cancerClass.fit(cancer_X_trainstd, cancer_Y_train)


# In[40]:


cancer_Y_pred = cancerClass.predict(cancer_X_teststd)


# In[41]:


cancer_Y_pred[0:9]


# In[42]:


cancer_cnf_matrix = confusion_matrix(cancer_Y_test, cancer_Y_pred)
cancer_cnf_matrix


# In[43]:


print("Accuracy:",metrics.accuracy_score(cancer_Y_test, cancer_Y_pred))
print("Precision:",metrics.precision_score(cancer_Y_test, cancer_Y_pred, pos_label="Benign"))
print("Recall:",metrics.recall_score(cancer_Y_test, cancer_Y_pred, pos_label="Benign"))


# In[44]:


class_names = [0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cancer_cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[48]:


C = [10, 1, .1, .001]
for c in C:
    clf = LogisticRegression(penalty ='l1', C=c, solver='liblinear')
    clf.fit(cancer_X_trainstd, cancer_Y_train)
    print('C:', c)
    print('Training accuracy: ', clf.score(cancer_X_trainstd, cancer_Y_train))
    print('Test accuracy: ', clf.score(cancer_X_teststd, cancer_Y_test))
    print(' ')


# In[49]:


cancer_kfold = KFold(n_splits=5, random_state=42, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, cancer_X, cancer_Y, cv=cancer_kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[50]:


cancer_kfold = KFold(n_splits=10, random_state=42, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, cancer_X, cancer_Y, cv=cancer_kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[51]:


for c in C:
    cancer_kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    model = LogisticRegression(penalty ='l1', C=c, solver='liblinear')
    results = cross_val_score(model, cancer_X, cancer_Y, cv=cancer_kfold)
    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[52]:


for c in C:
    cancer_kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    model = LogisticRegression(penalty ='l1', C=c, solver='liblinear')
    results = cross_val_score(model, cancer_X, cancer_Y, cv=cancer_kfold)
    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[ ]:




