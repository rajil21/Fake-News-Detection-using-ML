#!/usr/bin/env python
# coding: utf-8

# ## Importing the libraries 

# In[86]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sb


# ## Reading the dataset 

# In[3]:


df= pd.read_csv('news.csv')
df.head(3)


# In[4]:


labels=df.label
labels.unique()       # no.of unique values in the output column of the dataset df


# ## Preprocessing  and splitting the dataset 

# In[5]:


# Splitting the dataset into 20% testing data and 80% training data
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

To analyse,predict and to classify we need the input values as numbers,not as string therefore we are using the tfidf Vectorizer
which generates an matrix for each document(rows) for the dataframe 
# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer     # importing the function from the library 

#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# ## Passive Aggressive Classifier

# In[7]:


from sklearn.linear_model import PassiveAggressiveClassifier     


# In[95]:


#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=100)
pac.fit(tfidf_train,y_train)
accuracy=[]
#DataFlair - Predict on the test set and calculate accuracy
for i in range(10):
    y_pred_pac = pac.predict(tfidf_test)
    accuracy.append(accuracy_score(y_test,y_pred_pac))
score = np.mean(accuracy)
score = round(score*100,2)
print(f'Accuracy: {score}%')

#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred_pac, labels=['FAKE','REAL'])


# In[96]:


plt.figure(figsize=(4,4))
sb.heatmap(confusion_matrix(y_test,y_pred_LR),annot=True,fmt='.1f',linewidths=0.9,square=(2,2))
plt.xlabel('Actual')
plt.ylabel('predicted')
all_sample_title = 'Accuracy Score: {0} %'.format(score)
plt.title(all_sample_title,size=15);


# ## SVM 

# In[97]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(tfidf_train, y_train) 
y_pred_svm =clf.predict(tfidf_test)

score1=accuracy_score(y_test,y_pred_svm)
score1=round(score1*100,2)
print(f'Accuracy: {score1}%')

#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred_svm, labels=['FAKE','REAL'])


# In[98]:


plt.figure(figsize=(4,4))
sb.heatmap(confusion_matrix(y_test,y_pred_svm),annot=True,fmt='.1f',linewidths=0.9,square=(2,2))
plt.xlabel('Actual')
plt.ylabel('predicted')
all_sample_title = 'Accuracy Score: {0} %'.format(score1)
plt.title(all_sample_title,size=15);


# ## K-Nearest Neighbour 

# In[102]:


from sklearn.neighbors import KNeighborsClassifier
clusters=[]
accuracy=[]
for i in range(100,150):
    knn = KNeighborsClassifier(n_neighbors=i)        # creating an object
    knn.fit(tfidf_train,y_train)
    y_pred_knn = knn.predict(tfidf_test)
    clusters.append(i)
    accuracy.append(accuracy_score(y_test,y_pred_knn))
print(clusters)
print(accuracy)
print(max(accuracy))


# In[103]:


plt.title('Clusters vs accuracy')
plt.xlabel('Clusters(K)')
plt.ylabel('Accuracy')
plt.plot(clusters,accuracy)
plt.show()


# In[104]:


accuracy[23]


# In[105]:


score2  = round((accuracy[23])*100,2)

As we see above, for k = 123 the maximum accuracy value 85%  occurs   
# In[106]:


plt.figure(figsize=(4,4))
sb.heatmap(confusion_matrix(y_test,y_pred_knn),annot=True,fmt='.1f',linewidths=0.9,square=(2,2))
plt.xlabel('Actual')
plt.ylabel('predicted')
all_sample_title = 'Accuracy Score: {0} %'.format(score2)
plt.title(all_sample_title,size=15);


# ## Decision Tree 

# In[62]:


from sklearn.tree import DecisionTreeClassifier               # importing the required libraries 
from sklearn.tree import export_graphviz  
from sklearn.externals.six import StringIO

from IPython.display import Image 
import pydotplus

Decision Tree with maximum depth with its accuracy 
# In[107]:


tree = DecisionTreeClassifier(criterion='entropy')
tree.fit(tfidf_train,y_train)


# In[108]:


dot_data = StringIO()                                  # creating an object 
export_graphviz(tree,out_file=dot_data,                # exporting graph with few functions 
               filled=True,rounded=True,
               special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())     # creating a empty graph 
Image(graph.create_png(),width=800,height=1000)    


# In[117]:


y_pred_tree = tree.predict(tfidf_test)
score3 = round(accuracy_score(y_test,y_pred_tree)*100,2)
print(f'Accuracy Score: {score3} %')


# In[118]:


plt.figure(figsize=(4,4))
sb.heatmap(confusion_matrix(y_test,y_pred_tree),annot=True,fmt='.1f',linewidths=0.9,square=(2,2))
plt.xlabel('Actual')
plt.ylabel('predicted')
all_sample_title = 'Accuracy Score: {0} %'.format(score3)
plt.title(all_sample_title,size=15);

Decision Tree with maximum depth = 5 and with its accuracy 
# In[77]:


tree = DecisionTreeClassifier(criterion='entropy',max_depth=5)  # here the maximum depth is given as 5
tree.fit(tfidf_train,y_train)


# In[78]:


dot_data = StringIO()                                  # creating an object 
export_graphviz(tree,out_file=dot_data,                # exporting graph with few functions 
               filled=True,rounded=True,
               special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())     # creating a empty graph 
Image(graph.create_png(),width=800,height=1000)    


# In[79]:


y_pred_tree = tree.predict(tfidf_test)
print('Accuracy : ',accuracy_score(y_test,y_pred_tree))


# ## Logistic Regression 

# In[123]:


from sklearn import linear_model
regress = linear_model.LogisticRegression()  


# In[124]:


regress.fit(tfidf_train,y_train)
y_pred_LR = regress.predict(tfidf_test)
score4 = round(accuracy_score(y_pred_LR,y_test)*100,2)
print(f'Accuracy Score : {score4} %')


# In[125]:


print('Confustion Matrix\n ',confusion_matrix(y_pred_LR,y_test))


# In[126]:


plt.figure(figsize=(4,4))
sb.heatmap(confusion_matrix(y_test,y_pred_LR),annot=True,fmt='.1f',linewidths=0.9,square=(2,2))
plt.xlabel('Actual')
plt.ylabel('predicted')
all_sample_title = 'Accuracy Score: {0} %'.format(score4)
plt.title(all_sample_title,size=15);


# ## Conclusion 

# In[142]:


scores = [score,score1,score2,score3,score4]
class_methods = ['PAC','SVM','KNN','DT','LR']
plt.title('Classification Methods Vs Accuracy')
plt.xlabel('Accuracy %')
plt.ylabel('Classification Methods')
#plt.ylim()
plt.barh(class_methods,scores)
plt.show()


# In[ ]:




