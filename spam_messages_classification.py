#!/usr/bin/env python
# coding: utf-8

# # Spam message detection using Naive Bayes
# 

# 
# To filter mobile phone spam using the Naive Bayes algorithm

# 
# https://github.com/stedy/Machine-Learning-with-R-datasets

# In[39]:


import math
import numpy as np
import pandas as pd
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')

import string
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
import statsmodels.api as sm

# {{{The TfidfVectorizer will tokenize documents, learn the vocabulary and inverse document frequency
# weightings, and allow you to encode new documents. Alternately, if you already have a learned
# CountVectorizer, you can use it with a TfidfTransformer to just calculate the inverse document 
# frequencies and start encoding documents.}}}


# In[9]:


data = pd.read_csv('spamsms-1.csv',encoding='latin-1')
data


# In[10]:


data.drop(columns=[ 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
data


# In[11]:


data.type.value_counts()


# In[12]:


# data[data.type == 'ham'].valuescount
sns.countplot(x=data.type, data= data)


# In[13]:


print(data.drop('type',axis=1))


# In[14]:


#splitting document into words and tokenizing it. 
def text_process(x):
    return x.split()

bow_transformer = CountVectorizer(analyzer=text_process).fit(data['text'])
print(len(bow_transformer.vocabulary_))

#bow_transformer here seems to develop a vocabulary 
#of all the used words by tokenizing them. 


# In[16]:


print(bow_transformer.get_feature_names()[0])
print(bow_transformer.get_feature_names()[8555])


# In[19]:


data['length'] = data['text'].apply(lambda x:len(x))
data = data[data['length']>0]
data.info()
data.head()


# In[20]:


X = data['text']
y = data['type']


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=1)


# In[26]:


x_train


# In[30]:


messages_bow = bow_transformer.transform(x_train)
tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


# In[31]:


spam_detect_model = MultinomialNB().fit(messages_tfidf,y_train)


# In[32]:


test_bow = bow_transformer.transform(x_test)
tfidf_transformer = TfidfTransformer().fit(test_bow)
test_tfidf = tfidf_transformer.transform(test_bow)
print(test_tfidf.shape)


# In[33]:


y_pred = spam_detect_model.predict(test_tfidf)
print(y_pred)


# In[34]:


print('confusion matrix')
print(confusion_matrix(y_test,y_pred))


# In[37]:


cm = confusion_matrix(y_test,y_pred,labels=['ham','spam'])
cm_df = pd.DataFrame(cm, index = [i for i in ['1','0']], columns = [i for i in ['pred-1','pred-0']])
plt.figure(figsize = (7,5))
sns.heatmap(cm_df,annot=True)


# In[42]:


from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,y_pred,labels=['ham','spam']))


# In[ ]:




