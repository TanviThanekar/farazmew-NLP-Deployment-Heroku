#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("Amazon_Unlocked_Mobile.csv")


# In[3]:


df.head()


# In[5]:


df = df[["Rating","Reviews"]].head(5000)


# In[6]:


df = df[df["Rating"] != 3]


# In[7]:


df = df.dropna(axis ='rows')


# In[10]:


df.reset_index(inplace = True)


# In[14]:


df = df.drop(['index'], axis=1)


# In[16]:


import numpy as np
df["Rating"] = np.where(df["Rating"] > 3,1,0)


# In[18]:


X = df['Reviews']
y = df['Rating']


# In[20]:


from sklearn.feature_extraction.text import CountVectorizer
# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB


# In[25]:


from flask import Flask,render_template,url_for,request
from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib


# In[27]:


clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))


# In[28]:


filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)


# In[33]:


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)


# In[34]:


if __name__ == '__main__':
	app.run(debug=True)


# In[35]:





# In[ ]:




