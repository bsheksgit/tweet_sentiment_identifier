#!/usr/bin/env python
# coding: utf-8

# # Tweet sentiment analysis

# In[1]:


import pandas as pd  
import numpy as np                                          #data processing
import re                                                   #regular expression for data cleaning
from sklearn.feature_extraction.text import CountVectorizer #feature extraction
import nltk                                                 #natural language toolkit
from nltk.corpus import stopwords                           #stopwords
from nltk.stem import WordNetLemmatizer                     #text normalization
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score,mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 200)


# ### Importing Data

# In[3]:


train = pd.read_csv(r'train.csv')
train.head()


# In[4]:


train.info()


# In[5]:


# There are no null values in any columns


# ### We shall extract features from tweets. First we will remove all symbols from the text. Then we will apply contraction mapping. After that we will lemmatize the words and apply count vectorizer.

# In[6]:


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}


# In[ ]:


# Let's create a lemmatizer object


# In[7]:


lemmatizer = WordNetLemmatizer()


# In[8]:


def tweet_cleaner(text):
    newString=re.sub(r'@[A-Za-z0-9]+','',text)                     #removing user mentions
    newString=re.sub("#","",newString)                             #removing hashtag symbol
    newString= ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")]) #contraction mapping
    newString= re.sub(r'http\S+', '', newString)                   #removing links
    newString= re.sub(r"'s\b","",newString)                        #removing 's
    letters_only = re.sub("[^a-zA-Z]", " ", newString)             #Fetching out only letters
    lower_case = letters_only.lower()                              #converting everything to lowercase
    tokens = [w for w in lower_case.split() if not w in stop_words]#stopwords removal
    newString=''
    for i in tokens:                                                 
        newString=newString+lemmatizer.lemmatize(i)+' '            #converting words to lemma                               
    return newString.strip()  


# In[9]:


## Loading stopwords from another way since the there is some problem with target machine server.
eng = open(r"english",'r')
#eng.read()
stop_words = set(eng.read().split('\n'))
eng.close()  


# In[10]:


# Let's clean the tweets one by one


# In[11]:


cleaned_tweets = []
for i in train.tweet:
  cleaned_tweets.append(tweet_cleaner(i))
print(cleaned_tweets[:5])   #print top 5 records


# In[12]:


# Let's visualize the most commonly used words.
all_words = []
for line in cleaned_tweets:
    words = line.split()
    for word in words:
        all_words.append(word)
        
plt.figure(figsize=(12,5))
plt.title('Top 25 most common words')
plt.xticks(fontsize=13, rotation=90)
fd = nltk.FreqDist(all_words)
fd.plot(25,cumulative=False) 


# In[13]:


# iphone, apple, samsung, new, twitter, com, phone, sony, follow, pic etc are the most used words.


# In[14]:


train['cleaned_tweets'] = cleaned_tweets
train.head()


# # Let's now create a model using count Vectorizer

# In[15]:


vectorizer = CountVectorizer()
features= vectorizer.fit(train['cleaned_tweets'])
features = vectorizer.transform(train['cleaned_tweets'])
features.shape


# Let's split train and test sets

# In[16]:


targets = train['label']
inputs = features


# In[17]:


xtrain, xtest, ytrain, ytest = train_test_split(features,targets, random_state=42, test_size=0.2)


# ### Let's solve the classification using xgboost

# In[18]:


from xgboost import XGBClassifier


# In[19]:


xgb = XGBClassifier()


# In[20]:


xgb.fit(xtrain,ytrain)


# In[21]:


predictions = xgb.predict(xtest)


# In[22]:


f1_score(predictions,ytest)


# ### Let's apply these steps to the test set

# In[23]:


test = pd.read_csv(r'test.csv')
test.head()


# In[24]:


cleaned_tweets = []
for i in test.tweet:
  cleaned_tweets.append(tweet_cleaner(i))
print(cleaned_tweets[:5])   #print top 5 records
test['cleaned_tweets'] = cleaned_tweets
test.head()


# In[30]:


# We need to fit the count vectorizer only once since we need to have the same number of features in both train and test sets.


# In[25]:


# Count vectorizer
features_test = vectorizer.transform(test['cleaned_tweets'])
features_test.shape


# In[26]:


final_predictions = xgb.predict(features_test)


# In[27]:


submission_dict = {'id':test['id'], 'label': final_predictions}


# In[28]:


submission_file = pd.DataFrame(submission_dict)


# In[ ]:


# Creating a final submission file by exporting to csv file.


# In[29]:


submission_file.to_csv('submission_file.csv', index=False)


# In[ ]:




