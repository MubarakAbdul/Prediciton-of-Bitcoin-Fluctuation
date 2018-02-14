
# coding: utf-8

# In[223]:


import requests
import bs4 as bs
import pandas as pd
import numpy as np
import warnings
import re
import datetime
import time
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
#from __future__ import print_function, division, absolute_import #make compatible with Python 2 and Python 3


# In[182]:


#Scraping code for InvestFeed
record = pd.DataFrame()
for i in range(0,241):
    post=[]
    date=[]
    url = "https://www.investfeed.com/currency/BTC?page="+str(i+1)
    soup = bs.BeautifulSoup(requests.get(url).text, "html.parser")
    content = soup.find_all('div', attrs={'class': 'post-content'})
    date_time = soup.find_all('div',attrs={'class':'floated-right'})
    for x in range(len(content)):
        post.append(re.sub(r'\s+', ' ', content[x].text))
        date.append(re.sub(r'\s+',' ',date_time[x].text))
    if (i+1)%25 == 0:
        print('{} pages completed!'.format(i+1))
    t1 = pd.Series(date)
    t2 = pd.Series(post)
    temp_record = pd.concat([t1,t2],axis=1)
    record = pd.concat([record,temp_record]).reset_index(drop=True)


# In[183]:


record.shape


# In[191]:


record.head(25)


# In[131]:


#record = record.drop(record[record.index > 2264].index) #Elements beyond this had random content. 2264 captures all posts.


# In[187]:


record = record.rename(columns = {0:'Date',1:'Post'})


# In[188]:


record['Date'] = pd.to_datetime(record['Date']).dt.date #Converting from str to datetime obj and retaining just the dates


# In[192]:


record = record.iloc[23:].reset_index(drop=True) #Prices unavailable for Novmeber 30th


# In[193]:


col_names = ['Date','Price']
btc = pd.read_csv('btc_price.csv',header=None,names=col_names)


# In[194]:


print(btc.shape)
btc.head()


# In[195]:


#Setting labels for the Bitcoin prices to perform Sentiment Analysis
label=[]
for i in range(1,btc.shape[0]):
    if (btc.iat[i,1]>btc.iat[i-1,1]):
        btc.loc[btc.index[i],'Sentiment']=1
    else:
        btc.loc[btc.index[i],'Sentiment']=0
btc.loc[btc.index[0],'Sentiment']=0 #To match indices with btc. This false value will anyway go away when we trim BTC to match record df.
btc['Sentiment'] = btc['Sentiment'].astype(int)


# In[196]:


btc['Date'] = pd.to_datetime(btc['Date']).dt.date


# In[257]:


btc.tail()


# In[297]:


temp = record.merge(btc, on='Date', how='inner')


# In[298]:


sent = temp.drop('Price',1)


# In[299]:


sent.head()


# In[300]:


sent.shape


# In[301]:


print(sent.Sentiment.value_counts())
sent.Sentiment.hist(); 


# In[302]:


#Apply length function to the review column
lengths = sent.Post.apply(len)

print('Average character length of the posts are:')
print (np.mean(lengths))


# # NLP

# In[303]:


import nltk
#nltk.download()


# In[304]:


from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

eng = stopwords.words('English')


# In[305]:


test_post = sent['Post'][0]# the review used for initial analysis
print(test_post)


# In[281]:


print(len(sent_tokenize(test_post)))
sent_tokenize(test_post) # doesn't really split all sentences


# In[282]:


# Check if it does a better job if we add space after every period
test_post = test_post.replace('.','. ')

print(len(sent_tokenize(test_post)), end='\n\n') # number of sentences

# print all sentences on a new line
for sent in sent_tokenize(test_post):
    print(sent, end='\n\n')


# In[306]:


test_post = re.sub('[^a-zA-Z]',' ',test_post)
print(test_post) # remove special character


# In[284]:


test_post = test_post.lower()


# In[285]:


test_post


# In[287]:


test_post_words = test_post.split()
print(test_post_words[:10]) # tokenize and lower case
print(len(test_post_words))


# In[288]:


ps = PorterStemmer() #initialize Porter Stemmer object

ps_stems = []
for w in test_post_words:
    ps_stems.append(ps.stem(w))

print(' '.join(ps_stems)) # add all the stemmed words to one string


# In[293]:


#parts of speech tagging

token_tag = pos_tag(test_post_words)
token_tag[:10]


# In[294]:


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'


# In[295]:


#from nltk.stem import WordNetLemmatizer


wnl = WordNetLemmatizer()

wnl_stems = []
for pair in token_tag:
    res = wnl.lemmatize(pair[0],pos=get_wordnet_pos(pair[1]))
    wnl_stems.append(res)

print(' '.join(wnl_stems))


# In[313]:


def post_cleaner(post):
    
    #1. Use regex to find emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', post)
    
    #2. Remove punctuation
    post = re.sub("[^a-zA-Z]", " ",post)
    
    #3. Tokenize into words (all lower case)
    post = post.lower().split()
    
    #4. Remove stopwords
    eng_stopwords = set(stopwords.words("english"))
    post = [w for w in post if not w in eng_stopwords]
    
    #5. Join the review to one sentence
    post = ' '.join(post+emoticons)
    # add emoticons to the end

    return(post)


# In[326]:


get_ipython().run_cell_magic('time', '', '\nnum_posts = len(sent[\'Post\'])\n\npost_clean_original = []\n\nfor i in range(0,num_posts):\n    if( (i+1)%500 == 0 ):\n        # print progress\n        print("Done with %d posts" %(i+1)) \n    post_clean_original.append(post_cleaner(record[\'Post\'][i]))')


# In[331]:


get_ipython().run_cell_magic('time', '', '# Lemmatizer\n\npost_clean_wnl = []\n\nwnl = WordNetLemmatizer()\n\nfor i in range(0,num_posts):\n    if( (i+1)%500 == 0 ):\n        # print progress\n        print("Done with %d posts" %(i+1)) \n    \n    wnl_stems = []\n    token_tag = pos_tag(post_clean_original[i].split())\n    for pair in token_tag:\n        res = wnl.lemmatize(pair[0],pos=get_wordnet_pos(pair[1]))\n        wnl_stems.append(res)\n\n    post_clean_wnl.append(\' \'.join(wnl_stems))')


# In[333]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split

# put everything together in a function

def predict_sentiment(cleaned_posts, y=sent["Sentiment"]):

    print("Creating the bag of words model..\n")
    # CountVectorizer" is scikit-learn's bag of words tool, here we show more keywords 
    vectorizer = CountVectorizer(analyzer = "word",                                    tokenizer = None,                                     preprocessor = None,                                  stop_words = None,                                    max_features = 2000) 
    
    X_train, X_test, y_train, y_test = train_test_split(    cleaned_posts, y, random_state=0, test_size=0.2)

    # Then we use fit_transform() to fit the model / learn the vocabulary,
    # then transform the data into feature vectors.
    # The input should be a list of strings. .toarraty() converts to a numpy array
    
    train_bag = vectorizer.fit_transform(X_train).toarray()
    test_bag = vectorizer.transform(X_test).toarray()

    # You can extract the vocabulary created by CountVectorizer
    # by running print(vectorizer.get_feature_names())


    print("Training the random forest classifier..\n")
    # Initialize a Random Forest classifier with 75 trees
    forest = RandomForestClassifier(n_estimators = 50) 

    # Fit the forest to the training set, using the bag of words as 
    # features and the sentiment labels as the target variable
    forest = forest.fit(train_bag, y_train)


    train_predictions = forest.predict(train_bag)
    test_predictions = forest.predict(test_bag)
    
    train_acc = metrics.accuracy_score(y_train, train_predictions)
    valid_acc = metrics.accuracy_score(y_test, test_predictions)
    print("The training accuracy is: ", train_acc, "\n", "The validation accuracy is: ", valid_acc)
    print()
    print()
    #Extract feature importnace
    print('TOP TEN IMPORTANT FEATURES:')
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_10 = indices[:10]
    print([vectorizer.get_feature_names()[ind] for ind in top_10])


# In[335]:


predict_sentiment(post_clean_original) #Accuracy measure with original sentences


# In[336]:


predict_sentiment(post_clean_wnl) #Lemmatized sentences lead to lower accuracy

