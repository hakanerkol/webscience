#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import json
import string
import re
import seaborn as sns 
    
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

import nltk
from nltk.tokenize import RegexpTokenizer

from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize


# In[2]:


data = pd.read_json("tweets.json")


# In[3]:


# Extracting keys of each tweets from json file
print((data['tweets'][0]).keys())
print(" ")
print("Printing one sample tweet as a dict\n")
print((data['tweets'][0]))


# In[4]:


# Creating an empty list for text of tweets
tweets_txt = []
for i in range(len(data['tweets'])):
    tweets_txt.append((data['tweets'][i])['text'])


# In[5]:


df = pd.DataFrame()


# In[6]:


df['tweet'] = tweets_txt


# In[7]:


df.head()


# In[8]:


df['is_retweet'] = df['tweet'].apply(lambda x: x[:2]=='RT')
total_retwet = df['is_retweet'].sum()  # number of retweets
print("Total retweets are",total_retwet)


# In[9]:


df.head()


# In[10]:


df.to_csv('temp.csv')


# In[11]:


df = pd.read_csv('temp.csv')


# In[12]:


print("Unique tweets are",df.loc[df['is_retweet']].tweet.unique().size)


# In[13]:


df.applymap(lambda x: isinstance(x, list)).all()


# In[14]:


df.groupby(['tweet']).size().reset_index(name='counts')  .sort_values('counts', ascending=False).head(10)


# In[15]:


# number of times each tweet appears
counts = df.groupby(['tweet']).size()           .reset_index(name='counts')           .counts

# define bins for histogram
my_bins = np.arange(0,counts.max()+2, 1)-0.5

# plot histogram of tweet counts
plt.figure(figsize=(12,6))
plt.title("Grouping of tweets by retweeting (Frequency)")
plt.hist(counts, bins = my_bins)
plt.xlabels = np.arange(1,counts.max()+1, 1)
plt.xlabel('copies of each tweet')
plt.ylabel('frequency')
plt.yscale('log', nonposy='clip')
plt.show()


# In[16]:


def find_retweeted(tweet):
    '''This function will extract the twitter handles of retweed people'''
    return re.findall('(?<=RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

def find_mentioned(tweet):
    '''This function will extract the twitter handles of people mentioned in the tweet'''
    return re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)  

def find_hashtags(tweet):
    '''This function will extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)   


# In[17]:


# make new columns for retweeted usernames, mentioned usernames and hashtags
df['retweeted'] = df.tweet.apply(find_retweeted)
df['mentioned'] = df.tweet.apply(find_mentioned)
df['hashtags'] = df.tweet.apply(find_hashtags)


# In[18]:


df.head(100)


# In[19]:


# take the rows from the hashtag columns where there are actually hashtags
hashtags_list_df = df.loc[
                       df.hashtags.apply(
                           lambda hashtags_list: hashtags_list !=[]
                       ),['hashtags']]


# In[20]:


# create dataframe where each use of hashtag gets its own row
flattened_hashtags_df = pd.DataFrame(
    [hashtag for hashtags_list in hashtags_list_df.hashtags
    for hashtag in hashtags_list],
    columns=['hashtag'])


# In[21]:


# number of unique hashtags
flattened_hashtags_df['hashtag'].unique().size


# In[22]:


# count of appearances of each hashtag
popular_hashtags = flattened_hashtags_df.groupby('hashtag').size()                                        .reset_index(name='counts')                                        .sort_values('counts', ascending=False)                                        .reset_index(drop=True)
# popular_hashtags
print("Now printig most important / popular hastage amongst all tweets")
print(popular_hashtags[popular_hashtags['counts'] > 15])


# In[23]:


# number of times each hashtag appears
counts = flattened_hashtags_df.groupby(['hashtag']).size()                              .reset_index(name='counts')                              .counts

# define bins for histogram                              
my_bins = np.arange(0,counts.max()+2, 5)-0.5

# plot histogram of tweet counts
plt.figure()
plt.hist(counts, bins = my_bins)
plt.xlabels = np.arange(1,counts.max()+1, 1)
plt.xlabel('hashtag number of appearances')
plt.ylabel('frequency')
plt.yscale('log', nonposy='clip')
plt.show()


# In[24]:


# take hashtags which appear at least this amount of times
min_appearance = 10
# find popular hashtags - make into python set for efficiency
popular_hashtags_set = set(popular_hashtags[
                           popular_hashtags.counts>=min_appearance
                           ]['hashtag'])


# In[25]:


# make a new column with only the popular hashtags
hashtags_list_df['popular_hashtags'] = hashtags_list_df.hashtags.apply(
            lambda hashtag_list: [hashtag for hashtag in hashtag_list
                                  if hashtag in popular_hashtags_set])
# drop rows without popular hashtag
popular_hashtags_list_df = hashtags_list_df.loc[
            hashtags_list_df.popular_hashtags.apply(lambda hashtag_list: hashtag_list !=[])]


# In[26]:


# make new dataframe
hashtag_vector_df = popular_hashtags_list_df.loc[:, ['popular_hashtags']]

for hashtag in popular_hashtags_set:
    # make columns to encode presence of hashtags
    hashtag_vector_df['{}'.format(hashtag)] = hashtag_vector_df.popular_hashtags.apply(
        lambda hashtag_list: int(hashtag in hashtag_list))


# In[27]:


hashtag_vector_df 


# In[28]:


hashtag_matrix = hashtag_vector_df.drop('popular_hashtags', axis=1)
# calculate the correlation matrix
correlations = hashtag_matrix.corr()

# plot the correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(correlations,
    cmap='RdBu',
    vmin=-1,
    vmax=1,
    square = True,
    cbar_kws={'label':'correlation'})
plt.show()


# In[29]:


def remove_links(tweet):
    '''Takes a string and removes web links from it'''
    tweet = re.sub(r'http\S+', '', tweet) # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet) # rempve bitly links
    tweet = tweet.strip('[link]') # remove [links]
    return tweet

def remove_users(tweet):
    '''Takes a string and removes retweet and @user information'''
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove retweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove tweeted at
    return tweet


# In[30]:


my_stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

# cleaning master function
def clean_tweet(tweet, bigrams=False):
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = tweet.lower() # lower case
    tweet = re.sub('['+my_punctuation + ']+', ' ', tweet) # strip punctuation
    tweet = re.sub('\s+', ' ', tweet) #remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet) # remove numbers
    tweet_token_list = [word for word in tweet.split(' ')
                            if word not in my_stopwords] # remove stopwords

    tweet_token_list = [word_rooter(word) if '#' not in word else word
                        for word in tweet_token_list] # apply word rooter
    if bigrams:
        tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                            for i in range(len(tweet_token_list)-1)]
    tweet = ' '.join(tweet_token_list)
    return tweet


# In[31]:


df['clean_tweet'] = df.tweet.apply(clean_tweet)


# In[32]:


df.head()


# In[33]:


# Get the important users from the tweets
imp_users_df = df.loc[
                       df.retweeted.apply(
                           lambda imp_user_list: imp_user_list !=[]
                       ),['retweeted']]


# In[34]:


# create dataframe where each use of users gets its own row
flattened_users_df = pd.DataFrame(
    [user for users_list in imp_users_df.retweeted
    for user in users_list],
    columns=['retweeted'])


# In[35]:


popular_users = flattened_users_df.groupby('retweeted').size()                                        .reset_index(name='counts')                                        .sort_values('counts', ascending=False)                                        .reset_index(drop=True)
print("Important 10 users are as follows (whose tweets are highly popular/important)")
popular_users[0:10]


# In[36]:


# Topic Modeliing and Grouping the words

from sklearn.feature_extraction.text import CountVectorizer

# the vectorizer object will be used to transform text to vector form
vectorizer = CountVectorizer(max_df=0.9, min_df=25, token_pattern='\w+|\$[\d\.]+|\S+')

# apply transformation
tf = vectorizer.fit_transform(df['clean_tweet']).toarray()

# tf_feature_names tells us what word each column in the matric represents
tf_feature_names = vectorizer.get_feature_names()


# In[37]:



number_of_topics = 10

model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)


# In[38]:


model.fit(tf)


# In[39]:


def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


# In[40]:


no_top_words = 10
display_topics(model, tf_feature_names, no_top_words)

