#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install pyLDAvis


# In[1]:


from google.colab import drive
drive.mount("/content/drive")


# In[5]:



import pandas as pd
import numpy as np
data=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data/Amazon_reviews.csv')
print('data matrix (rows,column) :',data.shape)
data.head()


# In[6]:


data.tail()
#data.describe()


# In[7]:


data.isna().sum()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.gcf()

# Changing Seaborn Plot size
fig.set_size_inches(12, 5)
sns.set(font_scale=1)
#plt.figure(figsize=(12, 5))
sns.heatmap(data=data.isna(),yticklabels=False,cmap='coolwarm',cbar=False)


# In[9]:


#so drop all columns having MAX NAN values
col_to_drop=['reviews.id','reviews.didPurchase','reviews.userCity','reviews.userProvince']
#data=data.drop(['reviews.id','reviews.didPurchase','reviews.userCity','reviews.userProvince'],axis=1)
data=data.drop([*col_to_drop],axis=1)
print('after dropping column :',data.shape)


# In[11]:


# Number of unique values in each column, excluding NAN
data.nunique(dropna=True)


# In[12]:


# for our analysis, we need only reviews.sourceURLs, reviews.text, reviews.title and reviews.username

data=data[['reviews.sourceURLs', 'reviews.text', 'reviews.title','reviews.username']]
print(data.shape)
data.head()


# In[13]:


data.isna().sum()


# In[14]:


# Print the rows having NAN vlues 
nan_values = data[data.isna().any(axis=1)]
nan_values


# In[15]:


#lower casing the abstract so that 'text', 'Text, 'TEXT' are treated in the same way and also help in duplication while countint the frequency
data['reviews.text']=data['reviews.text'].str.lower()
data['reviews.text'].head()


# In[18]:


#removal pf punctuation
import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

data['reviews.text']=data['reviews.text'].astype(str).apply(lambda text: remove_punctuation(text))


# In[19]:


#removal of stop words.
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = (stopwords.words('english'))

STOPWORDS.extend(['from', 'subject','made','using','past','years','3d','year','study','showed','show','ha','time','new','use','useful','hence','objective','aim' 're', 'edu', 'use', 'ieee', 'elsevier', 'ltd', 'rights', 'also', 'find', 'may', 
                   'include'])
STOPWORDS


# In[20]:


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

data['reviews.text']=data['reviews.text'].apply(lambda text: remove_stopwords(text))


# In[21]:


data['reviews.text'].head()


# In[22]:


#converting abstarct to list
# Here dop_abstract will take dopant Abstract. for Scaffold data we write scaff_abstract instead of dop_abstract and do the same.
#scaff_abstract= scaffold.Abstract.Values.tolist()
data_text=data['reviews.text']


# In[23]:


import re
data_text= [re.sub("\'", "", str(abstract)) for abstract in data_text]
# Remove all non keyboard characters
data_text = [re.sub('[^A-Za-z0-9]+', " ", str(abstract)) for abstract in data_text]


# In[24]:


data_text


# In[25]:


import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess


# In[26]:


# split sentence into words
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

abstracts_words = list(sent_to_words(data_text))


# In[27]:


print(abstracts_words[:1])


# In[28]:


from gensim.models import Phrases
bigram = Phrases(abstracts_words, min_count=3, threshold=10) 
trigram = Phrases(bigram[abstracts_words], threshold=10)  


# In[29]:


# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[30]:


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# In[31]:


# Form Bigrams
abstracts_words_bigrams = make_bigrams(abstracts_words)

# Form Trigrams
abstracts_words_trigrams = make_trigrams(abstracts_words_bigrams)

abstracts_words_bigrams


# In[32]:


abstracts_words_trigrams


# In[33]:


import spacy


def lemmatization(texts, allowed_postags=['NOUN']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[34]:


#python -m spacy download en
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
#nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
# data_lemmatized = lemmatization(abstracts_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
data_lemmatized = lemmatization(abstracts_words_trigrams, allowed_postags=['NOUN'])


# In[35]:


print(data_lemmatized[:3])


# In[37]:


# Save data lemmatized in to pickle data persistence
import pickle

PKL_Loc = "C:/Users/LENOVO/Desktop/LDATopicModelling-main"
# open a file, where you ant to store the data
filename = 'dogs1'
file = open(filename,'wb')
#file = open(PKL_Loc + 'N_data_lemmatized_Sust_AND_Trans.pkl', 'wb')
# dump information to that file
pickle.dump(data_lemmatized, file)
# close the file
file.close()


# In[38]:


#create dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]


# In[39]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=50,
                                           alpha='auto',
                                           per_word_topics=True)


# In[40]:


from pprint import pprint
pprint(lda_model.show_topics(formatted=False))


# In[41]:


print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
from gensim.models import CoherenceModel
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[43]:


import pyLDAvis
import pyLDAvis.gensim_models 
pyLDAvis.enable_notebook()
m=pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)


# In[44]:


from wordcloud import WordCloud#, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
for t in range(lda_model.num_topics):
    plt.figure()
    cloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(" ".join(np.array(lda_model.show_topic(t, 200))[: , 0]))
    #plt.savefig('C:/Users/LENOVO/Desktop/LDATopicModelling-main/books_read.png')
    plt.imshow(cloud)

