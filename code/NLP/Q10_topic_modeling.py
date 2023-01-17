# Databricks notebook source
pip install wordcloud

# COMMAND ----------

pip install pyLDAvis 

# COMMAND ----------

#pip install pickle

# COMMAND ----------

pip install nltk

# COMMAND ----------

pip install gensim

# COMMAND ----------

glos_comm = spark.read.parquet("/FileStore/glossier/glossier_comments")

cols = ("author_cakeday","author_flair_css_class","author_flair_text","permalink","stickied","gilded","distinguished","can_gild","retrieved_on","edited")
glos_comm = glos_comm.drop(*cols)
glos_comm = glos_comm.withColumn("created_utc",glos_comm.created_utc.cast('timestamp'))

glos_comm.createOrReplaceTempView("glos_comm_vw")
glos_comm_final = glos_comm

glos_sub = spark.read.parquet("/FileStore/glossier/glossier_submissions")


cols = ("whitelist_status","url","thumbnail_width","thumbnail_height","thumbnail","third_party_tracking_2","third_party_tracking","third_party_trackers","suggested_sort",
       "secure_media_embed", "retrieved_on", "promoted_url", "parent_whitelist_status", "link_flair_text", "link_flair_css_class", "imp_pixel", "href_url", "gilded", "embed_url", 
       "author_flair_css_class", "author_cakeday","adserver_imp_pixel", "adserver_click_url", "secure_media_embed", "secure_media", "post_hint", "permalink", "original_link", 
       "mobile_ad_url", "embed_type", "domain_override", "domain", "author", "preview", "author_flair_text", "edited", "crosspost_parent_list", "media", "media_embed")
glos_sub = glos_sub.drop(*cols)

glos_sub_final = glos_sub.withColumn("created_utc",glos_sub.created_utc.cast('timestamp'))

# COMMAND ----------

top_prods = glos_comm_final.select("body")
new = glos_sub_final.select("title").alias("body")


# COMMAND ----------

df = top_prods.union(new)

# COMMAND ----------

import re
import pandas
dfnew = df.toPandas()
# removing punctuation
dfnew['text_processed'] = \
dfnew['body'].map(lambda x: re.sub('[,\.!?]', '', x))
# all words to lowercase
dfnew['text_processed'] = \
dfnew['text_processed'].map(lambda x: x.lower())
dfnew['text_processed'].head()

# COMMAND ----------

# importing the wordcloud library
from wordcloud import WordCloud
# join the different processed comments together.
long_string = ','.join(list(dfnew['text_processed'].values))
# creating WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# generating word cloud
wordcloud.generate(long_string)
# visualizing
wordcloud.to_image()

# COMMAND ----------

#print(long_string[0:10])
import pandas as pd
def Convert(string):
    li = list(string.split(" "))
    return li
lst = Convert(long_string)
df = pd.DataFrame(lst)

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.ml.feature import Tokenizer #, StopWordsRemover
sparkDF=spark.createDataFrame(df) 

sparkDF.printSchema()


# COMMAND ----------

## defining tokenizer
tokenizer = Tokenizer(outputCol="words")
tokenizer.setInputCol('0')


df_words_token = tokenizer.transform(sparkDF) #.head()
result = df_words_token.withColumn('indword', f.explode(f.col('words'))) \
  .groupBy('indword') \
  .count().sort('count', ascending=False).limit(20)

# COMMAND ----------

#preparing data for LDA Analysis
#transforming the textual data that will serve as input for training LDA model. tokenizing the text, removing stopwords and converting the tokenized object into a corpus and dictionary.

import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'bot', 'please', 'like','questions','moderators', 'subreddit','edu', 'use','automatically', 'names', 'sidebar','rules','message','performed','compose','welcome','action','include','sure','within','deleted','list','much', 'glossier', 'minutes','complete'])
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]
data = dfnew.text_processed.values.tolist()
data_words = list(sent_to_words(data))
# removing stop words
data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])

# COMMAND ----------

#print(data_words[0])
#import numpy as np
#counter = np.unique(data_words, return_counts=True)
#counter

#sparkDF1=spark.createDataFrame(data_words) 

#sparkDF1.printSchema()



# COMMAND ----------

import gensim.corpora as corpora
# creating dict
id2word = corpora.Dictionary(data_words)
#creating corpus
texts = data_words
#term doc frequency
corpus = [id2word.doc2bow(text) for text in texts]
print(corpus[:1][0][:30])

# COMMAND ----------

from pprint import pprint
# num of topics
num_topics = 10
#building LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# COMMAND ----------

import pyLDAvis.gensim_models
import pickle 
import pyLDAvis
import os
#visualizing topics
pyLDAvis.enable_notebook()
LDAvis_data_filepath = os.path.join('./ldavis_prepared_'+str(num_topics))


if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_'+ str(num_topics) +'.html')
LDAvis_prepared

# COMMAND ----------

for idx, topic in lda_model.show_topics(formatted=False, num_words= 10):
    print('Topic: {} \nWords: {}'.format(idx, '|'.join([w[0] for w in topic])))

# COMMAND ----------

cols = ['Topic Num', 'Top 10 Words']
lst = []
for idx, topic in lda_model.show_topics(formatted=False, num_words= 10):
    lst.append([idx+1, ' | '.join([w[0] for w in topic])])
df1 = pd.DataFrame(lst, columns=cols)
df1

# COMMAND ----------

df1.to_csv("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/code/data/csv/topicmodel_top_words.csv")
