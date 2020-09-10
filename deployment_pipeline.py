#!/usr/bin/env python
# coding: utf-8

# ### 1. Loading training dataset

# In[ ]:


import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, jsonify, request


# In[ ]:


# 1. Acquiring preprocessed_dataset
tbs_df = pd.read_csv('tbs_df.csv')
tbs_df = tbs_df.fillna(' ')


# ### 2. Tag Predictor

# In[ ]:


import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
from tensorflow.keras.layers import Input, Softmax, GRU, LSTM, RNN, Embedding, Dense, RepeatVector, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model

import numpy as np


# In[ ]:


# 1. Loading saved tokenizers
import pickle
handle = open('tag_predictor_tokenizer.pickle', 'rb')
token_text = pickle.load(handle)

handle = open('tag_predictor_token_tar.pickle', 'rb')
token_tar = pickle.load(handle)

text_vocab = token_text.word_index
tar_vocab = token_tar.word_index


# In[ ]:


# 2. loading saved w2v model
from gensim.models import Word2Vec
w2v_model_sg = Word2Vec.load("word2vec_sg.model")
len(w2v_model_sg.wv.vocab)


# In[ ]:


# 3. creating Embedding Matrix with Word2Vec representations
max_words = 20000
w2v_vocab = set(w2v_model_sg.wv.vocab)
embedding_matrix = np.random.normal(loc = 0, scale = 0.15, size = (max_words+1, 100))
for word, i in text_vocab.items():
    if i <= max_words and word in w2v_vocab:
      vector = w2v_model_sg[word]
    # if vector is not None:
      embedding_matrix[i] = vector
embedding_matrix.shape


# In[ ]:


# 4. Creating freezed 'Embedding layer'
from tensorflow.keras.initializers import Constant
text_embedding_layer = Embedding(input_dim = max_words+1, output_dim= 100, embeddings_initializer = Constant(embedding_matrix),
                               mask_zero = True, trainable = False, name = 'text_embed')


# In[ ]:


# 5. Constructing a model
tf.keras.backend.clear_session()

enc_inputs = Input(name = 'text_seq', shape = (250,))
enc_embed = text_embedding_layer(enc_inputs)
encoder = Bidirectional(GRU(name = 'ENCODER', units = 128, dropout = 0.2))

enc_out = encoder(enc_embed)

dec_lstm = GRU(name = 'DECODER', units = 256, dropout = 0.2, return_sequences= True, return_state= True)

repeat = RepeatVector(5)(enc_out)
dec_out, dec_hidden = dec_lstm(repeat)

dec_dense = Dense(units = len(tar_vocab)+1, activation = 'softmax')
out = dec_dense(dec_out)

model = Model(inputs = enc_inputs, outputs = out)
model.summary()


# In[ ]:


# 6. loading model weights
model.load_weights('weights--019--2.4615.hdf5')


# In[ ]:


# defining a function to remove stop_words
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.add('would')
stop_words.update([chr(c) for c in range(97, 123)])
# stop_words.remove('no'); stop_words.remove('not'); stop_words.remove('nor')

def stopwrd_removal(sent):
  lst = []
  for wrd in sent.split():
    if wrd not in stop_words:
      lst.append(wrd)
  return " ".join(lst)


# In[ ]:


# 7. input text preprocessor
def text_preprocessor(corpus, stop_word = False, remove_digits = False):
  clean_corpus = []
  for doc in corpus:
    # 1. remove html tags, html urls, replace html comparison operators
    clean_str = re.sub('<.*?>', '', doc)
    clean_str = clean_str.replace('&lt;', '<').replace('&gt;', '>').replace('&le;', '<=' ).replace('&ge;', '>=')

    # 2. remove latex i,e., mostly formulas since it's mathematics based dataset
    clean_str = re.sub('\$.*?\$', '', clean_str)

    # 3. all lowercase
    clean_str = clean_str.lower()

    # 4. decontractions
    clean_str = clean_str.replace("won't", "will not").replace("can\'t", "can not").replace("n\'t", " not").replace("\'re", " are").replace("\'s", " is").replace("\'d", " would").replace("\'ll", " will").replace("\'t", " not").replace("\'ve", " have").replace("\'m", " am")

    # 5. remove all special-characters other than alpha-numericals
    clean_str = re.sub('\W', ' ', clean_str)
    if remove_digits == True:
      clean_str = re.sub('\d', ' ', clean_str)

    # 6. Stop_word removal
    if stop_word == True:
      clean_str = stopwrd_removal(clean_str)

    # 7. remove all white-space i.e., \n, \t, and extra_spaces
    clean_str = re.sub('  +', ' ', clean_str)
    clean_str = clean_str.replace("\n", " ").replace("\t", " ").strip()

    clean_corpus.append(clean_str)

  return clean_corpus

def padded_sequence(clean_corpus):
    # 8. converting words into tokens (int)
    tokens = token_text.texts_to_sequences(clean_corpus)

    # 9. padding the sequence
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    sequence = pad_sequences(tokens, maxlen = 250, padding = 'post')

    return sequence


# In[ ]:


tar_vocab_reverse = {v:k for k,v in tar_vocab.items()}
def final_tag_prediction(corpus):
  """
  1. Text preprocessing on corpus
  2. Convert clean corpus to padded sequence
  3. Passes sequence through model and model generates output
  4. Converts model output into tags list of tags
  """
  # 10. model prediction
  clean_corpus = text_preprocessor(corpus,  stop_word = False, remove_digits = False)
  sequence = padded_sequence(clean_corpus)
  model_out = model.predict(sequence)

  # 11. converting model prediction to human readable tags
  final_lst = []
  for dp in model_out:
    tar_wrd_idx_lst = []
    for time_step in dp:
      tar_wrd_idx = np.argmax(time_step)
      tar_wrd = ('<' + tar_vocab_reverse[tar_wrd_idx] + '>')
      tar_wrd_idx_lst.append(tar_wrd)
      tar_wrd_idx_lst = list(set(tar_wrd_idx_lst))
    final_lst.append(tar_wrd_idx_lst)

  return final_lst


# In[ ]:


# 12. Creating a dictionary of {tag : datapoint_idx}
idx = []
unique_tags = list(set(np.concatenate([tbs_df['tag_pred1'].unique(), tbs_df['tag_pred2'].unique(), tbs_df['tag_pred3'].unique(), tbs_df['tag_pred4'].unique(), tbs_df['tag_pred5'].unique()], axis = 0)))
unique_tags.remove('-')
for tag in unique_tags:
  idx.append(list(tbs_df[(tbs_df['tag_pred1'] == tag) | (tbs_df['tag_pred2'] == tag) | (tbs_df['tag_pred3'] == tag) | (tbs_df['tag_pred4'] == tag) | (tbs_df['tag_pred5'] == tag)].index))
tag_idx_dict = dict(zip(unique_tags, idx))


# In[ ]:


# 13. final indices of tag_corpus
def tag_corpus(tags_pred):
  """This function takes predicted tags and returns indices of corrospoinding questions from dataset"""
  tag_corpus_idx = []
  for tag in tags_pred:
    tag_corpus_idx += tag_idx_dict[tag]
  return list(set(tag_corpus_idx))


# In[ ]:


# FINAL TESTING
corpus = [tbs_df['Title'][129792]]
tags_pred = final_tag_prediction(corpus)
tag_corpus_idx = tag_corpus(tags_pred[0])
print(len(tag_corpus_idx), tags_pred)


# ### 3. LDA

# In[ ]:


# 1. Loading LDA model and LDA_dictionary
from gensim.models.ldamodel import LdaModel

handle = open('LDA_dictionary.pickle', 'rb')
dictionary = pickle.load(handle)

ldamodel_title_body_tag = LdaModel.load('ldamodel_title_body_tag')


# In[ ]:


# 2. defining a final topic prediction function
def final_topic_prediction(corpus):
  clean_corpus = text_preprocessor(corpus, stop_word = True, remove_digits = True)
  tokens_corpus = [i.split(' ') for i in clean_corpus]
  BOW_corpus = [dictionary.doc2bow(i) for i in tokens_corpus]

  topics_pred = []
  for BOW_query in BOW_corpus:
    topic_proba_tuple = ldamodel_title_body_tag.get_document_topics(BOW_query, minimum_probability = 0.20)
    topics_pred.append([k for k,v in topic_proba_tuple])
  return topics_pred


# In[ ]:


# 3. Creating a dictionary of {topic_id : datapoint_idx}
idx = []
topics = list(set(np.concatenate([tbs_df['topic_pred1'].unique(), tbs_df['topic_pred2'].unique(), tbs_df['topic_pred3'].unique(), tbs_df['topic_pred4'].unique()], axis = 0)))
topics.remove(1000)
for topic in topics:
    idx.append(list(tbs_df[(tbs_df['topic_pred1'] == topic) | (tbs_df['topic_pred2'] == topic) | (tbs_df['topic_pred3'] == topic) | (tbs_df['topic_pred4'] == topic)].index))
topic_idx_dict = dict(zip(topics, idx))


# In[ ]:


# 4. final indices of tag_corpus
def topic_corpus(topics_pred):
  """This function takes predicted topics and returns indices of corrospoinding questions from dataset"""
  topic_corpus_idx = []
  for topic in topics_pred:
    topic_corpus_idx += topic_idx_dict[topic]
  return list(set(topic_corpus_idx))


# In[ ]:


# FINAL TESTING
topics_pred = final_topic_prediction([tbs_df['Title'].values[129792]])
topics_corpus_idx = topic_corpus(topics_pred[0])
print(len(topics_corpus_idx), topics_pred)


# ### 4. BM25

# In[ ]:


# get_ipython().system('pip install rank-bm25')
from rank_bm25 import BM25Okapi


# In[ ]:


# 1. preparing dataset for BM25 : truncated "title + body"
# title_body preprocessing
corpus =  tbs_df['combined_text'].values
title_body = text_preprocessor(corpus, remove_digits= True, stop_word=True)

# truncating title_body on 40 words
title_body = [' '.join(i.split(' ')[:40]) for i in title_body]

len(title_body)


# In[ ]:


# 2. Training BM25 model
train_tokens = [i.split(' ') for i in title_body]
bm25 = BM25Okapi(train_tokens)


# In[ ]:


# 3. Defining a final function
def BM25_corpus(query, train_data, n_results):
  # finding results indices
  query = text_preprocessor([query], remove_digits= True, stop_word=True)[0]
  tokenized_query = query.split(" ")
  idx = range(len(train_data))
  BM25_corpus_idx = bm25.get_top_n(tokenized_query, idx, n = n_results)

  # getting scores associated with each result
  doc_scores = bm25.get_scores(tokenized_query)
  BM25_scores = np.sort(doc_scores)[::-1][:n_results]

  return BM25_corpus_idx, BM25_scores


# In[ ]:


# Final testing
query = tbs_df['Title'][129792]
BM25_corpus_idx, BM25_scores = BM25_corpus(query, train_data = train_tokens, n_results = 10)
print('query :', query, '\n\nBM25_corpus_idx :', BM25_corpus_idx, '\n\nBM25_scores :', BM25_scores, '\n\nreults :', tbs_df.Title.values[BM25_corpus_idx])

# In[ ]:


len(set(tag_corpus_idx + topics_corpus_idx + BM25_corpus_idx))


# ### 5. Combining all corpus indices : 'tag_corpus' + 'topic_corpus' + 'BM25_corpus)

# In[ ]:


def all_results_idx(query):
  # 1. tag_predictor
  tags_pred = final_tag_prediction([query])
  tag_corpus_idx = tag_corpus(tags_pred[0])

  # 2. LDA - topic prediction
  topics_pred = final_topic_prediction([query + ' ' + ' '.join(tags_pred[0])]) # adding tags to query
  topics_corpus_idx = topic_corpus(topics_pred[0])

  # 3. BM25 results
  BM25_corpus_idx, BM25_scores = BM25_corpus(query, train_tokens, n_results = 100)

  all_idx = list(set(tag_corpus_idx + topics_corpus_idx + BM25_corpus_idx))
  return all_idx, dict(zip(BM25_corpus_idx, BM25_scores))


# In[ ]:


# testing
query = tbs_df['Title'][129792]
all_idx, BM25_dict = all_results_idx(query)
print(len(all_idx), query)


# ### 6.Sentence Embeddings
#

# ### 6.1 BERT

# In[ ]:


# 1. Loading BERT vector representation of all questions in the dataset
bert_embeddings = np.load('bert_train_out.npy')

# 2. Laoding pretrained BERT model
import tensorflow as tf
import tensorflow_hub as hub
# get_ipython().system('pip install transformers')
from transformers import BertTokenizer, TFBertModel

# Load pretrained model/tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')


# In[ ]:


# 3. BERT model
bert_input_ids = Input(name = 'bert_input_ids', shape = (44,), dtype = 'int64')
bert_attn_mask = Input(name = 'bert_attn_mask', shape = (44,), dtype = 'int64')
bert_token_typ = Input(name = 'bert_token_typ', shape = (44,), dtype = 'int64')

bert_output = bert_model([bert_input_ids, bert_attn_mask, bert_token_typ])
bert_output = bert_output[0][:,0,:]
# bert_output = bert_output[1][:]

BERT = Model(inputs = [bert_input_ids, bert_attn_mask, bert_token_typ], outputs = bert_output)
BERT.summary()


# In[ ]:


# 4. input text preprocessor
def text_preprocessor(corpus, stop_word = False, remove_digits = False):
  clean_corpus = []
  for doc in corpus:
    # 1. remove html tags, html urls, replace html comparison operators
    clean_str = re.sub('<.*?>', '', doc)
    clean_str = clean_str.replace('&lt;', '<').replace('&gt;', '>').replace('&le;', '<=' ).replace('&ge;', '>=')

    # 2. remove latex i,e., mostly formulas since it's mathematics based dataset
    clean_str = re.sub('\$.*?\$', '', clean_str)

    # 3. all lowercase
    clean_str = clean_str.lower()

    # 4. decontractions
    clean_str = clean_str.replace("won't", "will not").replace("can\'t", "can not").replace("n\'t", " not").replace("\'re", " are").replace("\'s", " is").replace("\'d", " would").replace("\'ll", " will").replace("\'t", " not").replace("\'ve", " have").replace("\'m", " am")

    # # 5. remove all special-characters other than alpha-numericals
    clean_str = re.sub('\W', ' ', clean_str)
    if remove_digits == True:
      clean_str = re.sub('\d', ' ', clean_str)

    # 6. Stop_word removal
    if stop_word == True:
      clean_str = stopwrd_removal(clean_str)

    # 7. remove all white-space i.e., \n, \t, and extra_spaces
    clean_str = re.sub('  +', ' ', clean_str)
    clean_str = clean_str.replace("\n", " ").replace("\t", " ").strip()

    clean_corpus.append(clean_str)

  return clean_corpus


# In[ ]:


# 5. A function to create vector representation of query
def BERT_sentence_vec(query):
  clean_query = text_preprocessor([query],  stop_word = False, remove_digits = False)
  tokens = bert_tokenizer.batch_encode_plus(clean_query, truncation = True, max_length =  44, pad_to_max_length = True)

  input_ids = np.array(tokens['input_ids'])
  attn_mask = np.array(tokens['attention_mask'])
  token_typ_ids = np.array(tokens['token_type_ids'])
  bert_out = BERT.predict([input_ids, attn_mask, token_typ_ids])
  return bert_out


# In[ ]:


# cross checking the current results with previous results
query = tbs_df['Title'][50]
print(sum(bert_embeddings[50] - BERT_sentence_vec(query)[0]), query)


# In[ ]:


# 6. Final testing of BERT model with query point
query = "How does deepmind's Atari game AI work?"
bert_out = BERT_sentence_vec(query)
print(bert_out.shape)


# ### 6.2. USE Embeddings

# In[ ]:


# 1. Loading USE vector representation of all questions in the dataset
use_embeddings = np.load('use_embeddings.npy')


# In[ ]:


# 2. Laoding pretrained USE model
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")


# In[ ]:


# 3. A function to create vector representation of query
def USE_sentence_vec(query):
  clean_query = text_preprocessor([query],  stop_word = False, remove_digits = False)
  use_out = use_model(clean_query)

  return use_out


# In[ ]:


# cross checking the current results with previous results
query = tbs_df['Title'][100000]
print(sum(use_embeddings[100000] - USE_sentence_vec(query)[0]), query)


# In[ ]:


# 4. Final testing of BERT model with query point
query = "How does deepmind's Atari game AI work?"
use_out = USE_sentence_vec(query)
print(use_out.shape)


# ### 7. Ranking : compute cos-sim based results

# In[ ]:


from numpy.linalg import norm
def cos_sim(a, b):
  cos_sim = np.dot(a, b)/(norm(a)*norm(b))
  return cos_sim

def inverse_euc_dist(a, b):
  euc_dist = norm(a-b)
  return 1/euc_dist


# **Final search results using : BERT embeddings**

# In[52]:


def BERT_results(query, n = 10):
  all_idx, BM25_dict = all_results_idx(query)
  BERT_corpus = bert_embeddings[all_idx]

  query_vector = BERT_sentence_vec(query)[0]
  cos_sim_lst = [cos_sim(query_vector, b) for b in BERT_corpus]

  d = dict(zip(all_idx, cos_sim_lst))
  s = sorted(d.items(), key=lambda x: x[1], reverse=True)
  cos_sim_idx = [i for i,j in s]

  return tbs_df.Title.values[cos_sim_idx][:n]


# In[54]:


# some sample queries to try:
# tensorflow vs pytorch # difference between tensorflow and pytorch # keras accuracy stuck # change keras backend
# what is the best deep learning library for scala # install nltk # optimizing overfitted models
query = "change keras backend"
results = BERT_results(query = query, n = 10)
print(query, '\n', results)


# **Final search results using : USE embeddings**

# In[57]:


def USE_results(query, n = 10):
  all_idx, BM25_dict = all_results_idx(query)
  BERT_corpus = use_embeddings[all_idx]

  query_vector = USE_sentence_vec(query)[0]
  cos_sim_lst = [cos_sim(query_vector, b) for b in BERT_corpus]

  d = dict(zip(all_idx, cos_sim_lst))
  s = sorted(d.items(), key=lambda x: x[1], reverse=True)
  cos_sim_idx = [i for i,j in s]

  return tbs_df.Title.values[cos_sim_idx][:10]


# In[58]:

query = "change keras backend"
results = USE_results(query = query, n = 10)
print(query, '\n', results)


# # Crucial points to note:
# **1. The purpose of this case study is to build simple search engine with LOW latency.**
#
# **2. By using 'right data structures' in our mechanism, we make it happen to get results under 800-900 milliseconds on a normal 8 gb machine.**
#
# # Observations :
#     1. pretrained BERT model is not trained to capture semantic relationships at first place. Its trained on two tasks : NSP (Next sentence prediction) and MLM (Masked language model).
#     2. The corpus on which BERT model is trained is general wikipidia data, But our stackoverflow corpus has all the mathematical, computer science and machine learning related technical terms.
#     3. first things first Universal sentence embedding model is trained to capture semantic relationships with contextual meaning.
#
# # Conclusion :
#     1. Hence BERT model fails to give good results as compare to USE model.
#     2. We need to fine tune bert model on our technical corpus to get good results with bert.
#
# - In our case USE embeddings outperformed. A searching mechanism with USE vectors gives great reults with 'semantic relationship' between query and resulting results.

# # 8. Comparison with stackoverflow.com results
#
# find all results here : https://imgur.com/a/9XRVEOd

# ### q1  =  'How to reverse a linked list in python'
# **top 5 results :**
#
#  <img src='https://i.imgur.com/rbCbWua.png' width="800">
#
#  **next 5 results :**
#
#  <img src='https://i.imgur.com/6B4bNU1.png' width="800">
#

# In[60]:

query = "How to reverse a linked list in python"
results = BERT_results(query = query, n = 10)
print(query, '\n', results)


# In[63]:

query = "How to reverse a linked list in python"
results = USE_results(query = query , n = 10)
print(query, '\n', results)


# ### q2  = 'valueerror'
#
# **top 5 results :**
#
# <img src='https://i.imgur.com/8SlM6lH.png' width="800">
#
# **next 5 results :**
#
#  <img src='https://i.imgur.com/AYcDUsI.png' width="800">

# In[64]:

query = "valueerror"
results = BERT_results(query = query, n = 10)
print(query, '\n', results)


# In[66]:

query = "valueerror"
results = USE_results(query = query, n = 10)
print(query, '\n', results)


# ### q3  = 'matplotlib'
#
# **top 5 results :**
#
# <img src='https://i.imgur.com/EryAZE7.png' width="800">
#
# **next 5 results :**
#
#  <img src='https://i.imgur.com/aBc6X0O.png' width="800">

# In[67]:

query = "matplotlib"
results = BERT_results(query = query, n = 10)
print(query, '\n', results)


# In[65]:

query = "matplotlib"
results = USE_results(query = query, n = 10)
print(query, '\n', results)


# In[ ]:


"""**Flask app**"""
import flask
app = Flask(__name__)

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  lst = request.form.to_dict()
  print(lst)
  results = USE_results(query = lst['query'], n = 10).tolist()
  return jsonify({'Search_results' : results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
