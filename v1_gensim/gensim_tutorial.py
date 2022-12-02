# TUTORIAL LINK:
# https://www.geeksforgeeks.org/nlp-gensim-tutorial-complete-guide-for-beginners/

import gensim
import os
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.test.utils import get_tmpfile
import gensim.downloader as api
from gensim.models.phrases import Phrases
from gensim.corpora import MmCorpus
from gensim.test.utils import get_tmpfile
from gensim import models
import numpy as np
from multiprocessing import cpu_count
from gensim.models.word2vec import Word2Vec
from gensim.models import doc2vec
from gensim.models import LdaModel, LdaMulticore
import gensim.downloader as api
from gensim.utils import simple_preprocess
from gensim.models import LsiModel
import gensim.downloader as api
from gensim.matutils import softcossim
from gensim import corpora
# from pattern.en import lemma
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import logging

# open the text file as an object
doc = open('sample_data.txt', encoding ='utf-8')

# preprocess the file to get a list of tokens
tokenized =[]
for sentence in doc.read().split('.'):
    # the simple_preprocess function returns a list of each sentence
    tokenized.append(simple_preprocess(sentence, deacc = True))

print(tokenized)

# storing the extracted tokens into the dictionary
my_dictionary = corpora.Dictionary(tokenized)
print(my_dictionary)

# save your dictionary to disk
my_dictionary.save('my_dictionary.dict')

# load back
# load_dict = corpora.Dictionary.load(my_dictionary.dict')
								
# save your dictionary as text file
tmp_fname = get_tmpfile("dictionary")
my_dictionary.save_as_text(tmp_fname)

# load your dictionary text file
load_dict = corpora.Dictionary.load_from_text(tmp_fname)

# converting to a bag of word corpus
BoW_corpus =[my_dictionary.doc2bow(doc, allow_update = True) for doc in tokenized]
print(BoW_corpus)

output_fname = get_tmpfile("BoW_corpus.mm")

# save corpus to disk
MmCorpus.serialize(output_fname, BoW_corpus)

# load corpus
load_corpus = MmCorpus(output_fname)

# Word weight in Bag of Words corpus
word_weight =[]
for doc in BoW_corpus:
    for id, freq in doc:
        word_weight.append([my_dictionary[id], freq])
    print(word_weight)

# create TF-IDF model
tfIdf = models.TfidfModel(BoW_corpus, smartirs ='ntc')

# TF-IDF Word Weight
weight_tfidf =[]
for doc in tfIdf[BoW_corpus]:
    for id, freq in doc:
        weight_tfidf.append([my_dictionary[id], np.around(freq, decimals = 3)])
    print(weight_tfidf)


# load the text8 dataset
dataset = api.load("text8")

# extract a list of words from the dataset
data =[]
for word in dataset:
    data.append(word)
			
# Bigram using Phraser Model			
bigram_model = Phrases(data, min_count = 3, threshold = 10)

print(bigram_model[data[0]])


# Trigram using Phraser Model
trigram_model = Phrases(bigram_model[data], threshold = 10)

# trigram
print(trigram_model[bigram_model[data[0]]])


# ------------------------------------------------------------------------------------------

# extract a list of words from the dataset
data =[]
for word in dataset:
    data.append(word)

# We will split the data into two parts
data_1 = data[:1200] # this is used to train the model
data_2 = data[1200:] # this part will be used to update the model

# Training the Word2Vec model
w2v_model = Word2Vec(data_1, min_count = 0, workers = cpu_count())

# word vector for the word "time"
print(w2v_model['time'])

# similar words to the word "time"
print(w2v_model.most_similar('time'))

# save your model
w2v_model.save('Word2VecModel')

# load your model
model = Word2Vec.load('Word2VecModel')

# build model vocabulary from a sequence of sentences
w2v_model.build_vocab(data_2, update = True)

# train word vectors
w2v_model.train(data_2, total_examples = w2v_model.corpus_count, epochs = w2v_model.iter)

print(w2v_model['time'])



# get dataset

data =[]
for w in dataset:
    data.append(w)

# To train the model we need a list of tagged documents
def tagged_document(list_of_ListOfWords):
    for x, ListOfWords in enumerate(list_of_ListOfWords):
	    yield doc2vec.TaggedDocument(ListOfWords, [x])

# training data
data_train = list(tagged_document(data))

# print trained dataset
print(data_train[:1])

# Initialize the model
d2v_model = doc2vec.Doc2Vec(vector_size = 40, min_count = 2, epochs = 30)

# build the vocabulary
d2v_model.build_vocab(data_train)

# Train Doc2Vec model
d2v_model.train(data_train, total_examples = d2v_model.corpus_count, epochs = d2v_model.epochs)

# Analyzing the output
Analyze = d2v_model.infer_vector(['violent', 'means', 'to', 'destroy'])
print(Analyze)


logging.basicConfig(format ='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level = logging.INFO)

# import stopwords
stop_words = stopwords.words('english')
# add stopwords
stop_words = stop_words + ['subject', 'com', 'are', 'edu', 'would', 'could']

# import the dataset
data = [w for w in dataset]

# Preparing the data
processed_data = []

# for x, doc in enumerate(data[:100]):
# 	doc_out = []
# 	for word in doc:
# 		if word not in stop_words: # to remove stopwords
# 			Lemmatized_Word = lemmatize(word, allowed_tags = re.compile('(NN|JJ|RB)')) # lemmatize
# 			if Lemmatized_Word:
# 				doc_out.append(Lemmatized_Word[0].split(b'/')[0].decode('utf-8'))
# 		else:
# 			continue
# 	processed_data.append(doc_out) # processed_data is a list of list of words

# # Print sample
# print(processed_data[0][:10])

# -------------------------------------------------------------------------

# create dictionary and corpus
dict = corpora.Dictionary(processed_data)
Corpus = [dict.doc2bow(l) for l in processed_data]

# Training
LDA_model = LdaModel(corpus = Corpus, num_topics = 5)
# save model
LDA_model.save('LDA_model.model')

# show topics
print(LDA_model.print_topics(-1))

# probability of a word belonging to a topic
LDA_model.get_term_topics('fire')

bow_list =['time', 'space', 'car']
# convert to bag of words format first
bow = LDA_model.id2word.doc2bow(bow_list)

# interpreting the data
doc_topics, word_topics, phi_values = LDA_model.get_document_topics(bow, per_word_topics = True)

# Training the model with LSI
LSI_model = LsiModel(corpus = Corpus, id2word = dict, num_topics = 7, decay = 0.5)

# Topics
print(LSI_model.print_topics(-1))

s1 = ' Afghanistan is an Asian country and capital is Kabul'.split()
s2 = 'India is an Asian country and capital is Delhi'.split()
s3 = 'Greece is an European country and capital is Athens'.split()

# load pre-trained model
word2vec_model = api.load('word2vec-google-news-300')

# Prepare the similarity matrix
similarity_matrix = word2vec_model.similarity_matrix(dict, tfidf = None, threshold = 0.0, exponent = 2.0, nonzero_limit = 100)

# Prepare a dictionary and a corpus.
docs = [s1, s2, s3]
dictionary = corpora.Dictionary(docs)

# Convert the sentences into bag-of-words vectors.
s1 = dictionary.doc2bow(s1)
s2 = dictionary.doc2bow(s2)
s3 = dictionary.doc2bow(s3)

# Compute soft cosine similarity
print(softcossim(s1, s2, similarity_matrix)) # similarity between s1 &s2

print(softcossim(s1, s3, similarity_matrix)) # similarity between s1 &s3

print(softcossim(s2, s3, similarity_matrix)) # similarity between s2 &s3


# probability of a word belonging to a topic
LDA_model.get_term_topics('fire')

bow_list =['time', 'space', 'car']
# convert to bag of words format first
bow = LDA_model.id2word.doc2bow(bow_list)

# interpreting the data
doc_topics, word_topics, phi_values = LDA_model.get_document_topics(bow, per_word_topics = True)

# Training the model with LSI
LSI_model = LsiModel(corpus = Corpus, id2word = dict, num_topics = 7, decay = 0.5)

# Topics
print(LSI_model.print_topics(-1))

# ---------------------------------------------------------------------------------------

import gensim.downloader as api
from gensim.matutils import softcossim
from gensim import corpora

s1 = ' Afghanistan is an Asian country and capital is Kabul'.split()
s2 = 'India is an Asian country and capital is Delhi'.split()
s3 = 'Greece is an European country and capital is Athens'.split()

# load pre-trained model
word2vec_model = api.load('word2vec-google-news-300')

# Prepare the similarity matrix
similarity_matrix = word2vec_model.similarity_matrix(dictionary, tfidf = None, threshold = 0.0, exponent = 2.0, nonzero_limit = 100)

# Prepare a dictionary and a corpus.
docs = [s1, s2, s3]
dictionary = corpora.Dictionary(docs)

# Convert the sentences into bag-of-words vectors.
s1 = dictionary.doc2bow(s1)
s2 = dictionary.doc2bow(s2)
s3 = dictionary.doc2bow(s3)

# Compute soft cosine similarity
print(softcossim(s1, s2, similarity_matrix)) # similarity between s1 &s2

print(softcossim(s1, s3, similarity_matrix)) # similarity between s1 &s3

print(softcossim(s2, s3, similarity_matrix)) # similarity between s2 &s3

# ----------------------------------------------------------------------------------------

import gensim.downloader as api
from gensim.matutils import softcossim
from gensim import corpora

s1 = ' Afghanistan is an Asian country and capital is Kabul'.split()
s2 = 'India is an Asian country and capital is Delhi'.split()
s3 = 'Greece is an European country and capital is Athens'.split()

# load pre-trained model
word2vec_model = api.load('word2vec-google-news-300')

# Prepare the similarity matrix
similarity_matrix = word2vec_model.similarity_matrix(dictionary, tfidf = None, threshold = 0.0, exponent = 2.0, nonzero_limit = 100)

# Prepare a dictionary and a corpus.
docs = [s1, s2, s3]
dictionary = corpora.Dictionary(docs)

# Convert the sentences into bag-of-words vectors.
s1 = dictionary.doc2bow(s1)
s2 = dictionary.doc2bow(s2)
s3 = dictionary.doc2bow(s3)

# Compute soft cosine similarity
print(softcossim(s1, s2, similarity_matrix)) # similarity between s1 &s2

print(softcossim(s1, s3, similarity_matrix)) # similarity between s1 &s3

print(softcossim(s2, s3, similarity_matrix)) # similarity between s2 &s3

# -----------------------------------------------------------------------------------------------

# Find Odd one out
print(word2vec_model.doesnt_match(['india', 'bhutan', 'china', 'mango']))
#> mango

# cosine distance between two words.
word2vec_model.distance('man', 'woman')

# cosine distances from given word or vector to other words.
word2vec_model.distances('king', ['queen', 'man', 'woman'])

# Compute cosine similarities
word2vec_model.cosine_similarities(word2vec_model['queen'],
											vectors_all =(word2vec_model['king'],
														word2vec_model['woman'],
														word2vec_model['man'],
														word2vec_model['king'] + word2vec_model['woman']))
# king + woman is very similar to queen.

# words closer to w1 than w2
word2vec_model.words_closer_than(w1 ='queen', w2 ='kingdom')

# top-N most similar words.
word2vec_model.most_similar(positive ='king', negative = None, topn = 5, restrict_vocab = None, indexer = None)

# top-N most similar words, using the multiplicative combination objective,
word2vec_model.most_similar_cosmul(positive ='queen', negative = None, topn = 5)


# -------------------------------------------------------------------

# from gensim.summarization import summarize, keywords
import os

text = " ".join((l for l in open('sample_data.txt', encoding ='utf-8')))

# Summarize the paragraph
# print(summarize(text, word_count = 25))


# Important keywords from the paragraph
# print(keywords(text))
