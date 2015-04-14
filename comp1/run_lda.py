import collections
import gensim
import os
import json
import time

"""
Parameters
"""

IMPORT_DIR = 'clean_docs'

# Model parameters
NUM_TOPICS = 38 # The number of topics to find.
NUM_PASSES = 50 # The number of passes to make over the corpus.

# Filter parameters
FILTER = False
NO_BELOW = 1
NO_ABOVE = 0.95

# Output parameters
N_TOP_WORDS = 10 # The number of top words to show per topic.
TOPIC_FILE = 'topics.txt'
CLASSIFICATION_PATH = 'classes.txt'

"""
Lazy iterator for accessing files. This allows us to access the files without
loading them all into memory.
"""
class MyCorpus(object):
    def __init__(self, directory):
        self.directory = directory

    def __iter__(self):
        for root, dirs, files in os.walk(self.directory):
            for f in files:
                path = os.path.join(root, f)
                text = ' '.join(open(path).readlines())
                yield text.lower().split(), path


"""
This is where gensim code starts.
"""

start_time = time.clock()

###
# Create the dictionary.
###

print 'Creating dictionary...'

dict_start_time = time.clock()
dictionary = gensim.corpora.Dictionary(text for text,_ in MyCorpus(IMPORT_DIR))

print '\t', dictionary
print '\tTime to create dictionary:', time.clock() - dict_start_time

###
# Filter the dictionary.
###

if FILTER:
    print '====='
    print 'Filtering dictionary...'

    filter_start_time = time.clock()
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)

    print '\t', dictionary
    print '\tTime to filter:', time.clock() - filter_start_time

###
# Serialize the corpus.
###

print '====='
print 'Serializing corpus...'

serialize_start_time = time.clock()
corpus = [dictionary.doc2bow(text) for text,_ in MyCorpus(IMPORT_DIR)]
gensim.corpora.MmCorpus.serialize('corpus.mm', corpus)
corpus = None
mm = gensim.corpora.MmCorpus('corpus.mm')

print '\t', mm
print '\tTime to serialize:', time.clock() - serialize_start_time

###
# Train the model.
###

print '====='
print 'Training model...'

train_start_time = time.clock() 
lda_model = gensim.models.ldamodel.LdaModel(corpus=mm, num_topics=NUM_TOPICS, \
    id2word=dictionary, update_every=0, passes=NUM_PASSES, alpha='auto')

print '\tTime to train', time.clock() - train_start_time

###
# Write the topics.
###

print '====='
print 'Writing topics...'

topic_writer = open(TOPIC_FILE, 'w+')
topic_writer.write(json.dumps(list(lda_model.alpha)) + '\n')
for topic_id in range(NUM_TOPICS):
    topic_string = str(topic_id) + ' '
    for p,w in lda_model.show_topic(topic_id, topn=N_TOP_WORDS):
        topic_string += str(p) + '*' + w + ' '
    topic_string += '\n'
    topic_writer.write(topic_string)
topic_writer.close()

###
# Classify the documents.
###

print '====='
print 'Classifying documents...'

classification_writer = open(CLASSIFICATION_PATH, 'w+')
for text,path in MyCorpus(IMPORT_DIR):
    classification_text = path + ' '
    for topic, percent in lda_model[dictionary.doc2bow(text)]:
        classification_text += str(topic) + ' ' + str(percent) + ' '
    classification_text += '\n'
    classification_writer.write(classification_text)
classification_writer.close()

print '====='
print 'Total time elapsed:', time.clock() - start_time
