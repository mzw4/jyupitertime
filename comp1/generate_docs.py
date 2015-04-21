import csv
import os
import random

"""
Min probability = 4.14662464754e-05.
Num words should be greater than 24116.
Max distinct words in speech is 4917.
"""
if __name__ == '__main__':

    words_to_keep = set(range(25000, 50000))

    NUM_SPEECHES = 2740
    DOC_DIR = 'generated_docs'
    WORDS_PER_DOC = 50000

    if not os.path.exists(DOC_DIR):
        os.mkdir(DOC_DIR)
    
    reader = csv.reader(open('speech_vectors.csv'), delimiter=',')

    min_probability = 2
    max_distinct_words_in_speech = 0

    # Read the probability matrix.
    probabilities = []
    for row_num, row in enumerate(reader):
        doc_probs = []
        for i, p_str in enumerate(row):
            p = float(p_str)
            if p != 0.0:
                doc_probs.append( (i,p) )

                if p < min_probability:
                    min_probability = p
        if len(doc_probs) > max_distinct_words_in_speech:
            max_distinct_words_in_speech = len(doc_probs)
        probabilities.append(doc_probs)
        print 'Reading csv:', 100.0 * (row_num+1) / NUM_SPEECHES, '%'

    assert len(probabilities) == NUM_SPEECHES
    print 'Min probability:', min_probability
    print 'Num words should be greater than?', 1.0/min_probability
    print 'Max distinct words in speech:', max_distinct_words_in_speech

    # Generate documents with the given word probabilities.
    for doc_num, document in enumerate(probabilities):
        words = []
        for a_word in range(WORDS_PER_DOC):
            current_prob = random.uniform(0,1)
            cumulative_prob = 0.0
            for word, word_prob in document:
                cumulative_prob += word_prob
                if current_prob < cumulative_prob:
                    # Only keep words not relating to sentiment.
                    if word in words_to_keep:
                       words.append(str(word))
                    break
        # Write the words to a file.
        doc_path = os.path.join(DOC_DIR, str(doc_num) + '.txt')
        open(doc_path, 'w+').write(' '.join(words))

        print 'Generating docs:', 100.0 * (doc_num+1) / NUM_SPEECHES, '%'
