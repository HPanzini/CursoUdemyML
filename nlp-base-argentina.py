#sacar el dataset de github -andando

import pandas as pd
url = 'https://raw.githubusercontent.com/HPanzini/CursoUdemyML/master/all%20Argentina%20q2%202020.csv?token=APABFG3CRX64KWGJILDHN6C7BJOQK'
dataset = pd.read_csv(url, index_col=0, quoting=3, error_bad_lines=False, warn_bad_lines=False)

#test raw data -andando

print(dataset.head(10))

#tokenizar

from nltk import sent_tokenize, word_tokenize
ejemplo = 'A Panchito lo operamos el 10 de agosto, en la veterinaria El Arca. El veterinario se llama Diego y es pelado. '

print(sent_tokenize(dataset))

print(word_tokenize(dataset))

#sacar stopwords

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("spanish"))
words = word_tokenize(ejemplo)

print(words)

filtered_sentence = []
for w in words:
    if not w in stop_words:
        filtered_sentence.append(w)

print(filtered_sentence)

# stemming

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

for w in word_tokenize(ejemplo):
    print(ps.stem(w))

# speech tagging ??????? 