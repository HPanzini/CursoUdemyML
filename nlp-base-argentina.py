#sacar el dataset de github -andando

import pandas as pd
import re
url = 'https://raw.githubusercontent.com/HPanzini/CursoUdemyML/master/all%20Argentina%20q2%202020.txt'
dataset = pd.read_csv(url, encoding='latin-1', sep='\t')
datastring = str(dataset)
#test raw data -andando

print(dataset)
print(type(dataset))

#limpiar texto

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1705):
    review= re.sub('[^a-zA-Z]', ' ', dataset['Texto'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words("spanish"))]
    review=' '.join(review)
    corpus.append(review)

print(corpus)

# modelo bag of words 
# max features toma las 1500 palabras mas comunes, sacando stopwords

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:, -1].values

# training sets y test sets
# test-size 0.20 indica que el 20% de nuestro doc va a ser usado para comprobar el funcionamiento de el ML
# random-state=0 nos dice que no se va a mezclar

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=0)

# entrenamiento (elegir modelo mas indicado evaluando accuracy vs test set)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# predecir resultados utilizando el test-set del dataset

import numpy as np
Y_pred = classifier.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))

#confusion matrix + accuracy score

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test,Y_pred)
print(cm)
accuracy_score(Y_test,Y_pred)

## predecir documento

urlpred = 'https://raw.githubusercontent.com/HPanzini/CursoUdemyML/master/Predecir.txt'
datapred = pd.read_csv(urlpred, encoding='latin-1', sep='\t')
print(new_review)
corpuspred = []
for i in range(0,5):
    new_review = re.sub('[^a-zA-Z]', ' ', datapred['Texto'][i])
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    new_review = [ps.stem(word) for word in new_review if not word in set(stopwords.words("spanish"))]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = cv.transform(new_corpus).toarray()
    new_y_pred = classifier.predict(new_X_test)
    print(new_y_pred)