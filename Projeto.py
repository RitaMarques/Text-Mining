import pandas as pd
import numpy as np
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
from copy import deepcopy
import unicodedata
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

basedir = r'./Corpora/train/'

#----------------------------------------------------------------------------------------------------------------
# IMPORT TRAIN FILES
#----------------------------------------------------------------------------------------------------------------
def import_folder_files(directory):
    f = []
    fulldir = basedir + directory
    for name, lista, files in os.walk(fulldir):

        for file in files:

            if file.endswith(".txt"):
                f1 = open(fulldir + '\\' + file, "r", encoding='utf-8')
                f1 = f1.read()
                f.append(f1)
    return f


AlmadaNegreiros = import_folder_files('AlmadaNegreiros')
Camilo = import_folder_files('CamiloCasteloBranco')
EcaQueiros = import_folder_files('EcaDeQueiros')
JoseRodriguesSantos = import_folder_files('JoseRodriguesSantos')
JoseSaramago = import_folder_files('JoseSaramago')
LuisaMarquesSilva = import_folder_files('LuisaMarquesSilva')

textos_labels = [[AlmadaNegreiros, 'Almada Negreiros'], [Camilo, 'Camilo Castelo Branco'],
                 [EcaQueiros, 'Eça de Queiros'],
                 [JoseRodriguesSantos, 'José Rodrigues dos Santos'], [JoseSaramago, 'José Saramago'],
                 [LuisaMarquesSilva, 'Luísa Marques Silva']]

df_original = pd.DataFrame(columns=['Label','Text'])

df_original = df_original[0:0]
for lista in textos_labels:
    df_aux = pd.DataFrame({'Label': lista[1],
                            'Text': lista[0]
                            })

    df_original = df_original.append(df_aux, ignore_index=True)

del AlmadaNegreiros, Camilo, EcaQueiros, JoseRodriguesSantos, JoseSaramago, LuisaMarquesSilva, df_aux, lista, \
    textos_labels, basedir

#----------------------------------------------------------------------------------------------------------------
# PRE - PROCESSING
#----------------------------------------------------------------------------------------------------------------
def normalize(s):
    return unicodedata.normalize('NFD', s)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

def feature_eng(texts_column):
    lista_wordsPsent = []

    for text in texts_column:
        soma = 0
        contagem_frases = 0

        for sentence in re.split(r"[\??\!?\.?]+", text):
            soma += len(sentence.strip().split(' '))
            contagem_frases += 1
        lista_wordsPsent.append(round(soma/contagem_frases,2))

    return lista_wordsPsent

def remove_spaces(text_column):
    '''removes multiple spaces'''
    text_column = text_column.apply(lambda x: re.sub(r"\s+", " ", x))
    return text_column


def clean(dataframe, stopwords_bol=True, stemmer_bol=True, lemmatizer_bol=False, punctuation_all=True):
    ''' 
    Does lowercase, stopwords, creates new features
    '''
    df = deepcopy(dataframe)

    # lowercase
    df['Text'] = df['Text'].str.lower()

    # remove tags
    df['Text'] = df['Text'].apply(lambda x: BeautifulSoup(x, "html.parser").get_text())

    # replace number with token
    df['Text'] = df['Text'].apply(lambda x: re.sub('\d+', 'NUMBER', x))

    # replace \n with a space
    df['Text'] = df['Text'].apply(lambda x: x.replace('\n', ' '))

    # removes multiple spaces
    df['Text'] = remove_spaces(df['Text'])

    # create feature nr médio palavras por frase
    df['WordsPerSentence'] = feature_eng(df['Text'])

    # remove "acentos"
    regexp = re.compile(r'\w\'\w')
    for idx, row in df.iterrows():
        sentence = []
        for word in row.Text.split(' '):
            if regexp.search(word):
                # see the accuracy with both cases
                    # replaces the ' with APOSTROPHE: dAPOSTROPHEangustia
                #word = re.sub("\'", 'APOSTROPHE', word)
                    # cuts the ' and adds a feature APOSTROPHE
                word = re.sub("\'", '', word) # em conjunto com o de baixo
                word = word + " APOSTROPHE" # caso o de cima aconteça
            sentence.append(normalize(word))
        df.iloc[idx, 1] = ' '.join(word for word in sentence)

    # re.split(r"\s+", text)                     SPLITTING A STRING WITH MULTIPLE SPACES
    # re.sub(r"\s+[a-zA-Z]\s+", " ", text)       REMOVING A SINGLE CHARACTER

    # replace ! and ? with token
    df['Text'] = df['Text'].apply(lambda x: re.sub('\?|\!', ' EXPRESSION', x))

    # remove all punctuation
    if punctuation_all == True:
        df["Text"] = df['Text'].str.replace('[^a-zA-Z]', ' ')

    # removes multiple spaces
    df['Text'] = remove_spaces(df['Text'])

    for idx, row in df.iterrows():

        # remove stopwords
        if stopwords_bol == True:
            stop = stopwords.words('portuguese') # create stopwords

            sentence = []
            for word in row.Text.split(' '):
                if word not in stop:
                    sentence.append(word)
            df.iloc[idx, 1] = ' '.join(word for word in sentence)

        # Stemmer
        if stemmer_bol == True:
            snowball_stemmer = SnowballStemmer('portuguese')
            
            df.iloc[idx, 1] = ' '.join(snowball_stemmer.stem(word)
                                    for word in row[1].split())

        # Lemmatizer
        elif lemmatizer_bol == True:
            lemma = WordNetLemmatizer()
            df.iloc[idx, 1] = ' '.join(lemma.lemmatize(word)
                                       for word in row[1].split())

    return df

df_cleaned = clean(df_original, stemmer_bol=True)
df_cleaned_feat = clean(df_original, stemmer_bol=False)

#----------------
# Split dataset
#----------------
X_train, X_test, y_train, y_test = train_test_split(
                df_cleaned['Text'], df_cleaned['Label'], test_size=0.20, stratify=df_cleaned['Label'], shuffle=True, random_state=1)

# nr of texts in train
X_train.shape
# nr of texts in test
X_test.shape

#------------------------------------------------------------------------------------------------------------
# LANGUAGE MODEL
#------------------------------------------------------------------------------------------------------------

#------------------------------------
# Bag of Words - binary
#------------------------------------
cv = CountVectorizer(
    max_df=0.9, 
    #max_features=10000, 
    ngram_range=(1,3),
    binary=True # only 0 and 1 for each word
)

X_train_cv = cv.fit_transform(X_train)

# we have to use the same vectorizer for the test set, as we used for the train set!!!
X_test_cv = cv.transform(X_test)

#------------------------
# TF-IDF
#------------------------
cv = CountVectorizer(
    max_df=0.9,
    #max_features=10000,
    ngram_range=(1,3),
    binary=False # counts per word
)
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

feature_names = cv.get_feature_names()

tfidf = TfidfTransformer()
tfidf.fit(X_train_cv)

tf_idf_vector = tfidf.transform(X_test_cv)


def extract_feature_scores(feature_names, document_vector):
    """
    Function that creates a dictionary with the TF-IDF score for each feature.
    :param feature_names: list with all the feature words.
    :param document_vector: vector containing the extracted features for a specific document

    :return: returns a sorted dictionary "feature":"score".
    """
    feature2score = {}
    for i in range(len(feature_names)):
        feature2score[feature_names[i]] = document_vector[0][i]
    return sorted(feature2score.items(), key=lambda kv: kv[1], reverse=True)

extract_feature_scores(feature_names, tf_idf_vector.toarray())[:30]

#--------------------------
# POS Tagging
#--------------------------
#nltk.download('mac_morpho')
nltk.corpus.mac_morpho.tagged_words()
#nltk.download('punkt')
tagger = nltk.data.load('tokenizers/punkt/portuguese.pickle')



#------------------------------------------------------------------------------------------------------------
# MACHINE LEARNING ALGORITHMS
#------------------------------------------------------------------------------------------------------------

#--------------------------
# KNN
#--------------------------
# Clustering the document with KNN classifier
modelknn = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='brute',
                                         metric='cosine')
modelknn.fit(X_train_cv,y_train)

predict = modelknn.predict(X_test_cv)


#--------------------------
# Results
#--------------------------
class1 = classification_report(predict, y_test)
print (classification_report(predict, y_test))

conf_matrix = confusion_matrix(predict, y_test)

# function to display confusion matrix
def plot_cm(confusion_matrix : np.array, classnames : list):
    """
    Function that creates a confusion matrix plot using the Wikipedia convention for the axis. 
    :param confusion_matrix: confusion matrix that will be plotted
    :param classnames: labels of the classes"""
    
    confusionmatrix = confusion_matrix
    class_names = classnames             

    fig, ax = plt.subplots()
    im = plt.imshow(confusionmatrix, cmap=plt.cm.cividis)
    plt.colorbar()

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, confusionmatrix[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Confusion Matrix")
    plt.xlabel('Targets')
    plt.ylabel('Predictions')
    plt.ylim(top=len(class_names)-0.5)  # adjust the top leaving bottom unchanged
    plt.ylim(bottom=-0.5)  # adjust the bottom leaving top unchanged
    return plt.show()


labels = ['Almada Negreiros','Camilo Castelo Branco','Eça de Queirós','José Rodrigues dos Santos',
          'José Saramago','Luísa Marques Silva']
plot_cm(conf_matrix,labels)


#--------------------------
# Random things
#--------------------------

# count the number of words per text
counts = []
for idx, row in df_original.iterrows():
    counts.append(len(row[1].split()))


# Split sentences in words
df_cleaned['String'] = df_cleaned.Text.str.split(' ')

# WORD COUNTER
def word_counter(df):
    """
    Function that receives a list of strings and returns the frequency of each word
    in the set of all strings.
    """
    counter = []
    for idx, row in df.iterrows():
        words_in_df = " ".join(row[2]).split()

        # Count all words
        counter.append(pd.Series(words_in_df).value_counts())

    return counter

counter = word_counter(df_cleaned)

# see the datasets
df_cleaned.iloc[0, 1]
df_original.iloc[0, 1]