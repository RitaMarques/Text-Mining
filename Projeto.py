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
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MinMaxScaler
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

def words_sentence(texts_column):
    '''returns a list with the averge number of words per sentence for each sample'''
    lista_wordsPsent = []
    lista_sentences_nr = []

    for text in texts_column:
        soma = 0
        contagem_frases = 0

        for sentence in re.split(r"[\??\!?\.?]+", text):
            soma += len(sentence.strip().split(' '))
            contagem_frases += 1
        
        lista_sentences_nr.append(contagem_frases)
        lista_wordsPsent.append(round(soma/contagem_frases,2))

    return lista_wordsPsent, lista_sentences_nr

def unique_words(text_column):
    '''returns a list with the number of unique words per sample'''
    unique_words = [len(set(text.split(' '))) for text in text_column]
    return unique_words


def remove_spaces(text_column):
    '''removes multiple spaces'''
    text_column = text_column.apply(lambda x: re.sub(r"\s+", " ", x))
    return text_column

def words_per_text(text_column):
    '''returns a list with the total nr of words per sample'''
    total_words = [len(text.split(' ')) for text in text_column]
    return total_words

def clean(dataframe, stopwords_bol=True, stemmer_bol=True, sampled_texts=False, punctuation_all=True):
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

    # if the texts don't have all the same length create feature with the total number of words per text
    if sampled_texts == False:
        df['TotalWords'] = words_per_text(df['Text'])

    # create feature avg number of words per sentence
    df['WordsPerSentence'],_ = words_sentence(df['Text'])
    
    # create feature number of sentences per sample
    _,df['Sentences'] = words_sentence(df['Text'])

    if sampled_texts == False:
        # create feature number of sentences per sample divided by the total words
        _,aux = words_sentence(df['Text'])
        df['Sentences_norm'] = [round(x/y,2) for x,y in zip(aux,df['TotalWords'])]

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

    if sampled_texts == True:
        # get feature with unique words per sample (includes stopwords)
        df['UniqueWords'] = unique_words(df['Text'])
    else:
        # get feature with unique words per sample (includes stopwords) divided by total words per text
        df['UniqueWords'] = [round(x/y,2) for x,y in zip(unique_words(df['Text']),words_per_text(df['Text']))]


    # create feature ExpressionSentences (nr of expressions per sentence)
    df['ExpressionSentences'] = 0

    for idx, row in df.iterrows():
        # populate ExpressionSentences feature
        df.loc[idx,'ExpressionSentences'] = round(row.Text.split(' ').count('EXPRESSION')/row.Sentences,2)
        
        # remove stopwords
        if stopwords_bol == True:
            stop = stopwords.words('portuguese') # create stopwords

            sentence = []
            for word in row.Text.split(' '):
                if word.strip() not in stop:
                    sentence.append(word.strip())
            df.iloc[idx, 1] = ' '.join(element for element in sentence)
    
    for idx, row in df.iterrows():
        # Stemmer
        if stemmer_bol == True:
            snowball_stemmer = SnowballStemmer('portuguese')
            
            df.iloc[idx, 1] = ' '.join(snowball_stemmer.stem(word)
                                    for word in row[1].split())

    if sampled_texts == False:
        df.drop(columns=['Sentences'], inplace=True)

    df.drop(columns=['TotalWords'], inplace=True)
    
    return df

df_cleaned = clean(df_original, stopwords_bol=False, stemmer_bol=True)

#----------------
# Split dataset
#----------------
X_train, X_val, y_train, y_val = train_test_split(
                df_cleaned['Text'], df_cleaned['Label'], test_size=0.20, stratify=df_cleaned['Label'], shuffle=True, random_state=1)

# nr of texts in train
X_train.shape
# nr of texts in test
X_val.shape

train_idx = list(X_train.index)
val_idx = list(X_val.index)
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
X_val_cv = cv.transform(X_val)


#------------------------------------------
# ADD FEATURES (run this section if wanted)
#------------------------------------------
# TRAIN
df_cleaned_train = df_cleaned.iloc[train_idx,:]
vocab = cv.get_feature_names()
X1 = pd.DataFrame(X_train_cv.toarray(), columns = vocab)

X1['Sentences_norm'] = list(df_cleaned_train['Sentences_norm'])
X1['Unique_words'] = list(df_cleaned_train['UniqueWords'])
X1['Expression_Sentences'] = list(df_cleaned_train['ExpressionSentences'])

aux = df_cleaned_train['WordsPerSentence'].values.reshape(-1, 1) #returns a numpy array
min_max_scaler = MinMaxScaler()
x1_scaled = min_max_scaler.fit_transform(aux).flatten().tolist()
X1['Words_Per_Sentence'] = [round(x,3) for x in x1_scaled]

X_sparse = sparse.csr_matrix(X1.values)

# VALIDATION
df_cleaned_val = df_cleaned.iloc[val_idx,:]
vocab = cv.get_feature_names()
X2 = pd.DataFrame(X_val_cv.toarray(), columns = vocab)

X2['Sentences_norm'] = list(df_cleaned_val['Sentences_norm'])
X2['Unique_words'] = list(df_cleaned_val['UniqueWords'])
X2['Expression_Sentences'] = list(df_cleaned_val['ExpressionSentences'])

aux = df_cleaned_val['WordsPerSentence'].values.reshape(-1, 1) #returns a numpy array
min_max_scaler = MinMaxScaler()
x2_scaled = min_max_scaler.fit_transform(aux).flatten().tolist()
X2['Words_Per_Sentence'] = [round(x,3) for x in x2_scaled]

X_sparse = sparse.csr_matrix(X2.values)


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
X_val_cv = cv.transform(X_val)

feature_names = cv.get_feature_names()

tfidf = TfidfTransformer()

X_train_cv = tfidf.fit_transform(X_train_cv)
X_val_cv = tfidf.transform(X_val_cv)


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
#nltk.corpus.mac_morpho.tagged_words()
#nltk.download('punkt')
#tagger = nltk.data.load('tokenizers/punkt/portuguese.pickle')



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

predict = modelknn.predict(X_val_cv)


#--------------------------
# Results
#--------------------------
class0 = classification_report(predict, y_val)
print (class1)

conf_matrix0 = confusion_matrix(predict, y_val)

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
plot_cm(conf_matrix0,labels)






#-------------
# TEST FILES
#-------------






















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
