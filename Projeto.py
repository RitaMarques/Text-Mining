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

#----------------------------------------------------------------------------------------------------------------
# IMPORT TRAIN FILES
#----------------------------------------------------------------------------------------------------------------
def import_folder_files(basedir, directory):
    f = []
    fulldir = basedir + directory
    for name, lista, files in os.walk(fulldir):

        for file in files:
            if file.endswith(".txt"):
                f1 = open(fulldir + '\\' + file, "r", encoding='utf-8')
                f1 = f1.read()
                f.append(f1)
    return f

def get_dataframe(basedir):
    '''Function that creates the dataframe with the respective Label and Text extracted from the directories'''
    AlmadaNegreiros = import_folder_files(basedir, 'AlmadaNegreiros')
    Camilo = import_folder_files(basedir, 'CamiloCasteloBranco')
    EcaQueiros = import_folder_files(basedir, 'EcaDeQueiros')
    JoseRodriguesSantos = import_folder_files(basedir, 'JoseRodriguesSantos')
    JoseSaramago = import_folder_files(basedir, 'JoseSaramago')
    LuisaMarquesSilva = import_folder_files(basedir, 'LuisaMarquesSilva')

    textos_labels = [[AlmadaNegreiros, 'Almada Negreiros'], [Camilo, 'Camilo Castelo Branco'],
                     [EcaQueiros, 'Eça de Queiros'],
                     [JoseRodriguesSantos, 'José Rodrigues dos Santos'], [JoseSaramago, 'José Saramago'],
                     [LuisaMarquesSilva, 'Luísa Marques Silva']]

    df = pd.DataFrame(columns=['Label', 'Text'])

    df = df[0:0]
    for lista in textos_labels:
        df_aux = pd.DataFrame({'Label': lista[1],
                                'Text': lista[0]
                                })

        df = df.append(df_aux, ignore_index=True)

    return df


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

def clean(dataframe, stopwords_bol=False, stemmer_bol=True, sampled_texts=False, punctuation_all=True):
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


#----------------------------------------------------------------------------------------------------------------
# Split dataset
#----------------------------------------------------------------------------------------------------------------
def split(df, test_size=0.2):
    '''The dataframe has at least 2 columns named Text and Label'''
    X_train, X_val, y_train, y_val = train_test_split(df['Text'], df['Label'], test_size=test_size,
                                                      stratify=df['Label'], shuffle=True, random_state=1)

    return X_train, X_val, y_train, y_val


#------------------------------------------------------------------------------------------------------------
# LANGUAGE MODEL
#------------------------------------------------------------------------------------------------------------
# See which features have higher weights for TF-IDF
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

def language_model(X_train, max_df=0.9, ngram=(1,3), BOW=True, TFIDF=False, binary=True):
    if BOW == True:
        #------------------------------------
        # Bag of Words - binary
        #------------------------------------
        cv = CountVectorizer(
            max_df=max_df,
            #max_features=10000,
            ngram_range=ngram, #(1,3)
            binary=binary# only 0 and 1 for each word
        )

        X_train_cv = cv.fit_transform(X_train)

        return cv, X_train_cv

    elif TFIDF == True:
        #------------------------
        # TF-IDF
        #------------------------
        cv, X_train_cv = language_model(X_train, max_df, ngram, BOW=True, binary=False)

        tfidf = TfidfTransformer()
        X_train_cv = tfidf.fit_transform(X_train_cv)

        return cv, X_train_cv, tfidf

    #--------------------------
    # POS Tagging
    #--------------------------
    #nltk.download('mac_morpho')
    #nltk.corpus.mac_morpho.tagged_words()
    #nltk.download('punkt')
    #tagger = nltk.data.load('tokenizers/punkt/portuguese.pickle')

#------------------------------------------
# ADD FEATURES
#------------------------------------------
def extra_features(df, X_data, cv, X_data_cv, testdata=None):
    '''Creates 4 new features and returns the dataframe with them'''
    if testdata == None:
        data_idx = list(X_data.index)

        data_df = df.iloc[data_idx, :].copy()
    else:
        data_df = df.copy()

    vocab = cv.get_feature_names()
    data_X = pd.DataFrame(X_data_cv.toarray(), columns=vocab)

    data_X['Sentences_norm'] = list(data_df['Sentences_norm'])
    data_X['Unique_words'] = list(data_df['UniqueWords'])
    data_X['Expression_Sentences'] = list(data_df['ExpressionSentences'])

    aux = data_df['WordsPerSentence'].values.reshape(-1, 1)  # returns a numpy array
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(aux).flatten().tolist()
    data_X['Words_Per_Sentence'] = [round(x, 3) for x in x_scaled]

    #X_sparse = sparse.csr_matrix(data_X.values)

    features = True

    return data_X, features


#------------------------------------------------------------------------------------------------------------
# MACHINE LEARNING ALGORITHMS
#------------------------------------------------------------------------------------------------------------
def ml_algorithm(X_train_cv, y_train, KNN=True):
    if KNN == True:
        #--------------------------
        # KNN
        #--------------------------
        # Clustering the document with KNN classifier
        modelknn = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='brute',
                                                 metric='cosine')
        modelknn.fit(X_train_cv, y_train)

        return modelknn


#------------------------------------------------------------------------------------------------------------
# RESULTS
#------------------------------------------------------------------------------------------------------------
# Function to display confusion matrix
def plot_cm(confusion_matrix: np.array, classnames: list):
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
    plt.ylim(top=len(class_names) - 0.5)  # adjust the top leaving bottom unchanged
    plt.ylim(bottom=-0.5)  # adjust the bottom leaving top unchanged
    return plt.show()


def predict(df, cv, model, x_data, y_data, features=None, vectorizer=None, testdata=None):
    """Function that transforms the data that we want to predict, does the final predictions according to the model
    chosen and returns the predictions, the classification measures and the confusion matrix """
    X_cv = cv.transform(x_data)

    if features != None:
        X_cv,_ = extra_features(df, x_data, cv, X_cv, testdata)

    if vectorizer != None:
        X_cv = vectorizer.transform(X_cv)

        feature_names = cv.get_feature_names()

        tf_idf_vector = vectorizer.transform(X_cv)

        scores = extract_feature_scores(feature_names, tf_idf_vector.toarray())[:30]

    data_predict = model.predict(X_cv)

    report = classification_report(data_predict, y_data)
    conf_matrix = confusion_matrix(data_predict, y_data)

    labels = ['Almada Negreiros', 'Camilo Castelo Branco', 'Eça de Queirós', 'José Rodrigues dos Santos',
              'José Saramago', 'Luísa Marques Silva']

    plot_cm(conf_matrix, labels)

    if vectorizer == None:
        return data_predict, report, conf_matrix
    else:
        return data_predict, report, conf_matrix, scores


#------------------------------------------------------------------------------------------------------------
# TRAIN PIPELINE
#------------------------------------------------------------------------------------------------------------
# ---- GET DATA
df_original = get_dataframe(r'./Corpora/train/')

# ---- CLEAN DATA
df_cleaned = clean(df_original, stopwords_bol=False, stemmer_bol=False)

# ---- SPLIT DATA
X_train, X_val, y_train, y_val = split(df_cleaned)

# ---- CHOOSE LANGUAGE MODEL
# If Bag of Words
cv, X_train_cv = language_model(X_train, max_df=0.9, ngram=(1,3), BOW=True, TFIDF=False, binary=True)
# If TF-IDF
# cv, X_train_cv, tfidf = language_model(X_train, max_df=0.9, ngram=(1,3), BOW=False, TFIDF=True, binary=False)

# ---- ADD EXTRA FEATURES
features = None
# If we want extra features, uncomment the following line
X_train_cv, features = extra_features(df_cleaned, X_train, cv, X_train_cv)

# ---- TRAIN MODEL
# KNN
modelknn = ml_algorithm(X_train_cv, y_train, KNN=True)

# ---- PREDICT
# If Bag-of-Words
data_predict, report, conf_matrix = predict(df_cleaned, cv, modelknn, X_val, y_val, features)
# If TF-IDF
#data_predict, report, conf_matrix, scores = predict(cv, modelknn, X_val, y_val, features, vectorizer=tfidf)


#------------------------------------------------------------------------------------------------------------
# TEST FILES
#------------------------------------------------------------------------------------------------------------
def test(testset, cv, modelknn, features, vectorizer=None):
    """Function that predicts our test data"""
    test_cleaned = clean(testset, stopwords_bol=False, stemmer_bol=False)

    return predict(test_cleaned, cv, modelknn, test_cleaned['Text'], test_cleaned['Label'], features,
                   vectorizer, testdata=True)

# 500 WORDS
df_test_500 = get_dataframe(r'./Corpora/test-IMPORT/500Palavras/')

data500_predict, report500, conf_matrix500 = test(df_test_500, cv, modelknn, features)


# 1000 WORDS
df_test_1000 = get_dataframe(r'./Corpora/test-IMPORT/1000Palavras/')

data1000_predict, report1000, conf_matrix1000 = test(df_test_1000, cv, modelknn, features)