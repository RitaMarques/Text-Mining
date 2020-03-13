import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
from copy import deepcopy
import unicodedata
import re

basedir = r'./Corpora/train/'

#------------------------------------------------------
# IMPORT TRAIN FILES
#------------------------------------------------------
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
    textos_labels

#------------------------------------------------------
# PRE - PROCESSING
#------------------------------------------------------
def normalize(s):
    return unicodedata.normalize('NFD', s)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

def clean(dataframe, stopwords_bol=True, stemmer_bol=True, lemmatizer_bol=False, punctuation_all=False):
    ''' 
    Does lowercase, stopwords
    '''
    df = deepcopy(dataframe)

    # lowercase
    df['Text'] = df['Text'].str.lower()

    # replace ! and ? with token
    df['Text'] = df['Text'].apply(lambda x: re.sub('\?|\!', ' EXPRESSION', x))

    # replace number with token
    df['Text'] = df['Text'].apply(lambda x: re.sub('\d+', 'NUMBER', x))

    # replace \n with a space
    df['Text'] = df['Text'].apply(lambda x: x.replace('\n', ' '))
    #df.iloc[idx, 1] = row[1].replace('\n', ' ')

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

    # remove all punctuation
    if punctuation_all == True:
        df["Text"] = df['Text'].str.replace('[^a-zA-Z]', ' ')

    # removes multiple spaces
    df['Text'] = df['Text'].apply(lambda x: re.sub(r"\s+", " ", x))

    for idx, row in df.iterrows():
        # remove tags
        df.iloc[idx, 1] = BeautifulSoup(row[1]).get_text()

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

df_cleaned = clean(df_original, stemmer_bol=False, lemmatizer_bol=False, punctuation_all=True)

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