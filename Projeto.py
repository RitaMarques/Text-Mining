import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
import re

#os.chdir(r'./Corpora/train')
basedir=r'./Corpora/train/'

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

df = pd.DataFrame(columns=['Label','Text'])

df = df[0:0]
for lista in textos_labels:
    df_aux = pd.DataFrame({'Label': lista[1],
                            'Text': lista[0]
                            })

    df = df.append(df_aux, ignore_index=True)

#------------------------------------------------------
# PRE - PROCESSING
#------------------------------------------------------
def clean(df, stopwords_bol=True, stemmer_bol=True, lemmatizer_bol=False, punctuation_all=False):
    ''' 
    Does lowercase, stopwords
    '''
    # lowercase
    df['Text'] = df['Text'].str.lower()

    # remove all punctuation
    if punctuation_all == True:
        df["Text"] = df['Text'].str.replace('[^a-zA-Z]', ' ')

    for idx, row in df.iterrows():
        # remove tags
        df.iloc[idx, 1] = BeautifulSoup(row[1]).get_text()

        # replace \n with a space
        df.iloc[idx, 1] = row[1].replace('\n', ' ')

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


    # split sentences in words
    #df['Text'] = df.Text.str.split(' ')

    return df


## TESTE stem
print(df.Text[0])
snowball_stemmer = SnowballStemmer('portuguese')
' '.join(snowball_stemmer.stem(word) for word in df.Text[0].split())

## TESTE lema
print(df.Text[0])
lemma = WordNetLemmatizer()
' '.join(lemma.lemmatize(word) for word in df.Text[0].split())


df_cleaned = clean(df, stopwords_bol=True, stemmer_bol=True, lemmatizer_bol=False, punctuation_all=True)