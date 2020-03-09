import pandas as pd
import os
import nltk

os.chdir(r'./Corpora/train')

# IMPORT TRAIN FILES


def import_folder_files(directory):
    f = []
    for name, lista,files in os.walk(directory):

        for file in files:

            if file.endswith(".txt"):
                f1 = open(directory + '\\' + file, "r", encoding='utf-8')
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

df = pd.DataFrame(columns=['Text','Label'])

df = df[0:0]
for lista in textos_labels:
    df_aux = pd.DataFrame({'Label': lista[1],
                            'Text': lista[0]
                            })

    df = df.append(df_aux, ignore_index=True)

# lowercase
df['Text'] = df['Text'].str.lower()

# stopwords
stopwords = nltk.corpus.stopwords.words('portuguese')
print(len(stopwords))

nltk.FreqDist()