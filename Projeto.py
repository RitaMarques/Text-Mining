import pandas as pd
import os

os.chdir(r'C:\Users\Sofia\OneDrive - NOVAIMS\Nova IMS\Mestrado\2º semestre\
        Text Mining\Projeto\Text-Mining\Corpora\train')

# IMPORT TRAIN FILES


def import_folder_files(directory):
    f = []
    for root, dirs, files in os.walk(directory):
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

for lista in textos_labels:
    df_aux = pd.DataFrame({'Text': lista[0],
                           'Label': lista[1]})

    df = df.append(df_aux, ignore_index=True)
