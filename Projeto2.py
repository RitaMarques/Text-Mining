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
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import random
import functools
import warnings
from tqdm import tqdm_notebook as tqdm
nltk.download('stopwords')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# conda install -c conda-forge tqdm
# conda install -c conda-forge ipywidgets

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
# SAMPLING
#----------------------------------------------------------------------------------------------------------------
''' Get a balanced set of samples
We get the least common denominator for the number of texts for each author this will be the number of samples 
per author, we then just have to divide this LCDM by the number of texts of the author to get the number of 
samples to get from each text for each author'''


def get_df_of_samples(df, multiplier, number_of_words, balanced=False):
    """ Receives a dataframe with columns Label and Text
    :param multiplier: number of samples per text or multiplier for balanced samples
    :param number_of_words: the size of each sample
    :param balanced: if the should supply the same number of samples per label (author)
    Returns a dataframe
    """
    uniquecount = pd.value_counts(df.Label)  # get the count of texts for each author

    # We use the greatest common divisor to get the least common denominator:
    def gcd(a, b):
        """Return greatest common divisor using Euclid's Algorithm."""
        while b:      
            a, b = b, a % b
        return a

    def lcm(a, b):
        """Return lowest common multiple."""
        return a * b // gcd(a, b)

    # we do it iteratively for each element in the list
    def lcmm(*args):
        """Return lcm of args."""   
        return functools.reduce(lcm, args)

    denom = lcmm(*uniquecount)

    # and we get the number of samples per author
    samplespertext = uniquecount.to_dict()
    for key in samplespertext:
        samplespertext[key] = int(denom/samplespertext[key])

    # Now we start to build our samples dataframe
    df_samples = pd.DataFrame(columns=['Label', 'Text'])
    
    def get_sample(words, sizeofsample):
        """Get a sample of size: sizeofsample , from a list of words: words
        returns a list of words of desired size.
        Uses the text as circular if:
        start point + sizeofsample > lenght of text given
        """
        start = random.randint(0, len(words))
        if (start + sizeofsample) > len(words):

            finalwords = words[start:]
            return ' '.join(finalwords + words[:(start + sizeofsample - len(words))])
        else:
            return ' '.join(words[start:start + sizeofsample])

        

    def create_samples(datadf, samplesdf, multiplier, sizeofsample, balanced=False):
        for index, row in datadf.iterrows():
            allwords=row[1].split()
            if balanced:
                for i in range(0, (samplespertext[row[0]]*multiplier)):  # balanced number of samples
                    samplesdf = samplesdf.append(pd.Series([row[0], get_sample(allwords, sizeofsample)],
                                                           index=datadf.columns), ignore_index=True)
            else:
                for i in range(0, multiplier):  # number of samples per text
                    samplesdf = samplesdf.append(pd.Series([row[0], get_sample(allwords, sizeofsample)],
                                                         index=datadf.columns), ignore_index=True)
        return samplesdf        

    return create_samples(df, df_samples, multiplier, number_of_words, balanced)


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

def language_model(X_train, max_df=0.9, ngram=(1,3), langmodel="BOW", binary=True):
    if langmodel == "BOW":
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

    elif langmodel =="TFIDF":
        #------------------------
        # TF-IDF
        #------------------------
        cv, X_train_cv = language_model(X_train, max_df, ngram, langmodel="BOW", binary=False)

        tfidf = TfidfTransformer()
        X_train_cv = tfidf.fit_transform(X_train_cv)

        return cv, X_train_cv, tfidf

    else:   #default to BOW
        print("no valid language model, defaulting to BOW")
        return language_model(X_train, max_df, ngram, langmodel="BOW", binary=binary)
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

    X_sparse = sparse.csr_matrix(data_X.values)

    features = True

    return X_sparse, features


#------------------------------------------------------------------------------------------------------------
# MACHINE LEARNING ALGORITHMS
#------------------------------------------------------------------------------------------------------------
class Classifier(object):
    """ Multi Class Classifier base class """

    def __init__(self, input_size, n_classes):
        """
        Initializes a matrix in which each column will be the Weights for a specific class.
        :param input_size: Number of features
        :param n_classes: Number of classes to classify the inputs
        """
        self.parameters = np.zeros((input_size + 1, n_classes))  # input_size +1 to include the Bias term

    def train(self, X, Y, devX, devY, epochs=20):
        """
        This trains the perceptron over a certain number of epoch and records the
            accuracy in Train and Dev sets along each epoch.
        :param X: numpy array with size DxN where D is the number of training examples
                 and N is the number of features.
        :param Y: numpy array with size D containing the correct labels for the training set
        :param devX (optional): same as X but for the dev set.
        :param devY (optional): same as Y but for the dev set.
        :param epochs (optional): number of epochs to run.
        """
        #train_accuracy = [self.evaluate(X, Y)]
        #dev_accuracy = [self.evaluate(devX, devY)]
        for epoch in range(epochs):
            for i in tqdm(range(X.shape[0])):
                self.update_weights(X[i, :].toarray(), Y[i])
            #outs_eval = self.evaluate(X, Y)
            outs_evaldev = self.evaluate(devX, devY)
            #train_accuracy.append(outs_eval[0])
            #dev_accuracy.append(outs_evaldev[0])
        return outs_evaldev[1]  # train_accuracy, dev_accuracy,
            # labels x_val

    def evaluate(self, X, Y):
        """
        Evaluates the error in a given set of examples.
        :param X: numpy array with size DxN where D is the number of examples to
                    evaluate and N is the number of features.
        :param Y: numpy array with size D containing the correct labels for the training set
        """
        correct_predictions = 0
        labels = []
        Y = Y.copy()
        Y.reset_index(inplace=True, drop=True)
        for i in range(X.shape[0]):
            y_pred = self.predict(X[i, :].toarray())
            labels.append(self.predict(X[i, :].toarray()))
            if Y[i] == y_pred:
                correct_predictions += 1
        return correct_predictions / X.shape[0], labels

    def plot_train(self, train_accuracy, dev_accuracy):
        """
        Function to Plot the accuracy of the Training set and Dev set per epoch.
        :param train_accuracy: list containing the accuracies of the train set.
        :param dev_accuracy: list containing the accuracies of the dev set.
        """
        x_axis = [epoch + 1 for epoch in range(len(train_accuracy))]
        plt.plot(x_axis, train_accuracy, '-g', linewidth=1, label='Train')
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        plt.plot(x_axis, dev_accuracy, 'b-', linewidth=1, label='Dev')
        plt.legend()
        plt.show()

class MultinomialLR(Classifier):
    """ Multinomial Logistic Regression """

    def __init__(self, input_size, n_classes, lr=0.001):
        """
        Initializes a matrix in which each column will be the Weights for a specific class.
        :param input_size: Number of features
        :param n_classes: Number of classes to classify the inputs
        """
        Classifier.__init__(self, input_size, n_classes)
        self.lr = lr

    def predict(self, input):
        """
        This function will add a Bias value to the received input, multiply the
            Weights corresponding to the different classes with the input vector, run
            a softmax function and choose the class that achieves an higher probability.
        :param x: numpy array with size 1xN where N = number of features.
        """
        return np.argmax(self.softmax(np.dot(np.append(input, [1]), self.parameters)))

    def softmax(self, x):
        """ Compute softmax values for each sets of scores in x."""
        return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)

    def update_weights(self, x, y):
        """
        Function that will take an input example and the true prediction and will update
            the model parameters.
        :param x: Array of size N where N its the number of features that the model takes as input.
        :param y: The int corresponding to the correct label.
        """
        linear = np.dot(np.append(x, [1]), self.parameters)
        predictions = self.softmax(linear)
        self.parameters = self.parameters - self.lr * (np.outer(predictions, np.append(x, [1])).T)
        self.parameters[:, y] = self.parameters[:, y] + self.lr * np.append(x, [1])


def ml_algorithm(X_train_cv, y_train, model="KNN",neighbors=7):
    if model == "KNN":
        #--------------------------
        # KNN
        #--------------------------
        # Classifying with KNN
        modelknn = KNeighborsClassifier(n_neighbors=neighbors, weights='distance', algorithm='brute',
                                                 metric='cosine')
        modelknn.fit(X_train_cv, y_train)

        return modelknn

    elif model == "MLRP":
        #--------------------------
        # Logistic Regression
        #--------------------------
        # Classifying with LR
        lr = MultinomialLR(X_train_cv.shape[1], len(np.unique(y_train)))

        return lr
    else:
        print("No valid model selected, defaulting to KNN")
        return ml_algorithm(X_train_cv, y_train, model="KNN",neighbors=neighbors)
    #Expandable to more models in the future

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


def predict(df, cv, trymodel,model, x_data, y_data, X_train_cv=None, y_train=None, features=None,
            vectorizer=None, testdata=None):
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

    if trymodel == "MLRP":
        data_predict = model.train(X=X_train_cv, Y=y_train, devX=x_data, devY=y_data, epochs=20)
    elif trymodel=="KNN": 
        data_predict = model.predict(X_cv)
    else:  #default to KNN, expandable to newmodels
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
def run_pipeline(sampled,multiply,words,balanced=True,stopwords=True,stemmer=False,
                    max_df=0.9, ngram=(1,3), langmodel="BOW", binary=True,
                    features=False,
                    trymodel="KNN",neighbors=5):
    df_original = get_dataframe(r'./Corpora/train/')


# ---- SAMPLE AND CLEAN
    if sampled:
        
        df_sampled = get_df_of_samples(df_original, multiplier=3, number_of_words=words, balanced=balanced)
        df_cleaned = clean(df_sampled, stopwords_bol=stopwords, stemmer_bol=stemmer)
    else:
        df_cleaned = clean(df_original, stopwords_bol=stopwords, stemmer_bol=stemmer)


# ---- SPLIT DATA
    X_train, X_val, y_train, y_val = split(df_cleaned)


# ---- CHOOSE LANGUAGE MODEL
    if langmodel=="TFIDF":
         cv, X_train_cv, tfidf = language_model(X_train, max_df, ngram, langmodel, binary)
    elif langmodel=="BOW": #Bag of Words
        cv, X_train_cv = language_model(X_train, max_df, ngram, langmodel, binary)


# ---- ADD EXTRA FEATURES
    if features:
        X_train_cv, features = extra_features(df_cleaned, X_train, cv, X_train_cv)
    else:
        features = None


# ---- TRAIN MODEL & PREDICT
    if trymodel=="KNN":

        in_use_model = ml_algorithm(X_train_cv, y_train, model="KNN",neighbors=5)
        if langmodel=="TFIDF":
            data_predict, report, conf_matrix, scores = predict(df_cleaned, cv,trymodel, in_use_model, X_val, y_val, features, vectorizer="tfidf")

        elif langmodel=="BOW": #Bag of Words
            data_predict, report, conf_matrix = predict(df_cleaned, cv, trymodel,  in_use_model, X_val, y_val, features)
        else:   #default toBag of Words
            data_predict, report, conf_matrix = predict(df_cleaned, cv, trymodel,  in_use_model, X_val, y_val, features)

    elif trymodel=="MLRP":
# If Multinomial Logistic Regression Perceptron
        in_use_model = ml_algorithm(X_train_cv, y_train, model="MLRP")
        data_predict, report, conf_matrix = predict(df_cleaned, cv,trymodel, model=in_use_model, x_data=X_val, y_data=y_val,
                                            X_train_cv=X_train_cv, y_train=y_train, features=features)
    
    print(report)
    return cv, in_use_model, features, X_train_cv

#------------------------------------------------------------------------------------------------------------
# TEST FILES
#------------------------------------------------------------------------------------------------------------
def test(testset, cv, modelknn, features, vectorizer=None):
    """Function that predicts our test data"""
    test_cleaned = clean(testset, stopwords_bol=False, stemmer_bol=False)

    return predict(test_cleaned, cv, modelknn, test_cleaned['Text'], test_cleaned['Label'], features,
                   vectorizer, testdata=True)

##################################################
# Define model conditions - can be changed in each model or set defaults
##################################################

modeltotest="KNN"           #option  "KNN,"MLRP"
langmodeltotest="BOW"       #options "BOW,TFIDF"
sampling=True               #wether to use sampling (T) or original
balancing=True              #balanced (T) or unbalanced sampling
stop_words=True              #wether to remove stopwords (T) or not
stemming=False              #Wether to apply a Stemmer (T) or not
max_df=0.9                  #CountVectorizer ignore terms that appear in more than (0.0-1) 0.0-100% of documents
ngram=(1,3)                 #range of n-grams to be extracted (min,max)
binary_vec=True             #Vectorizer counts or only notes presence (T)



# 500 WORDS
cv, in_use_model, features, X_train_cv= run_pipeline(sampling,multiply=5,words=500, balanced=balancing,     #sampling
               stopwords=stop_words,stemmer=stemming,                                                    #processing
               max_df=max_df, ngram=ngram,langmodel=langmodeltotest, binary=binary_vec,                  #language model     
               trymodel=modeltotest,neighbors=5                                                 #ml model           
                )
df_test_500 = get_dataframe(r'./Corpora/test-IMPORT/500Palavras/')
data500_predict, report500, conf_matrix500 = test(df_test_500, cv, modeltotest, in_use_model, features)

cv, in_use_model, features, X_train_cv= run_pipeline(sampled=sampling,multiply=3,words=1000,balanced=balancing,
                stopwords=stop_words,stemmer=stemming,                                
                max_df=max_df, ngram=ngram,langmodel=langmodeltotest, binary=binary_vec,        
                trymodel=modeltotest, neighbors=5                                              
                )
# 1000 WORDS
df_test_1000 = get_dataframe(r'./Corpora/test-IMPORT/1000Palavras/')
data1000_predict, report1000, conf_matrix1000 = test(df_test_1000, cv, modeltotest, in_use_model, features)


