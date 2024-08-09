import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
import re
import stanza
from sklearn.model_selection import train_test_split
from unidecode import unidecode
import torch

def main():
    df = pd.read_csv("violentometro2.csv")
    df = df.dropna()
    df2 = pd.read_csv("Guardado.csv")
    df2 = df2.dropna()
    df = pd.concat([df, df2], axis=0)
    regex_puntuacion = re.compile('[^\w\s]')
    df['Text'] = df['Text'].apply(lambda x: regex_puntuacion.sub('', x))
    df['Text'] = df['Text'].apply(lambda x: unidecode(regex_puntuacion.sub('', x)))
    df['Text'] = df['Text'].apply(lambda x: x.lower())
    nltk.download('stopwords')
    nltk.download('wordnet')
    from nltk.corpus import stopwords
    def limpiar_texto(texto):
        palabras_vacias = set(stopwords.words('spanish'))
        palabras = texto.split()
        palabras_sin_vacias = [palabra for palabra in palabras if palabra not in palabras_vacias]
        texto_sin_vacias = ' '.join(palabras_sin_vacias)
        return texto_sin_vacias
    df['Text'] = df['Text'].apply(limpiar_texto)
    stanza.download('es')
    nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')
    def lematizar(texto):
        doc = nlp(texto)
        palabras_lematizadas = [word.lemma for sent in doc.sentences for word in sent.words]
        texto_lematizado = ' '.join(palabras_lematizadas)
        return texto_lematizado
    import time
    # Configurar el uso de GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma', use_gpu=torch.cuda.is_available())

    def lematizar(texto):
        doc = nlp(texto)
        palabras_lematizadas = [word.lemma for sent in doc.sentences for word in sent.words]
        texto_lematizado = ' '.join(palabras_lematizadas)
        return texto_lematizado
    start_time = time.time()

    df['Text'] = df['Text'].apply(lematizar)
    end_time = time.time()
    execution_time = end_time - start_time

    df.to_csv("violentometro2.csv")
    columnas = {
        'Text': [],
        'Etiqueta': []
    }
    df2 = pd.DataFrame(columnas)
    df2.to_csv('Guardado.csv', index=False)
    return execution_time
