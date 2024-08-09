import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
import re
import stanza
from sklearn.model_selection import train_test_split
from unidecode import unidecode
import threading
import time

def main(numero):
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

    tiempo_final = time.time()
    if numero == 2:
        # Lematizar hilos
        n = len(df['Text'])
        grupo1 = df['Text'].iloc[:n//2]
        grupo2 = df['Text'].iloc[n//2:]


        # Crea una lista para almacenar los resultados
        resultados = [None] * n

        # Función que lematiza un grupo de textos en un hilo separado
        def lematizar_grupo(textos, inicio, fin):
            for i, texto in enumerate(textos):
                resultados[i + inicio] = lematizar(texto)

        # Crea los hilos y asigna los grupos de textos a cada uno
        hilo1 = threading.Thread(target=lematizar_grupo, args=(grupo1, 0, n//2))
        hilo2 = threading.Thread(target=lematizar_grupo, args=(grupo2, n//2, n))


        # Inicia la ejecución de los hilos
        start_time = time.time()
        
        hilo1.start()
        hilo2.start()

        # Espera a que ambos hilos finalicen
        hilo1.join()
        hilo2.join()

        end_time = time.time()
        execution_time = end_time - start_time
        tiempo_final = execution_time
        print("Tiempo de ejecución:", execution_time, "segundos")
    elif numero == 4:
        # Lematizar hilos    
        n = len(df['Text'])
        grupo1 = df['Text'].iloc[:n//4]
        grupo2 = df['Text'].iloc[n//4:2*n//4]
        grupo3 = df['Text'].iloc[2*n//4:3*n//4]
        grupo4 = df['Text'].iloc[3*n//4:]

        # Crea una lista para almacenar los resultados
        resultados = [None] * n

        # Función que lematiza un grupo de textos en un hilo separado
        def lematizar_grupo(textos, inicio, fin):
            for i, texto in enumerate(textos):
                resultados[i + inicio] = lematizar(texto)

        # Crea los hilos y asigna los grupos de textos a cada uno
        hilo1 = threading.Thread(target=lematizar_grupo, args=(grupo1, 0, n//4))
        hilo2 = threading.Thread(target=lematizar_grupo, args=(grupo2, n//4, 2*n//4))
        hilo3 = threading.Thread(target=lematizar_grupo, args=(grupo3, 2*n//4, 3*n//4))
        hilo4 = threading.Thread(target=lematizar_grupo, args=(grupo4, 3*n//4, n))

        # Inicia la ejecución de los hilos
        start_time = time.time()
        
        hilo1.start()
        hilo2.start()
        hilo3.start()
        hilo4.start()

        # Espera a que ambos hilos finalicen
        hilo1.join()
        hilo2.join()
        hilo3.join()
        hilo4.join()

        end_time = time.time()
        execution_time = end_time - start_time
        tiempo_final = execution_time
        print("Tiempo de ejecución:", execution_time, "segundos")
    elif numero == 6:
        # Lematizar hilos
        n = len(df['Text'])
        grupo1 = df['Text'].iloc[:n//6]
        grupo2 = df['Text'].iloc[n//6:2*n//6]
        grupo3 = df['Text'].iloc[2*n//6:3*n//6]
        grupo4 = df['Text'].iloc[3*n//6:4*n//6]
        grupo5 = df['Text'].iloc[4*n//6:5*n//6]
        grupo6 = df['Text'].iloc[5*n//6:]

        # Crea una lista para almacenar los resultados
        resultados = [None] * n

        # Función que lematiza un grupo de textos en un hilo separado
        def lematizar_grupo(textos, inicio, fin):
            for i, texto in enumerate(textos):
                resultados[i + inicio] = lematizar(texto)

        # Crea los hilos y asigna los grupos de textos a cada uno
        hilo1 = threading.Thread(target=lematizar_grupo, args=(grupo1, 0, n//6))
        hilo2 = threading.Thread(target=lematizar_grupo, args=(grupo2, n//6, 2*n//6))
        hilo3 = threading.Thread(target=lematizar_grupo, args=(grupo3, 2*n//6, 3*n//6))
        hilo4 = threading.Thread(target=lematizar_grupo, args=(grupo4, 3*n//6, 4*n//6))
        hilo5 = threading.Thread(target=lematizar_grupo, args=(grupo5, 4*n//6, 5*n//6))
        hilo6 = threading.Thread(target=lematizar_grupo, args=(grupo6, 5*n//6, n))

        # Inicia la ejecución de los hilos
        start_time = time.time()
        
        hilo1.start()
        hilo2.start()
        hilo3.start()
        hilo4.start()
        hilo5.start()
        hilo6.start()


        # Espera a que ambos hilos finalicen
        hilo1.join()
        hilo2.join()
        hilo3.join()
        hilo4.join()
        hilo5.join()
        hilo6.join()

        end_time = time.time()
        execution_time = end_time - start_time
        tiempo_final = execution_time
        print("Tiempo de ejecución:", execution_time, "segundos")

    df['Text'] = resultados

    df.to_csv('violentometro2.csv', index=False)
    columnas = {
        'Text': [],
        'Etiqueta': []
    }
    df2 = pd.DataFrame(columnas)
    df2.to_csv('Guardado.csv', index=False)
    return tiempo_final

    