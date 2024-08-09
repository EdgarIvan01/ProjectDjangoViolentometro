import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
import re
import stanza
from sklearn.model_selection import train_test_split
from unidecode import unidecode


def uso(texto):
    regex_puntuacion = re.compile('[^\w\s]')
    df = pd.read_csv("violentometro2.csv")
    
    df2 = pd.read_csv("Guardado.csv")
    df2 = df2.dropna()
    
    df = df.dropna()
    nltk.download('stopwords')
    nltk.download('wordnet')
    from nltk.corpus import stopwords
    stanza.download('es')
    def limpiar_texto(texto):
        palabras_vacias = set(stopwords.words('spanish'))
        palabras = texto.split()
        palabras_sin_vacias = [palabra for palabra in palabras if palabra not in palabras_vacias]
        texto_sin_vacias = ' '.join(palabras_sin_vacias)
        return texto_sin_vacias
    nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')

    def lematizar(texto):
        doc = nlp(texto)
        palabras_lematizadas = [word.lemma for sent in doc.sentences for word in sent.words]
        texto_lematizado = ' '.join(palabras_lematizadas)
        return texto_lematizado
    

    X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Etiqueta'], test_size=0.3, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    nuevo_texto = texto
    # quitar acentos y simbolos de puntuacion
    nuevo_texto1 = unidecode(regex_puntuacion.sub('', nuevo_texto.lower()))
    # Texto en minusculas
    nuevo_texto2 = nuevo_texto1.lower()
    # Limpiar stopwords
    nuevo_texto3 = limpiar_texto(nuevo_texto2)
    # Lematizar texto
    nuevo_texto4 = lematizar(nuevo_texto3)
    nuevo_textap=["1"]
    nuevo_textap[0]=nuevo_texto4
    nuevo_texto_vec = vectorizer.transform(nuevo_textap)
    y_pred = clf.predict(nuevo_texto_vec)
    print(y_pred)

    tex = np.array2string(y_pred)
    # Quitar corchetes y comillas
    tex = tex.strip("[]")
    tex = tex.replace("'", "")
    df2 = pd.concat([df2, pd.DataFrame({'Text': [texto], 'Etiqueta': [tex]})], ignore_index=True)
    df2.to_csv('Guardado.csv', index=False)

    return y_pred
#main("hola")