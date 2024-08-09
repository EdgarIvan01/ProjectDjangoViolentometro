import pandas as pd
columnas = {
    'Text': [],
    'Etiqueta': []
}
df = pd.DataFrame(columnas)
print(df)
df.to_csv('C:/Users/arman/OneDrive/Escritorio/Project/Guardado.csv', index=False)
