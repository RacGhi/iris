import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Cette application prédit le type des **fleurs d'iris**
""")

st.sidebar.header("Entrez les paramétres de l'Iris:")

def user_input_features():
    sepal_length = st.sidebar.slider('Longueur sépale', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Largeur sépale', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Longueur des pétales', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Largeur des pétales', 0.1, 2.5, 0.2)
    data = {'Longueur sépale': sepal_length,
            'Largeur sépale': sepal_width,
            'Longueur des pétales': petal_length,
            'Largeur des pétales': petal_width}
    features = pd.DataFrame(data, index=['Par'])
    return features

df = user_input_features()

st.subheader("Les paramétres choisis par l'utilisateur:")
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)


prediction_proba = pd.DataFrame(clf.predict_proba(df), columns =iris.target_names, index = ['Probabilité'] )



st.subheader("Le type d'Iris prédit: ")
df = pd.DataFrame(iris.target_names[prediction])
df.style.hide_index()
st.write(df.iloc[0])
#st.write(prediction)

st.subheader("Probabilités de prédiction par type d'Iris :")
st.write(prediction_proba)
