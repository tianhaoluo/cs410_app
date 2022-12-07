import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
from collections import Counter,defaultdict
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from os.path import exists as file_exists
import time
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output,dash_table,State  # pip install dash (version 2.0.0 or higher)
import dash
import dash_bootstrap_components as dbc
from dash_extensions import Lottie

with open("model.pkl","rb") as f:
    lr = pickle.load(f)

with open("wordvectorizer.pkl","rb") as f:
    tfidf = pickle.load(f)

def review_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem
    
    return words

def predict(new_review,wordvectorizer,model):
    words_test = [review_to_words(new_review)]
    X_test = wordvectorizer.transform([" ".join(review) for review in words_test])
    y_prob = model.predict_proba(X_test)[:,1]
    return y_prob[0]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

server = app.server

app.layout = dbc.Container([
    dbc.Row([
         dcc.Textarea(
        id='input',
        value='Try inputting a sentence to see if it is sarcastic!',
        style={'width': '100%', 'height': 300},
        )], className="g-0", justify='start'),
    dbc.Button("Submit",id='submit'),
    html.Div(id='output')
      # Horizontal:start,center,end,between,around
    ],fluid=True)


@app.callback(
    [
    Output(component_id="output",component_property='children')
    ],
    [
    Input(component_id="submit",component_property="n_clicks")
    ],
    [State(component_id="input",component_property='value')
    ],
    prevent_initial_call=True,
    )
def press(_,review):
    print(review)
    prob = predict(review,tfidf,lr)
    return ["Sarcastic probability:"+str(round(prob,3))]

if __name__ == '__main__':
    app.run_server(debug=True)