import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
from collections import Counter,defaultdict
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

# with open("model.pkl","rb") as f:
#     lr = pickle.load(f)

# with open("wordvectorizer.pkl","rb") as f:
#     tfidf = pickle.load(f)
df = pd.read_json("Sarcasm_Headlines_Dataset.json",lines=True)
y = df['is_sarcastic'].values

def review_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem
    
    return words

words_all = [review_to_words(review) for review in tqdm(df['headline'])]

def create_vocab(text,y_all):
    ind = 0
    vocab_all = {}
    doc_freq = defaultdict(set)
    term_freq = Counter()
    n_docs = len(text)
    dtm = defaultdict( lambda: defaultdict(Counter))
    for i,review in enumerate(text):
        if (i+1) % 2000 == 0:
            print(f"Processing review number {i+1}...")
        label = y_all[i]
        for word in review:
            if word not in vocab_all:
                vocab_all[word] = ind
                ind += 1
            doc_freq[word].add(i)
            term_freq[word] += 1
        for w1,w2 in zip(review,review[1:]):
            word = w1+" "+w2
            if word not in vocab_all:
                vocab_all[word] = ind
                ind += 1
            doc_freq[word].add(i)
            term_freq[word] += 1
        for w1,w2,w3 in zip(review,review[1:],review[2:]):
            word = w1+" "+w2+" "+w3
            if word not in vocab_all:
                vocab_all[word] = ind
                ind += 1
            doc_freq[word].add(i)
            term_freq[word] += 1
    valid_vocab = {}
    ind = 0
    print("Processing valid_vocab...")
    for word in vocab_all:
        if len(doc_freq[word]) > n_docs*0.5 or len(doc_freq[word]) < n_docs*0.001 or term_freq[word] < 10:
            continue
        valid_vocab[word] = ind
        ind += 1
    
    for i,review in enumerate(text):
        if (i+1) % 2000 == 0:
            print(f"Processing review number {i+1} for dtm...")
        label = y_all[i]
        for word in review:
            if word not in valid_vocab:
                continue
            dtm[word][label][i] += 1
        for w1,w2 in zip(review,review[1:]):
            word = w1+" "+w2
            if word not in valid_vocab:
                continue
            dtm[word][label][i] += 1
        for w1,w2,w3 in zip(review,review[1:],review[2:]):
            word = w1+" "+w2+" "+w3
            if word not in valid_vocab:
                continue
            dtm[word][label][i] += 1
    
    return valid_vocab,dtm

valid_vocab,dtm = create_vocab(words_all,y)
train = df[['headline']].copy()

n_train = train.shape[0]
words_train = [None]*n_train
for i,review in enumerate(train['headline']):
	if (i+1) % 5000 == 0:
		print(f"Tokenizing review {i+1}...")
	words_train[i] = review_to_words(review) 
#words_train = [review_to_words(review) for review in tqdm(train['review'])]



tfidf = TfidfVectorizer(vocabulary=valid_vocab,ngram_range=(1,3))
X_train = tfidf.fit_transform([" ".join(review) for review in words_train])


lr = LogisticRegression(solver='liblinear',penalty='l2',C=5,random_state=5064)
lr.fit(X_train,y)

y_prob = lr.predict_proba(X_train)[:,1]

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
