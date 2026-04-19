import os
from gensim.models import Word2Vec, FastText

def train_w2v(sentences, params, output_path):
    # Навчання Word2Vec моделі [cite: 65, 182]
    print("Тренування Word2Vec...")
    model = Word2Vec(sentences=sentences, **params)
    model.save(output_path)
    return model

def train_ft(sentences, params, output_path):
    # Навчання FastText моделі [cite: 66, 183]
    print("Тренування FastText...")
    model = FastText(sentences=sentences, **params)
    model.save(output_path)
    return model
