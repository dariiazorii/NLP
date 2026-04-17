import re
import numpy as np
import pandas as pd

def further_clean(text, custom_stop_words):
    """
    Додатково очищую текст від специфічного тендерного шуму, 
    який я виявила під час аналізу корпусу.
    """
    # Перетворюю на рядок, якщо прийшло число або NaN
    #text = str(text)
    
    # Видаляю номери (наприклад, №123) та просто цифри
    text = re.sub(r'№\s?\d+', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Токенізую та фільтрую за списком стоп-слів та довжиною
    tokens = [word for word in text.lower().split() 
              if len(word) > 2 and word not in custom_stop_words]
    
    return ' '.join(tokens)

def get_top_words(model, vectorizer, n_top_words=10):
    """
    Формую словник тем, де ключем є номер теми, 
    а значенням — список топ-слів.
    """
    words = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        topics[f"Topic {topic_idx}"] = [words[i] for i in top_features_ind]
    return topics

def get_top_documents(model_matrix, df, text_column='clean_text', n_docs=3):
    """
    Знаходжу найбільш релевантні документи для кожної теми 
    для подальшої ручної інтерпретації.
    """
    top_docs = {}
    for i in range(model_matrix.shape[1]):
        # Сортую індекси документів за спаданням ваги теми i
        top_indices = np.argsort(model_matrix[:, i])[::-1][:n_docs]
        # Зберігаю список текстів
        top_docs[i] = df.iloc[top_indices][text_column].values
    return top_docs
