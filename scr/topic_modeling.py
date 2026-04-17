import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from topic_utils import further_clean, get_top_words

# Повний список стоп-слів, який я визначила як оптимальний для Prozorro
CUSTOM_STOP_WORDS = [
   'згідно', 'пункт', 'додаток', 'номер', 'стаття', 'закон', 'україна', 
    'постанова', 'кабінет', 'міністр', 'особливість', 'закупівля', 'товар',
    'послуга', 'робота', 'код', 'дк', '021', '2015', 'очікувати', 'вартість',
    'предмет', 'договір', 'умова', 'період', 'день', 'місяць', 'рік', 'грн',
    'відповідно', 'затвердженого', 'здійснення', 'публічних', 'замовник',
    'надання', 'україни', 'обслуговування', 'послуг', 'замовника', 'року',
    'про', 'вул', 'мовою', 'енергії', 'тендерної'
]

def run_topic_modeling_pipeline(data_path='processed_v2.csv'):
    """
    Запускаю повний процес: завантаження, векторизація та моделювання.
    """
    print("--- Починаю роботу Pipeline (Topic Modeling) ---")
    
    # 1. Завантаження даних
    df = pd.read_csv(data_path)
    
    # 2. Очищення та препроцесинг
    print("Очищую корпус...")
    df['refined_text'] = df['clean_text'].apply(lambda x: further_clean(x, CUSTOM_STOP_WORDS))
    # Видаляю надто короткі тексти (менше 3 слів)
    df = df[df['refined_text'].apply(lambda x: len(x.split()) >= 3)]
    
    # 3. Векторизація для LSA (TF-IDF з біграмами)
    print("Векторизація TF-IDF (для LSA)...")
    tfidf_vec = TfidfVectorizer(max_df=0.85, min_df=5, analyzer='word', ngram_range=(1, 2))
    tfidf_matrix = tfidf_vec.fit_transform(df['refined_text'])
    
    # 4. Векторизація для LDA (Count)
    print("Векторизація Count (для LDA)...")
    count_vec = CountVectorizer(max_df=0.85, min_df=5, analyzer='word')
    count_matrix = count_vec.fit_transform(df['refined_text'])
    
    # 5. Побудова моделі LSA (k=10)
    print("Навчання LSA (k=10)...")
    lsa_model = TruncatedSVD(n_components=10, random_state=42)
    lsa_model.fit(tfidf_matrix)
    
    # 6. Побудова моделі LDA (k=10)
    print("Навчання LDA (k=10)...")
    lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
    lda_model.fit(count_matrix)
    
    # 7. Вивід результатів LSA для перевірки
    print("\n[РЕЗУЛЬТАТИ LSA TOP WORDS]")
    lsa_topics = get_top_words(lsa_model, tfidf_vec)
    for topic, words in lsa_topics.items():
        print(f"{topic}: {', '.join(words)}")
        
    print("\n--- Pipeline завершено успішно ---")
    return lsa_model, lda_model, tfidf_vec, count_vec, df

if __name__ == "__main__":
    run_topic_modeling_pipeline()
