import pandas as pd

def get_comparison_table(models_dict, words):
    """
    Створює таблицю порівняння найближчих сусідів для списку слів.
    """
    results = []
    for word in words:
        row = {"Word": word}
        for name, model in models_dict.items():
            try:
                # Отримуємо 5 найближчих сусідів
                neighbors = model.wv.most_similar(word, topn=5)
                neighbors_str = ", ".join([f"{n[0]} ({n[1]:.2f})" for n in neighbors])
                row[name] = neighbors_str
            except KeyError:
                row[name] = "Not in vocabulary"
        results.append(row)
    
    return pd.DataFrame(results)
