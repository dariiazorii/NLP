import spacy

def add_hybrid_rules(nlp):
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.get_pipe("entity_ruler")

    patterns = [
        # ПРАВИЛО 1: Текстові дати (12 жовтня 2022)
        {"label": "DATE", "pattern": [
            {"IS_DIGIT": True},
            {"LOWER": {"IN": ["січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня", "жовтня", "листопада", "грудня"]}},
            {"IS_DIGIT": True, "OP": "?"},
            {"LOWER": {"IN": ["р.", "року"]}, "OP": "?"}
        ]},

        # ПРАВИЛО 2: Гроші (сума + грн)
        # Ловимо будь-який токен, що стоїть перед "грн"
        {"label": "MONEY", "pattern": [
            {"LIKE_NUM": True},
            {"LOWER": "грн"}
        ]},

        # ПРАВИЛО 3: Спрощені ідентифікатори (номери, що починаються з №)
        {"label": "ID", "pattern": [
            {"TEXT": {"PREFIX": "№"}},
            {"IS_ASCII": False, "OP": "?"}
        ]},

        # ПРАВИЛО 4: Організації (БВПД, НСЗУ, ТОВ)
        # Використовуємо LOWER + IN - це працює миттєво і без помилок
        {"label": "ORG", "pattern": [
            {"LOWER": {"IN": ["бвпд", "нсзу", "кму", "тов", "пп", "ат", "пат"]}}
        ]},

        # ПРАВИЛО 5: Коди ДК (проста перевірка структури без regex)
        {"label": "CPV", "pattern": [
            {"LOWER": "код"}, {"LOWER": "дк"}, {"IS_PUNCT": True, "OP": "?"}, {"IS_DIGIT": True}
        ]}
    ]

    ruler.add_patterns(patterns)
    return nlp
