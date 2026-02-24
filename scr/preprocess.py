import re

def replace_homoglyphs(text: str) -> str:
    homoglyphs = {
        'a': 'а', 'c': 'с', 'e': 'е', 'i': 'і', 'j': 'й', 'o': 'о', 'p': 'р', 's': 'с', 'x': 'х', 'y': 'у',
        'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н', 'I': 'І', 'K': 'К', 'M': 'М', 'O': 'О', 'P': 'Р',
        'T': 'Т', 'X': 'Х'
    }
    # Захищаємо теги <EMAIL>, <PHONE>, <URL> від заміни гомогліфів
    parts = re.split(r'(<(?:EMAIL|PHONE|URL)>)', text)
    processed_parts = []
    for part in parts:
        if re.match(r'<(?:EMAIL|PHONE|URL)>', part):
            processed_parts.append(part) # Не чіпаємо тег
        else:
            table = str.maketrans(homoglyphs)
            processed_parts.append(part.translate(table))
    return "".join(processed_parts)

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r'<(?!(?:EMAIL|PHONE|URL)>)[^>]*>', ' ', text) # Видаляємо HTML, крім наших тегів
    text = re.sub(r'[\n\r\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_text(text: str) -> str:
    text = replace_homoglyphs(text)
    text = text.lower()
    text = re.sub(r"['’`‘]", "'", text)
    text = re.sub(r'[–—−]', '-', text)
    text = re.sub(r'[«»„“"”″‟]', '"', text)
    text = re.sub(r"''", '"', text)
    return text

def mask_pii(text: str) -> str:
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '<EMAIL>', text)
    text = re.sub(r'\+?\d{10,12}', '<PHONE>', text)
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    return text

def is_garbage(text: str) -> bool:
    clean_from_tags = re.sub(r'<(EMAIL|PHONE|URL)>', '', text).strip()
    if not clean_from_tags: return True
    has_letters = bool(re.search(r'[a-zA-Zа-яА-ЯіїєґІЇЄҐ]', clean_from_tags))
    if not has_letters: return True
    only_digits = re.sub(r'[^0-9]', '', clean_from_tags)
    if only_digits and not has_letters: return True
    return False

def sentence_split(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip(): return []
    abbr_pattern = r"(?<!" + r")(?<!".join(["м", "вул", "р", "стор", "обл", "грн", "ст", "д", "т"]) + r")"
    split_pattern = abbr_pattern + r"(?<!\d)\.(?!\d)|(?<=[!?])"
    sentences = re.split(split_pattern, text)
    return [s.strip() for s in sentences if s.strip()]

def preprocess(text: str) -> dict:
    cleaned = clean_text(text)
    normalized = normalize_text(cleaned)
    masked = mask_pii(normalized)
    
    word_count = len(masked.split())
    garbage = is_garbage(masked)
    is_valid = (not garbage) and (word_count >= 5)
    sentences = sentence_split(masked) if is_valid else []
    
    return {
        "clean_text": masked,
        "sentences": sentences,
        "word_count": word_count,
        "sentence_count": len(sentences),
        "is_garbage": garbage,
        "is_valid": is_valid
    }