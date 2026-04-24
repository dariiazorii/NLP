import spacy
from typing import List, Tuple, Dict, Any

class NerPipeline:
    def __init__(self, model_name: str = "uk_core_news_trf"):
        try:
            self.nlp = spacy.load(model_name)
            print(f"Пайплайн успішно завантажено: {model_name}")
        except OSError:
            raise OSError(f"Модель {model_name} не знайдена.")

    def process_text(self, text: str) -> Dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            return {"text": text, "entities": []}

        doc = self.nlp(text)
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
            for ent in doc.ents
        ]

        return {
            "text": text,
            "entities": entities
        }

    def get_supported_labels(self) -> List[str]:
        return self.nlp.pipe_labels.get("ner", [])
