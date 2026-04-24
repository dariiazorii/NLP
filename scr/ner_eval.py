import pandas as pd

class NerEvaluator:
    def __init__(self):
        self.stats = {}

    def _get_stat_template(self):
        return {"tp": 0, "fp": 0, "fn": 0}

    def evaluate(self, evaluation_set, pipeline):
        for item in evaluation_set:
            # Золотий стандарт
            gold_entities = set(tuple(e) for e in item["expected_entities"])
            
            # Передбачення моделі
            prediction = pipeline.process_text(item["text"])
            pred_entities = set((ent["text"], ent["label"]) for ent in prediction["entities"])

            # Збираємо всі унікальні типи сутностей
            gold_types = {e[1] for e in gold_entities}
            pred_types = {e[1] for e in pred_entities}
            all_types = gold_types | pred_types
            
            for t in all_types:
                if t not in self.stats:
                    self.stats[t] = self._get_stat_template()

            # True Positives: збіг тексту та мітки
            tp_set = gold_entities.intersection(pred_entities)
            for _, t in tp_set:
                self.stats[t]["tp"] += 1

            # False Positives: є в передбаченні, але немає в Gold
            fp_set = pred_entities - gold_entities
            for _, t in fp_set:
                self.stats[t]["fp"] += 1

            # False Negatives: пропущено моделлю
            fn_set = gold_entities - pred_entities
            for _, t in fn_set:
                self.stats[t]["fn"] += 1

    def get_report(self):
        report_data = []
        for label, s in self.stats.items():
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            report_data.append({
                "Entity Type": label,
                "Correct (TP)": tp,
                "Missed (FN)": fn,
                "False Pos (FP)": fp,
                "Precision": round(precision, 2),
                "Recall": round(recall, 2),
                "F1-Score": round(f1, 2)
            })
        
        return pd.DataFrame(report_data).sort_values(by="F1-Score", ascending=False)