import torch
import os
from bleurt_pytorch import (
    BleurtConfig,
    BleurtForSequenceClassification,
    BleurtTokenizer,
)


class Evaluator:
    def __init__(self, model_name="lucadiliello/BLEURT-20-D12"):
        self.config = BleurtConfig.from_pretrained(model_name)
        self.model = BleurtForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = BleurtTokenizer.from_pretrained(model_name)
        self.model.eval()

    def evaluate(self, references: list[str], candidates: list[str]) -> list[float]:
        assert isinstance(references, list) and isinstance(candidates, list)
        assert len(references) == len(candidates)

        with torch.no_grad():
            inputs = self.tokenizer(
                references, candidates, padding="longest", return_tensors="pt"
            )
            res = self.model(**inputs).logits.flatten().tolist()

        return res


if __name__ == "__main__":
    references = ["This is a test."]
    candidates = ["This is the test."]

    evaluator = Evaluator()
    scores = evaluator.evaluate(references, candidates)
    print(scores)
