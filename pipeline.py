from typing import Dict
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class TextClassificationPipeline:
    def __init__(self, model_name: str):
        """
        Initialize the pipeline with a pre-trained model and tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def __call__(self, inputs: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Perform inference on the input text.
        """
        text = inputs.get("text")
        if not text:
            raise ValueError("Input dictionary must contain a 'text' key.")

        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return {"probabilities": probabilities.numpy()}