"""Local TF-IDF and Logistic Regression solution prediction."""
from __future__ import annotations

import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "training_tickets.json"

class Predictor:
    """Train a local TF-IDF and Logistic Regression classifier."""

    def __init__(self, dataset=DATASET):
        try:
            examples = json.loads(Path(dataset).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            examples = []

        self.examples = [
            example for example in examples
            if isinstance(example, dict)
            and isinstance(example.get("subject"), str)
            and isinstance(example.get("description"), str)
            and isinstance(example.get("solution"), str)
            and example["solution"].strip()
        ]

        self.vectorizer = TfidfVectorizer()
        self.classifier = None
        self.training_warning = None

        labels = [
            example["solution"].strip()
            for example in self.examples
        ]
        texts = [
            f"{example['subject']} {example['description']}".strip()
            for example in self.examples
        ]

        if len(texts) < 2 or len(set(labels)) < 2:
            self.training_warning = (
                "At least two examples with two distinct solutions "
                "are required to train the classifier."
            )
            return

        try:
            features = self.vectorizer.fit_transform(texts)
            self.classifier = LogisticRegression(max_iter=1000)
            self.classifier.fit(features, labels)
        except ValueError as error:
            self.classifier = None
            self.training_warning = (
                f"Training data is unsuitable: {error}"
            )

        if self.classifier and len(set(labels)) == len(labels):
            self.training_warning = (
                "The dataset has only one example per solution, so "
                "generalization accuracy cannot be established."
            )

    def predict(self, text):
        """Return a predicted solution or a safe generic fallback."""
        fallback = (
            "Review the issue details and follow the relevant "
            "support procedure."
        )

        if (
            not isinstance(text, str)
            or not text.strip()
            or not self.classifier
        ):
            return fallback

        features = self.vectorizer.transform([text])

        if not features.nnz:
            return fallback

        return self.classifier.predict(features)[0]

