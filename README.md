# 🧠 Toxic Tweet Classification

This project performs toxic tweet classification using NLP techniques like TF-IDF vectorization and a Linear Support Vector Classifier (LinearSVC). The goal is to classify tweets as toxic or non-toxic.

---

## 📁 Files

- `Toxic_Tweet_Classification_LinearSVC.ipynb`: Jupyter notebook with step-by-step explanation and implementation.
- `Toxic_Tweet_Classification_LinearSVC.py`: Python script version of the notebook for quick execution.
- `submission_tuned.csv`: The output file containing predicted labels for the test dataset.
- `train_2kmZucJ.csv`, `test_oJQbWVk.csv`, `sample_submission_LnhVWA4.csv`: Input CSV files used for training and prediction.

---

## 🚀 Features

- Text cleaning with lemmatization and stopword removal.
- Combined word-level and character-level TF-IDF vectorization.
- Model training using `LinearSVC` with class balancing.
- Evaluation using cross-validation and F1-score.
- Final predictions saved in a submission CSV file.

---

## 🛠️ Requirements

Install the required Python libraries using:

```bash
pip install pandas scikit-learn nltk
```

Also run this once to download NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

---

## 📈 Usage

1. Place the dataset CSV files in the project folder.
2. Run the Jupyter notebook or Python script.
3. The model will output cross-validation scores, training F1 score, and generate a `submission_tuned.csv` file with predictions.

---

## 📊 Output

- **Cross-validation F1 Score** is printed during training.
- **Training F1 Score** helps to estimate model fit.
- **`submission_tuned.csv`** contains final test predictions in the format required for competition or evaluation.

---

## 📌 Note

This is a baseline TF-IDF + LinearSVC model. You can extend it by:
- Using BERT for contextual embeddings
- Applying data augmentation techniques
- Trying other classifiers or ensemble models

---
