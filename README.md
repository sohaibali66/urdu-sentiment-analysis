# Urdu Sentiment Analysis Using Machine Learning, Deep Learning and Transformer Models

A complete Natural Language Processing project that classifies Urdu text into **Positive (P)** and **Negative (N)** sentiment categories using traditional Machine Learning, Deep Learning, and pretrained Transformer models.

This project also includes text preprocessing, word-frequency analysis, bigram analysis, next-word prediction, model comparison, and a Tkinter desktop GUI for live Urdu sentiment prediction.

---

## Project Highlights

* Urdu text preprocessing using regular expressions
* Binary sentiment classification: Positive and Negative
* Neutral (`O`) records removed before training
* Word-frequency and bigram analysis
* Simple next-word prediction and bigram-based sentence generation
* Count Vectorizer and TF-IDF feature extraction
* Comparison of 7 sentiment-analysis approaches
* Tkinter GUI for testing custom Urdu sentences
* Accuracy, classification report, and confusion matrix evaluation

---

## Models Implemented

### Classical Machine Learning Models

| Model                         | Input Representation                       |
| ----------------------------- | ------------------------------------------ |
| Multinomial Naive Bayes       | Count Vectorizer with unigrams and bigrams |
| Logistic Regression           | TF-IDF with unigrams and bigrams           |
| Linear Support Vector Machine | TF-IDF with unigrams and bigrams           |

### Deep Learning Models

| Model      | Input Representation                |
| ---------- | ----------------------------------- |
| Simple RNN | Tokenized and padded word sequences |
| LSTM       | Tokenized and padded word sequences |

### Pretrained Transformer Models

| Model             | Usage in This Project                                |
| ----------------- | ---------------------------------------------------- |
| Multilingual BERT | Pretrained multilingual sentiment classifier         |
| XLM-RoBERTa       | Pretrained multilingual Twitter sentiment classifier |

> **Important Note:** BERT and XLM-RoBERTa are used as pretrained sentiment pipelines. They are not fine-tuned on this Urdu sentiment dataset.

---

## Complete System Workflow

```text
Urdu Sentiment Dataset (.tsv)
        ↓
Dataset Loading and Validation
        ↓
Remove Missing and Empty Records
        ↓
Keep P / N / O Labels
        ↓
Remove Neutral (O) Class
        ↓
Urdu Text Preprocessing
        ↓
Tokenization
        ↓
Word Frequency and Bigram Analysis
        ↓
Feature Extraction
• Count Vectorizer
• TF-IDF Vectorizer
        ↓
80% Training Data / 20% Testing Data
        ↓
Train Classical ML Models
• Naive Bayes
• Logistic Regression
• Linear SVM
        ↓
Prepare Word Sequences for Deep Learning
        ↓
Train Deep Learning Models
• RNN
• LSTM
        ↓
Load Pretrained Transformer Models
• BERT
• XLM-RoBERTa
        ↓
Evaluate All Models
• Accuracy
• Classification Report
• Confusion Matrix
        ↓
Generate Model Comparison Table
        ↓
Launch Tkinter GUI for Live Prediction
```

---

## Dataset

The project uses the file:

```text
urdu-sentiment-corpus-v1.tsv
```

The original dataset contains three sentiment labels:

| Label | Meaning                       |
| ----- | ----------------------------- |
| `P`   | Positive sentiment            |
| `N`   | Negative sentiment            |
| `O`   | Objective / Neutral sentiment |

For this project, the neutral class (`O`) is removed so that the system performs binary classification between Positive and Negative sentiment.

---

## Text Preprocessing

The preprocessing stage performs the following operations:

* Converts text into string format
* Removes punctuation and quotation marks
* Removes non-Urdu characters
* Removes unnecessary symbols and digits
* Normalizes repeated spaces
* Removes leading and trailing spaces
* Splits cleaned Urdu text into tokens

This creates a cleaner and more consistent text representation before feature extraction and model training.

---

## Linguistic Analysis Features

In addition to sentiment classification, the project performs basic NLP analysis:

* Top frequent Urdu words
* Bigram / collocation extraction
* Next-word guessing using bigram frequency
* Simple bigram-based sentence generation

These features help demonstrate fundamental Natural Language Processing concepts alongside the sentiment-analysis pipeline.

---

## Evaluation Metrics

Each model is evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Confusion Matrix**

The project prints detailed evaluation results in the terminal before opening the graphical user interface.

---

## Actual Model Results

The following results were generated from the current project execution:

| Model               | Accuracy |
| ------------------- | -------- |
| Naive Bayes         | 60.20%   |
| Logistic Regression | 59.18%   |
| Linear SVM          | 56.63%   |
| RNN                 | 47.96%   |
| LSTM                | 48.98%   |
| Multilingual BERT   | 59.69%   |
| XLM-RoBERTa         | 60.20%   |

### Result Summary

* **Naive Bayes** and **XLM-RoBERTa** achieved the highest accuracy in the current run.
* Classical Machine Learning models performed better than the RNN and LSTM models in this experiment.
* RNN and LSTM performance may vary because neural networks depend on random initialization, training epochs, dataset size, and hyperparameter settings.
* Transformer models are pretrained external models, so their binary label mapping should be interpreted carefully.

---

## Project Structure

```text
urdu-sentiment-analysis/
│
├── main.py
├── requirements.txt
├── urdu-sentiment-corpus-v1.tsv
├── README.md
├── .gitignore
│
├── docs/
│   └── urdu-sentiment-analysis-project-report.pdf
│
└── screenshots/
    ├── 01-dataset-and-feature-extraction.png
    ├── 02-classical-machine-learning-results.png
    ├── 03-deep-learning-results.png
    ├── 04-pretrained-transformer-results.png
    └── 05-final-model-comparison.png
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Hugging Face Transformers
* PyTorch
* Tkinter
* Natural Language Processing
* Machine Learning
* Deep Learning

---

## Installation

Clone or download this repository, then install the required Python libraries:

```bash
pip install -r requirements.txt
```

---

## How to Run

1. Make sure the dataset file is in the same folder as `main.py`.

```text
urdu-sentiment-corpus-v1.tsv
```

2. Install all required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the main Python file:

```bash
python main.py
```

4. The terminal will display:

* Sample tokens
* Frequent Urdu words
* Bigram analysis
* Accuracy of all models
* Classification reports
* Confusion matrices
* Final comparison table

5. After training and evaluation, the Tkinter GUI will open.

6. Enter an Urdu sentence and click **Predict Sentiment**.

7. The GUI will display predictions from:

* Naive Bayes
* Logistic Regression
* SVM
* RNN
* LSTM
* BERT
* XLM-RoBERTa

---

## Screenshots and Results

### Dataset and Feature Extraction

### Classical Machine Learning Results

### Deep Learning Results

### Pretrained Transformer Results

### Final Model Comparison

---

## Limitations

* Neutral sentiment is removed, so the current system does not predict a separate Neutral class.
* RNN and LSTM are trained for only five epochs.
* The dataset is relatively limited for deep-learning training.
* The transformer models are not fine-tuned on this exact Urdu dataset.
* The program retrains local models every time it starts.
* The GUI displays `P` and `N` labels instead of full confidence scores.

---

## Future Improvements

* Add three-class classification: Positive, Negative, and Neutral
* Apply Urdu stop-word removal and text normalization
* Add Urdu stemming or lemmatization
* Fine-tune multilingual BERT or XLM-RoBERTa on the dataset
* Save trained models using `joblib` and Keras model saving
* Add confidence scores and majority voting in the GUI
* Add graphical charts for model comparison
* Increase the dataset size with product reviews, educational feedback, restaurant reviews, and e-commerce comments

---

## Documentation

A detailed explanation of the project workflow, model architecture, evaluation process, GUI, limitations, and future work is available here:

[View Project Report](docs/urdu-sentiment-analysis-project-report.pdf)

---

## Author

**Sohaib Ali**
BS Artificial Intelligence Student
Email: [zebi65871@gmail.com](mailto:zebi65871@gmail.com)
