# ============================================
# Urdu Sentiment Analysis - Complete Script
# ============================================

import os
import re
import pandas as pd
import numpy as np
from collections import Counter
import tkinter as tk

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Dropout

from transformers import pipeline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

DATA_FILE = 'urdu-sentiment-corpus-v1.tsv'

# Load dataset and remove neutral labels

def load_dataset(file_path):
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    df.columns = ['text', 'label']
    df.dropna(inplace=True)
    df = df[df['text'].astype(str).str.strip() != '']
    df = df[df['label'].isin(['P', 'N', 'O'])]
    df = df[df['label'] != 'O']
    df.reset_index(drop=True, inplace=True)
    return df


def preprocess(text):
    text = str(text)
    text = re.sub(r"[،۔؟!؟«»“”\"'’\\]", ' ', text)
    text = re.sub(r'[^؀-ۿ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text):
    return text.split()


def build_bigram_counts(tokens_list):
    bigrams = Counter()
    for tokens in tokens_list:
        for i in range(len(tokens) - 1):
            bigrams[(tokens[i], tokens[i + 1])] += 1
    return bigrams


def guess_next_word(word, bigrams):
    candidates = [pair[1] for pair in bigrams if pair[0] == word]
    if not candidates:
        return 'No guess'
    return Counter(candidates).most_common(1)[0][0]


def generate_sentence(start_word, bigrams, length=5):
    sentence = [start_word]
    current = start_word
    for _ in range(length):
        next_word = guess_next_word(current, bigrams)
        if next_word == 'No guess':
            break
        sentence.append(next_word)
        current = next_word
    return ' '.join(sentence)


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\n{name} Accuracy: {accuracy:.4f}')
    print(f'\n{name} Classification Report:')
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    print(f'\n{name} Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    return accuracy


def build_rnn_model(vocab_size, embedding_dim, input_length):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=input_length))
    model.add(SimpleRNN(64))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_lstm_model(vocab_size, embedding_dim, input_length):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=input_length))
    model.add(LSTM(64))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def map_bert_label(result):
    label = result['label'].lower()
    if '1' in label or '2' in label:
        return 'N'
    if '4' in label or '5' in label:
        return 'P'
    return 'N'


def map_xlmr_label(result):
    label = result['label'].lower()
    if 'positive' in label:
        return 'P'
    return 'N'


def predict_nb(sentence, preprocess_fn, vectorizer, model):
    sentence = preprocess_fn(sentence)
    vec = vectorizer.transform([sentence])
    return model.predict(vec)[0]


def predict_lr(sentence, preprocess_fn, tfidf_vectorizer, model):
    sentence = preprocess_fn(sentence)
    vec = tfidf_vectorizer.transform([sentence])
    return model.predict(vec)[0]


def predict_svm(sentence, preprocess_fn, tfidf_vectorizer, model):
    sentence = preprocess_fn(sentence)
    vec = tfidf_vectorizer.transform([sentence])
    return model.predict(vec)[0]


def predict_rnn(sentence, preprocess_fn, tokenizer, max_len, model):
    sentence = preprocess_fn(sentence)
    seq = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = model.predict(padded, verbose=0)
    return 'P' if np.argmax(pred[0]) == 1 else 'N'


def predict_lstm(sentence, preprocess_fn, tokenizer, max_len, model):
    sentence = preprocess_fn(sentence)
    seq = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = model.predict(padded, verbose=0)
    return 'P' if np.argmax(pred[0]) == 1 else 'N'


def predict_bert(sentence, bert_pipe):
    result = bert_pipe(sentence)[0]
    return map_bert_label(result)


def predict_xlmr(sentence, xlmr_pipe):
    result = xlmr_pipe(sentence)[0]
    return map_xlmr_label(result)


def predict_sentiment(sentence, preprocess_fn, vectorizer, model):
    sentence = preprocess_fn(sentence)
    strong_negative = [
        'بہت خراب', 'بالکل خراب', 'خراب', 'ناراض', 'بد', 'نفرت',
        'بدترین', 'اچھا نہیں', 'ناقابلِ برداشت'
    ]
    strong_positive = [
        'بہت اچھا', 'زبردست', 'شاندار', 'خوش', 'اچھا',
        'بہترین', 'مزیدار', 'زبردست ہے'
    ]
    for phrase in strong_negative:
        if phrase in sentence:
            return 'N'
    for phrase in strong_positive:
        if phrase in sentence:
            return 'P'
    vec = vectorizer.transform([sentence])
    return model.predict(vec)[0]


def build_gui(predictors):
    root = tk.Tk()
    root.title('Urdu Sentiment Classifier')
    root.geometry('600x420')

    tk.Label(root, text='Urdu sentence likhen:', font=('Arial', 12)).pack(pady=(12, 4))
    entry = tk.Entry(root, width=64, font=('Arial', 12))
    entry.pack(pady=4)
    entry.focus()

    result_label = tk.Label(root, text='Prediction yahan ayegi.', font=('Arial', 12), fg='blue', justify='left')
    result_label.pack(pady=10)

    def on_predict():
        text = entry.get().strip()
        if not text:
            result_label.config(text='Pehle Urdu sentence likhen.')
            return
        output_lines = []
        for name, func in predictors.items():
            try:
                prediction = func(text)
            except Exception as exc:
                prediction = f'Error: {exc}'
            output_lines.append(f'{name}: {prediction}')
        result_label.config(text='\n'.join(output_lines))

    tk.Button(root, text='Predict Sentiment', command=on_predict, font=('Arial', 11), bg='#4CAF50', fg='white').pack(pady=8)
    root.mainloop()


def main():
    df = load_dataset(DATA_FILE)
    df['clean_text'] = df['text'].apply(preprocess)
    df['tokens'] = df['clean_text'].apply(tokenize)

    print('\nSample Tokens:')
    print(df[['clean_text', 'tokens']].head())

    all_words = [word for tokens in df['tokens'] for word in tokens]
    word_freq = Counter(all_words)
    print('\nTop 10 Frequent Words:')
    print(word_freq.most_common(10))

    bigrams = build_bigram_counts(df['tokens'].tolist())
    print('\nTop 10 Bigrams (Collocations):')
    print(bigrams.most_common(10))

    print('\nWord Guessing Example:')
    print('یہ →', guess_next_word('یہ', bigrams))
    print('\nGenerated Sentence:')
    print(generate_sentence('یہ', bigrams, length=5))

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), ngram_range=(1, 2))
    X_counts = vectorizer.fit_transform(df['clean_text'])
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), ngram_range=(1, 2))
    X_tfidf = tfidf_vectorizer.fit_transform(df['clean_text'])

    print('\nTotal CountVectorizer Features:', len(vectorizer.get_feature_names_out()))
    print('Total TF-IDF Features:', len(tfidf_vectorizer.get_feature_names_out()))

    X_train_counts, X_test_counts, X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
        X_counts, X_tfidf, df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    nb_model = MultinomialNB()
    nb_model.fit(X_train_counts, y_train)
    acc_nb = evaluate_model('Naive Bayes', nb_model, X_test_counts, y_test)

    lr_model = LogisticRegression(max_iter=1000, solver='liblinear')
    lr_model.fit(X_train_tfidf, y_train)
    acc_lr = evaluate_model('Logistic Regression', lr_model, X_test_tfidf, y_test)

    svm_model = LinearSVC(max_iter=10000)
    svm_model.fit(X_train_tfidf, y_train)
    acc_svm = evaluate_model('Linear SVM', svm_model, X_test_tfidf, y_test)

    texts = df['clean_text'].tolist()
    labels = df['label'].map({'N': 0, 'P': 1}).tolist()
    tokenizer_keras = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer_keras.fit_on_texts(texts)
    sequences = tokenizer_keras.texts_to_sequences(texts)
    max_len = 50
    X_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    y_array = np.array(labels)

    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
        X_sequences, y_array, test_size=0.2, random_state=42, stratify=y_array
    )

    vocab_size = min(len(tokenizer_keras.word_index) + 1, 10000)
    embedding_dim = 64

    rnn_model = build_rnn_model(vocab_size, embedding_dim, max_len)
    rnn_model.fit(X_train_seq, y_train_seq, epochs=5, batch_size=32, validation_split=0.1, verbose=0)
    rnn_preds = np.argmax(rnn_model.predict(X_test_seq, verbose=0), axis=1)
    acc_rnn = accuracy_score(y_test_seq, rnn_preds)
    print('\nRNN Accuracy:', acc_rnn)
    print('\nRNN Classification Report:')
    print(classification_report(y_test_seq, rnn_preds, target_names=['Negative', 'Positive']))
    print('\nRNN Confusion Matrix:')
    print(confusion_matrix(y_test_seq, rnn_preds))

    lstm_model = build_lstm_model(vocab_size, embedding_dim, max_len)
    lstm_model.fit(X_train_seq, y_train_seq, epochs=5, batch_size=32, validation_split=0.1, verbose=0)
    lstm_preds = np.argmax(lstm_model.predict(X_test_seq, verbose=0), axis=1)
    acc_lstm = accuracy_score(y_test_seq, lstm_preds)
    print('\nLSTM Accuracy:', acc_lstm)
    print('\nLSTM Classification Report:')
    print(classification_report(y_test_seq, lstm_preds, target_names=['Negative', 'Positive']))
    print('\nLSTM Confusion Matrix:')
    print(confusion_matrix(y_test_seq, lstm_preds))

    raw_train_texts, raw_test_texts, raw_train_labels, raw_test_labels = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    print('\nLoading pretrained sentiment transformers...')
    bert_pipe = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment', tokenizer='nlptown/bert-base-multilingual-uncased-sentiment', device=-1)
    xlmr_pipe = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment', tokenizer='cardiffnlp/twitter-xlm-roberta-base-sentiment', device=-1)

    print('\nSample Transformer Predictions:')
    sample_sentences = [
        'یہ بہت زبردست ہے',
        'یہ بہت خراب ہے',
        'میں خوش ہوں',
        'میں ناراض ہوں',
    ]
    for sentence in sample_sentences:
        print('BERT:', sentence, '→', predict_bert(sentence, bert_pipe))
        print('XLM-RoBERTa:', sentence, '→', predict_xlmr(sentence, xlmr_pipe))

    bert_preds = [predict_bert(text, bert_pipe) for text in raw_test_texts]
    xlmr_preds = [predict_xlmr(text, xlmr_pipe) for text in raw_test_texts]
    acc_bert = accuracy_score(raw_test_labels, bert_preds)
    acc_xlmr = accuracy_score(raw_test_labels, xlmr_preds)
    print('\nBERT Accuracy:', acc_bert)
    print('\nXLM-RoBERTa Accuracy:', acc_xlmr)

    comparison = [
        {'Model Name': 'Naive Bayes', 'Accuracy': f'{acc_nb:.4f}'},
        {'Model Name': 'Logistic Regression', 'Accuracy': f'{acc_lr:.4f}'},
        {'Model Name': 'Linear SVM', 'Accuracy': f'{acc_svm:.4f}'},
        {'Model Name': 'RNN', 'Accuracy': f'{acc_rnn:.4f}'},
        {'Model Name': 'LSTM', 'Accuracy': f'{acc_lstm:.4f}'},
        {'Model Name': 'BERT', 'Accuracy': f'{acc_bert:.4f}'},
        {'Model Name': 'XLM-RoBERTa', 'Accuracy': f'{acc_xlmr:.4f}'},
    ]
    comparison_df = pd.DataFrame(comparison)
    print('\nModel Comparison:')
    print(comparison_df.to_string(index=False))

    predictors = {
        'Naive Bayes': lambda text: predict_nb(text, preprocess, vectorizer, nb_model),
        'Logistic Regression': lambda text: predict_lr(text, preprocess, tfidf_vectorizer, lr_model),
        'SVM': lambda text: predict_svm(text, preprocess, tfidf_vectorizer, svm_model),
        'RNN': lambda text: predict_rnn(text, preprocess, tokenizer_keras, max_len, rnn_model),
        'LSTM': lambda text: predict_lstm(text, preprocess, tokenizer_keras, max_len, lstm_model),
        'BERT': lambda text: predict_bert(text, bert_pipe),
        'XLM-RoBERTa': lambda text: predict_xlmr(text, xlmr_pipe),
    }

    build_gui(predictors)


if __name__ == '__main__':
    main()
