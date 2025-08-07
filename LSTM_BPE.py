# source tf-venv/bin/activate
import re
import pandas as pd
import numpy as np
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

import nltk
from nltk.corpus import stopwords

from matplotlib import pyplot as plt
import seaborn as sns

from tokenizers import Tokenizer, models, trainers, pre_tokenizers 

# Load data
# path = './FinancialPhraseBank-v1.0/Sentences_AllAgree.txt'
path = './FinancialPhraseBank-v1.0/Sentences_50Agree.txt'

# file doesn't use UTF-8 encoding, but rather Latin-1 (ISO-8869-1)
with open(path, 'r', encoding='ISO-8859-1') as file:
    data = []
    for line in file:
        if '@' in line:
            text, label = line.rsplit('@', 1)
            data.append((text.strip(), label.strip()))
            
            # data.append((text, label))
    print(len(data))

df = pd.DataFrame(data, columns=['text', 'label'])

# Map string labels to numbers
label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
df['label'] = df['label'].map(label_map)

# check class balance
print(df['label'].value_counts())

# initialize a BPE tokenizer model (empty vocab)
tokenizer = Tokenizer(models.BPE())

# add a pre-tokenizer to split text into words (or punctuation)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# set up a trainer — define vocab size, special tokens
trainer = trainers.BpeTrainer(vocab_size=10000, special_tokens=["<unk>", "<pad>"])

# save just the text column to a UTF-8 file for tokenizer training
utf8_corpus_path = 'corpus_utf8.txt'
df['text'].to_csv(utf8_corpus_path, index=False, header=False, encoding='utf-8')

# train tokenizer on dataset
tokenizer.train([utf8_corpus_path], trainer)

# convert sentences from raw text to token IDs (integer values), for model training later
encoded_lines = [tokenizer.encode(t).ids for t in df['text']]

# Pad sequences to uniform length
# will dynamically set max length to longest sentence, or could hard code a max limit like 100
MAX_LEN = max(len(x) for x in encoded_lines) 
X = pad_sequences(encoded_lines, maxlen=MAX_LEN, padding='post')
y = np.array(df['label'])

# Train/test split of 80/20 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model parameters
# vocab_size = len(vocab)
vocab_size = tokenizer.get_vocab_size()
embedding_dim = 300

# Build mode, 128 neurons, used softmax activation for multi-class activation where exactly one class is correct
# Dropout 30% --> helps robustness
model = Sequential([
    # manual embedding with random initialisation
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=MAX_LEN),

    # single layer bidirectional LSTM: 128 nodes
    Bidirectional(LSTM(128, return_sequences=False)),

    # triple layer bidirectional LSTM: 128, 256, 128 nodes in layers 1, 2 and 3 respectively
    # 30% dropout at each layer for robust testing

    # Bidirectional(LSTM(128, return_sequences=True)),
    # Dropout(0.3),
    # Bidirectional(LSTM(256, return_sequences=True)),
    # Dropout(0.3),
    # Bidirectional(LSTM(128)),
    # Dropout(0.3),
    Dense(3, activation='softmax')  # 3 sentiment classes
])

# Compile model
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# use sparse categorial crossentropy as labels are integer-encoded
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# compute class weights, makes it so that positives are never guessed however
# makes it so that in training, misclassifying a minority class is penalized more than misclassifying a majority class

# balanced: tells scikit-learn to automatically compute weights using the inverse frequency of each class
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

# converts the weights from a numpy array to a dictionary 
# of the form {0: 1.2, 1: 0.5, 2: 2.3}
class_weights_dict = dict(enumerate(class_weights))

# change the neutral weight, allowed the model to guess all 3, but maybe should remove this line
# class_weights_dict[1] *= 0.5

print("class weights:", class_weights_dict)


# Train model
# keras will use the class_weight to scale the loss for each training example based on its class, 
# e.g. misclassifying a negative means the loss is multiplied by a larger weight compared to misclassifying a 
# positive, as negative is the minority class compared to positive
# discourages the model from ignoring minority classes
lstm_model_history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32, class_weight=class_weights_dict)

# Evaluate
score = model.evaluate(X_test, y_test, verbose=1)

print("Test score: ", score[0])
print("Test accuracy: ", score[1])

# for each sentence, return the model's computed predicted probabilities for each classification
# e.g. [0.2, 0.5, 0.3] -> [20% positive, 50% neutral, 30% negative]
y_pred_probs = model.predict(X_test)

# Convert to class labels, picks the class with the ghighest probability for each sample 
# e.g. [0.2, 0.5, 0.3] -> select index 1 = neutral
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, save_path, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


# cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
# print(cm)
plot_confusion_matrix(
    y_true=y_test,
    y_pred=y_pred,
    classes=['positive', 'neutral', 'negative'],
    save_path='cm.png'
)

# Classification report (includes F1)
report = classification_report(y_test, y_pred, target_names=['positive', 'neutral', 'negative'])
print("\nClassification Report:")
print(report)

# don't print in scientific notation, print in 4 dp. 
np.set_printoptions(suppress=True, precision=4)
with open('output.txt', 'w+') as f: 
    for i in range(len(y_pred)): 
        if y_pred[i] != y_test[i]: 
            prediction = [k for k, v in label_map.items() if v == y_pred[i]]
            truth = [k for k, v in label_map.items() if v == y_test[i]]

            sentence = []

            # for number_word in X_test: 
            #     word = [k for k, v in vocab.items() if v == number_word]
            #     sentence.append(word)

            # Reconstruct sentence from indices
            word_indices = X_test[i]
            id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
            word_indices = X_test[i]
            sentence = [id_to_token[idx] for idx in word_indices if idx != tokenizer.token_to_id("<pad>")]


            f.write(f'''
Sentence: {" ".join(sentence)}
Prediction: {prediction[0]}
Probabilities: {y_pred_probs[i]}
True: {truth[0]}
''')