# Financial-Sentiment-Classifier

Dataset obtained from: https://huggingface.co/datasets/takala/financial_phrasebank

These are the different LSTM models I've implemented to classify the financial sentiment.

LSTM: 
* LSTM.py: 
    * Tried with 1 layer: 128 nodes
    * Tried with 3 layers: 128, 256, 128 nodes
    * Tried with GloVe embedding
* LSTM_BPE:
    * Tried with Byte Pair Encoding instead of GloVe embedding
