### Financial-Sentiment-Classifier

This project involves building a LSTM (Long Short Term Memory) RNN (Recurrent Neural Network) that analyses financial texts and classifies the sentiment into three classes (positive, neutral, negative). I've attempted to analyse how different hyperparameters, embedding methods and design architectures affect the overall accuracy. 

The model will print out classification reports afterwards and the confusion matrices to analyse the model's accuracy as well as classification behaviours, and we used the macro-avg f1 score as the main scoring metric to judge between this LSTM model as well as other Machine Learning and transformer-based language models like FinBERT. 

Below are the different hyperparameters and embedding methods that I have implemented and compared. 

LSTM (Long Short Term Memory): 
* LSTM.py: 
    * Tried with 1 layer: 128 nodes
    * Tried with 3 layers: 128, 256, 128 nodes
    * Tried with [GloVe embedding](https://nlp.stanford.edu/projects/glove/)
         * GloVe embeddings: word embeddings pretrained from popular global corpa like Wikipedia, Twitter, News, etc. 
* LSTM_BPE:
    * Tried with Byte Pair Encoding instead of GloVe embedding

Dataset obtained from: https://huggingface.co/datasets/takala/financial_phrasebank
