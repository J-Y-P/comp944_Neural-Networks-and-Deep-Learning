#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

For question how your program works, and explain any design and training decisions you made along the way.For question how your program works, and explain any design and training decisions you made along the way.

We used a bidirectional LSTM and MSE loss function with Adam optimiser as our model, and Glove 6B 50 dim to process word vector.

First of all, we considered which network should we use to solve this problem. We tested three network, CNN, RNN and LSTM.
LSTM got the best performance. Another reason for we choose LSTM is that rating prediction based on the product reviews can
be viewed as a NLP problem, we can analyze the emotion of reviews to analyze rating. So LSTM may have better performance,
as LSTM can train the relationship of between words to predict the emotion or rating.

Next, for details of the LSTM net work. We use the bidirectional LSTM. Because we can the relationship between word and before and after based
on bidirectional LSTM. We used two layers LSTM as our final network structure, a higher layer network did not bring us a better accuracy.

For the processing of labels (Rating), as a rate can be viewed as a regression or a classification problem. We tried both ways to process the labels,
regression has a better performance. (Convert Label)As the value of word vector is lower than 1, we normalized the label by min-max normalization which increased about 3% scores.
We used MSE function as our loss function. We also tried MAE, because we thought the reason for the low accuracy may be that there are
many outliers in the data set. However, the performance of MAE did no better than MSE.
(Convert Net Output)After we get the output of our model, we convert it to final result by multiply by 4 and add 1, then through a round function.

We did not get a good performance improvement after trying different parameters and methods. Then we tried to process our input(Preprocessing).
Firstly, we print the distribution of words, found that there are lots of meaningless words(Stop Words), such as "is, we, the", so we add the stop words.
And we also deleted the punctuation and digits in the sentences which we think are not helpful for our model. Indeed, our score has increased by 4%.

These are what we are thinking and how we deal with those problems when we work on this assignment. Although, we did not get hign correctness and weighted score,
this is the highest score we can get so far. More processing on data set or tuning hyper parameters may improve the performance of our model in the future.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn
import string

###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # clean punctuation
    sample = [x.translate(str.maketrans('', '', string.punctuation+string.digits)) for x in sample]
    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """

    return batch

stopWords = {"it's",'ourselves', 'hers', 'between', 'yourself', 'there', 'about', 'once', 'during', 'out', 'they', 'own', 'an', 'be', 'some', 'for', 'do',
             'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
             'themselves', 'until', 'are', 'we', 'these', 'your', 'his', 'me', 'were', 'her', 'more', 'himself', 'this', 'our', 'their', 'both', 'up',
             'to', 'ours', 'had', 'she', 'when', 'at', 'them', 'and', 'been', 'have', 'in', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what',
             'over', 'why', 'so', 'can', 'did', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'only', 'myself', 'which', 'those', 'i',
             'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here'}
wordVectors = GloVe(name='6B', dim=50)

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################

def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """
    datasetLabel = torch.add(datasetLabel,-1)
    datasetLabel = torch.div(datasetLabel, 4)
    return datasetLabel

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    netOutput = torch.mul(netOutput, 4)
    netOutput = torch.add(netOutput, 1)


    return netOutput.round()

###########################################################################
################### The following determines the model ####################
###########################################################################

input_size=50
hidden_size=64
output_size=1
num_layer=3
class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """

    def __init__(self):
        super(network, self).__init__()
        # LSTM(input_size,hidden_size,num_layers)
        self.lstm = tnn.LSTM(input_size, hidden_size, num_layer, batch_first=True, bidirectional=True)
        self.fc = tnn.Sequential(
            tnn.Linear(hidden_size,15),
            tnn.ReLU(),
            tnn.Linear(15,output_size),
        )
        self.h_s = None
        self.h_c = None

    def forward(self, input, length):
        # input = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True)
        lstm_out, (h_n, c_n) = self.lstm(input)
	# use the output of final cell as our output
        output = h_n[-1, :, :]
        output = self.fc(output)
        output = output.view(-1)
        return output


class loss(tnn.Module):
    """
    Class for creating a custom loss function, if desired.
    You may remove/comment out this class if you are not using it.
    """

    def __init__(self):
        super(loss, self).__init__()

    def forward(self, output, target):
        pass

net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
# lossFunc = tnn.L1Loss()
lossFunc = tnn.MSELoss()
###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 1
batchSize = 64
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.01)