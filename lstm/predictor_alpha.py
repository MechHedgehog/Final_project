import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

start_time = time.time()

with open("token2int.pickle", "rb") as pckl:
    token2int = pickle.load(pckl)
with open("int2token.pickle", "rb") as pckl:
    int2token = pickle.load(pckl)

vocab_size = len(int2token)

class WordLSTM(nn.Module):

    def __init__(self, n_hidden=256, n_layers=4, drop_prob=0.3, lr=0.001):
        super().__init__()

        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.emb_layer = nn.Embedding(vocab_size, 200)

        ## define the LSTM
        self.lstm = nn.LSTM(200, n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)

        ## define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        ## define the fully-connected layer
        self.fc = nn.Linear(n_hidden, vocab_size)

    def forward(self, x, hidden):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## pass input through embedding layer
        embedded = self.emb_layer(x)

        ## Get the outputs and the new hidden state from the lstm
        lstm_output, hidden = self.lstm(embedded, hidden)

        ## pass through a dropout layer
        out = self.dropout(lstm_output)

        # out = out.contiguous().view(-1, self.n_hidden)
        out = out.reshape(-1, self.n_hidden)

        ## put "out" through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        ''' initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        # if GPU is available
        if (torch.cuda.is_available()):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())

        # if GPU is not available
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden


def predict(net, tkn, h=None):
    # tensor inputs
    x = np.array([[token2int[tkn]]])
    inputs = torch.from_numpy(x)
    inputs = torch.tensor(inputs).to(torch.int64)

    # push to GPU
    inputs = inputs.cuda()

    # detach hidden state from history
    h = tuple([each.data for each in h])

    # get the output of the model
    out, h = net(inputs, h)

    # get the token probabilities
    p = F.softmax(out, dim=1).data

    p = p.cpu()

    p = p.numpy()
    p = p.reshape(p.shape[1], )

    # get indices of top 3 values
    top_n_idx = p.argsort()[-3:][::-1]

    # randomly select one of the three indices
    sampled_token_index = top_n_idx[random.sample([0, 1, 2], 1)[0]]

    # return the encoded value of the predicted char and the hidden state
    return int2token[sampled_token_index], h


# function to generate text
def sample(net, size, prime):
    # push to GPU
    net.cuda()

    net.eval()

    # batch size is 1
    h = net.init_hidden(1)

    toks = prime.split()

    # predict next token
    for t in prime.split():
        token, h = predict(net, t, h)

    toks.append(token)

    # predict subsequent tokens
    for i in range(size - 1):
        token, h = predict(net, toks[-1], h)
        toks.append(token)

    return ' '.join(toks)

net = WordLSTM()
net.load_state_dict(torch.load("mode.pth"))

net.cuda()
print("--- %s seconds ---" % (time.time() - start_time))
print(sample(net, 1, prime="сколько"))