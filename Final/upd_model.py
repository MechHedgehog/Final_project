import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchtext.data import Field, BucketIterator, TabularDataset
import matplotlib.pyplot as plt
from pytools import EarlyStopping
from cls import tokin, Decoder, Encoder, Seq2Seq
rus = Field( tokenize=tokin, lower=True, init_token="<sos>", eos_token="<eos>")
train_data, validation_data, test_data = TabularDataset.splits(
                                        path="Final",
                                        train="mem.json",
                                        validation="validation.json",
                                        test="test.json",
                                        format="json",
                                        fields={"src" : ("src", rus), "trg" : ("trg",rus)}

)

rus.build_vocab(train_data, max_size=16384, min_freq=4)
print(len(rus.vocab))

### We're ready to define everything we need for training our Seq2Seq model ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

# Training hyperparameters
num_epochs = 100
learning_rate = 3e-4
batch_size = 3

# Model hyperparameters
input_size_encoder = len(rus.vocab)
input_size_decoder = len(rus.vocab)
output_size = len(rus.vocab)
encoder_embedding_size = 130
decoder_embedding_size = 130
hidden_size = 700
num_layers = 1
enc_dropout = 0.0
dec_dropout = 0.0

# Tensorboard to get nice loss plot

step = 0

train_iterator, validation_iterator,test_iterator = BucketIterator.splits( #?
    (train_data,validation_data,test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = rus.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


checkpoint = torch.load("checkpoint.tar")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []
patience = 20
# initialize the early_stopping object
early_stopping = EarlyStopping(patience=patience, verbose=True)



for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")
    model.eval()
    model.train()

    for batch_idx, batch in enumerate(train_iterator):

        inp_data = batch.src.to(device)
        target = batch.trg.to(device)
        output = model(inp_data, target)
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        train_losses.append(loss.item())
        step += 1
    model.eval()
    for batch in validation_iterator:
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)
        output = model(inp_data, target)
        #?
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        loss = criterion(output, target)
        valid_losses.append(loss.item())

    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    epoch_len = len(str(num_epochs))

    print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                 f'train_loss: {train_loss:.5f} ' +
                 f'valid_loss: {valid_loss:.5f}')

    print(print_msg)

    # clear lists to track next epoch
    train_losses = []
    valid_losses = []

    # early_stopping needs the validation loss to check if it has decresed,
    # and if it has, it will make a checkpoint of the current model
    early_stopping(valid_loss, model, optimizer)

    if early_stopping.early_stop:
        print("Early stopping")
        break
model.load_state_dict(torch.load('checkpoint.pt'))
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
plt.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses,label='Validation Loss')

# find position of lowest validation loss
minposs = avg_valid_losses.index(min(avg_valid_losses))+1
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 0.5) # consistent scale
plt.xlim(0, len(avg_train_losses)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('loss_plot.png', bbox_inches='tight')