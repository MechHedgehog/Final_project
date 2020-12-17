import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys
import torch.nn.functional as F

def pred(model, sentence,rus, device, max_length=50, fl=0):
    q = []

    bf = ""
    tokens = [token.lower() for token in sentence]
    if fl:
        bf = sentence.pop()
    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, "<sos>")
    tokens.append("<eos>")

    # Go through each german token and convert to an index
    text_to_indices = [rus.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        outputs_encoder, hiddens, cells = model.encoder(sentence_tensor)

    outputs = [rus.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hiddens, cells = model.decoder(
                previous_word, outputs_encoder, hiddens, cells
            )
            best_guess = output.argmax(1).item()
        p = F.softmax(output, dim=1).data

        p = p.cpu()

        p = p.numpy()
        p = p.reshape(p.shape[1], )

        # get indices of top values

        q.append(p.argsort()[:][::-1])
        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == rus.vocab.stoi["<eos>"]:
            break
    if fl:
        m = []
        for i in q[-2]:
            if bf in rus.vocab.itos[i][:len(bf)]:
                m.append(rus.vocab.itos[i])
        return m
    return [rus.vocab.itos[i] for i in q[-2]]

    # remove start token


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
