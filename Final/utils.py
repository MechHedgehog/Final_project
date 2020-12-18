import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys
import torch.nn.functional as F


def pred(model, sentence, rus, device, max_length=50, fl=0):
    '''
    This is the function that predicts next word basing on previouse ones
    Params:
    model - Model - here you need to input your pretrained model
    sentence - string - person's sentence
    device - device - previously chosen device
    rus - field - your corpus
    '''
    q = []
    bf = ""
    if fl:
        bf = sentence.pop()
    tokens = [token.lower() for token in sentence]

    if tokens == []:
        tokens = ["привет"]
    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, "<sos>")
    tokens.append("<eos>")

    # Go through each token and convert to an index
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
    ret = []
    for i in q[-2]:
        if rus.vocab.itos[i] != "<unk>":
            ret.append(i)
    if fl:
        m = []
        for i in ret:
            if bf in rus.vocab.itos[i][:len(bf)]:
                m.append(rus.vocab.itos[i])
        return m
    return [rus.vocab.itos[i] for i in ret]

def save_text(mem):
    with open("mem.json", "a", encoding="windows-1251") as f:
        for i in mem:
            if "{\"src\":\"" + " ".join(i[0:-1]) + "\",\"trg\":\"" + " ".join(i[1:]) + "\"}" != "{\"src\":\"\",\"trg\":\"\"}":
                f.write("{\"src\":\"" + " ".join(i[0:-1]) + "\",\"trg\":\"" + " ".join(i[1:]) + "\"}\n")
    print("memory saved")

