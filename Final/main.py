import re
import sys
import pyperclip
from cls import Encoder, Decoder, Seq2Seq, tokin
# The PyQT5 library will be used to build a graphical user interface
# To install, you can simply open the console in the folder and type "pip install PyQt5" and "pip install pyperclip"
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import torch
from torchtext.data import Field, TabularDataset
from utils import pred, save_text
import string


#choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_context = []
pred_on_con = []
mem = []

#make field of text
rus = Field( tokenize=tokin, lower = True, init_token = "<sos>", eos_token="<eos>")
train_data, validation_data, test_data = TabularDataset.splits(
                                        path=r'C:\Users\Acer\PycharmProjects\final_BLT',
                                        train='train.json',
                                        validation= 'validation.json',
                                        test='test.json',
                                        format = 'json',
                                        fields = {"src" : ("src", rus), "trg" : ("trg",rus)})


def de_pun(text):
    '''
    This function deletes punctuation and numbers
    Params: text - text (file)
    '''
    ou = []
    for i in text:
        l = i
        for j in string.punctuation+"0123456789":
            l = l.replace(j, "")
        ou.append(l)
    return ou


#build vocabulary
rus.build_vocab(train_data, max_size=16384, min_freq=4)

#trainig params
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
encoder_net = Encoder( input_size_encoder, encoder_embedding_size,
                      hidden_size, num_layers, enc_dropout).to(device)

decoder_net = Decoder( input_size_decoder, decoder_embedding_size,
    hidden_size, output_size, num_layers, dec_dropout,).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
checkpoint = torch.load("checkpoint.tar")
model.load_state_dict(checkpoint['model_state_dict'])

# Compared with version 1, the project structure has changed slightly. Now layout is used for more flexibility
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Автоподсказки")
        self.setStyleSheet("background: white;")

        layout = QGridLayout()
        self.setLayout(layout)
        self.current_words = current_context if len(current_context) > 2 else ["привет", "я", "а"]
        # Tooltips buttons
        self.pushButton_1 = QPushButton(self.current_words[0])
        self.pushButton_1.setStyleSheet("cursor:pointer;border:none;background:none;color:black;font-size:14px;")
        layout.addWidget(self.pushButton_1, 0, 0, 1, 1)
        self.pushButton_2 = QPushButton(self.current_words[1])
        self.pushButton_2.setStyleSheet("cursor:pointer;border:none;background:none;color:black;font-size:14px;")
        layout.addWidget(self.pushButton_2, 0, 1, 1, 1)
        self.pushButton_3 = QPushButton(self.current_words[2])
        self.pushButton_3.setStyleSheet("cursor:pointer;border:none;background:none;color:black;font-size:14px;")
        layout.addWidget(self.pushButton_3, 0, 2, 1, 1)

        # Input field
        self.lineEdit = QLineEdit()
        self.lineEdit.setStyleSheet("border:1px solid black;color:black;font-size:16px;min-width:400px;height:24px;")
        # Latin characters cannot be typed in the input field
        self.lineEdit.setValidator(QRegExpValidator(QRegExp("[^a-zA-Z]*"), self.lineEdit))
        layout.addWidget(self.lineEdit, 1, 0, 1, 3)
        self.lineEdit.setFocus()

        # Does not allow you to resize the window
        self.setFixedSize(450, 100)


    # Key press processing
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F5:
            save_text(mem)
            mem.clear()
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            pyperclip.copy(self.lineEdit.text())
            self.lineEdit.setText('')
            mem.append(de_pun(current_context))
            current_context.clear()
        if event.modifiers() and Qt.ControlModifier:
            if event.key() == Qt.Key_1:
                text_appender(self.pushButton_1.text())
            if event.key() == Qt.Key_2:
                text_appender(self.pushButton_2.text())
            if event.key() == Qt.Key_3:
                text_appender(self.pushButton_3.text())


    # tooltip update
    def reload_buttons(self, curent_words):
        if curent_words == []:
            curent_words = ["привет", "я", "а"]
        if len(self.lineEdit.text()) == 0 or len(self.lineEdit.text()) > 1 and self.lineEdit.text()[-2] in ".?!":
            self.pushButton_1.setText(curent_words[0].capitalize())
            self.pushButton_2.setText(curent_words[1].capitalize())
            self.pushButton_3.setText(curent_words[2].capitalize())
        else:
            self.pushButton_1.setText(curent_words[0])
            self.pushButton_2.setText(curent_words[1])
            self.pushButton_3.setText(curent_words[2])
        self.lineEdit.setFocus()


# app creation
app = QApplication(sys.argv)
# Initializing an App
window = MainWindow()


# Let's combine 3 tooltips in a list for convenience, there can be as many tooltips as you want
buttons = [window.pushButton_1, window.pushButton_2, window.pushButton_3]


def hide_buttons(show=True):
    for button in buttons:
        button.setDisabled(not show)
        button.setStyleSheet("cursor:pointer;border:none;background:none;font-size:14px;color:" +
                             ("#000" if show else "#fff"))


# The function is called after each change of the input field value
def text_checker():
    # input_text - gets the text of the input field
    global pred_on_con, current_context
    input_text = window.lineEdit.text().lower()
    if len(input_text) > 0:
        # Tooltips will only appear after you enter a word and space after it
        if input_text[-1] == ' ':

            try:
                #word is the last word entered
                pred_on_con.clear()
                word = re.findall(r'[а-яА-Я]+', input_text)[-1]

                current_context.append(word)
                current_words = pred(model, de_pun(current_context), rus, device, max_length=50)[1:]
                # By the condition of the task 3 random prepositions appear in the prompts
                for index, button in enumerate(buttons):
                    button.setText(current_words[index])

            except IndexError:
                hide_buttons()
        # Better text formatting when punctuation marks are inserted ".,?!"
        elif len(input_text) > 1 and input_text[-2] == " ":
            if input_text[-1] == ',':
                window.lineEdit.setText(window.lineEdit.text()[:-2] + window.lineEdit.text()[-1:] + " ")
            elif input_text[-1] in ".?!":
                window.lineEdit.setText(window.lineEdit.text()[:-2] + window.lineEdit.text()[-1:] + " ")
                mem.append(de_pun(current_context + [input_text.split(" ")[-1]]))
                current_context.clear()
                for button in buttons:
                    button.setText(button.text().capitalize())
            else:
                x = input_text.split(" ")[-1]

                current_words = []
                if pred_on_con != []:
                    for i in pred_on_con:
                        if x in i[0:len(x)]:
                            current_words.append(i)
                    if len(current_words) < 3:
                        current_words += pred_on_con
                else:
                    current_words = pred(model, de_pun(current_context + [x]), rus, device, max_length=50, fl=1)[1:]
                    pred_on_con = current_words
                for index, button in enumerate(buttons):
                    button.setText(current_words[index])
        elif input_text[-1] in ".?!":
            mem.append(de_pun(current_context + [input_text.split(" ")[-1]]))
            current_context.clear()
            for button in buttons:
                button.setText(button.text().capitalize())

        elif len(input_text) == 1:
            pred_on_con = pred(model, de_pun([input_text]), rus, device, max_length=50, fl=1)[1:]
            for index, button in enumerate(buttons):
                button.setText(pred_on_con[index])
        else:
            x = input_text.split(" ")[-1]

            current_words = []
            if pred_on_con != []:
                for i in pred_on_con:
                    if x in i[0:len(x)]:
                        current_words.append(i)
                if len(current_words) < 3:
                    current_words += pred_on_con
            else:
                if x in "приветая" and x == input_text:
                    current_words = pred(model, de_pun(current_context + [x]), rus, device, max_length=50)
                else:
                    current_words = pred(model, de_pun(current_context + [x]), rus, device, max_length=50, fl=1)
                pred_on_con = current_words
            for index, button in enumerate(buttons):
                button.setText(current_words[index])
    else:
        # The input field is now blank.
        hide_buttons(True)
        for button in buttons:
            button.setText(button.text().capitalize())


# Triggers when the button is clicked
def text_appender(text):

    if len(window.lineEdit.text()) > 0 and window.lineEdit.text()[-1] != " ":
        window.lineEdit.setText(" ".join(window.lineEdit.text().split(" ")[:-1]) + " " + text)
    else:
        window.lineEdit.setText(window.lineEdit.text() + text)


# Set the first values for the prompts
for index, button in enumerate(buttons):
    button.setCursor(QCursor(Qt.PointingHandCursor))
    button.setText(["привет", "я", "а"][index].capitalize())


# The text_checker function is called when you change the value of an input field
window.lineEdit.textChanged.connect(text_checker)
# When you click on the tooltip, the text_appender function is called with the corresponding value of the new word
window.pushButton_1.clicked.connect(lambda: text_appender(window.pushButton_1.text()))
window.pushButton_2.clicked.connect(lambda: text_appender(window.pushButton_2.text()))
window.pushButton_3.clicked.connect(lambda: text_appender(window.pushButton_3.text()))

# main loop
window.show()
sys.exit(app.exec_())
