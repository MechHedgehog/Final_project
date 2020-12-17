import re
import sys
import pyperclip
import random
# Для построения графического пользовательского интерфейса будет использоваться библиотека PyQT5
# Для установки можно просто открыть консоль в папке и ввести "pip install PyQt5" и "pip install pyperclip"
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# список предлогов - для последующей случайной выборки
random_words = ["в", "за", "от", "около", "между", "до", "для", "и", "с", "без", "к", "перед", "от", "по", "про"]


# По сравнению с 1 версией немного изменилась структура проекта. Теперь используется layout для большей гибкости
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Автоподсказки")
        self.setStyleSheet("background: white;")

        layout = QGridLayout()
        self.setLayout(layout)

        # Кнопки-подсказки
        self.pushButton_1 = QPushButton(random_words[0])
        self.pushButton_1.setStyleSheet("cursor:pointer;border:none;background:none;color:black;font-size:14px;")
        layout.addWidget(self.pushButton_1, 0, 0, 1, 1)
        self.pushButton_2 = QPushButton(random_words[1])
        self.pushButton_2.setStyleSheet("cursor:pointer;border:none;background:none;color:black;font-size:14px;")
        layout.addWidget(self.pushButton_2, 0, 1, 1, 1)
        self.pushButton_3 = QPushButton(random_words[2])
        self.pushButton_3.setStyleSheet("cursor:pointer;border:none;background:none;color:black;font-size:14px;")
        layout.addWidget(self.pushButton_3, 0, 2, 1, 1)

        # Поле ввода
        self.lineEdit = QLineEdit()
        self.lineEdit.setStyleSheet("border:1px solid black;color:black;font-size:16px;min-width:400px;height:24px;")
        # в поле ввода нельзя вводить латинские символы
        self.lineEdit.setValidator(QRegExpValidator(QRegExp("[^a-zA-Z]*"), self.lineEdit))
        layout.addWidget(self.lineEdit, 1, 0, 1, 3)
        self.lineEdit.setFocus()

        # Не позволяет изменять размеры окна
        self.setFixedSize(450, 100)

    # Обработка нажатия клавиш
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F5:
            self.reload_buttons()
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            pyperclip.copy(self.lineEdit.text())
            self.lineEdit.setText('')
        if event.modifiers() and Qt.ControlModifier:
            if event.key() == Qt.Key_1:
                self.lineEdit.setText(self.lineEdit.text() + self.pushButton_1.text() + ' ')
            if event.key() == Qt.Key_2:
                self.lineEdit.setText(self.lineEdit.text() + self.pushButton_2.text() + ' ')
            if event.key() == Qt.Key_3:
                self.lineEdit.setText(self.lineEdit.text() + self.pushButton_3.text() + ' ')

    # обновление подсказок
    def reload_buttons(self):
        curent_words = random.sample(random_words, 3)
        if len(self.lineEdit.text()) == 0 or len(self.lineEdit.text()) > 1 and self.lineEdit.text()[-2] in ".?!":
            self.pushButton_1.setText(curent_words[0].capitalize())
            self.pushButton_2.setText(curent_words[1].capitalize())
            self.pushButton_3.setText(curent_words[2].capitalize())
        else:
            self.pushButton_1.setText(curent_words[0])
            self.pushButton_2.setText(curent_words[1])
            self.pushButton_3.setText(curent_words[2])
        self.lineEdit.setFocus()


# Создание приложенич
app = QApplication(sys.argv)
# Инициализация приложения
window = MainWindow()


# Объединим 3 кнопки-подсказки в список для удобства, подсказок может быть сколько угодно
buttons = [window.pushButton_1, window.pushButton_2, window.pushButton_3]


# Не всегда подсказки нужны. Когда юзер вводит слово - подсказки скрываются
def hide_buttons(show=False):
    for button in buttons:
        button.setDisabled(not show)
        button.setStyleSheet("cursor:pointer;border:none;background:none;font-size:14px;color:" +
                             ("#000" if show else "#fff"))


# Функция вызывается после каждого изменения значения поля ввода
def text_checker():
    # input_text - получает текст поля ввода
    input_text = window.lineEdit.text()
    if len(input_text) > 0:
        # Подсказки будут появляться только после ввода слова и пробела после него
        if input_text[-1] == ' ':
            hide_buttons(True)
            try:
                # word - последнее введенное слово
                # word = re.findall(r'[а-яА-Я]+', input_text)[-1]
                # print(f'Последнее слово: {word}')
                # По условию задания в подсказках появляется 3 случайных предлога
                window.reload_buttons()
            except IndexError:
                hide_buttons()
        # Более красивое форматирование текста при вволе знаков препинания ".,?!"
        elif len(input_text) > 1 and input_text[-2] == " ":
            if input_text[-1] == ',':
                window.lineEdit.setText(window.lineEdit.text()[:-2] + window.lineEdit.text()[-1:] + " ")
            elif input_text[-1] in ".?!":
                window.lineEdit.setText(window.lineEdit.text()[:-2] + window.lineEdit.text()[-1:] + " ")
                for button in buttons:
                    button.setText(button.text().capitalize())
            else:
                hide_buttons()
        else:
            hide_buttons()
    else:
        # Сейчас поле ввода стало пустым
        hide_buttons(True)
        for button in buttons:
            button.setText(button.text().capitalize())


# Срабатывает при клике по кнопке
def text_appender(text):
    window.lineEdit.setText(window.lineEdit.text() + text + ' ')


# Устанавливаем первые значения для подсказок
for index, button in enumerate(buttons):
    button.setCursor(QCursor(Qt.PointingHandCursor))
    button.setText(random_words[index].capitalize())


# При изменении значения поля ввода вызывается функция text_checker
window.lineEdit.textChanged.connect(text_checker)
# При нажатии на копку-подсказку вызывается функция text_appender с соответсвующим значением нового слова
window.pushButton_1.clicked.connect(lambda: text_appender(window.pushButton_1.text()))
window.pushButton_2.clicked.connect(lambda: text_appender(window.pushButton_2.text()))
window.pushButton_3.clicked.connect(lambda: text_appender(window.pushButton_3.text()))

# Основной цикл
window.show()
sys.exit(app.exec_())
