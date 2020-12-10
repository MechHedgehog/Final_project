import re
import sys
import random
# Для построения графического пользовательского интерфейса будет использоваться библиотека PyQT5
# Для установки можно просто открыть консоль в папке и ввести "pip install -r req.txt"
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(571, 115)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("background-color: #ffffff; border: none;")
        self.centralwidget.setObjectName("centralwidget")
        # Поле ввода
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(0, 40, 571, 31))
        self.lineEdit.setStyleSheet("font-size: 16px;border: 1px solid #000000")
        self.lineEdit.setObjectName("lineEdit")

        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(0, 0, 571, 41))
        self.groupBox.setAutoFillBackground(False)
        self.groupBox.setStyleSheet("background-color: rgba(0, 0, 0, 0);font-size: 14px;border: none;")
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        # Кнопки подсказок
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(0, 0, 181, 41))
        self.pushButton.setStyleSheet("cursor: pointer;")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(190, 0, 181, 41))
        self.pushButton_2.setStyleSheet("")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(380, 0, 191, 41))
        self.pushButton_3.setStyleSheet("")
        self.pushButton_3.setObjectName("pushButton_3")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 571, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Автоподсказки"))


# Создание приложения
app = QtWidgets.QApplication(sys.argv)
# Инициализация приложения
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)

# список предлогов - для последующей случайной выборки
random_words = ["в", "за", "от", "около", "между", "до", "для", "и", "с", "без", "к", "перед", "от", "по", "про"]
# Объединим 3 кнопки-подсказки в список для удобства, подсказок может быть сколько угодно
buttons = [ui.pushButton, ui.pushButton_2, ui.pushButton_3]


# Не всегда подсказки нужны. Когда юзер вводит слово - подсказки скрываются
def hide_buttons(show=False):
    for button in buttons:
        button.setVisible(show)


# Функция вызывается после каждого изменения значения поля ввода
def text_checker():
    # input_text - получает текст поля ввода
    input_text = ui.lineEdit.text()
    if len(input_text) > 0:
        # Подсказки будут появляться только после ввода слова и пробела после него
        if input_text[-1] == ' ':
            hide_buttons(True)
            try:
                # word - последнее введенное слово
                word = re.findall(r'[а-яА-Я]+', input_text)[-1]
                print(f'Последнее слово: {word}')
                # По условию задания в подсказках появляется 3 случайных предлога
                curent_words = random.sample(random_words, 3)
                if len(input_text) > 1 and input_text[-2] in ".?!":
                    curent_words = list(map(lambda s: s.capitalize(), curent_words))
                for index, button in enumerate(buttons):
                    button.setText(curent_words[index])
                ui.lineEdit.setFocus()
            except IndexError:
                hide_buttons()
        # BONUS: более красивое форматирование текста при вволе знаков препинания ".,?!"
        elif len(input_text) > 1 and input_text[-2] == " ":
            if input_text[-1] == ',':
                ui.lineEdit.setText(ui.lineEdit.text()[:-2] + ui.lineEdit.text()[-1:] + " ")
            elif input_text[-1] in ".?!":
                ui.lineEdit.setText(ui.lineEdit.text()[:-2] + ui.lineEdit.text()[-1:] + " ")
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
    ui.lineEdit.setText(ui.lineEdit.text() + text + ' ')


# Устанавливаем первые значения для подсказок
for index, button in enumerate(buttons):
    button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
    button.setText(random_words[index].capitalize())


# При изменении значения поля ввода вызывается функция text_checker
ui.lineEdit.textChanged.connect(text_checker)
# При нажатии на копку-подсказку вызывается функция text_appender с соответсвующим значением нового слова
ui.pushButton.clicked.connect(lambda: text_appender(ui.pushButton.text()))
ui.pushButton_2.clicked.connect(lambda: text_appender(ui.pushButton_2.text()))
ui.pushButton_3.clicked.connect(lambda: text_appender(ui.pushButton_3.text()))

# Основной цикл
MainWindow.show()
sys.exit(app.exec_())
