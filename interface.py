import algorithm
import variable
import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *

class Window(QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.initUI()

    def initUI(self):

        self.setGeometry(600, 300, 700, 500)
        self.setWindowTitle("Alphabet Recognition")
        self.home()

    def home(self):

        self.tdstextbox = QLineEdit(self)
        self.tdstextbox.move(5, 5)
        self.tdstextbox.resize(110, 30)
        self.tdstextbox.setText(str(variable.NUM_IMAGE_IN_DATA_SET))

        self.tdsbtn = QPushButton("Number of TDS", self)
        self.tdsbtn.clicked.connect(self.getLearningRateText)
        self.tdsbtn.resize(100,30)
        self.tdsbtn.move(120, 5)

        self.netextbox = QLineEdit(self)
        self.netextbox.move(5, 45)
        self.netextbox.resize(110, 30)
        self.netextbox.setText(str(variable.NUM_EPOCHS))

        self.nebtn = QPushButton("Number Epochs", self)
        self.nebtn.clicked.connect(self.getNumEpochsText)
        self.nebtn.resize(100, 30)
        self.nebtn.move(120, 45)

        self.h1textbox = QLineEdit(self)
        self.h1textbox.move(5, 85)
        self.h1textbox.resize(110, 30)
        self.h1textbox.setText(str(variable.NUM_NODE_HIDDEN1))

        self.h1btn = QPushButton("Hidden 1 nodes", self)
        self.h1btn.clicked.connect(self.getHiddenOneText)
        self.h1btn.resize(100, 30)
        self.h1btn.move(120, 85)

        self.h2textbox = QLineEdit(self)
        self.h2textbox.move(5, 125)
        self.h2textbox.resize(110, 30)
        self.h2textbox.setText(str(variable.NUM_NODE_HIDDEN2))

        self.h2btn = QPushButton("Hidden 2 nodes", self)
        self.h2btn.clicked.connect(self.getHiddenTwoText)
        self.h2btn.resize(100, 30)
        self.h2btn.move(120, 125)

        self.thtextbox = QLineEdit(self)
        self.thtextbox.move(5, 165)
        self.thtextbox.resize(110, 30)
        self.thtextbox.setText(str(variable.THRESHOLD))

        self.thbtn = QPushButton("Threshold", self)
        self.thbtn.clicked.connect(self.getThresholdText)
        self.thbtn.resize(100, 30)
        self.thbtn.move(120, 165)

        self.lrtextbox = QLineEdit(self)
        self.lrtextbox.move(5, 205)
        self.lrtextbox.resize(110, 30)
        self.lrtextbox.setText(str(variable.ALPHA))

        self.lrbtn = QPushButton("Learning Rate", self)
        self.lrbtn.clicked.connect(self.getNumOfTDSText)
        self.lrbtn.resize(100, 30)
        self.lrbtn.move(120, 205)

        self.intextbox = QLineEdit(self)
        self.intextbox.move(5, 245)
        self.intextbox.resize(110, 30)
        self.intextbox.setText(str(variable.RATE))

        self.inbtn = QPushButton("Image Scramble Rate", self)
        self.inbtn.clicked.connect(self.getRateText)
        self.inbtn.resize(120, 30)
        self.inbtn.move(120, 245)

        self.gentdsbtn = QPushButton("Generate A-G Training data set", self)
        self.gentdsbtn.clicked.connect(self.generateAlphbetTDS)
        self.gentdsbtn.resize(200, 30)
        self.gentdsbtn.move(5, 285)

        self.trainbtn = QPushButton("Initial Weights and Train NN", self)
        self.trainbtn.clicked.connect(self.trainNN)
        self.trainbtn.resize(200, 30)
        self.trainbtn.move(5, 325)
        self.trainbtn.setDisabled(True)

        self.instruction = QTextEdit(self)
        self.instruction.move(5, 365)
        self.instruction.resize(200, 100)
        self.instruction.setDisabled(True)
        self.instruction.setText("Change and set above value by clicking the button next to it")

        self.comboxBox = QComboBox(self)
        self.comboxBox.addItem("A")
        self.comboxBox.addItem("B")
        self.comboxBox.addItem("C")
        self.comboxBox.addItem("D")
        self.comboxBox.addItem("E")
        self.comboxBox.addItem("F")
        self.comboxBox.addItem("G")
        self.comboxBox.addItem("H")
        self.comboxBox.addItem("I")
        self.comboxBox.addItem("J")
        self.comboxBox.move(300, 5)

        self.genalphabetbtn = QPushButton("Generate image & Recognize using NN", self)
        self.genalphabetbtn.clicked.connect(self.genAlphabetForTesting)
        self.genalphabetbtn.resize(200, 30)
        self.genalphabetbtn.move(410, 5)
        self.genalphabetbtn.setDisabled(True)
        self.table = QTableWidget(10, 8, self)

        for i in range(10):
            for j in range(8):
                value = 0
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, "")
                self.table.setItem(i, j, item)
                if value == 1:
                    self.table.item(i, j).setBackground(QColor(0, 0, 0))

        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        self.table.resize(305, 255)
        self.table.move(300, 55)

        self.outputtextbox = QTextEdit(self)
        self.outputtextbox.move(300, 320)
        self.outputtextbox.resize(250, 150)
        self.outputtextbox.setDisabled(True)
        self.outputtextbox.setText("Output Value???")

        self.show()

    @pyqtSlot()
    def getLearningRateText(self):
        number = self.lrtextbox.text()
        try:
            variable.ALPHA = float(number)
            print("learning rate => " + str(variable.ALPHA))
        except ValueError:
            print("Please enter numbers only, not text!!")

    def getNumEpochsText(self):
        number = self.netextbox.text()
        try:
            variable.NUM_EPOCHS = int(number)
            print("Number of epochs => " + str(variable.NUM_EPOCHS))
        except ValueError:
            print("Please enter numbers only, not text!!")

    def getHiddenOneText(self):
        number = self.h1textbox.text()
        try:
            variable.NUM_NODE_HIDDEN1 = int(number)
            print("Hidden one nodes => " + str(variable.NUM_NODE_HIDDEN1))
        except ValueError:
            print("Please enter numbers only, not text!!")

    def getHiddenTwoText(self):
        number = self.h2textbox.text()
        try:
            variable.NUM_NODE_HIDDEN2 = int(number)
            print("Hidden two nodes => " + str(variable.NUM_NODE_HIDDEN2))
        except ValueError:
            print("Please enter numbers only, not text!!")

    def getThresholdText(self):
        number = self.thtextbox.text()
        try:
            variable.THRESHOLD = float(number)
            print("Threshold => " + str(variable.THRESHOLD))
        except ValueError:
            print("Please enter numbers only, not text!!")

    def getNumOfTDSText(self):
        number = self.tdstextbox.text()
        try:
            variable.NUM_IMAGE_IN_DATA_SET = int(number)
            print("Number of Training Data Set => " + str(variable.NUM_IMAGE_IN_DATA_SET))
        except ValueError:
            print("Please enter numbers only, not text!!")

    def getRateText(self):
        number = self.intextbox.text()
        try:
            variable.RATE = float(number)
            print("Number of Image scramble rate => " + str(variable.RATE))
        except ValueError:
            print("Please enter numbers only, not text!!")

    def generateAlphbetTDS(self):
        algorithm.create_training_data_set(variable.image_A, variable.Train_data_set_A)
        algorithm.create_training_data_set(variable.image_B, variable.Train_data_set_B)
        algorithm.create_training_data_set(variable.image_C, variable.Train_data_set_C)
        algorithm.create_training_data_set(variable.image_D, variable.Train_data_set_D)
        algorithm.create_training_data_set(variable.image_E, variable.Train_data_set_E)
        algorithm.create_training_data_set(variable.image_F, variable.Train_data_set_F)
        algorithm.create_training_data_set(variable.image_G, variable.Train_data_set_G)

        print("Done generate %d images for each alphabet" % variable.NUM_IMAGE_IN_DATA_SET)
        self.instruction.setText("Done generate %d images for each alphabet" % variable.NUM_IMAGE_IN_DATA_SET)
        self.trainbtn.setDisabled(False)

    def trainNN(self):
        print("Start Training")
        self.instruction.setText("Start Training, process could take very long")
        self.lrbtn.setDisabled(True)
        self.nebtn.setDisabled(True)
        self.h1btn.setDisabled(True)
        self.h2btn.setDisabled(True)
        self.thbtn.setDisabled(True)
        self.tdsbtn.setDisabled(True)

        algorithm.init_weight()

        for i in range(variable.NUM_EPOCHS):
            print(i)
            for j in range(variable.NUM_IMAGE_IN_DATA_SET):
                algorithm.training_nn(0, variable.Train_data_set_A, j)
                algorithm.training_nn(1, variable.Train_data_set_B, j)
                algorithm.training_nn(2, variable.Train_data_set_C, j)
                algorithm.training_nn(3, variable.Train_data_set_D, j)
                algorithm.training_nn(4, variable.Train_data_set_E, j)
                algorithm.training_nn(5, variable.Train_data_set_F, j)
                algorithm.training_nn(6, variable.Train_data_set_G, j)

        print("Training Complete")
        print(variable.Weight_input_hidden1)
        print(variable.Weight_hidden1_hidden2)
        print(variable.Weight_hidden2_output)
        self.instruction.setText("Training Complete")

        self.lrbtn.setDisabled(False)
        self.nebtn.setDisabled(False)
        self.h1btn.setDisabled(False)
        self.h2btn.setDisabled(False)
        self.thbtn.setDisabled(False)
        self.tdsbtn.setDisabled(False)
        self.genalphabetbtn.setDisabled(False)

    def genAlphabetForTesting(self):
        alphabet = self.comboxBox.currentText()

        if alphabet.__eq__('A'):
            image = algorithm.scramble(variable.image_A, variable.RATE)

            for i in range(10):
                for j in range(8):
                    value = image[i][j]
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, "")
                    self.table.setItem(i, j, item)
                    if value == 1:
                        self.table.item(i, j).setBackground(QColor(0, 0, 0))
            self.showNN(image)

        elif alphabet.__eq__('B'):
            image = algorithm.scramble(variable.image_B, variable.RATE)

            for i in range(10):
                for j in range(8):
                    value = image[i][j]
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, "")
                    self.table.setItem(i, j, item)
                    if value == 1:
                        self.table.item(i, j).setBackground(QColor(0, 0, 0))
            self.showNN(image)

        elif alphabet.__eq__('C'):
            image = algorithm.scramble(variable.image_C, variable.RATE)

            for i in range(10):
                for j in range(8):
                    value = image[i][j]
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, "")
                    self.table.setItem(i, j, item)
                    if value == 1:
                        self.table.item(i, j).setBackground(QColor(0, 0, 0))
            self.showNN(image)

        elif alphabet.__eq__('D'):
            image = algorithm.scramble(variable.image_D, variable.RATE)

            for i in range(10):
                for j in range(8):
                    value = image[i][j]
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, "")
                    self.table.setItem(i, j, item)
                    if value == 1:
                        self.table.item(i, j).setBackground(QColor(0, 0, 0))
            self.showNN(image)

        elif alphabet.__eq__('E'):
            image = algorithm.scramble(variable.image_E, variable.RATE)

            for i in range(10):
                for j in range(8):
                    value = image[i][j]
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, "")
                    self.table.setItem(i, j, item)
                    if value == 1:
                        self.table.item(i, j).setBackground(QColor(0, 0, 0))
            self.showNN(image)

        elif alphabet.__eq__('F'):
            image = algorithm.scramble(variable.image_F, variable.RATE)

            for i in range(10):
                for j in range(8):
                    value = image[i][j]
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, "")
                    self.table.setItem(i, j, item)
                    if value == 1:
                        self.table.item(i, j).setBackground(QColor(0, 0, 0))
            self.showNN(image)

        elif alphabet.__eq__('G'):
            image = algorithm.scramble(variable.image_G, variable.RATE)

            for i in range(10):
                for j in range(8):
                    value = image[i][j]
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, "")
                    self.table.setItem(i, j, item)
                    if value == 1:
                        self.table.item(i, j).setBackground(QColor(0, 0, 0))
            self.showNN(image)

        elif alphabet.__eq__('H'):
            image = algorithm.scramble(variable.image_H, variable.RATE)

            for i in range(10):
                for j in range(8):
                    value = image[i][j]
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, "")
                    self.table.setItem(i, j, item)
                    if value == 1:
                        self.table.item(i, j).setBackground(QColor(0, 0, 0))
            self.showNN(image)

        elif alphabet.__eq__('I'):
            image = algorithm.scramble(variable.image_I, variable.RATE)

            for i in range(10):
                for j in range(8):
                    value = image[i][j]
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, "")
                    self.table.setItem(i, j, item)
                    if value == 1:
                        self.table.item(i, j).setBackground(QColor(0, 0, 0))
            self.showNN(image)

        elif alphabet.__eq__('J'):
            image = algorithm.scramble(variable.image_J, variable.RATE)

            for i in range(10):
                for j in range(8):
                    value = image[i][j]
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, "")
                    self.table.setItem(i, j, item)
                    if value == 1:
                        self.table.item(i, j).setBackground(QColor(0, 0, 0))
            self.showNN(image)

    def showNN(self, image):
        algorithm.o_input(image)
        algorithm.o_hidden1()
        algorithm.o_hidden2()
        algorithm.o_output()
        self.outputtextbox.setText("The outputs are \n=> %f\n=> %f\n=> %f\n=> %f\n=> %f\n=> %f" %
                                   (variable.Output_output[0], variable.Output_output[1], variable.Output_output[2],
                                   variable.Output_output[3], variable.Output_output[4], variable.Output_output[5]))
        self.outputtextbox.append(self.recog_alpha(variable.Output_output))

    def recog_alpha(self, output):
        result = []
        reco = ""
        for item in output:
            if(item > 0.99):
                result.append(1)
            else:
                result.append(0)

        reco = ""
        for i in range(len(variable.expect_output)):
            if variable.expect_output[i] == result:
                if i == 0:
                    reco = "This image is A"
                elif i == 1:
                    reco = "This image is B"
                elif i == 2:
                    reco = "This image is C"
                elif i == 3:
                    reco = "This image is D"
                elif i == 4:
                    reco = "This image is E"
                elif i == 5:
                    reco = "This image is F"
                elif i == 6:
                    reco = "This image is G"

        if reco.__eq__(""):
            return "Cannot recognize this image: This image is not between A - G"
        else:
            return reco




app = QApplication(sys.argv)
GUI = Window()
sys.exit(app.exec_())

