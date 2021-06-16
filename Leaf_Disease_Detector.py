# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'leaf_disease_detector.ui'
# Created by: PyQt5 UI code generator 5.15.4

# Author: Shagun Bhardwaj


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QWidget
from PyQt5.QtGui import QPixmap
import resources_rc
import tensorflow as tf
import resources_rc
import numpy as np
from PIL import Image
import pickle
import cv2
import sklearn
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

global model
global image_labels
# Load model
with tf.device('/cpu:0'):
    model = load_model('model.h5')


class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1000)
        font = QtGui.QFont()
        font.setKerning(True)
        MainWindow.setFont(font)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("background-image: url(:/images/bgd.png);")
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.browse = QtWidgets.QPushButton(self.centralwidget)
        self.browse.setGeometry(QtCore.QRect(1240, 210, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.browse.setFont(font)
        self.browse.setStyleSheet("background-image: url(:/images/image4.jpg);")
        self.browse.setObjectName("browse")
        self.browse.clicked.connect(self.browsefiles)

        self.imagepath = QtWidgets.QTextBrowser(self.centralwidget)
        self.imagepath.setGeometry(QtCore.QRect(500, 210, 721, 31))
        self.imagepath.setStyleSheet("background-image: url(:/images/image3.jpg);")
        self.imagepath.setObjectName("imagepath")
        
        self.detect = QtWidgets.QPushButton(self.centralwidget)
        self.detect.setGeometry(QtCore.QRect(900, 570, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.detect.setFont(font)
        self.detect.setStyleSheet("background-image: url(:/images/image4.jpg);")
        self.detect.setObjectName("detect")
        self.detect.clicked.connect(self.detectdisease)
        
        self.resultlabel = QtWidgets.QLabel(self.centralwidget)
        self.resultlabel.setGeometry(QtCore.QRect(970, 720, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.resultlabel.setFont(font)
        self.resultlabel.setStyleSheet("background-image: url(:/images/images6.jpeg);")
        self.resultlabel.setTextFormat(QtCore.Qt.RichText)
        self.resultlabel.setObjectName("resultlabel")
        
        self.resultop = QtWidgets.QTextBrowser(self.centralwidget)
        self.resultop.setGeometry(QtCore.QRect(1480, 710, 311, 171))
        self.resultop.setStyleSheet("background-image: url(:/images/image3.jpg);")
        self.resultop.setObjectName("resultop")
        
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(830, 280, 256, 256))
        self.image.setStyleSheet("background-image: url(:/images/image3.jpg);")
        self.image.setText("")
        self.image.setScaledContents(True)
        self.image.setObjectName("image")
        
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(700, 20, 531, 121))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(28)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.title.setFont(font)
        self.title.setStyleSheet("background-image: url(:/images/images6.jpeg);\n"
                                 "font: 75 28pt \"Times New Roman\";")
        self.title.setTextFormat(QtCore.Qt.RichText)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")

        self.ClassificationReport = QtWidgets.QLabel(self.centralwidget)
        self.ClassificationReport.setGeometry(QtCore.QRect(1480, 660, 311, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.ClassificationReport.setFont(font)
        self.ClassificationReport.setStyleSheet("background-image: url(:/images/images6.jpeg);")
        self.ClassificationReport.setTextFormat(QtCore.Qt.RichText)
        self.ClassificationReport.setObjectName("ClassificationReport")
        
        self.Resultpred = QtWidgets.QLabel(self.centralwidget)
        self.Resultpred.setGeometry(QtCore.QRect(1080, 720, 341, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.Resultpred.setFont(font)
        self.Resultpred.setStyleSheet("background-image: url(:/images/images6.jpeg);")
        self.Resultpred.setTextFormat(QtCore.Qt.RichText)
        self.Resultpred.setObjectName("Resultpred")
        
        self.image_2 = QtWidgets.QLabel(self.centralwidget)
        self.image_2.setGeometry(QtCore.QRect(220, 650, 256, 256))
        self.image_2.setStyleSheet("background-image: url(:/images/image3.jpg);")
        self.image_2.setText("")
        self.image_2.setScaledContents(True)
        self.image_2.setObjectName("image_2")
        
        self.image_3 = QtWidgets.QLabel(self.centralwidget)
        self.image_3.setGeometry(QtCore.QRect(620, 650, 256, 256))
        self.image_3.setStyleSheet("background-image: url(:/images/image3.jpg);")
        self.image_3.setText("")
        self.image_3.setScaledContents(True)
        self.image_3.setObjectName("image_3")
        
        self.resultlabel_2 = QtWidgets.QLabel(self.centralwidget)
        self.resultlabel_2.setGeometry(QtCore.QRect(200, 600, 301, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.resultlabel_2.setFont(font)
        self.resultlabel_2.setStyleSheet("background-image: url(:/images/images6.jpeg);")
        self.resultlabel_2.setTextFormat(QtCore.Qt.RichText)
        self.resultlabel_2.setObjectName("resultlabel_2")
        
        self.resultlabel_3 = QtWidgets.QLabel(self.centralwidget)
        self.resultlabel_3.setGeometry(QtCore.QRect(620, 600, 251, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.resultlabel_3.setFont(font)
        self.resultlabel_3.setStyleSheet("background-image: url(:/images/images6.jpeg);")
        self.resultlabel_3.setTextFormat(QtCore.Qt.RichText)
        self.resultlabel_3.setObjectName("resultlabel_3")
        
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(1130, 510, 111, 21))
        self.checkBox.setStyleSheet("background-image: url(:/images/image3.jpg);")
        self.checkBox.setObjectName("checkBox")
        
        self.imagepath.raise_()
        self.detect.raise_()
        self.resultlabel.raise_()
        self.resultop.raise_()
        self.image.raise_()
        self.browse.raise_()
        self.title.raise_()
        self.ClassificationReport.raise_()
        self.Resultpred.raise_()
        self.image_2.raise_()
        self.image_3.raise_()
        self.resultlabel_2.raise_()
        self.resultlabel_3.raise_()
        self.checkBox.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Leaf Disease Detector"))
        MainWindow.setWhatsThis(_translate("MainWindow", "<html><head/><body><p><img src=\":/images/images (1).jpeg\"/></p></body></html>"))
        self.browse.setText(_translate("MainWindow", "Browse Image"))
        self.imagepath.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.detect.setText(_translate("MainWindow", "Detect Disease"))
        self.resultlabel.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; color:#c9e265;\">Result:</span></p></body></html>"))
        self.resultop.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.title.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600; color:#c9e265;\">Leaf Disease Detection</span></p></body></html>"))
        self.ClassificationReport.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; color:#c9e265;\">Classification Report</span></p></body></html>"))
        self.Resultpred.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#c9e265;\"><br/></span></p></body></html>"))
        self.resultlabel_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; color:#c9e265;\">Canny Edge Detection</span></p></body></html>"))
        self.resultlabel_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; color:#c9e265;\">Bounding Box</span></p></body></html>"))
        self.checkBox.setText(_translate("MainWindow", "Watermarked"))

    def browsefiles(self):
        global filename
        filename = QFileDialog.getOpenFileName(self, "Open File", "C:\\", "Image files (*.jpg *.png *.gif)")
        pix = QPixmap(filename[0])
        self.image.setPixmap(pix.scaled(self.image.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.imagepath.setText(filename[0])

    def detectdisease(self):
        im=Image.open(filename[0])
        im=im.resize((384,384))
        im=np.expand_dims(im,axis=0)
        im=np.array(im)
        im=im/255
        predictions = model.predict(im)
        mp = predictions[0]
        ind = np.argmax(mp)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.resultop.setFont(font)
        results={0:'Healthy', 1:'Multiple Diseases', 2:'Rust', 3:'Scab'}
        self.resultop.setText('Classification Probability:\n\nHealthy:  '+ str(mp[0])+'\nMultiple Diseases:  '+ str(mp[1])+'\nRust:  '+ str(mp[2])+'\nScab:  '+ str(mp[3]))
        font.setPointSize(24)
        self.Resultpred.setFont(font)
        self.Resultpred.setText(str(results[ind]))
        self.Resultpred.setStyleSheet("background-image: url(:/images/images6.jpeg);\n""color: #c9e265;")

        img1 = cv2.imread(filename[0]) 
        emb_img = img1.copy()
        wm = self.checkBox.isChecked()
        gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_img, 100, 180)
        
        if wm == True:
            erosion = cv2.erode(gray_img, (5,5),iterations = 70)
            edges1 = cv2.Canny(erosion, 100, 200)
        else:
            edges1 = edges

        edge_coors = []
        for i in range(edges1.shape[0]):
            for j in range(edges1.shape[1]):
                if edges1[i][j] != 0:
                    edge_coors.append((i, j))
    
        row_min = edge_coors[np.argsort([coor[0] for coor in edge_coors])[0]][0]
        row_max = edge_coors[np.argsort([coor[0] for coor in edge_coors])[-1]][0]
        col_min = edge_coors[np.argsort([coor[1] for coor in edge_coors])[0]][1]
        col_max = edge_coors[np.argsort([coor[1] for coor in edge_coors])[-1]][1]
        new_img = img1[row_min:row_max, col_min:col_max]
    
        emb_img[row_min:row_min+10, col_min:col_max] = [0, 0, 255]
        emb_img[row_max-10:row_max, col_min:col_max] = [0, 0, 255]
        emb_img[row_min:row_max, col_min:col_min+10] = [0, 0, 255]
        emb_img[row_min:row_max, col_max-10:col_max] = [0, 0, 255]

        cv2.imwrite("C:/Users/Shagun/Pictures/edges.jpg", edges)        
        cv2.imwrite("C:/Users/Shagun/Pictures/emb_img.jpg", emb_img)
        
        pix1 = QPixmap("C:/Users/Shagun/Pictures/edges.jpg")
        self.image_2.setPixmap(pix1.scaled(self.image_2.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        
        pix2 = QPixmap("C:/Users/Shagun/Pictures/emb_img.jpg")
        self.image_3.setPixmap(pix2.scaled(self.image_3.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
