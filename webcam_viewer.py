from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox, 
                             QMenu, QPushButton, QRadioButton, QVBoxLayout, QWidget, QSlider, QFileDialog)
from PyQt5.QtGui import QPixmap
import sys

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QSettings
import numpy as np



def my_brightness_contrast_LUT(low_val, high_val,gamma=1): 

    _chan = np.arange(256)
    img_map = np.dstack((_chan,_chan,_chan))
    
    if not isinstance(low_val,np.ndarray):
        inBlack  = low_val*np.array([1.0, 1.0, 1.0], dtype=np.float32)
        inWhite  = high_val*np.array([1.0, 1.0, 1.0], dtype=np.float32)
    else:
        inBlack  = low_val.astype('float32')
        inWhite  = high_val.astype('float32')
    
    inGamma  = gamma*np.array([1.0, 1.0, 1.0], dtype=np.float32)
    outBlack = np.array([0, 0, 0], dtype=np.float32)
    outWhite = np.array([255, 255, 255], dtype=np.float32)

    img_map = np.clip( (img_map - inBlack) / (inWhite - inBlack), 0, 255 )                            
    img_map = ( img_map ** (1/inGamma) ) *  (outWhite - outBlack) + outBlack
    img_map = np.clip( img_map, 0, 255).astype(np.uint8)
    
    return img_map


settings = QSettings('./parameters.ini', QSettings.IniFormat)
frame_width = settings.value('CAMERA/frame_width', 1920)
frame_height = settings.value('CAMERA/frame_height', 1080)
camera_num = settings.value('CAMERA/camera_num', 0)
default_file = settings.value('FILE/default_file','./image_out')

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        
        self.lut = my_brightness_contrast_LUT(0,255)
        self.fnum = 0
        self.fname = default_file
        
        self.cap = cv2.VideoCapture(camera_num,cv2.CAP_MSMF);
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        
    def run(self):
        # capture from web cam
        while self._run_flag:
            ret, frame = self.cap.read()
            frame_adj = cv2.LUT(frame,self.lut)
            if ret:
                self.change_pixmap_signal.emit(frame_adj.astype('uint8'))
        # shut down capture system
        self.cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

    @pyqtSlot(np.ndarray)
    def update_LUT(self, inputs):
        if len(inputs) == 2:
            low_val,high_val = inputs[0],inputs[1]
            self.lut = my_brightness_contrast_LUT(low_val,high_val)
        if len(inputs) == 6:
            low_val = inputs[::2]
            high_val = inputs[1::2]
            self.lut = my_brightness_contrast_LUT(low_val,high_val)
    
    @pyqtSlot(str)
    def update_output_file(self, fname):
        self.fname = fname
        self.fnum = 0
        
    @pyqtSlot()
    def capture_frame(self):
        ret, frame = self.cap.read()
        frame_adj = cv2.LUT(frame,self.lut)
        _output_file = self.fname+'_%04d.jpg'%self.fnum
        while os.path.isfile(_output_file):
            self.fnum+=1
            _output_file =  self.fname+'_%04d.jpg'%self.fnum
        
        cv2.imwrite(_output_file,frame_adj, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(self.fname+'_%04d_base.jpg'%self.fnum,frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        self.fnum+=1
    

class App(QWidget):
    update_LUT = pyqtSignal(np.ndarray)
    capture_frame = pyqtSignal()
    update_output_file = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Viewer")
        

        
        self.disply_width = frame_width
        self.display_height = frame_height
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        #self.textLabel = QLabel('Webcam')

        grid = QGridLayout()
        
        self.All_max_val = 255
        self.All_min_val = 0
        self.Red_max_val = 255
        self.Red_min_val = 0
        self.Green_max_val = 255
        self.Green_min_val = 0 
        self.Blue_max_val = 255
        self.Blue_min_val = 0
        button_heights = 75
       
        
        self.All_max = self.createStandardSlider(self.All_max_val)
        self.All_min = self.createStandardSlider(self.All_min_val)
        self.Red_max = self.createStandardSlider(self.Red_max_val)
        self.Red_min = self.createStandardSlider(self.Red_min_val)
        self.Green_max = self.createStandardSlider(self.Green_max_val)
        self.Green_min = self.createStandardSlider(self.Green_min_val)
        self.Blue_max = self.createStandardSlider(self.Blue_max_val)
        self.Blue_min = self.createStandardSlider(self.Blue_min_val)
        
        self.All_group = self.createSliderGroup(self.All_max,self.All_min,"All")
        self.Red_group = self.createSliderGroup(self.Red_max,self.Red_min,"Red")
        self.Green_group = self.createSliderGroup(self.Green_max,self.Green_min,"Green")
        self.Blue_group = self.createSliderGroup(self.Blue_max,self.Blue_min,"Blue")
        
        self.Save_button = QPushButton("save image")
        self.Path_button = QPushButton("output path")
        self.Path_button.setFixedHeight(button_heights)
        self.Save_button.setFixedHeight(button_heights)
        
        
        grid.addWidget(self.All_group, 0, 0,1,1)
        grid.addWidget(self.Red_group, 0, 1,1,1)
        grid.addWidget(self.Green_group, 1, 0,1,1)
        grid.addWidget(self.Blue_group, 1, 1,1,1)
        grid.addWidget(self.Path_button,0,10,1,1)
        grid.addWidget(self.Save_button,1,10,1,1)
        
        
        self.All_max.sliderMoved.connect(lambda: self.sliderMoved("All_max"))
        self.All_min.sliderMoved.connect(lambda: self.sliderMoved("All_min"))
        self.All_max.sliderReleased.connect(lambda: self.sliderReleased("All_max"))
        self.All_min.sliderReleased.connect(lambda: self.sliderReleased("All_min"))
        
        self.Red_max.sliderMoved.connect(lambda: self.sliderMoved("Red_max"))
        self.Red_min.sliderMoved.connect(lambda: self.sliderMoved("Red_min"))
        self.Red_max.sliderReleased.connect(lambda: self.sliderReleased("Red_max"))
        self.Red_min.sliderReleased.connect(lambda: self.sliderReleased("Red_min"))

        self.Green_max.sliderMoved.connect(lambda: self.sliderMoved("Green_max"))
        self.Green_min.sliderMoved.connect(lambda: self.sliderMoved("Green_min"))
        self.Green_max.sliderReleased.connect(lambda: self.sliderReleased("Green_max"))
        self.Green_min.sliderReleased.connect(lambda: self.sliderReleased("Green_min"))
        
        self.Blue_max.sliderMoved.connect(lambda: self.sliderMoved("Blue_max"))
        self.Blue_min.sliderMoved.connect(lambda: self.sliderMoved("Blue_min"))
        self.Blue_max.sliderReleased.connect(lambda: self.sliderReleased("Blue_max"))
        self.Blue_min.sliderReleased.connect(lambda: self.sliderReleased("Blue_min"))
        
        self.Path_button.pressed.connect(self.pathPressed)
        self.Save_button.pressed.connect(self.capture_frame)

        
        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        #vbox.addWidget(self.textLabel)
        vbox.addLayout(grid)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)
        
        
        # create the video capture thread
        self.thread = VideoThread()
        #
        self.update_LUT.connect(self.thread.update_LUT)
        self.capture_frame.connect(self.thread.capture_frame)
        self.update_output_file.connect(self.thread.update_output_file)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()
        
    def pathPressed(self):
        file_out,_ = QFileDialog.getSaveFileName(self, 'Output file')
        print(file_out)
        self.update_output_file.emit(file_out)
    
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def sliderReleased(self,change_id):
    
        if "All" in change_id:
            self.All_max.sliderMoved.disconnect()
            self.All_min.sliderMoved.disconnect()
            self.All_max.setValue(self.All_max_val)
            self.All_min.setValue(self.All_min_val)
            self.All_max.sliderMoved.connect(lambda: self.sliderMoved("All_max"))
            self.All_min.sliderMoved.connect(lambda: self.sliderMoved("All_min"))
            
        if "Red" in change_id:
            self.Red_max.sliderMoved.disconnect()
            self.Red_min.sliderMoved.disconnect()
            self.Red_max.setValue(self.Red_max_val)
            self.Red_min.setValue(self.Red_min_val)
            self.Red_max.sliderMoved.connect(lambda: self.sliderMoved("Red_max"))
            self.Red_min.sliderMoved.connect(lambda: self.sliderMoved("Red_min"))
            
            
        if "Green" in change_id:
            self.Green_max.sliderMoved.disconnect()
            self.Green_min.sliderMoved.disconnect()
            self.Green_max.setValue(self.Green_max_val)
            self.Green_min.setValue(self.Green_min_val)
            self.Green_max.sliderMoved.connect(lambda: self.sliderMoved("Green_max"))
            self.Green_min.sliderMoved.connect(lambda: self.sliderMoved("Green_min"))
            
        if "Blue" in change_id:
            self.Blue_max.sliderMoved.disconnect()
            self.Blue_min.sliderMoved.disconnect()
            self.Blue_max.setValue(self.Blue_max_val)
            self.Blue_min.setValue(self.Blue_min_val)
            self.Blue_max.sliderMoved.connect(lambda: self.sliderMoved("Blue_max"))
            self.Blue_min.sliderMoved.connect(lambda: self.sliderMoved("Blue_min"))
        
    def sliderMoved(self,change_id):
        
        All_max=self.All_max.value()
        All_min=self.All_min.value()
        Red_max=self.Red_max.value()
        Red_min=self.Red_min.value()
        Green_max=self.Green_max.value()
        Green_min=self.Green_min.value()    
        Blue_max=self.Blue_max.value()
        Blue_min=self.Blue_min.value()            
        
        if change_id == "All_max":
            
            if All_max <= All_min:
                All_max=All_min+1
                self.All_max.setValue(All_max)

            self.Red_max.setValue(All_max)
            self.Red_max_val=All_max
            self.Green_max.setValue(All_max)
            self.Green_max_val=All_max
            self.Blue_max.setValue(All_max)
            self.Blue_max_val=All_max
            
        if change_id == "All_min":
            
            if All_min >= All_max:
                All_min=All_max-1
                self.All_min.setValue(All_min)
            
            self.Red_min.setValue(All_min)
            self.Red_min_val=All_min
            self.Green_min.setValue(All_min)
            self.Green_min_val=All_min
            self.Blue_min.setValue(All_min)
            self.Blue_min_val=All_min
            
        if change_id == "Red_max":
            
            if Red_max <= Red_min:
                Red_max=Red_min+1
                self.Red_max.setValue(Red_max)
                
        if change_id == "Red_min":
            
            if Red_min >= Red_max:
                Red_min=Red_max-1
                self.Red_min.setValue(Red_min)
                        
        if change_id == "Green_max":
            
            if Green_max <= Green_min:
                Green_max=Green_min+1
                self.Green_max.setValue(Green_max)
                
        if change_id == "Green_min":
            
            if Green_min >= Green_max:
                Green_min=Green_max-1
                self.Green_min.setValue(Green_min)
        
        if change_id == "Blue_max":
            
            if Blue_max <= Blue_min:
                Blue_max=Blue_min+1
                self.Blue_max.setValue(Blue_max)
                
        if change_id == "Blue_min":
            
            if Blue_min >= Blue_max:
                Blue_min=Blue_max-1
                self.Blue_min.setValue(Blue_min)
                
        self.All_max_val=All_max
        self.All_min_val=All_min
        self.Red_max_val=Red_max
        self.Red_min_val=Red_min
        self.Green_max_val=Green_max
        self.Green_min_val=Green_min
        self.Blue_max_val=Blue_max
        self.Blue_min_val=Blue_min

        self.update_LUT.emit(np.array([self.Red_min_val,self.Red_max_val,
                                        self.Green_min_val,self.Green_max_val,
                                        self.Blue_min_val,self.Blue_max_val]))
        
    def createStandardSlider(self,value):
        slider = QSlider(Qt.Horizontal)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setMinimum(0);
        slider.setMaximum(255);
        slider.setValue(value)
        slider.setTickInterval(50)
        slider.setSingleStep(1)
        return slider
        
    def createSliderGroup(self, slider1,slider2, box_name="Slider Example"):
        groupBox = QGroupBox(box_name)

        vbox = QVBoxLayout()
        vbox.addWidget(slider1)
        vbox.addWidget(slider2)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox
        
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())