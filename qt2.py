from PySide2.QtWidgets import  QApplication,QMainWindow,QPushButton,QPlainTextEdit
import cv2
from yolo import YOLO
from PIL import Image
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog

yolo = YOLO()
#p = psutil.Process()
def handlecalc():
    print('打开摄像头')
    videoSourceIndex = 0
    cap = cv2.VideoCapture(cv2.CAP_DSHOW + videoSourceIndex)

    fps = 0.0
    while(True):
        t1 = time.time()
    # 读取某一帧
        ref,frame=cap.read()
    # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
        frame = Image.fromarray(np.uint8(frame))

    # 进行检测
        frame = np.array(yolo.detect_image(frame))

    # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %.2f"%(fps))
        frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video",frame)


        c= cv2.waitKey(30) & 0xff
        if c==27:
            cap.release()
            break


def handlecalt():
    print('打开视频')
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    print(file_path)

    cap = cv2.VideoCapture(file_path)
    fps = 0.0
    while (True):
        t1 = time.time()
        # 读取某一帧
        ref, frame = cap.read()
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))

        # 进行检测
        frame = np.array(yolo.detect_image(frame))

        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)

        textEdit.setPlaceholderText("video")

        c = cv2.waitKey(30) & 0xff
        if c == 27:
            cap.release()
            break

'''def handlecalf():
    p.suspend()'''

def handlecald():
    exit(0)

def handlecalu(self):
    self.cap.release()
    print('关闭摄像头')
def handlecalf(self):
    self.cap.release()
    print('关闭视频')





app = QApplication([])

windows = QMainWindow()
windows.resize(800,700)
windows.move(800,610)
windows.setWindowTitle('手势识别会议系统')

textEdit =QPlainTextEdit(windows)
textEdit.setPlaceholderText("请输入会议标题")
textEdit.move(10,25)
textEdit.resize(300,35)

'''button = QPushButton('暂停',windows)
button.move(670,540)'''
#button.clicked.connect(handlecalf)

'''button = QPushButton('开始',windows)
button.move(670,510)'''
#button.clicked.connect(handlecalt)


button = QPushButton('退出',windows)
button.move(670,570)
button.clicked.connect(handlecald)

button = QPushButton('打开视频',windows)
button.move(10,110)
button.clicked.connect(handlecalt)

button = QPushButton('打开摄像头',windows)
button.move(10,190)
button.clicked.connect(handlecalc)

'''button = QPushButton('关闭摄像头',windows)
button.move(10,230)
button.clicked.connect(handlecalf)

button = QPushButton('关闭视频',windows)
button.move(10,150)
button.clicked.connect(handlecalu)'''

windows.show()

app.exec_()
