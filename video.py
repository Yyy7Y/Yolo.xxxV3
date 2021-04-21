#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
#from PySide2.QtWidgets import  QApplication,QMainWindow,QPushButton,QPlainTextEdit
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
yolo = YOLO()

#def 函数
#def handlecalc():
#capture = cv2.VideoCapture(0)

#def handlecalt():
   # info = textEdit.toPlainText()
# 调用摄像头
#capture=cv2.VideoCapture(0) # capture=cv2.VideoCapture("1.mp4")
capture=cv2.VideoCapture("V:/pycharm/pycharmproject/YOLO_gesture/yolo3-pytorch-master/img/texttt.mp4")

fps = 0.0
while(True):
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            capture.release()
            break






'''app = QApplication([])

windows = QMainWindow()
windows.resize(500,400)
windows.move(300,310)
windows.setWindowTitle('手势识别会议系统')

textEdit =QPlainTextEdit(windows)
textEdit.setPlaceholderText("请输入会议标题")
textEdit.move(10,25)
textEdit.resize(300,35)

button = QPushButton('开始',windows)
button.move(370,340)

button = QPushButton('结束',windows)
button.move(370,370)

button = QPushButton('打开视频',windows)
button.move(10,110)


button = QPushButton('打开摄像头',windows)
button.move(10,180)
#button.clicked.connect(handlecalc)

windows.show()

app.exec_()'''