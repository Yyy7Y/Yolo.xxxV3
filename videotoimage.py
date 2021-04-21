import cv2
cap = cv2.VideoCapture('V:/pycharm/pingyi/suofang1.mp4')
flag, frame = cap.read()

c=1
timeF=5 #视频帧计数间隔频率
index = 1
while True:
    if flag == False:
        break

    x ,y = frame.shape[0:2]  #height width
    frame = cv2.resize(frame, (int(y),int(x)))  #改变图片的尺寸
    cv2.imshow('video', frame)  # 以图片展示视频
    #将视频中的图片保存到本地
    if (c%timeF==0):
        cv2.imwrite('V:/pycharm/fuyangben/suofang1.%s.jpg'%index, frame)
    c = c + 1
    if ord('q') == cv2.waitKey(1):#输入‘q’键退出，每秒展示10张图片
        break
    flag, frame = cap.read()
    index += 1
#资源释放
cv2.destroyAllWindows()
cap.release()


'''
import cv2 as cv
cap=cv.VideoCapture(0)
c=1
timeF=10 #视频帧计数间隔频率
while (True):
    ret,frame=cap.read()
    # cv.imshow('frame',frame)
    if (c%timeF==0):
        cv.imshow('frame', frame)
        resize_image=cv.resize(frame,(160,160))
        cv.imwrite('name/'+str(int(c/10))+'.jpg',resize_image)#存储为图像
    c=c+1
    if c%100==0:
        break
    if cv.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()
'''''