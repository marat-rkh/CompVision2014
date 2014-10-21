import cv2
import numpy as np
from Tkinter import *
import os.path

globalImg = None

def showImg(saveName):
    cv2.imshow('detected', globalImg)
    k = cv2.waitKey(0) & 0xFF
    if not saveName is None:
        cv2.imwrite(saveName, globalImg)
    else:
        print("Save failed: name is not provided")
    cv2.destroyAllWindows()

def normalize(i):
    if(i.get() % 2 == 0):
        return i.get() + 1
    return i.get()

def normalizeInput():
    gKerWidth.set(normalize(gKerWidth))
    gKerHeight.set(normalize(gKerHeight))
    lKer.set(normalize(lKer))

def loadImg():
    if not os.path.exists('text.bmp'):
        print("'text.bmp' file not found")
        return
    global globalImg
    globalImg = cv2.imread('text.bmp', 0)

def detectText():
    normalizeInput()
    global globalImg
    globalImg = cv2.GaussianBlur(globalImg, (gKerWidth.get(), gKerHeight.get()), 0)
    globalImg = cv2.Laplacian(globalImg, cv2.CV_64F, ksize=lKer.get())
    save()

def doErode():
    normalizeInput()
    eKer = np.ones((eKerW.get(), eKerH.get()), np.uint8)
    global globalImg
    globalImg = cv2.erode(globalImg, eKer, iterations=1)
    save()

def doDilate():
    normalizeInput()
    dKer = np.ones((dKerW.get(), dKerH.get()), np.uint8)
    global globalImg
    globalImg = cv2.dilate(globalImg, dKer, iterations=1)
    save()

def save():
    # cv2.imwrite('detected.bmp', globalImg)
    showImg('detected.bmp')
    print("Result stored in file 'detected.bmp'")

# creating controls
master = Tk()

gKerWidth = Scale(master, from_=0, to=100, orient=HORIZONTAL, length=300, label='Gaussian kernel width')
gKerWidth.set(21)
gKerWidth.pack()

gKerHeight = Scale(master, from_=0, to=100, orient=HORIZONTAL, length=300, label='Gaussian kernel height')
gKerHeight.set(23)
gKerHeight.pack()

lKer = Scale(master, from_=0, to=100, orient=HORIZONTAL, length=300, label='Laplacian kernel size')
lKer.set(11)
lKer.pack()

Button(master, text='Detect', command=detectText).pack()

eKerW = Scale(master, from_=0, to=100, orient=HORIZONTAL, length=300, label='Erode kernel width')
eKerW.set(5)
eKerW.pack()

eKerH = Scale(master, from_=0, to=100, orient=HORIZONTAL, length=300, label='Erode kernel height')
eKerH.set(5)
eKerH.pack()

Button(master, text='Erode', command=doErode).pack()

dKerW = Scale(master, from_=0, to=100, orient=HORIZONTAL, length=300, label='Dilate kernel width')
dKerW.set(8)
dKerW.pack()

dKerH = Scale(master, from_=0, to=100, orient=HORIZONTAL, length=300, label='Dilate kernel height')
dKerH.set(7)
dKerH.pack()

Button(master, text='Dilate', command=doDilate).pack()
Button(master, text='Reset', command=loadImg).pack()

# start the program
loadImg()
mainloop()