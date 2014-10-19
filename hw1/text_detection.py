import cv2
import numpy
from Tkinter import *
import os.path

def showImg(img, saveName):
    cv2.imshow('detected', img)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('s'):
        if not saveName is None:
            cv2.imwrite(saveName, img)
        else:
            print("Save failed: name is not provided")
    cv2.destroyAllWindows()

def normalizeInput():
    if(gKerWidth.get() % 2 == 0):
        gKerWidth.set(gKerWidth.get() + 1)
    if(gKerHeight.get() % 2 == 0):
        gKerHeight.set(gKerHeight.get() + 1)
    if(lKer.get() % 2 == 0):
        lKer.set(lKer.get() + 1)

def detectText():
    normalizeInput()
    if not os.path.exists('text.bmp'):
        print("'text.bmp' file not found")
        return
    img = cv2.imread('text.bmp', 0)
    blured = cv2.GaussianBlur(img, (gKerWidth.get(), gKerHeight.get()), 0)
    detected = cv2.Laplacian(blured, cv2.CV_64F, ksize=lKer.get())
    cv2.imwrite('detected.bmp', detected)
    print("Result stored in file 'detected.bmp'")

master = Tk()

labelStrVar = StringVar()
label = Label(master, textvariable=labelStrVar, relief=RAISED, wraplength=300)
labelStrVar.set("This program detects text on the sample image. Set desired parameters and press 'Detect' " + 
                "(defaults are the best for the current task, so just press 'Detect'). Image with name " + 
                "'text.bmp' will be loaded from disk (it must be placed in the current folder), processed " + 
                "and result will be stored in the current folder")
label.pack()

gKerWidth = Scale(master, from_=0, to=100, orient=HORIZONTAL, length=300, label='Gaussian kernel width')
gKerWidth.set(39)
gKerWidth.pack()

gKerHeight = Scale(master, from_=0, to=100, orient=HORIZONTAL, length=300, label='Gaussian kernel height')
gKerHeight.set(21)
gKerHeight.pack()

lKer = Scale(master, from_=0, to=100, orient=HORIZONTAL, length=300, label='Laplacian kernel size')
lKer.set(11)
lKer.pack()

Button(master, text='Detect', command=detectText).pack()

mainloop()