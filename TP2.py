import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import numpy as np
import time

def rotateImage(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg

#----------------parametros de calibração obtidos com o toolkit:---------------------------------------------------------
# Calibration results (with uncertainties):
#
# Focal Length:          fc = [ 568.45087   563.55473 ] ± [ 22.45520   20.58540 ]
# Principal point:       cc = [ 312.86352   207.41694 ] ± [ 8.63367   17.37837 ]
# Skew:             alpha_c = [ 0.00000 ] ± [ 0.00000  ]   => angle of pixel axes = 90.00000 ± 0.00000 degrees
# Distortion:            kc = [ 0.10977   -0.29957   -0.00915   -0.00096  0.00000 ] ± [ 0.02947   0.10084   0.00685   0.00569  0.00000 ]
# Pixel error:          err = [ 0.23016   0.28244 ]
#
# Note: The numerical errors are approximately three times the standard deviations (for reference).
#------------------------------------------------------------------------------------------------------------------------



#---------------------------MAIN CODE-----------------------------------------------------------------------------------

img = cv2.imread('alvo.jpg', 0) #Lê alvo
crop_img = img[:, 3:-4]  #transforma alvo em figura quadrada


#cria 4 alvos para comparar no futuro
alvo1=np.copy(crop_img)
alvo2= rotateImage(alvo1,90)
alvo3= rotateImage(alvo2,90)
alvo4= rotateImage(alvo3,90)

#captura video
cap = cv2.VideoCapture('entrada.avi')

#captura dimensões do video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


rows = frame_height  # salva em variável rows numero de linhas
cols = frame_width   # salva em variável cols numero de colunas

tamXalvo = alvo1.shape[1]   #salva tamanho do alvo em X
tamYalvo = alvo1.shape[0]   #salva tamanho do alvo em Y
contaframes=0
framesCalibrar=[]

while (1):
    contaframes+=1
    # reads frames from a camera
    ret, frame = cap.read()
    if not ret: break  #se chegar no final sai da rotina
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = np.float32(gray)
    original = np.copy(frame)
    mask1= cv2.Canny(frame,100,250)
    imageContours, contours =  cv2.findContours(mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('Video', frame)


    k = cv2.waitKey(5) & 0xFF   #se pressionar esc sai da rotina
    if k == 27:
        break

    approx=[]
    for i in imageContours:
        if cv2.isContourConvex(i):
            approx.append(i)

    frame = cv2.drawContours(frame, imageContours, -1, (0, 255, 0), 1)
    cv2.imshow('Contornos', frame)

cap.release()


print('valores: ',framesCalibrar)