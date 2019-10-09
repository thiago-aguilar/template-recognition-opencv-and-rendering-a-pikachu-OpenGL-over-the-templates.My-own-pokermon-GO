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
    gray = np.float32(gray)
    original = np.copy(frame)
    cv2.imshow('Video', frame)

    k = cv2.waitKey(5) & 0xFF   #se pressionar esc sai da rotina
    if k == 27:
        break
    if k == 32:
        time.sleep(5)
        framesCalibrar.append(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print (cap.get(cv2.CAP_PROP_POS_FRAMES))

    frames=[277, 472, 599, 753, 942, 1125, 1448, 1815]

    if contaframes in frames:
        cv2.imwrite('frame%d.jpg' % contaframes, frame)



cap.release()


print('valores: ',framesCalibrar)