import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import numpy as np
import time

def compara1(imageA, imageB):
    # implementando metodo meanSquaredError
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err



def retorna_final(homografias,original,final):
# HOMOGRAFIA------------------------------------------
    for i in homografias:
        # value1=i
        # aux1=np.array([[i[0][1],i[0][0]],[i[1][1],i[1][0]],[i[2][1],i[2][0]],[i[3][1],i[3][0]]])
        print(i)
        aux1 = np.array([[i[0][1], i[0][0]], [i[3][1], i[3][0]], [i[2][1], i[2][0]], [i[1][1], i[1][0]]])
        aux2 = np.array([[0, 0], [0, tamXimg], [tamYimg, tamXimg], [tamYimg, 0]])
        h, status = cv2.findHomography(aux1, aux2)
        im_dst = cv2.warpPerspective(original, h, (tamYimg, tamXimg))
        cv2.imshow('HOMOGRAFIA', im_dst)

        if METODO == 0:
            erro1 = compara1(im_dst, alvo1)
            erro2 = compara1(im_dst, alvo2)
            erro3 = compara1(im_dst, alvo3)
            erro4 = compara1(im_dst, alvo4)

            if erro1 <= 12000:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (255, 0, 0), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (0, 255, 0), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (0, 0, 255), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (255, 255, 0), 2)
            if erro2 <= 12000:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (255, 255, 0), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (255, 0, 0), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (0, 255, 0), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (0, 0, 255), 2)
            if erro3 <= 12000:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (0, 0, 255), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (255, 255, 0), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (255, 0, 0), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (0, 255, 0), 2)
            if erro4 <= 12000:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (0, 255, 0), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (0, 0, 255), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (255, 255, 0), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (255, 0, 0), 2)
        else:
            erro1 = compara2(im_dst, alvo1)
            erro2 = compara2(im_dst, alvo2)
            erro3 = compara2(im_dst, alvo3)
            erro4 = compara2(im_dst, alvo4)

            if erro1 >= 0.75:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (255, 0, 0), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (0, 255, 0), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (0, 0, 255), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (255, 255, 0), 2)
            if erro2 >= 0.75:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (255, 255, 0), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (255, 0, 0), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (0, 255, 0), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (0, 0, 255), 2)
            if erro3 >= 0.75:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (0, 0, 255), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (255, 255, 0), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (255, 0, 0), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (0, 255, 0), 2)
            if erro4 >= 0.75:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (0, 255, 0), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (0, 0, 255), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (255, 255, 0), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (255, 0, 0), 2)
    return final



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

def find_rectangle(approx,original,homografias):
    for contour in imageContours:
        epsilon = 0.009 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if len(approx) == 4:
            cv2.putText(original, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0))
            cv2.drawContours(original, [approx], -1, (0, 255, 255), 1)
            cv2.imshow('Contornos', original)
            homografias.append(approx)
            print("pontos")
            print(approx)
    return approx,original


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
    final = np.copy(original)

    mask1= cv2.Canny(frame,100,250)
    imageContours, contours =  cv2.findContours(mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('Video', frame)


    k = cv2.waitKey(5) & 0xFF   #se pressionar esc sai da rotina

    if k == 27:
        break
    print(np.shape(imageContours))

    homografias=[]
    approx=[]
   # for i in imageContours:
   #     if cv2.isContourConvex(i):
   #         approx.append(i)
    approx, original = find_rectangle(approx,original,homografias)
    #print(np.shape(homografias))
    #print(homografias)

################# descomentar linha abaixo e consertar(mudar o formato do vetor dos pontos para homografia) ######
    #final = retorna_final(homografias, frame, final)



    print(np.shape(imageContours))
    #frame = cv2.drawContours(gray, imageContours, -1, (255, 255), 1)
    #frame = cv2.drawContours(frame, imageContours, -1, (0, 255, 0), 1)
    #cv2.imshow('Contornos', frame)
    cv2.imshow('Contornos', original)
    print(np.shape(img))

cap.release()


print('valores: ',framesCalibrar)