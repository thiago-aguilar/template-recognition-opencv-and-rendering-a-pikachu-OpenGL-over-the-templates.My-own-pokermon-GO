import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import numpy as np
import time
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL import *
from objloader import *

def espelha(i, erro1m, erro2m, erro3m, erro4m):
    if erro1m < 10500 or erro3m < 10500:
        guarda1 = i[0]
        guarda2 = i[2]
        i[0]= i[1]
        i[2]= i[3]
        i[1]=guarda1
        i[3]=guarda2
    elif erro2m < 10500 or erro4m < 10500:
        guarda1 = i[0]
        guarda2 = i[2]
        i[0]= i[3]
        i[2]= i[1]
        i[1]=guarda2
        i[3]=guarda1
    return i

def compara1(imageA, imageB):
    # implementando metodo meanSquaredError
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err

def dist_euclidiana(v1, v2):
	dim, soma = len(v1), 0
	for i in range(dim):
		soma += math.pow(v1[i] - v2[i], 2)
	return math.sqrt(soma)


def  compara2(imageA, imageB):
    #implementando metodo CCORR

    numerador = np.sum((imageA.astype("float") * imageB.astype("float")))
    denominador1 = np.sum ( imageA.astype("float")**2)
    denominador2 = np.sum ( imageB.astype("float")**2)
    denominador = math.sqrt(denominador1*denominador2)
    err=numerador/denominador
    # err = numerador
    return err



def retorna_final(homografias,gray,final):
# HOMOGRAFIA------------------------------------------
    pontosPnP = []
    for i in homografias:
        # value1=i
        # aux1=np.array([[i[0][1],i[0][0]],[i[1][1],i[1][0]],[i[2][1],i[2][0]],[i[3][1],i[3][0]]])
        # print(i)
        aux1 = np.array([[i[0][1], i[0][0]], [i[3][1], i[3][0]], [i[2][1], i[2][0]], [i[1][1], i[1][0]]])
        aux2 = np.array([[0, 0], [0, tamXalvo], [tamYalvo, tamXalvo], [tamYalvo, 0]])
        h, status = cv2.findHomography(aux1, aux2)
        im_dst = cv2.warpPerspective(gray, h, (tamYalvo, tamXalvo))
        # cv2.imshow('HOMOGRAFIA', im_dst)
        METODO=1
        if METODO == 1:
            erro1m = compara1(im_dst, alvo1m)
            erro2m = compara1(im_dst, alvo2m)
            erro3m = compara1(im_dst, alvo3m)
            erro4m = compara1(im_dst, alvo4m)
            if erro1m <= 9500 or erro2m <= 9500 or erro3m <= 9500 or erro4m <= 9500:
                i=espelha(i,erro1m,erro2m,erro3m,erro4m)
                aux1 = np.array([[i[0][1], i[0][0]], [i[3][1], i[3][0]], [i[2][1], i[2][0]], [i[1][1], i[1][0]]])
                aux2 = np.array([[0, 0], [0, tamXalvo], [tamYalvo, tamXalvo], [tamYalvo, 0]])
                h, status = cv2.findHomography(aux1, aux2)
                im_dst = cv2.warpPerspective(gray, h, (tamYalvo, tamXalvo))
            erro1 = compara1(im_dst, alvo1)
            erro2 = compara1(im_dst, alvo2)
            erro3 = compara1(im_dst, alvo3)
            erro4 = compara1(im_dst, alvo4)

                # i=espelha(i)


            if erro1 <= 9500:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (255, 0, 0), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (0, 255, 0), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (0, 0, 255), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (255, 255, 0), 2)
                pontosPnP.append([[i[2][1], i[2][0]],[i[1][1], i[1][0]],[i[0][1],i[0][0]],[i[3][1], i[3][0]]])
            if erro2 <= 10500:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (255, 255, 0), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (255, 0, 0), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (0, 255, 0), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (0, 0, 255), 2)
                pontosPnP.append([[i[1][1], i[1][0]],[i[0][1],i[0][0]],[i[3][1], i[3][0]],[i[2][1], i[2][0]]])
            if erro3 <= 10500:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (0, 0, 255), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (255, 255, 0), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (255, 0, 0), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (0, 255, 0), 2)
                pontosPnP.append([[i[0][1],i[0][0]],[i[3][1], i[3][0]],[i[2][1], i[2][0]],[i[1][1], i[1][0]]])
            if erro4 <= 10500:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (0, 255, 0), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (0, 0, 255), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (255, 255, 0), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (255, 0, 0), 2)
                pontosPnP.append([[i[3][1], i[3][0]],[i[2][1], i[2][0]],[i[1][1], i[1][0]],[i[0][1],i[0][0]]])
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
    return final, pontosPnP



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
    pontos=[]
    for contour in imageContours:
        epsilon = 0.009 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if len(approx) == 4:
            cv2.putText(original, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0))
            cv2.drawContours(original, [approx], -1, (0, 255, 255), 1)
            # cv2.imshow('Contornos', original)
            pontos.append([approx[0][0],approx[1][0],approx[2][0],approx[3][0]])
            # print("pontos")
            # print(approx)
    return approx,original, pontos



def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def normalize_rows(x: np.ndarray):
    """
    function that normalizes each row of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by row) numpy matrix.
    """
    return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)

def functionPnP(pontospnp, objectPoints, cameraMatrix, distCoeffs):
    matrizRotacao = []
    matrizTanslacao = []
    m=[]
    for i in pontospnp:
        imagePoints = np.asarray([i[0],i[1],i[2],i[3]])
        ret, rvec, tvec, = cv2.solvePnP(objectPoints.astype(float), imagePoints.astype(float), cameraMatrix, distCoeffs)

        # tvec[2]=tvec[2]*-1
        # axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)
        # imgpts, jac = cv2.projectPoints(axis, rvec, tvec , cameraMatrix, distCoeffs)
        # imgpts=imgpts.astype(int)
        # img = draw(frame,imagePoints,imgpts)
        # cv2.imshow('img',img)
        tvec=tvec/math.sqrt((cameraMatrix[0][0]*cameraMatrix[0][0]+cameraMatrix[1][1]*cameraMatrix[1][1])*0.05)
        rotm=cv2.Rodrigues(rvec)[0]
        matrizRotacao.append(rotm)
        matrizTanslacao.append(tvec)
        # matrizAuxiliar = np.transpose(np.array([[rotm[0][0],rotm[0][1],rotm[0][2],tvec[0]],
        #             [rotm[1][0],rotm[1][1],rotm[1][2],tvec[1]],
        #             [rotm[2][0],rotm[2][1],rotm[2][2],tvec[2]],
        #             [       0.0,       0.0,       0.0,   1.0]]))
        matrizAuxiliar = np.array([[rotm[0][0], rotm[0][1], rotm[0][2], tvec[0]],
                                                [rotm[1][0], rotm[1][1], rotm[1][2], tvec[1]],
                                                [rotm[2][0], rotm[2][1], rotm[2][2], tvec[2]],
                                                [0.0, 0.0, 0.0, 1.0]])

        # matrizAuxiliar = np.array([[1.0, 0.0, 0.0, -20.0],
        #                            [0.0, 1.0, 0.0, -30.0],
        #                            [0.0, 0.0, 1.0, -50.0],
        #                            [0.0, 0.0, 0.0, 1.0]])

        matrizAuxiliar = matrizAuxiliar*np.array(
            [[1.0, 1.0, 1.0, 1.0],
             [-1.0, -1.0, -1.0, -1.0],
             [-1.0, -1.0, -1.0, -1.0],
             [1.0, 1.0, 1.0, 1.0]]
        )
        m.append(matrizAuxiliar)

    return m, matrizRotacao, matrizTanslacao


def initOpenGL(cameraMatrix, dimensions):

    (tamX, tamY) = dimensions

    glClearColor(0.0,0.0,0.0,0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    # glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    # lightAmbient  = [1.0, 1.0, 1.0, 1.0]
    # lightDiffuse  = [1.0, 1.0, 1.0, 1.0]
    # lightPosition = [1.0, 1.0, 1.0, 1.0]
    # glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient)
    # glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse)
    # glLightfv(GL_LIGHT0, GL_POSITION, lightPosition)
    # glEnable(GL_LIGHT0)
    # glEnable(GL_LIGHTING)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    Fx= cameraMatrix[0,0]
    Fy= cameraMatrix[1,1]
    FieldOfView= 2*np.arctan(0.5*tamY/Fy)*180/np.pi
    aspect= (tamX*Fy)/(tamY*Fx)
    gluPerspective(FieldOfView, aspect, 0.1, 300.0)
    # gluLookAt(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    glMatrixMode(GL_MODELVIEW)

    obj= OBJ("Pikachu.obj", swapyz=True)
    # print('Ok')
    glEnable(GL_TEXTURE_2D)
    background_id= glGenTextures(1)
    return obj, background_id


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
cameraMatrix= np.array([[568.45087,0,312.86352],[0,563.55473,207.41694],[0,0,1]])
distCoeffs=np.array([ 0.10977 ,  -0.29957 ,  -0.00915  , -0.00096 , 0.00000 ])


#---------------------------MAIN CODE-----------------------------------------------------------------------------------

img = cv2.imread('alvo.jpg', 0) #Lê alvo
crop_img = img[:, 3:-4]  #transforma alvo em figura quadrada

METODO=0
#cria 4 alvos para comparar no futuro
alvo1=np.copy(crop_img)
alvo2= rotateImage(alvo1,90)
alvo3= rotateImage(alvo2,90)
alvo4= rotateImage(alvo3,90)

alvo1m=cv2.flip(alvo1,0)
alvo2m= rotateImage(alvo1m,90)
alvo3m= rotateImage(alvo2m,90)
alvo4m= rotateImage(alvo3m,90)
# cv2.imshow('1', alvo1)
# cv2.imshow('2', alvo2)
# cv2.imshow('3', alvo3)
# cv2.imshow('4', alvo4)
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
# framesCalibrar=[]

contaRoda=0
glutInit()
glutInitWindowSize(cols, rows)
glutCreateWindow("3D")

glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
glClearColor(0.0, 0.0, 0.0, 0.0)
obj, background_id= initOpenGL(cameraMatrix,(cols,rows))
contapnpsAtual=[]
guardapnps=[]
contaframes=0
intervaloFrame=10
while (1):
    contaframes+=1
    # reads frames from a camera
    ret, frame = cap.read()
    if not ret: break  #se chegar no final sai da rotina
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = np.float32(gray)
    original = np.copy(frame)
    final = np.copy(original)

    mask1= cv2.Canny(frame,150,200)
    imageContours, contours =  cv2.findContours(mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('Video', frame)


    k = cv2.waitKey(5) & 0xFF   #se pressionar esc sai da rotina

    if k == 27:
        break
    # print(np.shape(imageContours))

    homografias=[]
    approx=[]
   # for i in imageContours:
   #     if cv2.isContourConvex(i):
   #         approx.append(i)
    approx, original ,quinas= find_rectangle(approx,original,homografias)
    #print(np.shape(homografias))
    #print(homografias)

################# descomentar linha abaixo e consertar(mudar o formato do vetor dos pontos para homografia) ######
    # conta=0
    # homografias=[]
    # quinas = []
    # for i in approx:
    #     quinas.append(i)
    #     if conta == 3:
    #         homografias.append([[quinas[0][0][0],quinas[0][0][1]],[quinas[1][0][0], quinas[1][0][1]],[quinas[2][0][0], quinas[2][0][1]],[quinas[3][0][0], quinas[3][0][1]]])
    #         quinas = []
    #         conta=-1
    #     conta=conta+1
    for i in quinas:
        for j in i:
            guarda=j[1]
            j[1]=j[0]
            j[0]=guarda

    final, pontospnp = retorna_final(quinas, gray, final)

    # objectPoints=np.array([[+tamXalvo,+tamYalvo,0],[-tamYalvo/2,-tamXalvo/2,0],[-tamYalvo/2,+tamXalvo/2,0],,[+tamXalvo,-tamYalvo,0]])
    # objectPoints = np.array( [[-tamXalvo/2, -tamYalvo/2, 0],[tamXalvo/2,-tamYalvo/2,0],[tamXalvo/2,tamYalvo/2,0],[-tamXalvo/2,tamYalvo/2,0]])
    var=tamXalvo
    objectPoints = np.array(
        [[var / 2, var/ 2, 0], [-var / 2, var / 2, 0],[-var / 2,  -var/ 2, 0],[var / 2, -var / 2, 0]])
    # objectPoints = np.array(
    #     [[0, 0, 0], [0, var, 0], [var , var , 0], [var , 0, 0]])
    matRotacao=[]

    vecTranslacao=[]

    pontospnp2=[]
    if len(pontospnp)!=0: pontospnp2.append(pontospnp[0])

    for i in pontospnp:
        for j in i:
            for k in pontospnp2:
                for y in k:
                    if dist_euclidiana(j, y) < 5: flag = 0
        if flag == 1:
            pontospnp2.append(i)
        else:
            flag = 1

    if contaframes<intervaloFrame: contapnpsAtual.append(len(pontospnp2))
    else :
        contapnpsAtual.append(len(pontospnp2))
        contapnpsAtual.pop(0)

    if len(pontospnp2)== max(contapnpsAtual):
        guardapnps=[]
        for i in pontospnp2:
            guardapnps.append(i)
    else:
        for i in guardapnps:
            if i in pontospnp2:
                a=0
            else:
                pontospnp2.append(i)

    pontospnp3 = []
    if len(pontospnp2) != 0: pontospnp3.append(pontospnp2[0])

    for i in pontospnp2:
        for j in i:
            for k in pontospnp3:
                for y in k:
                    if dist_euclidiana(j, y) < 5: flag = 0
        if flag == 1:
            pontospnp3.append(i)
        else:
            flag = 1

    M,matRotacao,vecTranslacao = functionPnP(pontospnp3, objectPoints, cameraMatrix, distCoeffs)

    # for i in

    #print(np.shape(imageContours))
    #frame = cv2.drawContours(gray, imageContours, -1, (255, 255), 1)
    #frame = cv2.drawContours(frame, imageContours, -1, (0, 255, 0), 1)
    # cv2.imshow('Contornos1', frame)
    # cv2.imshow('Contornos', final)
    # cv2.imshow('Contornos2', original)
    # print(np.shape(img))

    #-------------INICIO-DO-OPENGL---------------------

    background = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    background = cv2.flip(background,0)
    # cv2.imshow('Contornos2', background)
    height, width, channels = background.shape
    background= np.fromstring(background.tostring(), background.dtype, height * width * channels)
    background.shape = (height, width, channels)


    glEnable(GL_TEXTURE_2D)

    glBindTexture(GL_TEXTURE_2D, background_id)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, background)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # glActiveTexture(GL_TEXTURE0)

    glGenerateMipmap(GL_TEXTURE_2D)


    # glDepthMask(GL_FALSE)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, width, 0, height)

    glMatrixMode(GL_MODELVIEW)

    glBindTexture(GL_TEXTURE_2D, background_id)

    glPushMatrix()
    glBegin(GL_QUADS)
    glTexCoord2i(0,0)
    glVertex2i(0,0)
    glTexCoord2i(1,0)
    glVertex2i(width,0)
    glTexCoord2i(1,1)
    glVertex2i(width, height)
    glTexCoord2i(0,1)
    glVertex2i(0,height)
    glEnd()
    glFlush()
    glutSwapBuffers()

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glDepthMask(GL_TRUE)
    glDisable(GL_TEXTURE_2D)


    for i in M:
        # glBegin()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glLoadMatrixf(np.transpose(i))
        mult = 5
        # glTranslate(aux1*mult, -aux2*mult, -aux3*mult)
        # glRotatef(5, 60, 0, 1)
        # glRotatef(10*contaRoda, 90, 30, 0)
        contaRoda = contaRoda + 2
        glCallList(obj.gl_list)
        # glEnd()
        glFlush()
        glutSwapBuffers()
        glPopMatrix()


    # glutSwapBuffers()

    # for i in M:
    #     glMatrixMode(GL_MODELVIEW)
    #     glLoadIdentity()
    #     glLoadMatrixf(i)





cap.release()

