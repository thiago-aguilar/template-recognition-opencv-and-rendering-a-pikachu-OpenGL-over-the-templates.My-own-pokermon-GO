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

glutInit()
img = cv2.imread('alvo.jpg',cv2.IMREAD_COLOR)
background = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dimensions= (img.shape[1],img.shape[0])

glMatrixMode(GL_PROJECTION);
glPushMatrix();
glLoadIdentity();
glOrtho(0.0, glutGet(GLUT_WINDOW_WIDTH), 0.0, glutGet(GLUT_WINDOW_HEIGHT), -1.0, 1.0);
glMatrixMode(GL_MODELVIEW);
glPushMatrix();

glLoadIdentity();
glDisable(GL_LIGHTING);

glColor3f(1, 1, 1);
glEnable(GL_TEXTURE_2D);
glBindTexture(GL_TEXTURE_2D, mark_textures[0].id);


glBegin(GL_QUADS);
glTexCoord2f(0, 0);
glVertex3f(0, 0, 0);
glTexCoord2f(0, 1);
glVertex3f(0, 100, 0);
glTexCoord2f(1, 1);
glVertex3f(100, 100, 0);
glTexCoord2f(1, 0);
glVertex3f(100, 0, 0);
glEnd();

glDisable(GL_TEXTURE_2D);
glPopMatrix();

glMatrixMode(GL_PROJECTION);
glPopMatrix();

glMatrixMode(GL_MODELVIEW);