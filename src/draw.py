import cv2 
import math


# Colors are in BGR
WHITE = (255, 255, 255); BLACK = (0, 0, 0)
BLUE = (255, 0, 0); RED = (0, 0, 255); GREEN = (0, 255, 0); GREY = (128, 128, 128)
#Aircraft
AIRFOIL_THICKNESS = 5
rFus = 10 #radius of fuselage 
tRat = 0.5 #tail to aircraft ratio

def draw_F16(screen, viper, WIDTH, HEIGHT, DEPTH, F16_RADIUS):
    #draw the rearview 
    x = viper.NX[10] #width, x, east
    y = HEIGHT-viper.NX[11] #height, y, up
    z = viper.NX[9]
    psi = viper.NX[5]; theta = viper.NX[4]; phi = viper.NX[3]
    #build from nose to tail
    #canopy
    cv2.circle(screen, (int(x+rFus*math.sin(psi)), int(y-rFus*math.sin(theta))), int(F16_RADIUS/4), WHITE, -1) 
    #wings
    cv2.line(screen, (int(x),int(y)), (int(x+F16_RADIUS*math.cos(phi)),int(y+F16_RADIUS*math.sin(phi))), GREY, AIRFOIL_THICKNESS) 
    cv2.line(screen, (int(x),int(y)), (int(x-F16_RADIUS*math.cos(phi)),int(y-F16_RADIUS*math.sin(phi))), GREY, AIRFOIL_THICKNESS)
    #tail
    rTail = F16_RADIUS*tRat
    cv2.line(screen, (int(x),int(y)), (int(x+rTail*math.sin(phi)),int(y-rTail*math.cos(phi))), GREY, AIRFOIL_THICKNESS)
    #exhaust
    cv2.circle(screen, (int(x),int(y)), int(F16_RADIUS/4), RED, -1) 
    #boundary
    cv2.circle(screen, (int(x),int(y)), int(F16_RADIUS), RED, 1)   

    #draw the top view
    sRad = F16_RADIUS*WIDTH/DEPTH #scaled radius
    sX = x*WIDTH/DEPTH ; sZ = z*WIDTH/DEPTH #scaled x and z
    cv2.circle(screen, (int(WIDTH+sX), int(DEPTH*WIDTH/DEPTH-sZ)), int(sRad), RED, 1)

def draw_walls(screen, WALL_WIDTH, WIDTH, HEIGHT, DEPTH):
    #draw the rearview 
    # Top 
    cv2.rectangle(screen, (0,0), (WIDTH, WALL_WIDTH), BLUE, -1)
    # Bottom
    cv2.rectangle(screen, (0,HEIGHT-WALL_WIDTH), (WIDTH, HEIGHT), BLUE, -1)
    # Left
    cv2.rectangle(screen, (0,0), (WALL_WIDTH, HEIGHT), BLUE, -1)
    # Right
    cv2.rectangle(screen, (WIDTH-WALL_WIDTH,0), (WIDTH, HEIGHT), BLUE, -1)
    #draw the top view
    #divider
    cv2.line(screen, (WIDTH,0), (WIDTH,HEIGHT), BLACK, 10)
    #left guy
    sWW = WALL_WIDTH*WIDTH/DEPTH #scaled wall width
    cv2.rectangle(screen, (WIDTH,0), (int(WIDTH+sWW), HEIGHT), BLUE, -1)
    #right guy
    cv2.rectangle(screen, (int(WIDTH+WIDTH*WIDTH/DEPTH-sWW),0),(int(WIDTH+WIDTH*WIDTH/DEPTH),HEIGHT), BLUE, -1)

    return screen

def draw_target(screen, target_x, target_y, target_z, WIDTH,  DEPTH):
    #draw the rearview 
    cv2.circle(screen, (int(target_x), int(target_y)), 5, GREEN, -1)
    #draw the top view
    sX = target_x*WIDTH/DEPTH ; sZ = target_z*WIDTH/DEPTH #scaled x and z
    cv2.circle(screen, (int(WIDTH+sX), int(DEPTH*WIDTH/DEPTH-sZ)), 3, GREEN, -1)
    cv2.circle(screen, (int(WIDTH+sX), int(DEPTH*WIDTH/DEPTH-sZ)), 3, RED, 1)

    return screen


