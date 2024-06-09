import numpy as np
import math
import cv2

WHITE = (255, 255, 255); BLACK = (0, 0, 0)
BLUE = (255, 0, 0); RED = (0, 0, 255); GREEN = (0, 255, 0); GREY = (128, 128, 128)
#Aircraft
SUBSENSOR_THICKNESS = 1
HISTORY = 5 #how far back to record the sensor readings
class sensor ():
    def __init__(self,
                 range = np.array([0,500]) ,
                 azimuth = np.array([-45,45]),
                 elevation = np.array([-45,45]), #inverse depression angle, up is positive, 0 is at horizon
                  #degrees left/right nose 
                 AEstep = 45, #degrees, stedegrees of azimuth and elevation
                 ):
        super(sensor, self).__init__()
        self.range = range 
        self.azimuth = azimuth
        self.elevation = elevation
        self.AEstep = AEstep
        self.subsensors = self.build_subsensor() #has [az, el, dist]
        self.history = np.zeros(self.subsensors.shape+(HISTORY,)) #az, el, dist, time

    def build_subsensor(self):
        #create vectors along azimuth and elevation at density steps
        aDots = 1+(self.azimuth[1] - self.azimuth[0]) / self.AEstep
        eDots = 1+(self.elevation[1] - self.elevation[0]) / self.AEstep
        nodes = np.zeros((int(eDots * aDots), 3))
        nodes = nodes.reshape((int(eDots),int(aDots), 3)) #third dimension is the distance to the wall
        for a in range(self.azimuth[0],self.azimuth[1]+1,self.AEstep):
            for e in range(self.elevation[0],self.elevation[1]+1,self.AEstep):
                ai = int((a-self.azimuth[0])/self.AEstep) # scale to index
                ei = int((e-self.elevation[0])/self.AEstep) # scale to index
                nodes[ai][ei][0:2] = [a,e]
        return nodes
    
    def draw_sensor(screen, x,y,z, psi, theta, phi, subs, rho, wall):
        dy = wall.height-y #where to plot the y
        for a in range(subs.shape[0]):
            for e in range(subs.shape[1]):
                #xyz of sensor from polar coordinates
                az = math.radians(subs[a][e][0]); el = math.radians(subs[a][e][1])
                #find xyz of sensor via polarish coordinates
                x_end = x + rho * math.cos(el)*math.sin(az)
                z_end = z + rho * math.cos(el)*math.cos(az)
                y_end = y + rho * math.sin(el)
                #rotate to aircraft frame
                basis = np.array([x_end-x, y_end-y, z_end-z])
                x_r, y_r, z_r = sensor.rotate_to_body(basis.reshape(3,1), 
                                                      psi, theta, phi)
                x_r += x; y_r += y; z_r += z #to align it with the aircraft

                #get distances while it's rotated
                subs[a][e][2] = sensor.get_distances(np.array([x, y, z, x_r, y_r, z_r]),
                                                     rho,wall) 
                #plot the rear view
                dy_end = wall.height - y_r
                cv2.line(screen, (int(x),int(dy)), (min(wall.width,int(x_r)),int(dy_end)), GREEN, SUBSENSOR_THICKNESS)
                #plot the top view
                sX = x*wall.width/wall.depth ; sZ = z*wall.width/wall.depth
                sX_end = sX + (x_r-x)*wall.width/wall.depth ; sZ_end = sZ + (z_r-z)*wall.width/wall.depth
                cv2.line(screen, (int(wall.width+sX), int(wall.height-sZ)), 
                         (int(wall.width+sX_end), int(wall.height-sZ_end)), GREEN, SUBSENSOR_THICKNESS)

    def rotate_to_body(base , psi, theta, phi):
        #rotate to aircraft frame
        Rz = np.array([[math.cos(phi), math.sin(phi), 0],
                       [-math.sin(phi), math.cos(phi), 0],
                       [0,0,1]])
        #negative pitch because the Ph is up, not down as in NED 
        Ry = np.array([[math.cos(psi), 0, -math.sin(psi)], 
                       [0,1,0], 
                       [math.sin(psi), 0, math.cos(psi)]])
        Rx = np.array([[1,0,0],
                       [0, math.cos(theta), math.sin(theta)],
                       [0, -math.sin(theta), math.cos(theta)]])
        #rotate yaw > pitch > roll
        Ryaw = Ry.T @ base #rotate about negative y
        Rpitch = Rx @ Ryaw #rotate about x
        Rroll = Rz @ Rpitch #rotate about z
        return Rroll[0][0], Rroll[1][0], Rroll[2][0]

    def get_distances(sensor, rho, wall):
        #unpack the sensor
        x, y, z, x_r, y_r, z_r = sensor
        #get the distances to the walls
        #loop t from 1 to range of the laser
        for t in range(1, int(rho)):
            #find the intersection point
            #***find min, make sure within range, pass back distance, else farfield 
            this_x = x + t/rho*(x_r-x)
            this_y = y + t/rho*(y_r-y)
            this_z = z + t/rho*(z_r-z)
            #check if it crossed any boundaries and report distance if it did
            if this_x < wall.wall_width or this_x > wall.width-wall.wall_width:
                return t
            if this_y < wall.wall_width or this_y > wall.height-wall.wall_width:
                return t
            #for z, if it hits depth, return the range
            if this_z > wall.depth:
                return rho
        
        return rho #farfield

    def subs2echomap(self, viper, subs, wall):
        x = viper.NX[10]; y = viper.NX[11]; z = viper.NX[9]
        psi = viper.NX[5]; theta = viper.NX[4]; phi = viper.NX[3]
        rho = self.range[1]
        #create the echomap
        echomap = np.zeros(subs.shape[0]*subs.shape[1])
        for a in range(subs.shape[0]):
            for e in range(subs.shape[1]):
                #xyz of sensor from polar coordinates
                az = math.radians(subs[a][e][0]); el = math.radians(subs[a][e][1])
                #find xyz of sensor via polarish coordinates
                x_end = x + rho * math.cos(el)*math.sin(az)
                z_end = z + rho * math.cos(el)*math.cos(az)
                y_end = y + rho * math.sin(el)
                #rotate to aircraft frame
                basis = np.array([x_end-x, y_end-y, z_end-z])
                x_r, y_r, z_r = sensor.rotate_to_body(basis.reshape(3,1), 
                                                      psi, theta, phi)
                x_r += x; y_r += y; z_r += z #to align it with the aircraft

                echomap[a*subs.shape[1]+e] = sensor.get_distances(np.array([x, y, z, x_r, y_r, z_r]),
                                                     rho,wall) 

        return echomap

