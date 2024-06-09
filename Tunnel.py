import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2
import time
from src.draw import draw_F16, draw_walls, draw_target
from src.sensor import sensor 
from src.wall import Wall
import math 
from jax_f16.f16 import F16
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot  

# Environment 
WIDTH = 500; HEIGHT = 500; DEPTH = 10000; WALL_WIDTH = 50 
timestep = 0.01 #should be 0.01 #how long wait in between frames 
SEED = 42
# Targets 
tgtStep = 100; TARGET_POSITION_0 = [500 ,250, 250] #north (z), east (x), altitude (y)
REWARD0 = 0; HISTORY = 2 #think of this like the state history
TARGET_REWARD = [REWARD0 + i * 100 for i in range(1+math.floor((DEPTH-TARGET_POSITION_0[0])/tgtStep))]
# Aircraft
F16_RADIUS = 50 #imaginary circle around the aircraft 
F16_POSITION_0 = [0,250,250] #north (z), east (x), altitude (y)
F16_JUMP = 0.05 #pixels at each step, 0.05 gives 3 rad/s roll input to about 1/2 rotation per second 
uMin = np.array([-5, -10, -10, -100]); uMax = np.array([10, 10 , 10, 100]) #Nz, Ps, Nyr, Thr

def check_hit_wall(x_pos, y_pos):
	l = x_pos - WALL_WIDTH - F16_RADIUS < 0 
	r = WIDTH - x_pos - WALL_WIDTH - F16_RADIUS < 0 
	u = y_pos - WALL_WIDTH - F16_RADIUS < 0 
	d = HEIGHT - y_pos - WALL_WIDTH - F16_RADIUS < 0 

	if any([l,r,u,d]):
		return True
	else:
		return False 
	
def new_target(self):
	self.targetz += tgtStep
	self.targetx =  250 
	self.targety =  250 

class Tunnel(gym.Env):
	def __init__(self, gpsOut=False):
		super(Tunnel, self).__init__()
		# environment init
		self.action_space = spaces.Box(low=uMin, high=uMax, 
								 shape=(4,), dtype=np.float32)
		self.observation_space = spaces.Box(low=-max(WIDTH,HEIGHT),
								high=max(WIDTH,HEIGHT), 
								shape=(3,3,3,HISTORY), dtype=np.float32)
		self.targetz, self.targetx, self.targety = TARGET_POSITION_0 #z, x, y 
		self.targets = 0 #number of targets hit
		self.score = REWARD0
		self.wall = Wall(WIDTH, HEIGHT, DEPTH, WALL_WIDTH)
		
		# aircraft init
		self.viper = F16() #viper is the object within jax-f16
		self.viper.sensor = sensor()
		self.viper.NX , self.viper.NU = self.viper.trim_state(), self.viper.trim_control() #trim out all angles (may not need)
		# NX[9] = north (z), NX[10] = east (x), NX[11] = altitude (y)
		self.viper.NX[9], self.viper.NX[10], self.viper.NX[11]  = F16_POSITION_0 
		#autopilot init (not required)
		self.viper.wpt = [(250, DEPTH, 250)] #Note: waypoints in y, z, x NOT as in aircraft
		self.viper.autopilot = WaypointAutopilot(self.viper.wpt)
		self.viper.autopilot.waypoint_index = 0

	def step(self, action):
		#Display F16
		cv2.imshow('yz (rearview)',self.img)
		cv2.waitKey(1)
		self.img = np.zeros((HEIGHT,int(WIDTH+WIDTH*WIDTH/DEPTH),3),dtype='uint8')

		draw_target(self.img, self.targetx, self.targety, self.targetz, WIDTH,  DEPTH)
		
		sensor.draw_sensor(self.img, self.viper.NX[10], self.viper.NX[11], self.viper.NX[9],
			  				self.viper.NX[5], self.viper.NX[4], self.viper.NX[3],
			  				self.viper.sensor.subsensors,self.viper.sensor.range[1], self.wall)
		
		draw_F16(self.img,self.viper, WIDTH, HEIGHT, DEPTH, F16_RADIUS)
		draw_walls(self.img, WALL_WIDTH, WIDTH, HEIGHT, DEPTH)
		
		#execute the waiting
		t_end = time.time() + timestep
		k = -1
		while time.time() < t_end:
			if k == -1:
				k = cv2.waitKey(1)
			else:
				continue
		
		# action is the control input
		# for RL control
		self.viper.NU = action 
		# for autopilot control 
		#self.viper.NU = np.array(self.viper.autopilot.get_u_ref(self.viper.autopilot, self.viper.NX))
		# Move F16 in environment 
		xdot = self.viper.xdot(self.viper.NX,self.viper.NU)
		xdot = np.array(xdot)
		self.viper.NX = self.viper.NX + xdot*F16_JUMP

		# check if end of tunnel 
		if(self.viper.NX[9]>DEPTH): #gets to the end of the tunnel
			self.done = True
		# check if hit wall
		if check_hit_wall(self.viper.NX[10], self.viper.NX[11]):
			#print('YOU HIT THE WALL')
			self.done = True
		# check if hit target
		if self.viper.NX[9]>self.targetz:
			self.targets+=1
			new_target(self)

		echomap = sensor.subs2echomap(self.viper.sensor, self.viper,
								self.viper.sensor.subsensors,self.wall)
		state_new = np.concatenate((echomap, self.viper.NX)).reshape(-1,1)
		self.score = np.sqrt((self.viper.NX[10]-self.targetx)**2 + (self.viper.NX[11]-self.targety)**2)
		return	state_new, self.score, self.done, False, {}
	
	def reset(self, seed=SEED):
		# reset environment
		self.img = np.zeros((WIDTH,HEIGHT,3),dtype='uint8') 
		self.score = REWARD0 #reward
		self.targets = 0 #number of targets hit
		self.done = False #game over
		# reset aircraft
		self.viper.NX , self.viper.NU = self.viper.trim_state(), self.viper.trim_control()
		self.viper.NX[9], self.viper.NX[10], self.viper.NX[11]  = F16_POSITION_0 #set the depth = north = z
		self.targetz, self.targetx, self.targety = TARGET_POSITION_0
		#reset sensor 
		self.viper.sensor.history = np.zeros(self.viper.sensor.subsensors.shape+(HISTORY,)) #reset history
		self.viper.sensor.subsensors = sensor.build_subsensor(self.viper.sensor) #has [az, el, dist]
		echomap = sensor.subs2echomap(self.viper.sensor, self.viper,
								self.viper.sensor.subsensors, self.wall)
		state_new = np.concatenate((echomap, self.viper.NX)).reshape(-1,1)
		return state_new, {}

if __name__ == "__main__":
	env = Tunnel()
	episodes = 600
	for episode in range(episodes):
		done = False
		obs = env.reset()
		#actions: [Nz (+aft stick), Ps (+right stick), Nyr (+right rudder), Thr (throttle)]
		#action = env.action_space.sample() #if you'd like random actions
		action = np.array([0.01,0,0,0]) 
		while True:
			obs, reward, done, conc, info = env.step(action)
			if done:
				break








