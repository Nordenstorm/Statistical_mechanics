import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm



vmax=2

p=0.1

period=3

cars=3

def car_speed(x,v,d):#Function to calculate car x and car v


	if vmax>v:

		v=v+1

	if d<=v:

		v=d-1

	if v>0:

		if random.uniform(0,1)>1-p:

			v=v-1

	if v<0:

		v=0

	x=x+v

	return x,v

def main(time=50):

	"""Here i just define a couple of list to keep track of things in a easy way"""
	car_x=[]# Car postistion 
	car_v=[]# Car speed
	CAR=[]# Store all car posistions throgh time for plots
	TIME=[]# Store time

	for j in range(0,cars):#Add cars and make list for car posistion and speed

		CAR.append([])
		car_x.append(j)
		car_v.append(0)


	for i in range (0,time):# MAIN LOOP to make the cars move

		TIME.append(i)#Save time

		for j in range(0,cars):# First step find distance to next car to store for future update

			if j<cars-1:

				d=(car_x[j+1]-car_x[j]+period)%period#Finds the distance to the car in front

			else:

				d=(car_x[j]-car_x[0]+period)%period#Finds the distance to the car in front


			car_x[j],car_v[j]=car_speed(car_x[j],car_v[j],d)#Update speed

			if car_x[j]>=period: #If car at reached end of the road

				car_x[j]=car_x[j]-period

			CAR[j].append(car_x[j])#APPEND NEW DATA

			print(car_v)

	return CAR,TIME

def plot(CAR,time):

	plt.figure()
	plt.xlabel('Position on road')
	plt.ylabel('Time')

	for i in range(0,len(CAR)):

		plt.plot(CAR[i],time,linestyle = '--',marker='D')
	plt.title("Road plot ")
	plt.tight_layout()  # adapt the plot area tot the text with larger fonts 
	plt.show()

CAR,time=main(50)



plot(CAR,time)



