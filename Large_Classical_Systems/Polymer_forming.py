import random
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

def polymer_avoid(length):

	#First define list with cordinates, We also define a inital place for the polymer to begin

	X=[0]
	Y=[0]


	#Second run a code that add polymers

	G=0

	while len(X)!=length:

		reset=False

		Blocked_by_polymer=False

		x=X[-1]

		y=Y[-1]

		direction=random.random()*4

		direction=int(direction)

		if direction==0:

			x=x+1

		elif direction==1:

			x=x-1

		elif direction==2:

			y=y+1

		elif direction==3:

			y=y-1

		for i in range(0,len(X)):

			if x==X[i] and y==Y[i]:

				Blocked_by_polymer=True

				reset=True

		if reset==True:

			X=[0]
			Y=[0]

		if Blocked_by_polymer==False:

			X.append(x)

			Y.append(y)

			G=0

	return X,Y

def polymer_avoid_better(length):

	#First define list with cordinates, We also define a inital place for the polymer to begin

	X=[0]
	Y=[0]


	#Second run a code that add polymers

	G=0

	while len(X)!=length:

		reset=False

		Blocked_by_polymer=False

		x=X[-1]

		y=Y[-1]

		direction=random.random()*4

		direction=int(direction)

		if len(X)>1:

			Run=1

			while x==X[-2] and y==Y[-2] or Run==1:

				Run=0

				if direction==0:

					x=x+1

				elif direction==1:

					x=x-1

				elif direction==2:

					y=y+1

				elif direction==3:

					y=y-1

		else:

				if direction==0:

					x=x+1

				elif direction==1:

					x=x-1

				elif direction==2:

					y=y+1

				elif direction==3:

					y=y-1

		for i in range(0,len(X)):

			if x==X[i] and y==Y[i]:

				Blocked_by_polymer=True

				reset=True

		if reset==True:

			X=[0]
			Y=[0]

		if Blocked_by_polymer==False:

			X.append(x)

			Y.append(y)

			G=0

	return X,Y


def plot_polymer(X,Y):

	c = np.linspace(0, 1, len(X))

	plt.figure()
	plt.xlabel('x')
	plt.ylabel('y')

	plt.plot(X, Y,linestyle='-')

	plt.scatter(X, Y, c=c,marker='s')

	plt.colorbar()
	plt.title("Polymer shape of size "+str(len(X)))

    
	plt.tight_layout()  # adapt the plot area tot the text with larger fonts 
	plt.show()


X,Y=polymer_avoid_better(30)


plot_polymer(X,Y)




