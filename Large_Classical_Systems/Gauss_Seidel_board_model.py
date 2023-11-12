import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

L = 10

# The grid is n+1 points along x and y, including boundary points 0 and n
n = 10

# The grid spacing is L/n

# The number of iterations
nsteps = 100
# Initialize the grid to 0
v = np.zeros((n+1, n+1))


# Set the boundary conditions
for i in range(1,n):
    v[0,i] = 10
    v[n,i] = 10
    v[i,0] = 10
    v[i,n] = 10

# Set the interior points
for i in range(1,len(v)-1):
    for j in range(1,len(v[0])-1):
        v[i,j]=0
# Set center potential




fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(v, cmap=None, interpolation='nearest')
fig.colorbar(im)

Convergce=[]
# checker=1: no checkboard, checker=2: checkerboard (note: n should be even)
checker = 1

# perform one step of relaxation
def relax(n, v, checker):
    for check in range(0,checker):
        for x in range(1,n):
            for y in range(1,n):
                if (x*(n+1) + y) % checker == check:
                    if (x+y)%2==0:
                        v[x,y] = (v[x-1][y] + v[x+1][y] + v[x][y-1] + v[x][y+1])*0.25
       
        for x in range(1,n):
            for y in range(1,n):
                if (x*(n+1) + y) % checker == check:
                    if (x+y)%2==1:
                        v[x,y] = (v[x-1][y] + v[x+1][y] + v[x][y-1] + v[x][y+1])*0.25


        V_tot=810
        SUM=0

        for i in range(1,len(v)-1):

            for j in range(1,len(v[0])-1):
                SUM=SUM+v[i,j]

        print(SUM)

        Convergce.append(SUM/V_tot)

        return v
def update(step):

    global n, v, checker

    # FuncAnimation calls update several times with step=0,
    # so we needs to skip the update with step=0 to get
    # the correct number of steps 
    if step > 0:
        relax(n, v, checker)

    im.set_array(v)
    #print(im)
    return im,

# we generate nsteps+1 frames, because frame=0 is skipped (see above)
anim = animation.FuncAnimation(fig, update, frames=nsteps+1, interval=200, blit=True, repeat=False)
plt.show()

for i in range (150):

    v=relax(n, v, checker)

_1_=[]
_99_=[]
for i in range(0,len(Convergce)):
    _1_.append(1)
    _99_.append(0.99)



plt.figure()
plt.plot(Convergce)
plt.plot(_1_)
plt.plot(_99_)
plt.xlabel("Step Updates")
plt.ylabel("Current Potential/Correct potential")
plt.title("Plot of convergent behaviour : Boundary 10-10-10-0, interior V=1")
plt.legend(["Current Potential/Correct potential","1","0,99"])
plt.show()





