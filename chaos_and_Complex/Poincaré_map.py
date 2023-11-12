#!/bin/python3

# Python simulation of a double pendulum with real time animation.
# BH, MP, AJ 2020-10-27, latest version 2021-11-02.
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import animation
from pylab import *
import numpy as np


"""
    This script simulates and animates a double pendulum.
    Classes are similar to the ones of pendolum_template.py. The main differences are:
    - coordinates are obtained from the total energy value E (look at what functions
        Oscillator.p1squaredFromH and Oscillator.__init__ do)
    - you are asked to implement the expression for the derivatives of the Hamiltonian 
        w.r.t. coordinates p1 and p2
    - you are asked to check when the conditions to produce the Poincare' map are
        satisfied and append the coordinates' values to some container to then plot
"""

# Global constants
G = 9.8  # gravitational acceleration

pi=math.pi

Global_poincare_q1=[]
Global_poincare_p1=[]

Lista=[]


# Kinetic energy
def Ekin(osc):
    return 1 / (2.0 * osc.m * osc.L * osc.L) * (osc.p1 * osc.p1 + 2.0 * osc.p2 * osc.p2 - 2.0 * osc.p1 * osc.p2 * cos(osc.q1 - osc.q2)) / (1 + (sin(osc.q1 - osc.q2)) ** 2)

# Potential energy
def Epot(osc):
    return osc.m * G * osc.L * (3 - 2 * math.cos(osc.q1) - math.cos(osc.q2))


# Class that holds the parameter and state of a double pendulum
class Oscillator:

    def p2squaredFromH(self):
        return (self.E - Epot(self)) * (1 + (sin(self.q1 - self.q2)) ** 2) * self.m * self.L * self.L

    # Initial condition is [q1, q2, p1, p2]; p2 is however re-obtained based on the value of E
    # therefore you can use any value for init_cond[3]


    def __init__(self, m=1, L=1, t0=0, E=1, init_cond=[0.0, 0.0, 0.0, -1.0]) :

        self.m = m      # mass of the pendulum bob
        self.L = L      # arm length
        self.t = t0     # the initial time
        self.E = E      # total conserved energy
        self.q1 = init_cond[0]
        self.q2 = init_cond[1]
        self.p1 = init_cond[2]
        self.p2 = -1.0
        while (self.p2 < 0):
            # Comment the two following lines in case you want to exactly prescribe values to q1 and q2
            # However, be sure that there exists a value of p2 compatible with the imposed total energy E!
            self.q1 = math.pi * (2 * np.random.random() - 1)
            self.q2 = math.pi * (2 * np.random.random() - 1)
            p2squared = self.p2squaredFromH()
            if (p2squared >= 0):
                self.p2 = math.sqrt(p2squared)
        self.q2_prev = self.q2
        self.init_cond = [self.q1,self.q2,self.p1,self.p2]  # Variable for init_cond Poincare
        print("Initialization:")
        print("E  = "+str(self.E))
        print("q1 = "+str(self.q1))
        print("q2 = "+str(self.q2))
        print("p1 = "+str(self.p1))
        print("p2 = "+str(self.p2))
        Lista.append(str(self.q1))




# Class for storing observables for an oscillator
class Observables:

    def __init__(self):
        self.time = []          # list to store time
        self.q1list = []        # list to store q1
        self.q2list = []        # list to store q2
        self.p1list = []        # list to store p1
        self.p2list = []        # list to store p2
        self.epot = []          # list to store potential energy
        self.ekin = []          # list to store kinetic energy
        self.etot = []          # list to store total energy
        self.poincare_q1 = []   # list to store q1 for Poincare plot
        self.poincare_p1 = []   # list to store p1 for Poincare plot


# Derivate of H with respect to p1
def dHdp1(q1, q2, p1, p2, m, L):
    # TODO: Write and return the formula for the derivative of H with respect to p1 here
    return (p1-p2*math.cos(q2-q1))/(m*(L**2)*((sin(q2-q1)**2) + 1))
    #((p1-cos(q2-q1)*p2)/(L*L*m*(sin(q2-q1)-1)))


# Derivate of H with respect to p2
def dHdp2(q1, q2, p1, p2, m, L):
    # TODO: Write and return the formula for the derivative of H with respect to p2 here
    return (2*p2-p1*math.cos(q2-q1))/(m*(L**2)*((sin(q2-q1)**2) + 1))
    #-(2*p2-cos(q2-q1)*p1)/(L*L*m*(sin(q2-q1)-1))


# Derivate of H with respect to q1
def dHdq1(q1, q2, p1, p2, m, L):
    return 1 / (2.0 * m * L * L) * (
        -2 * (p1 * p1 + 2 * p2 * p2) * math.cos(q1 - q2) + p1 * p2 * (4 + 2 * (math.cos(q1 - q2)) ** 2)) * math.sin(
            q1 - q2) / (1 + (math.sin(q1 - q2)) ** 2) ** 2 + m * G * L * 2.0 * math.sin(q1)


# Derivate of H with respect to q2
def dHdq2(q1, q2, p1, p2, m, L):
    return 1 / (2.0 * m * L * L) * (
        2 * (p1 * p1 + 2 * p2 * p2) * math.cos(q1 - q2) - p1 * p2 * (4 + 2 * (math.cos(q1 - q2)) ** 2)) * math.sin(q1 - q2) / (
            1 + (math.sin(q1 - q2)) ** 2) ** 2 + m * G * L * math.sin(q2)


class BaseIntegrator:

    def __init__(self, dt=0.003):
        self.dt = dt    # time step

    def integrate(self,
                  osc,
                  obs, 
                  ):

        """ Perform a single integration step """
        self.timestep(osc, obs)

        """ Append observables to their lists """
        obs.time.append(osc.t)
        obs.q1list.append(osc.q1)
        obs.q2list.append(osc.q2)
        obs.p1list.append(osc.p1)
        obs.p2list.append(osc.p2)
        obs.epot.append(Epot(osc))
        obs.ekin.append(Ekin(osc))
        obs.etot.append(Epot(osc) + Ekin(osc))
        # TODO: Append values for the Poincare map

 

        if len(obs.q2list) > 1: # Poincare conditions
            if obs.q2list[-2] < 0 and osc.q2 > 0 and osc.p2 > 0:
                obs.poincare_q1.append(osc.q1)
                obs.poincare_p1.append(osc.p1)




    def timestep(self, osc, obs):
        """ Virtual function: implemented by the child classes """
        pass


# Euler-Richardson integrator
class EulerRichardsonIntegrator(BaseIntegrator):
    def timestep(self, osc, obs):

        dt = self.dt
        osc.t += dt
        # TODO: Add integration here
        # Now we got Hamiltonians This will change the nature of the acceleration and

        ####


        v1=dHdp1(osc.q1, osc.q2, osc.p1, osc.p2, osc.m, osc.L)*self.dt*0.5
        v2=dHdp2(osc.q1, osc.q2, osc.p1, osc.p2, osc.m, osc.L)*self.dt*0.5
        a1=-1*(dHdq1(osc.q1, osc.q2, osc.p1, osc.p2, osc.m, osc.L))*self.dt*0.5
        a2=-1*(dHdq2(osc.q1, osc.q2, osc.p1, osc.p2, osc.m, osc.L))*self.dt*0.5

        dq1_mid = dHdp1(osc.q1+v1, osc.q2+v2, osc.p1+a1, osc.p2+a2, osc.m, osc.L)
        dq2_mid = dHdp2(osc.q1+v1, osc.q2+v2, osc.p1+a1, osc.p2+a2, osc.m, osc.L)
        dp1_mid = -1*dHdq1(osc.q1+v1, osc.q2+v2, osc.p1+a1, osc.p2+a2, osc.m, osc.L)
        dp2_mid = -1*dHdq2(osc.q1+v1, osc.q2+v2, osc.p1+a1, osc.p2+a2, osc.m, osc.L)
        
        osc.q1 = osc.q1 + dq1_mid*dt
        osc.q2 = osc.q2 + dq2_mid*dt
        osc.p1 = osc.p1 + dp1_mid*dt
        osc.p2 = osc.p2 + dp2_mid*dt

        



class RK4Integrator(BaseIntegrator):
    def timestep(self, osc, obs):

        def Deriative_of_dH(a, b, deriv):  # a and b is a_i and b_i. Calculates derivitives
            if deriv == "dq1":
                return dHdp1(osc.q1+a, osc.q2+a, osc.p1+b, osc.p2+b, osc.m, osc.L)
            elif deriv == "dq2":
                return dHdp2(osc.q1+a, osc.q2+a, osc.p1+b, osc.p2+b, osc.m, osc.L)
            elif deriv == "dp1":
                return -1*dHdq1(osc.q1+a, osc.q2+a, osc.p1+b, osc.p2+b, osc.m, osc.L)
            elif deriv == "dp2":
                return -1*dHdq2(osc.q1+a, osc.q2+a, osc.p1+b, osc.p2+b, osc.m, osc.L)

        dt = self.dt
        D_List = ["dq1","dq2","dp1","dp2"]
        # TODO: Add integration here
        # First step in Runge Kutta is to update acceleration then after that update velocity

        Runge_Kutta_Matrix = []  # Runge_Kutta_Matrix[derivitive][number for a and b][a or b]
        for i in range(0,len(D_List)-2):    # Calculates a_i and b_i for each derivitive
            a1, b1 = Deriative_of_dH(0,0,D_List[i])*dt, Deriative_of_dH(0,0,D_List[i+2])*dt #a1,b1
            a2, b2 = Deriative_of_dH(a1/2,b1/2,D_List[i])*dt, Deriative_of_dH(a1/2,b1/2,D_List[i+2])*dt
            a3, b3 = Deriative_of_dH(a2/2,b2/2,D_List[i])*dt, Deriative_of_dH(a1/2,b2/2,D_List[i+2])*dt
            a4, b4 = Deriative_of_dH(a3,b3,D_List[i])*dt, Deriative_of_dH(a3,b3,D_List[i+2])*dt
            Runge_Kutta_Matrix.append([[a1,b1],[a2,b2],[a3,b3],[a4,b4]])


        osc.q1 = osc.q1 + (Runge_Kutta_Matrix[0][0][0]+2*Runge_Kutta_Matrix[0][1][0]+2*Runge_Kutta_Matrix[0][2][0]+Runge_Kutta_Matrix[0][3][0])/6
        osc.q2 = osc.q2 + (Runge_Kutta_Matrix[1][0][0]+2*Runge_Kutta_Matrix[1][1][0]+2*Runge_Kutta_Matrix[1][2][0]+Runge_Kutta_Matrix[1][3][0])/6
        osc.p1 = osc.p1 + (Runge_Kutta_Matrix[0][0][1]+2*Runge_Kutta_Matrix[0][1][1]+2*Runge_Kutta_Matrix[0][2][1]+Runge_Kutta_Matrix[0][3][1])/6
        osc.p2 = osc.p2 + (Runge_Kutta_Matrix[1][0][1]+2*Runge_Kutta_Matrix[1][1][1]+2*Runge_Kutta_Matrix[1][2][1]+Runge_Kutta_Matrix[1][3][1])/6

        dt = self.dt
        
        osc.t += dt




class Simulation:

    def reset(self, osc=Oscillator()) :
        self.oscillator = osc
        self.obs = Observables()

    def __init__(self, osc=Oscillator()) :
        self.reset(osc)

    def save_pointclaire(self) :

        Global_poincare_p1.append(self.obs.poincare_p1)
        Global_poincare_q1.append(self.obs.poincare_q1)

    def run(self,
            integrator,
            tmax=150.,   # final time
            outfile='energy1.pdf'
            ):

        n = int(tmax / integrator.dt)

        for it in range(n):
            integrator.integrate(self.oscillator, self.obs)

        # If you experience problems visualizing the animation and/or
        # the following figures comment out the next line 

        self.save_pointclaire()

 
# It's good practice to encapsulate the script execution in 
# a main() function (e.g. for profiling reasons)

Energy=5


def main() :

    

    for i in range (-2,3):



        # Be sure you are passing the correct initial conditions!
        oscillator = Oscillator(m=1, L=1, t0=0, E=Energy) 


        # Create the simulation object for your oscillator instance:
        simulation = Simulation(oscillator)

        # Run the simulation using the various integration schemes (that you are asked to implement):
        #simulation.run(integrator=EulerRichardsonIntegrator(),  tmax=40)

        simulation.run(integrator=EulerRichardsonIntegrator())


    plt.figure()
    plt.xlabel('q1')
    plt.ylabel('p1')
    
    plt.plot(Global_poincare_q1[0],Global_poincare_p1[0],
    Global_poincare_q1[1],Global_poincare_p1[1],
    Global_poincare_q1[2],Global_poincare_p1[2],
    Global_poincare_q1[3],Global_poincare_p1[3],
    Global_poincare_q1[4],Global_poincare_p1[4],
    linestyle = 'None',marker='D')
    plt.legend((str(Lista[0]),str(Lista[1]),str(Lista[2]),str(Lista[3]),str(Lista[4])))
    plt.title("Poincare Map, With Energy "+str(Energy))
    """
    
    plt.plot(Global_poincare_q1[0],Global_poincare_p1[0],
    Global_poincare_q1[1],Global_poincare_p1[1]
    ,linestyle = 'None',marker='D')
    plt.legend(('0','1.1'))
    plt.title("Poincare Map, E=15")

     """
    
    plt.tight_layout()  # adapt the plot area tot the text with larger fonts 
    plt.show()


# Calling 'main()' if the script is executed.
# If the script is instead just imported, main is not called (this can be useful if you want to
# write another script importing and utilizing the functions and classes defined in this one)
if __name__ == "__main__" :
    main()
