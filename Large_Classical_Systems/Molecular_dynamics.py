import numpy as np
import matplotlib.pyplot as plt

# Everyone will start their gas in the same initial configuration.
# ----------------------------------------------------------------
def InitPositionCubic(Ncube, L):
    """Places Ncube^3 atoms in a cubic box; returns position vector"""
    N = Ncube**3
    position = np.zeros((N,3))
    rs = L/Ncube
    roffset = L/2 - rs/2
    n = 0
    # Note: you can rewrite this using the `itertools.product()` function
    for x in range(0, Ncube):
        for y in range(0, Ncube):
            for z in range(0, Ncube):
                if n < N:
                    position[n, 0] = rs*x - roffset 
                    position[n, 1] = rs*y - roffset
                    position[n, 2] = rs*z - roffset 
                n += 1
    return position

def InitVelocity(N, T0, mass=1., seed=1):
    dim = 3
    np.random.seed(seed)
    # generate N x dim array of random numbers, and shift to be [-0.5, 0.5)
    velocity = np.random.random((N,dim))-0.5
    sumV = np.sum(velocity, axis=0)/N  # get the average along the first axis
    velocity -= sumV  # subtract off sumV, so there is no net momentum
    KE = np.sum(velocity*velocity)  # calculate the total of V^2
    vscale = np.sqrt(dim*N*T0/(mass*KE))  # calculate a scaling factor
    velocity *= vscale  # rescale
    return velocity

# The simulation will require most of the functions you have already 
# implemented above. If it helps you debug, feel free to copy and
# paste the code here.

# We have written the Verlet time-stepping functions for you below, 
# `h` is the time step.
# -----------------------------------------------------------------

def VerletNextR(r_t, v_t, a_t, h):
    """Return new positions after one Verlet step"""
    # Note that these are vector quantities.
    # Numpy loops over the coordinates for us.
    r_t_plus_h = r_t + v_t*h + 0.5*a_t*h*h
    return r_t_plus_h

def VerletNextV(v_t,a_t,a_t_plus_h,h):
    """Return new velocities after one Verlet step"""
    # Note that these are vector quantities.
    # Numpy loops over the coordinates for us.
    v_t_plus_h = v_t + (1/2)*(a_t+a_t_plus_h)*h
    return v_t_plus_h
 

# Main Loop.
# ------------------------------------------------------------------------

# R, V, and A are the position, velocity, and acceleration of the atoms
# respectively. nR, nV, and nA are the next positions, velocities, etc.
# There are missing pieces in the code below that you will need to fill in.
# These are marked below with comments:

def simulate(Ncube, T0, L, M, steps, h,p,T_drift):
    """Initialize and run a simulation in a Ncube**3 box, for steps"""
    N = Ncube**3
    R = InitPositionCubic(Ncube, L)
    V = InitVelocity(N, T0, M)
    A = np.zeros((N,3))
    E = np.zeros(steps)
    V_sum= []
    T=[]

    nu=p/h
    nudt=nu*h
    for t in range(0, steps):
        E[t] = my_kinetic_energy(V, M)
        distance_table=my_disp_in_box(R, L)
        rij=calculate_distance_matrix(R,L)
        E[t] += my_potential_energy(rij)## calculate potential energy contribution
        pos_in_box=my_pos_in_box(R, L)
        F = all_my_forces(pos_in_box, L)## calculate forces; should be a function that returns an N x 3 array
        A = F/M
        nR = VerletNextR(R, V, A, h)
        nR = my_pos_in_box(nR, L)  ## from PrairieLearn HW
        nF = all_my_forces(nR, L)## calculate forces with new positions nR
        nA = nF/M
        nV = VerletNextV(V, A, nA, h)
        T.append(calc_temp(V,M))
        # update positions:
        R, V = nR, nV

        V=anderson_thermostat(V,T_drift,M,nudt)

        V_sum.append(M*np.sum(V))
        


    return E,V_sum,R,T
# Main Loop for all V
# ------------------------------------------------------------------------

# R, V, and A are the position, velocity, and acceleration of the atoms
# respectively. nR, nV, and nA are the next positions, velocities, etc.
# There are missing pieces in the code below that you will need to fill in.
# These are marked below with comments:

def simulate_give_all_V(Ncube, T0, L, M, steps, h,p):
    """Initialize and run a simulation in a Ncube**3 box, for steps"""
    N = Ncube**3
    R = InitPositionCubic(Ncube, L)
    V = InitVelocity(N, T0, M)
    A = np.zeros((N,3))

    V_all=[]

    nu=p/h
    nudt=nu*h
    for t in range(0, steps):

        distance_table=my_disp_in_box(R, L)
        rij=calculate_distance_matrix(R,L)
        pos_in_box=my_pos_in_box(R, L)
        F = all_my_forces(pos_in_box, L)## calculate forces; should be a function that returns an N x 3 array
        A = F/M
        nR = VerletNextR(R, V, A, h)
        nR = my_pos_in_box(nR, L)  ## from PrairieLearn HW
        nF = all_my_forces(nR, L)## calculate forces with new positions nR
        nA = nF/M
        nV = VerletNextV(V, A, nA, h)
        # update positions:
        R, V = nR, nV
        V=anderson_thermostat(V,T0,M,nudt)
        V_all.append(V)
        


    return np.array(V_all)

# You may adjust the gas properties here.
# ---------------------------------------
#My functions 

def my_pos_in_box(pos, lbox):

    return np.remainder(pos+lbox/2, lbox) -lbox/2

def my_disp_in_box(drij, lbox):

    return np.remainder(drij+lbox/2, lbox) -lbox/2

def my_kinetic_energy(vel, mass):
  
    return (mass/2)*np.sum(vel**2)



def my_potential_energy(rij):

    N_c,N_r=np.shape(rij)
    rij=rij[np.triu_indices(N_c, k = 1)]
    PE = np.sum(4*((1/rij)**6)*((1/rij)**6-1))
    
    return PE

def calculate_distance_matrix(R,lbox):
    Row,Col=np.shape(R)
    rij=np.zeros((Row,Row))

    for i,x_1 in enumerate(R):
        for j,x_2 in enumerate(R):
            r=my_disp_in_box(x_1-x_2,lbox)
            rij[i][j]=np.sqrt(((r[0])**2+(r[1])**2+(r[2])**2))

    return rij

def my_force_on(i, pos, lbox):
    
    rij=pos-pos[i,:]
    rij=np.delete(rij.T,i,1).T
    rij=np.remainder(rij+lbox/2, lbox) -lbox/2
    
    force=np.array([0.0,0.0,0.0])

    
    for n,x in enumerate(rij):
        r_2=(x[0]**2+x[1]**2+x[2]**2)
        R=(1/r_2)**3
        force[0]=force[0]-24*(1/r_2)*R*(2*R-1)*x[0]
        force[1]=force[1]-24*(1/r_2)*R*(2*R-1)*x[1]
        force[2]=force[2]-24*(1/r_2)*R*(2*R-1)*x[2]
        

    return force

def all_my_forces(pos, lbox):
    R,C=np.shape(pos)
    forces=np.zeros((R,C))

    for n,x in enumerate(pos):

        forces[n]=my_force_on(n, pos, lbox)
        

    return forces

# You may adjust the gas properties here.
# ---------------------------------------
# Here is code for numerics 
def calculate_std(data):

    N=len(data)
    
    mean=calculate_mean(data)
    std_sum=0
    for n,x in enumerate(data):
        std_sum=std_sum+(mean-x)**2

    std=(std_sum/((N-1)))**(1/2)
    return std
def calculate_mean(data):

    N=len(data)

    summa=0

    for n,x in enumerate(data):
        summa=summa+x

    mean = summa/len(data)
    return mean
# Analyse data

def my_pair_correlation(dists, natom, nbins, dr, lbox): #Calculate pair correlator

    histogram = np.histogram(dists, bins=nbins, range=(0, nbins*dr))
    r = (histogram[1] + dr/2)[:-1] # centers of the bins
    omega=lbox**3
    con_=(1/2)*(natom*(natom-1))/omega
    surf_vol = ((4/3)*np.pi)*(np.power((r+dr/2),3)-np.power((r-dr/2),3))
    surf_mass= np.array(con_*surf_vol)
    g_r=[]

    for i in range(0,len(surf_mass)):
        g_r.append(histogram[0][i]/surf_mass[i])

    return g_r,r

def my_legal_kvecs(maxn, lbox):#Domain of k values we decide to investigare

    c=2*np.pi/lbox
    kvecs = []
    for i in range(0,maxn+1):
        for j in range(0,maxn+1):
            for k in range(0,maxn+1):
                kvecs.append([c*i,c*j,c*k])

    return np.array(kvecs)

def my_calc_rhok(kvecs, pos):#Fourie transform posistions ensamble

    rho=[]
    for n,k in enumerate(kvecs):
        sum_=0
        for nat,pos_part in enumerate(pos):
            sum_=sum_+np.exp(-1j*np.dot(k,pos_part))
        rho.append(sum_)

    return rho


def my_calc_sk(kvecs, pos): #Create S function. Mode of rho's

    Natoms,ndim=pos.shape
    rho=my_calc_rhok(kvecs, pos)
    S=[]
    for n,rhoi in enumerate(rho):
        S.append(rhoi*np.conj(rhoi)/Natoms)
    return S

def my_calc_vacf0(all_vel, t):#Correlators of speed between atoms
    First_time_step=all_vel[0]
    T_time_step=all_vel[t]
    vacf=0
    N=len(First_time_step)
    for i in range(0,N):
        vacf=vacf+np.dot(First_time_step[i],T_time_step[i])
    vacf=vacf/N

    return vacf

def my_diffusion_constant(vacf):#Calculate diffusion constant

    dt = 0.032 #NOTE Do not change this value.
    D=(1/3)*np.trapz(vacf)*dt
    return D

def anderson_thermostat(V,T,M,nudt):
    V_new=[]
    for n,v in enumerate(V):
        prob=np.random.uniform(low=0.0, high=1)

        if prob<nudt:
            V_new.append(np.array([np.random.normal(loc=0.0, scale=np.sqrt(T/M)),
                        np.random.normal(loc=0.0, scale=np.sqrt(T/M)),
                        np.random.normal(loc=0.0, scale=np.sqrt(T/M))]))
        else:
            V_new.append(v)

    return np.array(V_new)

def plot_S(kvecs):
    kmags  = [np.linalg.norm(kvec) for kvec in kvecs]
    sk_arr = np.array(sk_list) # convert to numpy array if not already so 

    # average S(k) if multiple k-vectors have the same magnitude
    unique_kmags = np.unique(kmags)
    unique_sk    = np.zeros(len(unique_kmags))
    for iukmag in range(len(unique_kmags)):
        kmag    = unique_kmags[iukmag]
        idx2avg = np.where(kmags==kmag)
        unique_sk[iukmag] = np.mean(sk_arr[idx2avg])
    # end for iukmag
      
    # visualize
    plt.plot(unique_kmags,unique_sk)


def calc_temp(V,M):
    T=0

    for n,v in enumerate(V):
        T=T+(v[0]**2+v[1]**2+v[2]**2)
    T=M*T/(3*len(V))

    return T



"""Here I have only plotting and data processing"""
def Show_momentum_space_after_thermostat_impl(E,P,R):

    T0,L,N,M,h,steps,Ncube,p,nbins,dr=standard()
    Title="Temperature ="+str(T0)+", steps=" + str(steps)  + ", $\eta \Delta t =$" + str(p)
    plt.plot(P,label="P Momentum ")
    plt.xlabel("N (Time step)")
    plt.ylabel("P (Momentum)")
    plt.title(Title)
    lbox=L
    plt.legend()
    plt.show()

def Show_g_r_function(E,P,R):

    T0,L,N,M,h,steps,Ncube,p,nbins,dr=standard()
    Title="Temperature ="+str(T0)+", steps=" + str(steps) + ", nbins=" + str(nbins) + ", $\eta \Delta t =$" + str(p)
    lbox=L

    distance_matrix=calculate_distance_matrix(R,lbox)
    distance_flatten=distance_matrix.flatten()

    g_r,r=my_pair_correlation( distance_flatten, N, nbins, dr, lbox)

    plt.plot(r*dr,g_r,label="g_r")
    plt.xlabel("r (distance)")
    plt.ylabel("g (Correlation)")
    plt.title(Title)
    plt.legend()
    plt.show()

def Show_reciprocal_space(E,P,R):

    T0,L,N,M,h,steps,Ncube,p,nbins,dr=standard()
    Title="Temperature ="+str(T0)+", steps=" + str(steps) + ", nbins=" + str(nbins) + ", $\eta \Delta t =$" + str(p)
    lbox=L

    distance_matrix=calculate_distance_matrix(R,lbox)
    distance_flatten=distance_matrix.flatten()
    g_r,skrot=my_pair_correlation( distance_flatten, N, nbins, dr, lbox)

    kvecs=my_legal_kvecs(5,lbox)
    rhos=my_calc_rhok(kvecs,R)
    sk_list=my_calc_sk(kvecs, R)


    kmags  = [np.linalg.norm(kvec) for kvec in kvecs]
    sk_arr = np.array(sk_list) # convert to numpy array if not already so 

    # average S(k) if multiple k-vectors have the same magnitude
    unique_kmags = np.unique(kmags)
    unique_sk    = np.zeros(len(unique_kmags))
    for iukmag in range(len(unique_kmags)):
        kmag    = unique_kmags[iukmag]
        idx2avg = np.where(kmags==kmag)
        unique_sk[iukmag] = np.mean(sk_arr[idx2avg])
    # end for iukmag


    plt.xlabel("norm(k) (k vecs, n_max=5)")
    plt.ylabel("S (Reciprocal mode average)")
    plt.title(Title)
    plt.plot(unique_kmags,unique_sk)
    plt.legend()
    plt.show()

def Calc_diffusion_over_time():# Calculate diffusion constant variation
    T0,L,N,M,h,steps,Ncube,p,nbins,dr=standard()
    Title="Temperature ="+str(T0)+", steps=" + str(steps) + ", $\eta \Delta t =$" + str(p)
    V_all=simulate_give_all_V(Ncube, T0, L, M, steps,h,p)
    vacfl=[]
    for i in range(1,steps):
        vacf=np.array(my_calc_vacf0(V_all, i ))
        vacfl.append(vacf)
    Diffusion=my_diffusion_constant(vacfl)
    plt.xlabel("T (Timestep)")
    plt.ylabel("vacf (Correlation over time in v)")
    plt.title(Title)
    plt.plot(vacfl)
    plt.legend()
    plt.show()
    print(Diffusion)
    return Diffusion

"""Values for simulations"""
def standard():
    T0 = 0.728     # temperature
    L = 4.2323167 # box length
    N = 64        # number of particles
    M = 48.0      # mass of each particle
    h = 0.032      # time step size
    steps = 200  # number of time steps
    Ncube = 4
    p=0.01
    nbins=10
    dr= np.sqrt(3)*L/(nbins*2)

    return T0,L,N,M,h,steps,Ncube,p,nbins,dr
#Show_momentum_space_after_thermostat_impl
#Show_g_r_function
#Show_reciprocal_space
#Calc_diffusion_over_time


T0,L,N,M,h,steps,Ncube,p,nbins,dr=standard()
T_drift=T0
E,P,R,T=simulate(Ncube, T0, L, M, steps, h,p, T_drift)


Show_g_r_function(E,P,R)


