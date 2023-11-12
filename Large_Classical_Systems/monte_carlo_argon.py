import numpy as np
import matplotlib.pyplot as plt


"""Smart monte carlo"""



def minimum_image(r, L):
  return r - L*np.round(r / L)
  
def log_trial_probability(fold, fnew, eta, sigma, beta):
  """ Calculate the logarithm of the ``proposal'' contribution to
  acceptance probability in a smart Monte Carlo move.

  Args:
  fold (np.array): shape (ndim), force before move
  fnew (np.array): shape (ndim), force after move
  eta (np.array): shape (ndim), Gaussian random move vector
  sigma (float): width of Gaussian random numbers (eta0)
  beta (float): inverse temperature
  Return:
  float: ln(T(r'->r)/T(r->r'))
  """




  tau = 0.5*sigma**2
  rdelta=eta+fold*tau*beta
  T_r_to_rp = - (np.linalg.norm(rdelta-tau*beta*fold)**2)/(4*tau)
  T_rp_to_r = - (np.linalg.norm(-rdelta-tau*beta*fnew)**2)/(4*tau)

  return T_rp_to_r-T_r_to_rp

def potential_energy(i, pos, L, rc):
  """
  Args:
  i (int): particle index
  pos (np.array): particle positions, shape (number of atoms, 3)
  L (float): cubic box side length
  rc (float): potential cutoff radius
  Return:
  float: potential energy for particle i
  """
  vshift = 4*rc**(-6)*(rc**(-6)-1)  # potential shift
  r = minimum_image(pos[i] - pos,L)
  # r[j] is the same as the minimum image of pos[i] - pos[j]
  # it has shape (number of atoms, 3)

  # complete this function


  r=np.delete(r,i,0)

  pe=0

  for n,x in enumerate(r):
    
    R=np.linalg.norm(x)

    if R<rc:
      pe = pe + np.sum(4*((1/R)**6)*((1/R)**6-1))-vshift
    
  return pe

def my_loga_symmetric(vold, vnew, beta):
    """ Calculate the logarithm of the acceptance probability
    given the energies before and after a symmetric move.

    Args:
        vold (float): potential energy before move
        vnew (float): potential energy after move
        beta (float): inverse temperature
        Return:
        float: ln(A(x->x')), logorithm of the acceptance probability
    """


    return -beta*(vnew-vold)

def disp_in_box(drij, lbox):
  nint = np.around(drij/lbox)
  return drij-lbox*nint

def my_force_i(i, pos, lbox, rc):
  """ Calculate the force experienced by particle i 

  Args:
  i (int): particle index
  pos (np.array): particle positions, shape (natom, ndim)
  lbox (float): cubic box side length
  rc (float): potential cutoff radius
  Return:
  np.array: shape (ndim,), force on particle i
  """
  drij = disp_in_box(pos[i] - pos[np.newaxis, :], lbox)[0]
  # drij has shape (natom, ndim). It contains the displacement
  #  \vec{r}_i - \vec{r}_j for all j in range(natom)
  drij=np.delete(drij,i,0)

  natom, ndim = pos.shape
  # complete this function
  force=np.array([0.0,0.0,0.0])
  for n,x in enumerate(drij):

    r_2=np.linalg.norm(x)**2

    if np.sqrt(r_2)<rc:
      R=(1/r_2)**3
      force[0]=force[0]+24*(1/r_2)*R*(2*R-1)*x[0]
      force[1]=force[1]+24*(1/r_2)*R*(2*R-1)*x[1]
      force[2]=force[2]+24*(1/r_2)*R*(2*R-1)*x[2]

  return force

def pos_in_box(pos, lbox):
  return (pos+lbox/2.) % lbox - lbox/2.

def my_mc_sweep_smart(pos, lbox, rc, beta, eta, acc_check, sig):
  """ Perform one Monte Carlo sweep using smart Monte Carlo moves

  Args:
    pos (np.array): particle positions, shape (natom, ndim)
    lbox (float): cubic box side length
    rc (float): potential cutoff radius
    beta (float): inverse temperature
    eta (np.array): shape (natom, ndim), array of Gaussian random
     numbers, one for each single-particle move
    acc_check (np.array): shape (natom), array of uniform random
     numbers, one for each single-particle move
    sig (float): width of Gaussian random numbers (eta)
  Return:
    (int, float): (naccept, de), the number of accepted
    single-particle moves, and the change in energy
  """
  tau = 0.5*sig**2
  natom, ndim = pos.shape
  naccept = 0
  de = 0

  # complete this function
  for n in range(0,natom):

    fold=my_force_i(n,pos,lbox,rc)
    vold=potential_energy(n,pos,lbox,rc)

    pos[n]=pos[n]+eta[n]+tau*beta*fold

    fnew = my_force_i(n,pos,lbox,rc)
    vnew = potential_energy(n, pos, lbox, rc)

    w=my_loga_symmetric(vold,vnew,beta)+log_trial_probability(fold, fnew, eta[n], sig, beta)
    p=my_loga_symmetric(vold,vnew,beta)

    cond=min(1,np.exp(w))

    if cond > acc_check[n]:
      naccept=naccept+1
      de=de-p/beta
    else:
      pos[n]=pos[n]-eta[n]-tau*beta*fold

  return naccept, de
#######################

"""Monte carlo"""





def my_mc_sweep_not_smart(pos, lbox, rc, beta, eta, acc_check):
    """ Perform one Monte Carlo sweep

    Args:
    pos (np.array): particle positions, shape (natom, ndim)
    lbox (float): cubic box side length
    rc (float): potential cutoff radius
    beta (float): inverse temperature
    eta (np.array): shape (natom, ndim), array of Gaussian random
    numbers, one for each single-particle move
    acc_check (np.array): shape (natom), array of uniform random
    numbers, one for each single-particle move
    Return:
    (int, float): (naccept, de), the number of accepted
    single-particle moves, and the change in energy
    """
    natom, ndim = pos.shape
    naccept = 0
    de = 0

    for n in range(0,natom):
        vold=potential_energy(n,pos,lbox,rc)
        pos[n]=pos[n]+eta[n]
        vnew = potential_energy(n, pos, lbox, rc)

        w=my_loga_symmetric(vold,vnew,beta)
        cond=min(1,np.exp(w))

        if cond > acc_check[n]:
            naccept=naccept+1
            de=de-w/beta
        else:
            pos[n]=pos[n]-eta[n]    

    return naccept, de




###################################

def calculate_distance_matrix(R,lbox):
    Row,Col=np.shape(R)
    rij=np.zeros((Row,Row))

    for i,x_1 in enumerate(R):
        for j,x_2 in enumerate(R):
            r=disp_in_box(x_1-x_2,lbox)
            rij[i][j]=np.sqrt(((r[0])**2+(r[1])**2+(r[2])**2))

    return rij

def my_total_potential_energy(rij):

    N_c,N_r=np.shape(rij)
    rij=rij[np.triu_indices(N_c, k = 1)]
    PE = np.sum(4*((1/rij)**6)*((1/rij)**6-1))
    
    return PE





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


def Show_reciprocal_space(R):

    L,M,steps,Ncube,sigma,T,rho,rc,nbins,dr=standard()
    N=Ncube**3
    Title="Temperature ="+str(T)+", steps=" + str(steps) + ", nbins=" + str(nbins) + ", $\sigma t =$" + str(sigma)
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
    unique_sk=unique_sk/sum(unique_sk)
    plt.plot(unique_kmags,unique_sk)
    plt.legend()
    plt.show()


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

def my_pos_in_box(pos, lbox):

    return np.remainder(pos+lbox/2, lbox) -lbox/2

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
def standard():

    L = 4.0 # box length
    M = 48.0      # mass of each particle
    steps = 100  # number of time steps
    Ncube = 4
    sigma=0.07
    T = 2.0 # Temperature; beta = 0.5; for canonical ensemble.
    rho = 1.0 # number density
    rc= 2.0
    nbins=10
    dr= np.sqrt(3)*L/(nbins*2)


    return L,M,steps,Ncube,sigma,T,rho,rc,nbins,dr

def main_simulation(L,M,steps,Ncube,sigma,T,rho,rc,nbins,dr):

    
    """Initialize and run a simulation in a Ncube**3 box, for steps"""

    N=Ncube**3
    R = InitPositionCubic(Ncube, L)
    beta=1/T
    potential_energy_list=[]
    summor=0
    for i in range(steps):

        acc_check=np.random.uniform(low=0.0,high=1.0,size=64)
        eta=np.random.normal(loc=0.0, scale=sigma, size=(64,3))
        naccept, de = my_mc_sweep_not_smart(R, L, rc, beta, eta, acc_check)
        summor=summor+naccept
        R=my_pos_in_box(R, L)
        ###Calculate energy 
        rij=calculate_distance_matrix(R,L)
        potential_energy_list.append(my_total_potential_energy(rij))


    print(summor/(64*steps))
    ###Pair correlatotion
    distance_matrix=calculate_distance_matrix(R,L)
    distance_flatten=distance_matrix.flatten()
    g_r,r=my_pair_correlation( distance_flatten, N, nbins, dr, L)


    return R,g_r,r,potential_energy_list



L,M,steps,Ncube,sigma,T,rho,rc,nbins,dr=standard()

R,g_r,r,potential_energy_list=main_simulation(L,M,steps,Ncube,sigma,T,rho,rc,nbins,dr)



plt.plot(potential_energy_list,label="Energy")
plt.title("Potential energy")
plt.legend()
plt.show()
plt.plot(r,g_r,label="g_r")
plt.title("Correlation")
plt.legend()
plt.show()

Show_reciprocal_space(R)



