import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group


def timestep(psi,L):
    for i in range(int(L/2)):
        U = unitary_group.rvs(4)
        U = np.reshape(U, (2,2,2,2))
        psi = np.tensordot(U,psi,axes=([0,1],[2*i,2*i+1]))
    for i in range(int(L/2-1)):
        U = unitary_group.rvs(4)
        U = np.reshape(U, (2,2,2,2))
        psi = np.tensordot(U,psi,axes=([0,1],[2*i+1,2*i+2]))
    psi_ent = np.reshape(psi, (int(2**(L/2)),int(2**(L/2))))
    lamb = np.linalg.svd(psi_ent, full_matrices=False)[1]
    S_one = 0
    for i in range(len(lamb)):
        S_one = S_one - (lamb[i]**2)*np.log(lamb[i]**2)
    psi=psi/np.linalg.norm(psi)
    return psi,S_one

def psi_generator(N):
        
    for i in range(0,N):
        if i==0:
            psi=np.array([1,0])
        else:
            fi=np.array([1,0])
            psi=np.tensordot(psi, fi, axes=0)
    #Now first index of psi[x] is the first spin state. The second index psi[.][x] is the second spin and so on
    return psi

def main():
    N_list=np.array([2,4,6,8,10])

    T = 16
    samples = 100
    base = np.zeros(2).T
    color=["red","blue","green","orange","purple"]
    Inf_S=[]
    for i in range(0,len(N_list)):
        Inf_S.append([])
        for j in range(0,T+1):
            Inf_S[i].append(np.log(2**(N_list[i]/2)))

    for k in range(0,len(N_list)):
        print(k)
        n = N_list[k]
        S_av = np.zeros((samples, T))
        S = [0]
        for rep in range(samples):
            psi=psi_generator(n)
            for time in range(T):
                psi,S_one=timestep(psi,n)
                S_av[rep, time] = S_one

        for i in range(0,samples):
            for time in range(T):
                if i==0:
                    S.append(S_av[i,time]/samples)

                else:
                    S[time+1]=S[time+1]+S_av[i,time]/samples

        name="Entropy of half " +str(N_list[k]) + " chain"
        name1="ininity case"
        plt.plot(S,label=name,c=color[k])

    plt.plot(Inf_S[0], linestyle='dashed',c="red")
    plt.plot(Inf_S[1], linestyle='dashed',c="blue")
    plt.plot(Inf_S[2], linestyle='dashed',c="green")
    plt.plot(Inf_S[3], linestyle='dashed',c="orange")
    plt.plot(Inf_S[4], linestyle='dashed',c="purple")
    plt.xlabel("time (t)")
    plt.ylabel("entropy -S(t)")

    plt.legend()
    plt.show()
main()