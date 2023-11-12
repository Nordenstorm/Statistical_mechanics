import numpy as np
import matplotlib.pyplot as plt
import random as ran

def numerical_analysis_approache(ro,gamma,omega,n,dt,t,sigma_p,sigma_m,sigma_z):
	t_=0
	ro_list=[]

	while t_<t-dt:

		t_+=dt
		sigma_m_p=np.matmul(sigma_m,sigma_p)
		sigma_p_m=np.matmul(sigma_p,sigma_m)
		commutator=np.matmul(sigma_z,ro)-np.matmul(ro,sigma_z)
		lindbladian_1=gamma*(n+1)*((np.matmul(sigma_m,np.matmul(ro,sigma_p)))-(1/2)*(np.matmul(ro,sigma_p_m)+np.matmul(sigma_p_m,ro)))
		lindbladian=gamma*n*((np.matmul(sigma_p,np.matmul(ro,sigma_m)))-(1/2)*(np.matmul(ro,sigma_m_p)+np.matmul(sigma_m_p,ro)))


		ro = ro + dt*(-1j*omega*gamma*commutator/2+lindbladian_1+lindbladian)
		ro_list.append(ro)
	return ro_list

def plot_numerical(ro_list,t_space):
	ro_list_restruct_real=[[],[],[],[]]
	for i in range(0,len(ro_list)):
		ro_list_restruct_real[0].append(np.real(ro_list[i][0][0]))
		ro_list_restruct_real[1].append(np.real(ro_list[i][0][1]))
		ro_list_restruct_real[2].append(np.real(ro_list[i][1][0]))
		ro_list_restruct_real[3].append(np.real(ro_list[i][1][1]))


	plt.plot(ro_list_restruct_real[0],label=r'Re ($\rho$_{ee}) Numerical')
	#plt.plot(ro_list_restruct_real[1],label=r'Re ($\rho$_{eg}) Numerical')
	#plt.plot(ro_list_restruct_real[2],label=r'Re ($\rho$_{ge}) Numerical')
	plt.plot(ro_list_restruct_real[3],label=r'Re ($\rho$_{gg}) Numerical')

	ro_list_restruct_imag=[[],[],[],[]]
	for i in range(0,len(ro_list)):
		ro_list_restruct_imag[0].append(np.imag(ro_list[i][0][0]))
		ro_list_restruct_imag[1].append(np.imag(ro_list[i][0][1]))
		ro_list_restruct_imag[2].append(np.imag(ro_list[i][1][0]))
		ro_list_restruct_imag[3].append(np.imag(ro_list[i][1][1]))


	plt.plot(ro_list_restruct_imag[1],label=r'Imag ($\rho$_{eg}) Numerical')
	plt.plot(ro_list_restruct_imag[2],label=r'Imag ($\rho$_{ge}) Numerical')
def plot_numerical_markov(ro_list,t_space):
	ro_list_restruct_real=[[],[],[],[]]
	for i in range(0,len(ro_list)):
		ro_list_restruct_real[0].append(np.real(ro_list[i][0][0]))
		ro_list_restruct_real[1].append(np.real(ro_list[i][0][1]))
		ro_list_restruct_real[2].append(np.real(ro_list[i][1][0]))
		ro_list_restruct_real[3].append(np.real(ro_list[i][1][1]))


	plt.plot(ro_list_restruct_real[0],label=r'Re ($\rho$_{ee}) Markov')
	#plt.plot(ro_list_restruct_real[1],label=r'Re ($\rho$_{eg}) Markov')
	#plt.plot(ro_list_restruct_real[2],label=r'Re ($\rho$_{ge}) Markov')
	plt.plot(ro_list_restruct_real[3],label=r'Re ($\rho$_{gg}) Markov')

	ro_list_restruct_imag=[[],[],[],[]]
	for i in range(0,len(ro_list)):
		ro_list_restruct_imag[0].append(np.imag(ro_list[i][0][0]))
		ro_list_restruct_imag[1].append(np.imag(ro_list[i][0][1]))
		ro_list_restruct_imag[2].append(np.imag(ro_list[i][1][0]))
		ro_list_restruct_imag[3].append(np.imag(ro_list[i][1][1]))


	plt.plot(ro_list_restruct_imag[1],label=r'Imag ($\rho$_{eg}) Markov')
	plt.plot(ro_list_restruct_imag[2],label=r'Imag ($\rho$_{ge}) Markov')

def Monte_carlo_approache(psi,gamma,omega,n,dt,t,sigma_p,sigma_m,sigma_z,M):
	psi_base=psi
	L_1=np.sqrt(gamma*(n+1))* sigma_m
	L_2=np.sqrt(gamma*n)* sigma_p
	L_1_dag=np.conj(L_1.T)
	L_2_dag=np.conj(L_2.T)
	H=(1/2)*omega*sigma_z
	J=H-1j/2*np.matmul(L_1_dag,L_1)-1j/2*np.matmul(L_2_dag,L_2)
	t_=0

	psi_List_e=[]
	psi_List_g=[]
	for i in range(0,M):
		if i%100==0:
			print(i)
		psi_List_e.append([])
		psi_List_g.append([])
		t_=0
		psi=psi_base

		while t_<t:
			psi_List_e[i].append(psi[0])
			psi_List_g[i].append(psi[1])
			t_=t_+dt
			r=ran.uniform(0,1)
			bra=np.conj(psi.T)

			p_1=dt*np.matmul(bra,np.matmul(np.matmul(L_1_dag,L_1),psi))
			p_2=dt*np.matmul(bra,np.matmul(np.matmul(L_2_dag,L_2),psi))

			p_0=1-p_1-p_2

			if p_1>r:
				psi=np.matmul(L_1,psi)
			elif p_1+p_2>r:
				psi=np.matmul(L_2,psi)
			else:
				psi=psi-1j*dt*np.matmul(J,psi)

			bra=np.conj(psi.T)
			psi=psi/(np.sqrt(np.matmul(bra,psi)))
	rho_list_average=[]
	rho_t=[]
	for i in range(0,len(psi_List_e[0])):
		rho_list_average.append(np.array([[0,0],[0,0]]))

	for i in range(0,M):
		rho_t.append([])

	for j in range(0,len(psi_List_e[0])):
		for i in range(0,M):

			psi=np.array([psi_List_e[i][j],psi_List_g[i][j]])
			rho=np.array([[psi[0]*np.conj(psi[0]),psi[1]*np.conj(psi[0])],[psi[0]*np.conj(psi[1]),psi[1]*np.conj(psi[1])]])

			rho_t[i].append(rho)

	for i in range(0,M):
		for j in range(0,len(psi_List_e[0])):

			rho_list_average[j]=rho_list_average[j]+rho_t[i][j]/M

	return rho_list_average

def main():

	ro=np.array([[1,1],[1,1]])
	psi=np.array([1,1])*1/np.sqrt(2)
	ro=ro*1/2
	gamma=1
	omega=3
	n=10
	dt=0.001
	t=1
	M=100 # Monte carlo runs
	
	sigma_p=np.array([[0,1],[0,0]])
	sigma_m=np.array([[0,0],[1,0]])
	sigma_z=np.array([[1,0],[0,-1]])


	rho_list_average=Monte_carlo_approache(psi,gamma,omega,n,dt,t,sigma_p,sigma_m,sigma_z,M)

	t_space=np.linspace(0,t,int(1/(2*dt)))
	plot_numerical_markov(rho_list_average,t_space)


	
	ro_list=numerical_analysis_approache(ro,gamma,omega,n,dt,t,sigma_p,sigma_m,sigma_z)

	dt=0.0002
	t_space=np.linspace(0,t,int(1/dt))
	plot_numerical(ro_list,t_space)
	plt.legend()
	plt.show()
	


main()



