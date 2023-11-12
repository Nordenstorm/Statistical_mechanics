import numpy as np
import matplotlib.pyplot as plt

def master_equation(M,P,T=5000):
	t=np.linspace(0,T,num=T)

	P_lists=[]
	for i in range(0,len(P)):
		P_lists.append([])
		P_lists[i].append(P[i])

	for i in range(0,T-1):
		P=P+M.dot(P)*0.01
		for j in range(0,len(P)):
			P_lists[j].append(P[j])
	for i in range (0,len(P_lists)):

		name=str("P_a")
		plt.plot(t,P_lists[i],label=r'$P_{}$'.format(i+1))
	print(P)
	plt.legend()
	plt.show()

a=1
b=1
c=0.01
d=0.5


M=np.array([
[-a,b,0,0],
[0,-b,c,0],
[0,0,-c,d],
[a,0,0,-d]
])
P=np.array([1,0,0,0])
P=P.T

master_equation(M,P)
