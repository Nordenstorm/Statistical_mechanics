import numpy as np
import matplotlib.pyplot as plt

def wiener_one():
	dt = 0.1;
	Nsteps = 100;
	W = [0]
	t = [0]
	for s in range(Nsteps):
	    W.append(W[-1]+np.sqrt(dt)*np.random.normal())
	    t.append(t[-1]+dt)
	W = np.array(W)
	plt.plot(t,W)
def wiener_many(dt ,Nsteps, Nsamples):
	Nsamples=Nsamples-1
	for sample in range(Nsamples):
	    W = [0]
	    t = [0]
	    for s in range(Nsteps):
	        W.append(W[-1]+np.sqrt(dt)*np.random.normal())
	        t.append(t[-1]+dt)
	    W = np.array(W)
	    plt.plot(t,W)
def mean_and_variance(t,dt,Nsamples):
    Nsteps = int(t/dt)
    W_at_t = []
    for sample in range(Nsamples):
        W = [0]
        t = [0]
        for s in range(Nsteps):
            W.append(W[-1]+np.sqrt(dt)*np.random.normal())
            t.append(t[-1]+dt)
        W_at_t.append(W[-1])
    W_at_t = np.array(W_at_t)
    return np.mean(W_at_t),np.var(W_at_t)
def plot_mean_variance():
	dt_list = np.linspace(0.001,0.1,num=40,endpoint=True)
	t = 10
	Nsamples = 40
	Wmean_list = []
	Wvar_list = []
	for dt in dt_list:
	    Wmean, Wvar = mean_and_variance(t,dt,Nsamples)
	    Wmean_list.append(Wmean)
	    Wvar_list.append(Wvar)
	plt.plot(dt_list,Wmean_list)
	plt.plot(dt_list,Wvar_list)
def euler_stocastic(V_inital,Dri,Fri,dt,Nsteps):

	dt = dt
	Nsteps = Nsteps
	t = [0]
	V = [V_inital]

	for s in range(Nsteps):
	    W=np.random.normal(loc=0.0, scale=np.sqrt(dt))
	    v_new=V[-1]-V[-1]*dt*Fri+np.sqrt(Dri)*W
	    V.append(v_new)
	    t.append(t[-1]+dt)
	return V,t
def many_euler_stocastic(Nsamples,V_inital,Dri,Fri,dt,Nsteps):
	Vmatrix=[]
	for i in range(Nsamples):
		V,t=euler_stocastic(V_inital,Dri,Fri,dt,Nsteps)
		Vmatrix.append(V)
	return Vmatrix,t
def mean_and_variance_euler_stocastic(Nsamples,V_inital,Dri,Fri,dt,Nsteps):
	Vmatrix,t=many_euler_stocastic(Nsamples,V_inital,Dri,Fri,dt,Nsteps)
	Vmatrix=np.array(Vmatrix)
	V_mean=[]
	V_variance=[]
	for i in range(0,len(t)):

		col=Vmatrix[:, i]
		V_mean.append(np.mean(col))
		V_variance.append(np.var(col))
	V_analytical=[]
	for i in t:
		V_analytical.append((Dri/(2*Fri))*(1-np.exp(-2*Fri*i)))
	#plt.plot(t,V_mean, label="Mean_Numerical")
	error_mean=0
	for i in range(0,len(V_variance)):
		error_mean=V_variance[i]-V_analytical[i]
	error_mean=error_mean/len(V_variance)
	print(error_mean)
	plt.plot(t,V_variance, label="Variance of V: Numerical")
	plt.plot(t,V_analytical, label="Variance of V: Analytical")
def mean_and_variance_euler_stocastic_x_pos(Nsamples,V_inital,Dri,Fri,dt,Nsteps):
	Vmatrix,t=many_euler_stocastic(Nsamples,V_inital,Dri,Fri,dt,Nsteps)
	Vmatrix=np.array(Vmatrix)
	Xmatrix=[]

	X_variance=[]
	for i in range(0,len(t)):
		Xmatrix.append([0])
		for j in range(0,len(Vmatrix[0])):
			Xmatrix[i].append(Xmatrix[i][-1]+Vmatrix[i][j]*dt)
	Xmatrix=np.array(Xmatrix)

	for i in range(0,len(t)):
		col=Xmatrix[:, i]
		X_variance.append(np.var(col))

	X_analytical=[]
	for i in t:
		X_analytical.append((Dri/(Fri**2))*(i+(1/Fri)*(np.exp(-Fri*i)-1)-(1/(2*Fri))*((1-np.exp(-Fri*i))**2)))
	error_mean=0
	for i in range(0,len(X_variance)):
		error_mean=X_variance[i]-X_analytical[i]
	error_mean=error_mean/len(X_variance)
	print(error_mean)

	plt.plot(t,X_variance, label="Variance of X: Numerical")
	plt.plot(t,X_analytical, label="Variance of X: Analytical")

mean_and_variance_euler_stocastic(40,1,1,10,0.0005,6000)
plt.legend()
plt.show()

"""
mean_and_variance_euler_stocastic_x_pos(4000,1,1,10,0.005,600)
plt.legend()
plt.show()
"""



