import numpy as np 
from scipy import optimize

# solve a non_linear system of equations g_1(x)=0 .... g_M(x)=0 
#set G to be a column vectors of the functions G_i = g_i
#J_G is the Jacobian.. J_G[i,j]=dg_i/dx^j 
#Solves the system by minimizing F(x)=(1/2) sum_{i=1}^2  ||g_i^(x)||^2 by implicit gradient descent...


def evaluate_functions(G,x): #takes an array of functions g_{ij}=G_{i}{j} and a point 'x', returns a numpy array G(x)_{ij} = g_{ij}(x)
	if G.ndim == 2:

		Gx = np.empty(G.shape)
		for i in np.arange(G.shape[0]):
			for j in np.arange(G.shape[1]):
				Gx[i,j] = G[i,j](x)

		return Gx

	else:
		Gx = np.empty(G.shape)
		for i in np.arange(G.shape[0]):
			Gx[i] = G[i](x)

		return Gx


def IGD_zeroes(G, J_G, x_0, delta, epsilon):

	def F(x):
		return np.sum(evaluate_functions(G,x)**2)


	d = J_G.shape[1]
	M = G.shape[0]
	I = np.identity(d)
	x = np.copy(x_0)
	F_old = F(x)
	nit = 0
	done = False
	


	
	def grad_F_and_HF(x):
		J_G_new = evaluate_functions(J_G, x)
		G_new = evaluate_functions(G,x)
		grad_F = np.dot(G_new, J_G_new)
		HF = np.dot(np.transpose(J_G_new),J_G_new)
		return (grad_F, HF)


	#SOLVING
	while done == False:
		nit += 1
		decreasing = False
		while decreasing == False:
			(grad_F,HF) = grad_F_and_HF(x)
			delta = delta/2
			x_new = x - delta*np.linalg.solve((I+delta * HF), grad_F)
			F_new = F(x_new)
			decreasing = F(x_new) <= F(x)
		x = np.copy(x_new)
		done = (np.abs(F(x)) <= epsilon) or (nit >= 200)
		delta = 1.9 * 2**nit * delta 

	print("number of iterations" + str(nit))
	print("delta =" + str(delta))
	print("F(x) = " + str(F(x)))
	print("x = " + str(x))

	return x











