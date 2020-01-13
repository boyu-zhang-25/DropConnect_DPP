import numpy as np 


def f1(c,M,v_star):

	return (np.arcsin(c**2/(1+c))*M*(v_star)**2)/np.pi

def f2(c,M,v_star):
	return (M*(v_star)**2)/float(6)

def f3(c,M,v_star):

	return (2*np.arcsin(c/np.sqrt(2+2*c))*M*(v_star)**2)/np.pi


def err(c,M,v_star):

	return f1(c,M,v_star)+ f2(c,M,v_star)- f3(c,M,v_star)


c = float(83)/500
M = 2
v_star = 4

print(err(c,M,v_star))