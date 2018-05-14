from elliptic import Mathieu
import matplotlib as mpl
mpl.use('Tkagg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from scipy.special import gamma, hyp2f1, factorial

plt.close('all')

beta = 0.5
gm = 1./np.sqrt(1-beta**2)
cond = 1./2.5e-8
mu = 4*np.pi*1e-7

b = 0.035
qr = 0.001
a_axis = b*(1+qr)/(1-qr)
F = np.sqrt(a_axis**2-b**2)
ecc = F/a_axis
Z0 = 120*np.pi
mu0 = np.arccosh(1/ecc)

nterms_l = 40
nterms_p = 40
nterms_r = 40
nterms_t = 40
nterms = 40

nterms_lp = np.max((nterms_l,nterms_p))
nterms_rt = np.max((nterms_r,nterms_t))


z = []
f_vec = np.logspace(6,10,100)

for freq in f_vec:

	omega = 2*np.pi*freq
	Zs = np.sqrt(mu*omega/(2*cond))
	k0 = omega/c
	kz = omega/(c*beta)
	kt2 = k0**2-kz**2

	ce0=np.zeros((1,nterms_lp))
	cepi2=np.zeros((1,nterms_lp))
	Cemu0=np.zeros((1,nterms_lp))
	Ce0=np.zeros((1,nterms_lp))
	proj_rt=np.zeros((nterms_rt,nterms_rt))

	sumt_v=np.zeros((1,nterms_t))
	sumr_v=np.zeros((1,nterms_r))
	sump_v=np.zeros((1,nterms_p))
	suml_v=np.zeros((1,nterms_l))

	qv = -kt2*F**2/4
	mathieu = Mathieu(q = qv,nterms = nterms)
	p_v = np.arange(nterms)
	l_v = np.arange(nterms)
	ce0 = Mathieu(z = 0,q = qv,nterms=nterms).ce
	cepi2 = Mathieu(z = np.pi/2, q = qv, nterms=nterms).ce
	Cemu0 = np.real( Mathieu(z = np.pi/2-1j*mu0,q = qv,nterms = nterms).ce)
	Cemu0 = np.array([(-1)**l * el for l , el in enumerate(Cemu0)])
	Ce0 =  Mathieu(z = np.pi/2,q = qv,nterms = nterms).ce
	Ce0 = np.array([(-1)**l * el for l, el in enumerate(Ce0) ])

	for r in np.arange(nterms_rt):
	    for t in np.arange(nterms_rt):
		proj1=np.pi*np.sqrt(2)*np.exp(-(2*np.abs(r-t)+1)*mu0)*gamma(1./2+np.abs(r-t))/\
		     (gamma(1./2)*factorial(np.abs(r-t)))*hyp2f1(1./2,1./2+np.abs(r-t),1+np.abs(r-t),np.exp(-4*mu0))
		proj2=np.pi*np.sqrt(2)*np.exp(-(2*(r+t)+1)*mu0)*gamma(1./2+r+t)/\
						    (gamma(1./2)*factorial(r+t))*hyp2f1(1./2,1./2+r+t,1+r+t,np.exp(-4*mu0))
		proj_rt[r,t] = (-1)**r * (-1)**t * (proj1+proj2)

	if 0:
		suml=0;
		for l in np.arange(nterms_l):
		    sump = 0
		    for p in np.arange(nterms_p):
			sumr = 0
			for r in np.arange(nterms_r):
			    A2r2p = mathieu.A[r,p,0]
			    sumt = 0
			    for t in np.arange(nterms_t):
				A2t2l = mathieu.A[t,l,0]
				sumt+=A2t2l*proj_rt[r,t]
			    
			    sumr+=A2r2p*sumt
			
			sump+=ce0[p]*Ce0[p]/Cemu0[p]*sumr;
		    
		    suml+=(-1)**l*cepi2[l]*ce0[l]/Cemu0[l]*sump;
	else:
	    A2r2p = mathieu.A[:,:,0].T
	    A2t2l = mathieu.A[:,:,0].T

	    sumt = np.dot(A2t2l,proj_rt.T) 
	    
	    sumr = np.dot(A2r2p,sumt.T)

	    sump = np.dot((ce0*Ce0/Cemu0).T,sumr)
		
	    suml= np.dot(((-1)**l_v.reshape(1,-1)*(cepi2*ce0/Cemu0).T),sump.T)
		
	
	z.append(Zs*suml*np.sqrt(2)/(np.pi**2*F))
   
z=np.squeeze(np.asarray(z))


plt.figure()
plt.plot(f_vec, np.real(z))
plt.show()
