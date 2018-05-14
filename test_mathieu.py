from elliptic import Mathieu
import matplotlib as mpl
mpl.use('Tkagg')
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
### Plot ce_2n

q = 1.
phi = np.linspace(0, 2*np.pi, 500)
nterms = 50

ce2n=[]
mcn=[]
for phi_ in phi:
	mathieu = Mathieu(z = phi_, q = q, nterms = nterms)
	ce2n.append(mathieu.ce[:,0])
	mcn.append(mathieu.mcn[:,0])

ce2n = np.array(ce2n)
mcn = np.array(mcn)

plt.figure()
for n in np.arange(0,4):
	plt.plot(phi,ce2n[:,n], label = '2n=$%d$'%n)	

plt.xlabel('$\phi$')
plt.ylabel('$ce_{2n} (\phi,q)$')
plt.title('q = %f, %d terms truncation'%(q,nterms))
plt.legend()
plt.savefig('ce2n_func.png')

### Plot characteristic values

q = np.linspace(0,10,100)
phi = 0.
nterms = 50

ce2n=[]
mcn=[]
for q_ in q:
	mathieu = Mathieu(z = phi, q = q_, nterms = nterms)
	ce2n.append(mathieu.ce[:,0])
	mcn.append(mathieu.mcn[:,0])

ce2n = np.array(ce2n)
mcn = np.array(mcn)


plt.figure()
for n in np.arange(0,4):
	plt.plot(q, mcn[:,n], label = '2n=$%d$'%n)	

plt.xlabel('$q$')
plt.ylabel('$a_{2n}(-q)$')
plt.title('$\phi$ = %f, %d terms truncation'%(phi,nterms))
plt.legend()
plt.savefig('ce2n_mcn.png')

### Plot Ce_2n

q = 0.1
mu = np.linspace(0, 1, 500)
nterms = 50

Ce2n=[]
mcn=[]
for mu_ in mu:
	mathieu = Mathieu(z = mu_, q = q, nterms = nterms)
	Ce2n.append(mathieu.Ce_pp[:,0])
	mcn.append(mathieu.mcn[:,0])

Ce2n = np.array(Ce2n)
mcn = np.array(mcn)

plt.figure()
for n in np.arange(0,1):
	plt.plot(mu,Ce2n[:,n], label = '2n=$%d$'%n)	

plt.xlabel('$\mu$')
plt.ylabel('$Ce_{2n} (\mu, -q)$')
plt.title('q = %f, %d terms truncation'%(q,nterms))
plt.legend()
#plt.ylim(0,10)
plt.savefig('Ce2n_func.png')

### Plot ce_2n+1

q = 1.
phi = np.linspace(0, 2*np.pi, 500)
nterms = 10

ce2n=[]
mcn=[]
for phi_ in phi:
	mathieu = Mathieu(z = phi_, q = q, nterms = nterms)
	ce2n.append(mathieu.ce[:,1])
	mcn.append(mathieu.mcn[:,1])

ce2n = np.array(ce2n)
mcn = np.array(mcn)

plt.figure()
for n in np.arange(0,4):
	plt.plot(phi,ce2n[:,n], label = '2n+1=$%d$'%n)	

plt.xlabel('$\phi$')
plt.ylabel('$ce_{2n+1} (\phi,q)$')
plt.title('q = %f, %d terms truncation'%(q,nterms))
plt.legend()
plt.savefig('ce2np1_func.png')

M=np.empty((nterms,nterms),dtype=complex)
for in_, n_ in enumerate(np.arange(nterms)):
	for im_, m_ in enumerate(np.arange(nterms)):

		M[in_,im_]=np.trapz(ce2n[:,n_]*ce2n[:,m_],phi)

x = np.arange(nterms)
y = x

X,Y = np.meshgrid(x,y)
plt.figure();
plt.contourf(X,Y,np.log10(M/np.pi));
cbar = plt.colorbar()
cbar.set_label('Log10($<ce_{2n+1},ce_{2m+1}>$)')

plt.xlabel('Order n')
plt.ylabel('Order m')
plt.savefig('ce2np1_norm.png')


