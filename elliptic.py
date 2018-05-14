class Mathieu:
	def __init__(self, z = 1.+0*1j, q = 1., nterms = 50):
		
		# Computes azimuthal mathieu ce_2n(phi, q) function of complex argument q.
		#
		# Input:
		# z = 
		# q = argument of mathieu function (can be complex)
		# ord = maximum order to which compute the mathieu functions (it is counting modulo 2n, i.e. ord = 4 will give ce_2n up to ce_8) 
		# nterms = number of terms for truncation of the eigenvalue problem 
		#
		# Return:
		# mcn = mathieu characteristic values (eigenvalues)
		# A: mathieu characteristic vectors (eigenvectors), i.e. coefficients of mathieu functions
		# Done functions:
		# ce_2n(q) -> direct implementation of recursive system
		# Ce_2n(-q) -> from ce_2n(q)
		# ce_2n+1(q) -> direct implementation

		import os
		import numpy as np
		import scipy.linalg as la

		A = np.empty((nterms,nterms,2),dtype=complex)
		mcn = np.empty((nterms,2),dtype=complex)
		coeff = np.zeros((nterms,nterms),dtype=complex)
		ce = np.zeros((nterms,2),dtype=complex)
		Ce = np.zeros((nterms,2),dtype=complex)
		Ce_p = np.zeros((nterms,2),dtype=complex)
		Ce_pp = np.zeros((nterms,2),dtype=complex)


		# A/B axis=0: subscript, index related to infinite sum
		# A/B axis=1: superscript, n in order=2n+1
		# A/B axis=2: 0(even) or 1(odd) 

		ford = np.arange(0,nterms+1)
		voff = np.ones((nterms-1,),dtype=complex)*q
		voffm =  np.diag(voff,k=-1) +  np.diag(voff,k=+1)
		coeff += np.diag((2*ford[0:nterms])**2,k=0)
		coeff += voffm
		coeff[1,0] *= np.sqrt(2.0)
		coeff[0,1] *= np.sqrt(2.0)

		# compute eigenvalues (ntermsathieu characteristic numbers) and 
		# eigenvectors (ntermsathieu coefficients) 
		mcn[:,0],A[:,:,0] = la.eig(coeff)
		A[0,:,0]/= np.sqrt(2) # first coeff divided by sqrt(2)
		A[:,:,0] /= np.sqrt(A[0,:,0]*A[0,:,0] + np.sum(A[:,:,0]*A[:,:,0],axis=0))


		for n_ in np.arange(nterms):
			r = np.arange(0,nterms,1)
			ce[n_,0] = np.dot( A[:,n_,0], np.cos(2*r*z)) # A are already corresponding to 2r. 
			Ce[n_,0] = (-1)**n_ * np.dot( A[:,n_,0], (-1)**r * np.cosh(2*r*z)) # A are already corresponding to 2r. 
			Ce_p[n_,0] = (-1)**n_ * np.dot( A[:,n_,0], (-1)**r * 2*r * np.sinh(2*r*z)) # A are already corresponding to 2r. 
			Ce_pp[n_,0] = (-1)**n_ * np.dot( A[:,n_,0], (-1)**(r) * 4*r**2* np.cosh(2*r*z)) # A are already corresponding to 2r. 
 
		ford = np.arange(0,nterms+1)
		voff = np.ones((nterms-1,),dtype=complex)*q
		voffm =  np.diag(voff,k=-1) +  np.diag(voff,k=+1)
		coeff += np.diag((2*ford[0:nterms]+1)**2,k=0)
		coeff += voffm
		coeff[0,0] += q 

		# compute eigenvalues (ntermsathieu characteristic numbers) and 
		# eigenvectors (ntermsathieu coefficients) 
		mcn[:,1],A[:,:,1] = la.eig(coeff)
		A[:,:,1] /=  np.sum(A[:,:,1]*A[:,:,1],axis=0)


		for n_ in np.arange(nterms):
			r = np.arange(0,nterms,1)
			ce[n_,1] = np.dot( A[:,n_,1], np.cos((2*r+1)*z)) # A are already corresponding to 2r. 
 
	
		self.mcn = mcn
		self.A = A
		self.ce = ce
		self.Ce = Ce
		self.Ce_p = Ce_p
		self.Ce_pp = Ce_pp
