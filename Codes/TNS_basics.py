import os
import numpy as np
import scipy as sp
from scipy.sparse import kron
from dataclasses import dataclass, field
from time import time
import h5py
import torch as to


#---------------------------------------------------------------
@dataclass
class MPO:
	def __init__(self, L, d, chi=None, can=None):
		self.L = L
		self.d = d
		self.canonical = can
		self.TN = {}
		self.midEnt = 0
		self.chi = 2**L
		if chi!=None:
			self.chi = chi

	def BondDim(self):
		bondDim = []
		if self.canonical == "L":
			for l in range(self.L-1):
				bondDim.append(self.TN[l].shape[-1])
		elif self.canonical == "G-L":
			for l in range(1,self.L):
				bondDim.append(self.TN["Lam",l].shape[-1])
		return(bondDim)

	## Transforms the MPO to a matrix
	def ToMatrix(self):
		L = self.L
		O = self.TN
		if self.canonical == "G-L":
			M = to.tensordot(to.diag(O["Lam",0]), O["B",0], dims=([1], [2])) * np.sqrt(2) 
			for l in range(1,L): 
				M = to.tensordot(M, to.tensordot(to.diag(O["Lam",l]), O["B",l], dims=([1], [2])), dims=([2*l+1], [0])) * np.sqrt(2)
			source = tuple(np.concatenate( (np.arange(1,2*L+1,2), np.arange(2,2*L+1,2)) ))
			destin = tuple(np.arange(1,2*L+1))
			M = to.movedim(M, source, destin)[0,...,0].reshape(2**L,2**L)
		return(M)

#---------------------------------------------------------------
@dataclass
class MPS:
	def __init__(self, L, d, chi, can=None):
		self.L = L
		self.d = d
		self.canonical = can
		self.TN = {}
		self.midEnt = 0
		self.chi = 2**L
		if chi!=None:
			self.chi = chi

	def BondDim(self):
		bondDim = []
		for l in range(self.L-1):
			bondDim.append(self.TN[l].shape[-1])
		return(bondDim)

#---------------------------------------------------------------
@dataclass
class data:
	gates: np.ndarray = np.array(())
	locH: np.ndarray = np.array(())
	params: dict = field(default_factory= lambda: {
	})

#---------------------------------------------------------------

def safe_inverse(x, eps=1e-14):
	return x / (x ** 2 + eps)

class SVD(to.autograd.Function):
	@staticmethod
	def forward(self, A):
		U, S, V = to.linalg.svd(A, full_matrices=False)
		self.save_for_backward(U, S, V)
		return U, S, V

	@staticmethod
	def backward(self, dU, dS, dV):
		U, S, V = self.saved_tensors

		V = V.t().conj()
		dV = dV.t().conj()

		Vd = V.t().conj()
		Ud = U.t().conj()
		M = U.size(0)
		N = V.size(0)
		NS = len(S)

		F = (S - S[:, None])
		F = safe_inverse(F)
		F.diagonal().fill_(0)

		G = (S + S[:, None])
		G.diagonal().fill_(np.inf)
		G = 1./G

		UdU = Ud @ dU
		VdV = Vd @ dV

		Su = (F + G)*(UdU - UdU.t().conj()) / 2
		Sv = (F - G)*(VdV - VdV.t().conj()) / 2.

		dA = U @ (Su + Sv + to.diag(dS)) @ Vd
		if (M>NS):
			dA = dA + (to.eye(M, dtype=dU.dtype, device=dU.device) - U@Ud) @ (dU/S) @ Vd
		if (N>NS):
			dA = dA + (U/S) @ dV.t().conj() @ (to.eye(N, dtype=dU.dtype, device=dU.device) - V@Vd)

		if to.is_complex(U):
			L = UdU.diagonal(0, -2, -1)
			to.real(L).zero_()
			to.imag(L).mul_(1./S)
			dA += U * L.unsqueeze(-2) @ Vd

		return dA

#---------------------------------------------------------------
## Compress an MPS by series of SVD
def Compress_RMPS(State):
	Compressed_Psi = MPS(State.L, State.d)

	s, vh = np.ones(1), np.ones((1,State.TN[0].shape[1]))
	mask = s>1e-15
	for l in range(State.L):
		M = np.transpose(np.tensordot(np.diag(s[mask]) @ vh[mask,:] / np.linalg.norm(s), State.TN[l], (1,1)), axes=(1,0,2))
		A = M.reshape(State.d * M.shape[1], -1)
		u, s, vh = np.linalg.svd(A, full_matrices=False)

		mask = s>1e-15

		Compressed_Psi.TN[l] = u[:,mask].reshape(State.d,M.shape[1],-1)

	s, u = np.ones(1), np.ones((1,1))
	mask = [True]
	for l in range(State.L-1, -1, -1):
		M = np.tensordot(Compressed_Psi.TN[l], u[:,mask] @ np.diag(s[mask]) / np.linalg.norm(s), (2,0))
		A = M.reshape(-1, Compressed_Psi.d * M.shape[2])
		u, s, vh = np.linalg.svd(A, full_matrices=False)

		mask = s>1e-15

		Compressed_Psi.TN[l] = vh[mask,:].reshape(Compressed_Psi.d,-1,M.shape[2])
	Compressed_Psi.canonical = "R"

	return(Compressed_Psi)

#---------------------------------------------------------------
## Check if the MPS is L or R canonical
def CheckCan(Psi, ch):
	res = False
	nTN = len(Psi)
	if ch == "L":
		norm = 0
		for l in range(nTN-1):
			## For MPS
			if len(Psi[l].shape) == 3:
				A = np.tensordot(Psi[l].conjugate(), Psi[l], ((0,1),(0,1)))
				norm += 1-np.allclose(A, np.identity(Psi[l].shape[2]))
			## For MPO
			elif len(Psi[l].shape) == 4:
				A = np.tensordot(Psi[l].conjugate(), Psi[l], ((0,1,2),(0,1,2)))
				norm += 1-np.allclose(A, np.identity(Psi[l].shape[3]))
	elif ch == "R":
		norm = 0
		for l in range(nTN):
			## For MPS
			if len(Psi[l].shape) == 3:
				A = np.tensordot(Psi[l], Psi[l].conjugate(), ((0,2),(0,2)))
				norm += 1-np.allclose(A, np.identity(Psi[l].shape[1]))
			## For MPO
			elif len(Psi[l].shape) == 4:
				A = np.tensordot(Psi[l], Psi[l].conjugate(), ((0,1,3),(0,1,3)))
				norm += 1-np.allclose(A, np.identity(Psi[l].shape[2]))
	if norm == 0:
		res = True

	return(res)

#---------------------------------------------------------------
## Defines a product-state like MPO (Bond dim. = 1)
def IdMPO(d, L, chi, norm=1, **kwargs):
	Psi = MPO(L, d, chi=chi, can="G-L")
	for l in range(L):
	        Psi.TN["B",l] = np.identity(d).reshape((d,d,1,1)) / np.sqrt(2)**norm
	        Psi.TN["Lam",l] = np.ones((1)) ## Lambda 0 is the dummy index

	return(Psi)

#---------------------------------------------------------------
## Defines a product-state MPS (Bond dim. = 1)
def IdMPS(d, L, chi, norm=1, **kwargs):
	Psi = MPS(L, d, chi=chi, can="G-L")
	for l in range(L):
	        Psi.TN["B",l] = np.ones((d,1,1)) / np.sqrt(2)**norm
	        Psi.TN["Lam",l] = np.ones((1)) ## Lambda 0 is the dummy index

	return(Psi)

#---------------------------------------------------------------
## Load an MPS or MPO from an h5 dataset named "file."h5
def LoadTN(file):
	try:
		A = h5py.File(file)
	except OSError:
		B = None
		print('No file named', file)
	else:
		B = {}
		for key in A.keys():
			B[key] = np.array(A[key])
		A.close()
	return(B)

#---------------------------------------------------------------
## Transforms an MPS or MPO stored with numpy arrays to torch tensors
def Np2To(A, repres="TN",  grads=True):
	B = {}
	if repres == "TN":
		for key in A.keys():
			if grads:	
				key2 = tuple(key+(0,))
				B[key2] = (to.tensor(A[key]).cdouble()).requires_grad_()
			else:
				B[key] = (to.tensor(A[key]).cdouble())
	elif repres == "Exact":
		if grads:	
			 B["Op",0] = (to.tensor(A["Op"]).cdouble()).requires_grad_()
		else:
			 B["Op",0] = (to.tensor(A["Op"]).cdouble())

	return(B)

#---------------------------------------------------------------
## Creates an MPS with random matrices
def Rand_MPS(L, d, m, save=True, reuse=False, **kwargs):
        Psi = {}
        for l in range(int(L/2)):
                if d**(l-1) > m:
                        Psi[l] = Psi[l-1]/np.linalg.norm(Psi[l])
                        Psi[-1-l] = Psi[-l]
                else:
                        Psi[l] = np.random.random((d, min(d**l,m), min(d**(l+1),m)))
                        Psi[-l-1] = np.random.random((d, min(d**(l+1),m), min(d**(l),m)))

        ## Save the current starting random MPS
        if save==True:
                for i in range(sys_info.params['L']):
                        np.savetxt('Psi0/Psi0_%s.dat'%i, Psi[i].reshape(Psi[i].shape[0]*Psi[i].shape[1], Psi[i].shape[2]))
        ## Re-use a previously defined random MPS
        if reuse == True:
                for i in range(sys_info.params['L']):
                        Psi[i] = np.genfromtxt('Psi0/Psi0_%s.dat'%i).reshape(sys_info.params['d'], -1, Psi[i].shape[2])
        return(Psi)

#---------------------------------------------------------------
## Transforms an MPS to a right canonical form with series of QR
def LCanon_MPS(State):
	LcanMPS = MPS(State.L,State.d,can="L")
	for l in range(0,State.L):
                ## Phys. index absorbed in columns for QR in R-norm
		Na, Nb = State.TN[l].shape[0] * State.TN[l].shape[1], State.TN[l].shape[2]

                ## Dag. of B if we are in R-norm. 
		Q, R = np.linalg.qr(np.reshape(State.TN[l], (Na, Nb)))

                ## Re-extend the phys. index
		LcanMPS.TN[l] = np.transpose(np.reshape(Q, (State.d, int(Na/State.d), -1)), axes=(0,1,2))

		## We devide by the Froebenius norm of R to keep the norm of the operator close to 1 (See Discussion in Ref. Phys. Rev. B 95, 035129 (2017))
		if l+1 <= State.L-1:
			State.TN[l+1] = np.tensordot(R[:,:], State.TN[l+1], (1,1)) #/ np.linalg.norm(R)

#	RcanMPS.TN[l-1] = State.TN[l-1]

	return(LcanMPS)

#---------------------------------------------------------------
## Transforms an MPS to a right canonical form with series of QR
def RCanon_MPS(State):
	RcanMPS = MPS(State.L,State.d,can="R")
	for l in range(State.L-1,-1,-1):
		## Phys. index absorbed in columns for QR in R-norm
		Na, Nb = State.TN[l].shape[1], State.TN[l].shape[0] * State.TN[l].shape[2]

		## Dag. of B if we are in R-norm. 
		Q, R = np.linalg.qr(np.reshape(np.transpose(State.TN[l],(0,2,1)), (Nb, Na)))

		## Re-extend the phys. index
		RcanMPS.TN[l] = np.transpose(np.reshape(Q, (State.d, int(Nb/State.d),-1)), axes=(0,2,1))

		## We devide by the Froebenius norm of R to keep the norm of the operator close to 1 (See Discussion in Ref. Phys. Rev. B 95, 035129 (2017))
		if l-1 >= 0:
			State.TN[l-1] = State.TN[l-1] @ R[:,:].T #/ np.linalg.norm(R)

#	RcanMPS.TN[l-1] = State.TN[l-1]

	return(RcanMPS)

#---------------------------------------------------------------
## Transforms a matrix to a left-can. MPO with SVDs
def Matrix_to_LMPO(mat, L, d, chi = False, threshold = False, **kwargs):
	LMPO = MPO(L,d,"L")

	## Matrix should be in binary-alphabetical order
	mat = mat.reshape((d,)*2*L)

	ind = [0,L]
	order, a = np.zeros(2*L, dtype=int), np.arange(0,2*L)
	mask, mask2 = np.ones(2*L, dtype=bool), np.ones(2*L, dtype=bool)
	mask[[0,1]], mask2[[0,L]] = False, False
	order[[0,1]], order[mask] = ind, a[mask2]

	r1 = 1
	## Put the indices i,i' (for the MPO) together
	mat_order = np.transpose(mat, axes=order)
	new = mat_order.reshape(d*d*r1,-1)
	u, s, vh = np.linalg.svd(new, full_matrices=False)
	if threshold != False:
		mask = s/np.linalg.norm(s)>threshold
		r2 = np.sum(mask)
		LMPO.TN[0] = u[:,mask].reshape(d, d, 1, r2)
		mat = (np.diag(s[:r2]) @ vh[mask,:]).reshape((-1,)+(d,)*2*(L-1)) #/ np.linalg.norm(s)
	else:
		r2 = d**2
		LMPO.TN[0] = u.reshape(d, d, 1, r2)
		mat = (np.diag(s) @ vh).reshape((-1,)+(d,)*2*(L-1)) #/ np.linalg.norm(s)
	r1 = r2

	for l in range(1,L):
		ind = [1,L-l+1]
		order, a = np.zeros(2*(L-l)+1, dtype=int), np.arange(0,2*(L-l)+1)
		mask, mask2 = np.ones(2*(L-l)+1, dtype=bool), np.ones(2*(L-l)+1, dtype=bool)
		mask[[0,1]], mask2[[1,L-l+1]] = False, False
		order[[0,1]], order[mask] = ind, a[mask2]

		## Put the indices i,i' together
		mat_order = np.transpose(mat, axes=order)

		r2 = min(d**(2*(l+1)),d**(2*(L-l-1)))
		new = mat_order.reshape(d*d*r1,-1)
		u, s, vh = np.linalg.svd(new, full_matrices=False)
		if threshold != False:
			mask = s/np.linalg.norm(s) > threshold
			r2 = np.sum(mask)
			LMPO.TN[l] = np.transpose(u[:,mask].reshape(r1, d, d, r2), axes=(1,2,0,3))
			mat = (np.diag(s[:r2]) @ vh[mask,:]).reshape((-1,)+(d,)*2*(L-l-1)) #/ np.linalg.norm(s)
		else:
			LMPO.TN[l] = np.transpose(u.reshape(r1, d, d, r2), axes=(1,2,0,3))
			mat = (np.diag(s) @ vh).reshape((-1,)+(d,)*2*(L-l-1)) #/ np.linalg.norm(s)
		r1 = r2
	return(LMPO)

#---------------------------------------------------------------
## Pauli matrices in a dict.
def Paulis():
	paulis = {}
	paulis['sx'] = to.tensor(((0,1),(1,0)) )
	paulis['sy'] = to.tensor(((0,-1),(1,0))) * 1.j
	paulis['sz'] = to.tensor(((1,0),(0,-1)))
	paulis['sxsx'] = to.kron( to.tensor(((0,1),(1,0))),      to.tensor(((0,1),(1,0))))
	paulis['sysy'] = to.kron( to.tensor(((0,-1),(1,0)))*1.j, to.tensor(((0,-1),(1,0)))*1.j)
	paulis['szsz'] = to.kron( to.tensor(((1,0),(0,-1))),     to.tensor(((1,0),(0,-1))))
	return(paulis)

#---------------------------------------------------------------
## Transforms a vector to a left-can. MPS with SVDs
def Vector_to_LMPS(vec, L, d, chi = False, threshold = False, **kwargs):
	LMPS = MPS(L,d,"L")

	r1, r2 = 1,1
	mat = vec.reshape(d*r1,-1)
	u, s, vh = np.linalg.svd(mat, full_matrices=False)
	LMPS.TN[0] = u.reshape(d, 1, 2)
	vec = (np.diag(s) @ vh).reshape(-1) #/ np.linalg.norm(s)
	r1 = 2

	for l in range(1,L):
		r2 = min(d**(l+1),d**(L-l-1))
		new = vec.reshape(d*r1,-1)
		u, s, vh = np.linalg.svd(new, full_matrices=False)
		if l == int(L/2):
			Ent = np.sum(s**4)
		if threshold != False:
			mask = s>threshold
			r2 = np.sum(mask)
			LMPS.TN[l] = u[:,mask].reshape(d, r1, r2)
			vec = (np.diag(s[:r2]) @ vh[mask,:]).reshape(-1) #/ np.linalg.norm(s)
		elif chi != False:
			LMPS.TN[l] = u.reshape(d, r1, r2)
			vec = (np.diag(s) @ vh).reshape(-1) #/ np.linalg.norm(s)
		else:
			LMPS.TN[l] = u.reshape(d, r1, r2)
			vec = (np.diag(s) @ vh).reshape(-1) #/ np.linalg.norm(s)
		r1 = r2
	LMPS.Ent = -np.log(Ent)

	return(LMPS)

#---------------------------------------------------------------
## Multiplication of an MPS by an MPO
def MPO_MPS(Op, State):
	new_MPS = MPS(State.L, State.d)
	for l in range(State.L):
		A = np.transpose(np.tensordot(Op.TN[l], State.TN[l], ((1,0))), axes=(3,1,0,2,4))
		x1, x2 = A.shape[0]*A.shape[1], A.shape[-1] * A.shape[-2]
		#new_MPS.TN[l] = np.transpose(A.reshape(x1,State.d,x2), axes=(1,0,2))
		B = np.transpose(A.reshape(x1, State.d, A.shape[-2],  A.shape[-1]), axes=(3,2,1,0))
		C = B.reshape(x2,State.d,x1)
		new_MPS.TN[l] = np.transpose(C, axes=(1,2,0))
	new_MPS.canonical = False

	return(new_MPS)

#---------------------------------------------------------------
## Save an MPS or MPO to an h5 dataset named file.h5
def SaveTN(A, file):

	hf = h5py.File('%s.h5'%file, 'w')
	for key in A.keys():
		flatkey = "".join(np.array(["".join(str(elem)) for elem in key]))
		hf.create_dataset(flatkey, data=A[key])
	hf.close()
	print("\nSaved tensor at %s.h5 \n"%file)
	return()

#---------------------------------------------------------------
## Returns the Trace of the MPO product A+ B(Tr[A+B])
def TraceMPS(A, B):
	if A.canonical == "L":
		res = np.ones((1,1))
		for l in range(A.L):
			## We want to compare the B matrices of both MPOs
			res = np.tensordot(res, A.TN[l].conjugate(), (0,1))
			res = np.tensordot(res, B.TN[l], ((0,1),(1,0)))

		res = np.tensordot(res.conjugate(), res, (0,0))[0,0]

	elif A.canonical == "R":
		res = np.ones((1,1))
		for l in range(A.L-1,-1,-1):
			print(A.TN[l].shape, B.TN[l].shape, res.shape)
			## We want to compare the B matrices of both MPOs
			res = np.tensordot(A.TN[l].conj(), res, (2,1))
			res = np.tensordot(B.TN[l], res, ((0,2),(0,2)))

		res = np.tensordot(res.conjugate(), res, (0,0))[0,0]

	return(res)
#---------------------------------------------------------------
## Returns the Trace of the MPO product A+ B(Tr[A+B])
def TraceMPOs(A, B):
	res = np.ones((1,1))
	if A.canonical == "G-L": 
		for l in range(A.L):
			## We want to compare the B matrices of both MPOs
			res = np.tensordot(res, A.TN["B", l].conjugate(), (0,2))
			res = np.tensordot(res, B.TN["B", l], ((0,2,1),(2,1,0)))
	else:
		for l in range(A.L):
			## We want to compare the B matrices of both MPOs
			res = np.tensordot(res, A.TN[l].conjugate(), (0,2))
			res = np.tensordot(res, B.TN[l], ((0,2,1),(2,1,0)))

	res = np.tensordot(res.conjugate(), res, (0,0))[0,0]

	return(res)

#---------------------------------------------------------------
## Returns the Trace of the MPO product A+ B(Tr[A+B]), given pyTorch tensors
def torchTraceMPOs(A, B):
	res1, res2 = {}, {}
	res1['B',0] = to.ones((1,1), requires_grad=True, dtype=to.cdouble)
	res2['B',0] = to.ones((1,1), requires_grad=True, dtype=to.cdouble)
	for key in A.keys():
		## We want to compare the B matrices of both MPOs
		if key[0] == "B":
			res1[key[0],key[1]+1] = to.tensordot(res2[key], A[key].conj(), dims=([0],[2]))
			res2[key[0],key[1]+1] = to.tensordot(res1[key[0],key[1]+1], B[key], dims=([0,2,1],[2,1,0]))

	Trace = to.tensordot(res2["B",key[1]+1].conj(), res2["B",key[1]+1], dims=([0],[0]))[0,0]

	return(Trace)
