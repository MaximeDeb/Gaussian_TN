import sys
import time as t
import torch as to
import numpy as np
import scipy as sp
import scipy.linalg as spLA
from multiprocessing import Pool

sys.path.append('/home/maxime/Travail/Modules')
import Mods_TNS.HamilMPO as H_MPO
import Mods_TNS.TNS_basics as TNSb


## Tester differences entre calcul sur GPU et sparse

#---------------------------------------------------------------
## Applies a one-body gate. on a state (or an operator) Psi
def Apply_OBG(U, Psi, l):
        ## Applies evol. on the given qubit
        newPsi = np.moveaxis(np.tensordot(U, Psi, (0,l)), 0, l)
        return(newPsi)

#---------------------------------------------------------------
## Applies an n-body unitary gate U on a state (or an operator) Psi
def Apply_2BG(U, Psi, ij, sys_info):
	L = sys_info.params['L']

	order = np.zeros(len(Psi.shape), dtype=int)
	a = np.arange(0,len(Psi.shape))
	mask = (a != ij[0]) * (a != ij[1])
	order[ij] = np.array((0,1))
	order[mask] = a[2:]

	## Applies evol. on the corresponding qbits, and replace the dimensions in the correct order
	newPsi = np.transpose(np.tensordot(U, Psi, ((2,3), (ij[0],ij[1]))), axes = order)
	return(newPsi)

#---------------------------------------------------------------
## Applies an n-body unitary gate U on a state (or an operator) Psi
def Apply_nBG(U, Psi, ij, sys_info):
	L = sys_info.params['L']

	n = ij.shape[0]

	## Tensordot changes the order of the dimensions after application of the gate, we put it back in a sorted order (0,1,2, etc.)
	order = np.zeros(len(Psi.shape), dtype=int)
	a = np.arange(0,len(Psi.shape))
	mask = np.ones(len(Psi.shape), dtype=bool)
	mask[ij] = False
	order[ij] = np.arange(n)
	order[mask] = a[n:]
	ordU = np.arange(n,2*n,1)

	## Applies evol. on the corresponding qbits, and replace the dimensions in the correct order
	newPsi = np.transpose(np.tensordot(U, Psi, (ordU,ij)), axes = order)

	return(newPsi)

#---------------------------------------------------------------
## Applies a two-body unitary gate U on a state (or an operator) Psi
def torchApply_1BG(U, Psi, i, L, reverse=False):
	if reverse==False:
		order = np.zeros(L, dtype=int)
		a = np.arange(0,L)
		mask = (a != i)
		order[i], order[mask] = 0, a[1:]
		newPsi = to.movedim(to.tensordot(U, Psi, dims=([1],[i])), list(order), list(a))
	else:
		order = np.zeros(L, dtype=int)
		a = np.arange(0,L) + L
		mask = (a != i)
		order[i-L], order[mask] = 2*L-1, a[:-1]
		newPsi = to.movedim(to.tensordot(Psi, U, dims=([i], [0])), list(order), list(a))

	return(newPsi)

#---------------------------------------------------------------
## Applies a two-body unitary gate U on a state (or an operator) Psi
def torchApply_2BG(U, Psi, ij, L):
	## Tensordot changes the order of the dimensions after application of the gate, we put it back in a sorted order (0,1,2, etc.)
	order = np.zeros(L, dtype=int)
	a = np.arange(0,L)
	mask = (a != ij[0]) * (a != ij[1])
	order[ij] = np.array((0,1))
	order[mask] = a[2:]

	## Applies evol. on the corresponding qbits, and replace the dimensions in the correct order
	newPsi = to.movedim(to.tensordot(U, Psi, dims=([2,3],[ij[0],ij[1]])), list(order), list(a))

	return(newPsi)

#---------------------------------------------------------------
## Defines an identity Fock state 
def IdOperator(d, L, **kwargs):
	dim = np.ones(2*L,dtype=int) * d # Everyone has the same local dimension
	Id = np.identity((d)**L, dtype=complex)
	Psi = {"Op":Id.reshape(dim) }
	return(Psi)

#---------------------------------------------------------------
## Defines an identity operator in Fock space 
def IdState(L):
	dim = np.ones(L,dtype=int) * d # Everyone has the same local dimension
	Psi = np.zeros(dim, dtype=complex)
	return(Psi)

#---------------------------------------------------------------
## Returns the Trace of the product A+ B(Tr[A+B])
def TraceOps(A, B):
	L = int(len(B.shape)/2)
	## 2 exact representations
	if type(A) == type(B): 
		order = np.arange(0, len(A.shape), 1)
		orderT = order.copy()
		#orderT[int(len(A.shape)/2):], orderT[:int(len(A.shape)/2)] = order[:int(len(A.shape)/2)], order[int(len(A.shape)/2):] 
		res = np.tensordot(A.conjugate(), B, (orderT, order)) / 2**L
	
	## 1 exact & 1 TNS (A is the TNS)
	else:
		res = B.reshape(((1,)+B.shape))
		for key in A.keys():
			## We want to compare the B matrices of both MPOs
			if key[0] == "B":
				i = key[1]
				res = np.tensordot(A[key].conjugate(), res, ((2,1,0),(0,L-i+1,1)))

	return(res)

#---------------------------------------------------------------
## Returns the Trace of the product A+ B(Tr[A+B])
def torchTraceOps(A, B):
	L = int(len(B.shape)/2)

	## 2 exact representations
	if type(A) == type(B): 
		order = list(np.arange(0, len(A.shape), 1))
		res = to.tensordot(A.conj(), B, dims=(order, order)) / 2**L
		return(res)
	
	## 1 exact & 1 TNS (A is the TNS)
	else:
		L = int(len(B.size())/2)
		res = {0:B.reshape(((1,)+B.size()))}
		for key in A.keys():
			## We want to compare the B matrices of both MPOs
			if key[0] == "B":
				i = key[1]
				res[i+1] = to.tensordot(A[key].conj(), res[i], dims=([2,1,0],[0,L-i+1,1]))

		return(res[i+1])

#---------------------------------------------------------------
## Apply a series of commuting gates 
def Trott_step_exact(Psi, dt, sys_info, step, hamil=None):
	d, L = sys_info.params['d'], sys_info.params['L']
	## Defines the MPOs
	if hamil==None:
		if step in sys_info.locH:
			U = spLA.expm(-1.j * sys_info.locH[step] * dt).reshape((d,)*sys_info.locH["sh"+step], order="C")
			diffG = False
		else:
			diffG = True
	else:
		U, H = H_MPO.LocalHamil(hamil, dt, **sys_info.params)
	## We apply each gate of the given Trotter step
	for i, gate in enumerate(sys_info.gates[step]):
		if diffG:
			U = spLA.expm(-1.j * sys_info.locH[step,i] * dt).reshape((d,)*sys_info.locH["sh"+step], order="C")
		Psi = Apply_nBG(U, Psi, gate, sys_info)

	return(Psi)
