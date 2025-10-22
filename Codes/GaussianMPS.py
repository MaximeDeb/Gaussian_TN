import sys
import numpy as np
import scipy as sp

##      ----------------------------------------------------------------------
##      ----------------------------------------------------------------------

class Fermionic_Operators():
    def __init__(self, L):
        sp_i  = np.array(((0,0), (1,0)),  dtype=complex)  ## can be sigma+
        sm_i = np.array( ((0,1), (0,0)),  dtype=complex)  ## can be sigma-
        sz_i  = np.array(((1,0), (0,-1)), dtype=complex) ## can be sz

        ## We define the occuation super-operators in the many-body basis
        self.sp  = np.zeros(L, dtype=object)
        self.sm = np.zeros(L, dtype=object)
        self.sz = np.zeros(L, dtype=object)

        self.c = np.zeros(L, dtype=object)
        self.n = np.zeros(L, dtype=object)
        self.gam = np.zeros(2*L, dtype=object)

        Pstring = np.identity(2**L)
        for i in range(L):
            I1 = np.identity(2**i)
            I2 = np.identity(2**(L-i-1))

            sp_i_L = np.kron(I1, np.kron(sp_i, I2))
            sm_i_L = np.kron(I1, np.kron(sm_i, I2))
            sz_i_L = np.kron(I1, np.kron(sz_i, I2))

            self.sp[i] = sp_i_L
            self.sm[i] = sm_i_L
            self.sz[i] = sz_i_L

            self.c[i]  = (Pstring @ sm_i_L)
            self.n[i]  = 0.5 * (np.identity(2**L) - sz_i_L)
            
            self.gam[2*i-1] =  self.c[i].conj().T + self.c[i]
            self.gam[2*i] =  -1j * (self.c[i].conj().T - self.c[i])

            Pstring = Pstring @ sz_i_L


##      ----------------------------------------------------------------------
##      ----------------------------------------------------------------------

class GaussTN:
    '''
    '''
    def __init__(self, L):
        '''
        '''
        self.L = L
        self.BondDim = np.zeros(L+1, dtype=int)
        MPS = {}
        for l in range(L):
            MPS["B%s"%l] = np.zeros((1,2,1), dtype=complex) 
            MPS["B%s"%l][0,1,0] = 1 ## Start in the state |0000...> = |\psi_G>
            MPS["Lam%s"%l] = np.ones((1,1), dtype=complex) 
            
            self.BondDim[l] = 1
        self.BondDim[l+1] = 1
        self.MPS = MPS
        
        self.frame = np.identity(L, dtype=complex) ## Start with the frame being identity
        self.occupations = np.zeros(L) ## Start with everyone empty -> vacuum state. Update to initialize the state.
        
##      ----------------------------------------------------------------------
##      ----------------------------------------------------------------------


def Apply_NG_Gate(Psi, U, l1, l2): ## Only for local gates
    MPS = Psi.MPS
    
    lshape, rshape = MPS["B%s"%l1].shape[0], MPS["B%s"%l2].shape[-1]

    PhiB = np.tensordot(MPS["B%s"%l1], MPS["B%s"%l2], ([2],[0])) ## Legs correctly ordered
    
    ## UPhi = U * PhiB
    UPhiB = np.tensordot(U, PhiB, ([2,3],[1,2]))
    UPhiB = np.moveaxis(UPhiB, [2,0,1,3], [0,1,2,3]) ## Swap back indices

    ## Splits the tensor back to an MPS
    Phi = np.tensordot(MPS["Lam%s"%l1], PhiB, ([1],[0]))
    
    Phi  = Phi.reshape(lshape * 2, -1)
    PhiB = PhiB.reshape(lshape * 2, -1)
    
    U, S, Vh = np.linalg.svd(Phi, full_matrices=False)
    print(S)
    S = S / np.linalg.norm(S)
    mask = S > 1e-10
    S = S[mask]
    B2 = Vh[:,mask]
    B1 = np.reshape(PhiB, shape=(PhiB.shape[0] * PhiB.shape[1], -1)) @ B2.conj().T

    Psi.MPS["B%s"%l1], Psi.MPS["Lam%s"%l2], Psi.MPS["B%s"%l2] = B1.reshape(lshape, 2, -1), np.diag(S), B2.reshape(-1, 2, rshape)
    Psi.BondDim[l2] = S.shape[0]
    
    return

##      ----------------------------------------------------------------------
##      ----------------------------------------------------------------------

def Apply_G_Gate(Psi, U):
    Psi.frame = U @ Psi.frame

    return

##      ----------------------------------------------------------------------
##      ----------------------------------------------------------------------

L = 8

Psi = GaussTN(L)
Ops = Fermionic_Operators(L)

## Nearest neighbor hamiltonian
H = np.zeros((L,L), dtype=float)
i = np.arange(0,L)
H[i[1:], i[:-1]] = 1
G = H + H.conj().T

## Non-Gaussian gate
n_loc = np.array(((0,0),(0,1.)), dtype=complex)

nG = sp.linalg.expm(-1j * np.pi/4 * np.kron(n_loc, n_loc) ).reshape(2,2,2,2)

##      ----------------------------------------------------------------------
##      ----------------------------------------------------------------------

Apply_G_Gate(Psi, G)
Apply_NG_Gate(Psi, nG, 0, 1)
Apply_NG_Gate(Psi, nG, 1, 2)

print(Psi.BondDim)

Psi0 = np.zeros(2**L, dtype=complex)
Psi0[1] = 1

G = np.identity(L)
g = np.zeros((2**L,2**L),dtype=complex)
for i in range(L):
    for j in range(L):
        g += G[i,j] * Ops.c[i].T @ Ops.c[j]    
Psi_t = sp.linalg.expm(-1j * g) @ Psi0

ng = np.kron(nG.reshape(4,4), np.identity(2**(L-2)))
Psi_t = ng @ Psi_t
print(Psi_t)

