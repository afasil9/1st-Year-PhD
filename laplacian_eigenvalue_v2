#%%
import dolfinx, ufl
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as la
from mpi4py import MPI
from dolfinx import mesh
from ufl import form
from dolfinx.fem import form

n=2
domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n, mesh.CellType.quadrilateral)

FE=ufl.FiniteElement("Lagrange", domain.ufl_cell(),1)

V=dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))
u=ufl.TrialFunction(V)
v=ufl.TestFunction(V)

# Defining the forms
a=form(ufl.inner(ufl.grad(u),ufl.grad(v))*ufl.dx)
b=form(ufl.inner(u,v)*ufl.dx)

def petsc2array(v):
    s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
    return s
#%%
# Boundary Condtions

dim = domain.topology.dim - 1
domain.topology.create_connectivity(dim, domain.topology.dim)


boundary = np.where(np.array(dolfinx.cpp.mesh.exterior_facet_indices(domain.topology)) == 1)[0]
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, dim, boundary)

N = V.dofmap.index_map.size_global * V.dofmap.index_map_bs


freeinds = np.setdiff1d(range(N),boundary_dofs,assume_unique=True).astype(np.int32)


#%%
from dolfinx.fem.petsc import assemble_matrix
# A = dolfinx.fem.assemble_matrix(a)

A = assemble_matrix(a)
A.assemble()

ai, aj, av = A.getValuesCSR()
A = sps.csr_matrix((av, aj, ai))[freeinds,:][:,freeinds]

na = A.toarray()

B = assemble_matrix(b)
B.assemble()

bi, bj, bv = B.getValuesCSR()
B = sps.csr_matrix((bv, bj, bi))[freeinds,:][:,freeinds]

nb = B.toarray()

# na = petsc2array(A)
# nb = petsc2array(B)
#%%

import scipy.sparse.linalg as la

# Solver
k=5
vals, vecs = la.eigs(na, k=k, M=nb, sigma=0)
ths=np.pi**2*np.array([2,5,5,8,10])

print("Theoretical value | Calculated one | Difference")
for i in range(k): print(f"{ths[i]:1.14f}"+" | "+f"{np.real(vals[i]):.4f}"+f"{np.imag(vals[i]):+.0e}"+"j | "+f"{np.abs(ths[i]-vals[i]):00.2e}")
# %%
