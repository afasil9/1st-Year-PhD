#%%
from ufl import dx, curl, inner, TrialFunction, TestFunction
import numpy as np
from dolfinx import fem, mesh
from dolfinx.fem import (dirichletbc, Function, FunctionSpace, form,
                         VectorFunctionSpace, locate_dofs_topological)
from mpi4py import MPI
from dolfinx.fem.petsc import assemble_matrix
from petsc4py import PETSc
from dolfinx.mesh import (create_rectangle, locate_entities_boundary,
                          CellType, GhostMode, DiagonalType)
import sys
from scipy import linalg
from scipy.sparse.linalg import eigs


n = 1 # Mesh Size
domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
gdim = domain.geometry.dim
fdim = domain.topology.dim - 1

facets=mesh.locate_entities_boundary(domain,fdim,lambda x:np.full(x.shape[1],True,dtype=bool))

V = fem.FunctionSpace(domain, ("Lagrange", 1, (gdim,)))
# V = fem.FunctionSpace(domain, ("N1curl", 1))

dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)

# cond = fem.Function(V)
# cond.interpolate(lambda x: np.vstack((np.sin(np.pi*x[0]), np.sin(np.pi*x[1]))))
# bcs = fem.dirichletbc(cond, dofs)

# Define problem
u = TrialFunction(V)
v = TestFunction(V)
a = form(inner(curl(u), curl(v)) * dx)
b = form(inner(u, v) * dx)

# Assemble matrices
A = assemble_matrix(a,[bcs])
A.assemble()
# Zero rows of boundary DOFs of B. See [1]
B = assemble_matrix(b, [bcs], diagonal =0.0)
B.assemble()

# View Matrix
viewer_a = PETSc.Viewer().createASCII('assembled_matrix_A.txt', 'w')
A.view(viewer_a)

viewer_b = PETSc.Viewer().createASCII('assembled_matrix_B.txt', 'w')
B.view(viewer_b)

# # 
def petsc2array(v):
    s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
    return s


np_a = petsc2array(A)
np_b = petsc2array(B)

# np.fill_diagonal(M, np.where(np.diagonal(M) == 0, 1e-12, np.diagonal(M)))

# test = eigs(A = np_a, k =12, M= np_b)

# %%
