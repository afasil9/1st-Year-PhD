#%%
from ufl import dx, curl, inner, TrialFunction, TestFunction
import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI
from scipy import linalg
from scipy.sparse.linalg import eigs


n = 1 # Mesh Size
domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
gdim = domain.geometry.dim
fdim = domain.topology.dim - 1

facets=mesh.locate_entities_boundary(domain,fdim,lambda x:np.full(x.shape[1],True,dtype=bool))

V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
# V = fem.FunctionSpace(domain, ("N1curl", 1))

dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)

bcs = fem.dirichletbc(np.array([0.0, 0.0], dtype=np.float64), dofs, V)

# Define problem
u = TrialFunction(V)
v = TestFunction(V)
a = fem.form(inner(curl(u), curl(v)) * dx)
b = fem.form(inner(u, v) * dx)

# Assemble matrices
A = fem.assemble_matrix(a,[bcs])
# Zero rows of boundary DOFs of B. See [1]
B = fem.assemble_matrix(b, [bcs])

Asp = A.to_scipy()
Bsp = B.to_scipy()


w, v = eigs(Asp, k=6, M=Bsp)

print(w)