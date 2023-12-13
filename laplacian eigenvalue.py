#%%
from ufl import dx, curl, inner, TrialFunction, TestFunction, SpatialCoordinate, grad
import ufl
import numpy as np
from dolfinx import fem, mesh
from dolfinx.fem import (dirichletbc, Function, FunctionSpace, form,
                         VectorFunctionSpace, locate_dofs_topological)
from mpi4py import MPI
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc
from dolfinx.mesh import (create_unit_square, locate_entities_boundary,
                          CellType, GhostMode, DiagonalType)
from petsc4py.PETSc import ScalarType
import sys
from scipy import linalg
from scipy.sparse.linalg import eigs
from dolfinx.fem.petsc import LinearProblem

n = 2
domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
x = SpatialCoordinate(domain)
gdim = domain.geometry.dim
fdim = domain.topology.dim - 1

facets = mesh.locate_entities_boundary(domain, fdim, marker= lambda x:np.logical_or(np.isclose(x[0], 0.0),
        np.isclose(x[0], 1.0)))

V = fem.FunctionSpace(domain, ("Lagrange",1))
dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

#Funciton spaces

u = TrialFunction(V)
v = TestFunction(V)
a = form(inner(grad(u), grad(v)) * dx)  


sp = fem.create_sparsity_pattern(a)

# f = 6 #10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
L = form(inner(u, v) * dx)

A = assemble_matrix(a, bcs = [bc])
A.assemble()

B = assemble_matrix(L)
B.assemble()

def petsc2array(v):
    s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
    return s

np_a = petsc2array(A)
np_b = petsc2array(B)

#np_b = B.getArray() #Use this if B was a vector

# Identiy 

len = np_a.shape[0]
I = np.identity(len)

from scipy.linalg import eigvals

eigvals(np_a)

# %%
# Eigenvalue stuff Attempt 1

from slepc4py import SLEPc

n_eigs = 12

eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
eps.setOperators(A, B)
eps.setType(SLEPc.EPS.Type.ARNOLDI)
eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eps.setWhichEigenpairs(eps.Which.TARGET_MAGNITUDE)

st = eps.getST()
st.setType(SLEPc.ST.Type.SINVERT)

eps.setDimensions(n_eigs, PETSc.DECIDE, PETSc.DECIDE)
eps.setFromOptions()
eps.solve()

its = eps.getIterationNumber()
eps_type = eps.getType()
n_ev, n_cv, mpd = eps.getDimensions()
tol, max_it = eps.getTolerances()
n_conv = eps.getConverged()

computed_eigenvalues = []
for i in range(min(n_conv, n_eigs)):
    lmbda = eps.getEigenvalue(i)
    computed_eigenvalues.append(np.round(np.real(lmbda), 1))

print(f"Number of iterations: {its}")
print(f"Solution method: {eps_type}")
print(f"Number of requested eigenvalues: {n_ev}")
print(f"Stopping condition: tol={tol}, maxit={max_it}")
print(f"Number of converged eigenpairs: {n_conv}")

print(np.sort(computed_eigenvalues))

# %%
#Attempt 2

n = 10 # No of Eigenvalues

eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
eigensolver.setOperators(A,B)
eigensolver.setType(SLEPc.EPS.Type.JD)
eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eigensolver.setWhichEigenpairs(eigensolver.Which.SMALLEST_REAL)
eigensolver.setDimensions(n)
eigensolver.solve()
vr, vi = A.getVecs()


for i in range(1,eigensolver.getConverged()):
    lmbda = eigensolver.getEigenpair(i, vr, vi)
    print(i)
    print(lmbda/(np.pi))
# %%
