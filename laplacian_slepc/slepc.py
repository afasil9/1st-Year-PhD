#%%
# Import modules
from ufl import dx, curl, inner, TrialFunction, TestFunction, SpatialCoordinate, grad
import ufl
import numpy as np
from dolfinx import fem, mesh
from dolfinx.fem import (dirichletbc, Function, FunctionSpace, form,
                         VectorFunctionSpace, locate_dofs_topological)
from mpi4py import MPI
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc
from dolfinx.mesh import (create_unit_square, locate_entities_boundary, create_rectangle,
                          CellType, GhostMode, DiagonalType)
from petsc4py.PETSc import ScalarType
import sys
from scipy import linalg
from scipy.sparse.linalg import eigs
from dolfinx.fem.petsc import LinearProblem
import sys, slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc

n = 100

domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n, mesh.CellType.triangle)

def boundary(x):
    return boundary_lr(x) | boundary_tb(x)

def boundary_lr(x): #Setting Left and Right boundaries to 0
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)


def boundary_tb(x): #Setting Top and bottom boundaries to 0
    return np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)


x = SpatialCoordinate(domain)
gdim = domain.geometry.dim
fdim = domain.topology.dim - 1

# facets = mesh.locate_entities_boundary(domain, fdim, marker= lambda x:np.logical_or(np.isclose(x[0], 0.0),
        # np.isclose(x[0], 1.0)))

facets = locate_entities_boundary(domain, fdim, boundary)

V = fem.FunctionSpace(domain, ("Lagrange",1))

N = V.dofmap.index_map.size_global
dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)
freeinds = np.setdiff1d(range(N),dofs,assume_unique=True).astype(np.int32)

bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

#Function spaces

u = TrialFunction(V)
v = TestFunction(V)
a = form(inner(grad(u), grad(v)) * dx)  

L = form(inner(u, v) * dx)

A = assemble_matrix(a, bcs = [bc])
A.assemble()

# A.view()

B = assemble_matrix(L, bcs = [bc])
B.assemble()

# def petsc2array(v):
#     s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
#     return s

# np_a = petsc2array(A)[freeinds,:][:, freeinds]#Values of the matrix that are not part of the BC / boundary.
# np_b = petsc2array(B)[freeinds,:][:, freeinds]

#%%
#SLEPc version

E = SLEPc.EPS().create(PETSc.COMM_WORLD)
E.create()

E.setOperators(A,B)
E.setProblemType(SLEPc.EPS.ProblemType.GHEP)

n_eigs = 10

E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE)

E.setDimensions(n_eigs, PETSc.DECIDE, PETSc.DECIDE)

E.setFromOptions()
E.solve()

history = []
def monitor(eps, its, nconv, eig, err):
    if nconv<len(err): history.append(err[nconv])
E.setMonitor(monitor)


Print = PETSc.Sys.Print

Print()
Print("******************************")
Print("*** SLEPc Solution Results ***")
Print("******************************")
Print()

its = E.getIterationNumber()
Print( "Number of iterations of the method: %d" % its )

eps_type = E.getType()
Print( "Solution method: %s" % eps_type )

nev, ncv, mpd = E.getDimensions()
Print( "Number of requested eigenvalues: %d" % nev )

tol, maxit = E.getTolerances()
Print( "Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit) )

nconv = E.getConverged()
Print( "Number of converged eigenpairs %d" % nconv )

if nconv > 0:
  # Create the results vectors
  vr, vi = A.createVecs()
  #
  Print()
  Print("        k          ||Ax-kx||/||kx|| ")
  Print("----------------- ------------------")
  for i in range(min(nconv, n_eigs)):
    k = E.getEigenpair(i, vr, vi)
    error = E.computeError(i)
    if k.imag != 0.0:
      Print( " %9f%+9f j %12g" % (k.real, k.imag, error) )
    else:
      Print( " %12f       %12g" % (k.real, error) )
  Print()

# %%
