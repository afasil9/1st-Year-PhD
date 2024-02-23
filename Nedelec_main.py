#%%
# Import neccesary modules

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import ds, dx, grad, inner, curl
from dolfinx import fem, io, mesh, plot, default_scalar_type
from dolfinx.fem import dirichletbc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from ufl.core.expr import Expr
from petsc4py import PETSc
from ufl import SpatialCoordinate, as_vector, sin, pi, curl
from dolfinx.fem import (assemble_scalar, form, VectorFunctionSpace, Function)
from matplotlib import pyplot as plt
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.fem import petsc

#%%
# 3D Code

n = 10 # Mesh Size
domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
gdim = domain.geometry.dim
top_dim = gdim - 1 #Topological dimension 

V = fem.FunctionSpace(domain, ("N1curl", 1, (gdim,)))

facets = mesh.locate_entities_boundary(domain, dim=(domain.topology.dim - 1),
                                       marker=lambda x: np.isclose(x[0], 0.0)|np.isclose(x[0], 1.0)|np.isclose(x[1], 0.0)|np.isclose(x[1], 1.0)|
                                                                      np.isclose(x[2], 0.0)|np.isclose(x[2], 1.0))
dofs = fem.locate_dofs_topological(V=V, entity_dim=top_dim, entities=facets)

alpha_in = 1 #Permeability
beta_in = 1 #Electric permitivity

alpha = fem.Constant(domain, default_scalar_type(alpha_in))
beta = fem.Constant(domain, default_scalar_type(beta_in))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = ufl.as_vector((0,0.1,0))  #Source Current  

a = form(inner(alpha * curl(u), curl(v)) * dx + inner(beta * u, v) * dx)
L = form(inner(f,v) * dx)

cond = fem.Function(V)
cond.x.array[:]= 0      #Dirichlet Boundary Conditions
bc = fem.dirichletbc(cond, dofs)

# Solver steps

A = assemble_matrix(a, bcs = [bc])
A.assemble()

b = petsc.assemble_vector(L)
petsc.apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, [bc])

uh = fem.Function(V)
uh.vector.duplicate()

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.setFromOptions()
solver.setTolerances(rtol=1e-8)

# Set the initial guess to zero
# uh.vector[:] = 0

solver.solve(b, uh.vector)

#Extract solution 
uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# #%%
# #Post Processing

# # Interpolating Magnetic Vector potential to DG space for post pro 
# V0 = fem.FunctionSpace(domain, ("Discontinuous Lagrange", 1, (gdim,)))
# vfun = fem.Function(V0, dtype=default_scalar_type)
# vfun.interpolate(uh)

# # Computing B Field
# VB = fem.FunctionSpace(domain, ("Discontinuous Lagrange", 1, (gdim,)))
# B = fem.Function(VB)
# B_3D = curl(uh)
# Bexpr = fem.Expression(B_3D, VB.element.interpolation_points())
# B.interpolate(Bexpr)

# # Output to bp file
# from dolfinx.io import VTXWriter
# with VTXWriter(domain.comm, "output_nedelec.bp", vfun , "bp4") as f:
#     f.write(0.0)

#%%

#Exact Solutions

def exact(n):
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    gdim = domain.geometry.dim
    top_dim = gdim - 1 #Topological dimension 

    V = fem.FunctionSpace(domain, ("N1curl", 1, (gdim,)))

    facets = mesh.locate_entities_boundary(domain, dim=(domain.topology.dim - 1),
                                        marker=lambda x: np.isclose(x[0], 0.0)|np.isclose(x[0], 1.0)|np.isclose(x[1], 0.0)|np.isclose(x[1], 1.0)|
                                                                        np.isclose(x[2], 0.0)|np.isclose(x[2], 1.0))
    dofs = fem.locate_dofs_topological(V, entity_dim=top_dim, entities=facets)

    alpha_in = 1 #Permeability
    beta_in = 1 #Electric permitivity

    alpha = fem.Constant(domain, default_scalar_type(alpha_in))
    beta = fem.Constant(domain, default_scalar_type(beta_in))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = SpatialCoordinate(domain)
    uex = as_vector((sin(pi*x[0]), sin(pi*x[1]), sin(pi*x[2])))
    f = curl(curl(uex)) + uex

    a = form(inner(alpha * curl(u), curl(v)) * dx + inner(beta * u, v) * dx)
    L = form(inner(f,v) * dx)

    cond = fem.Function(V)
    cond.x.array[:]= 0      #Dirichlet Boundary Conditions
    bc = fem.dirichletbc(cond, dofs)

    # # Solver steps

    A = assemble_matrix(a, bcs = [bc])
    A.assemble()

    b = petsc.assemble_vector(L)
    petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                    mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)
    uh.vector.duplicate()

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setFromOptions()
    solver.setTolerances(rtol=1e-8)

    # Set the initial guess to zero
    # uh.vector[:] = 0

    solver.solve(b, uh.vector)

    #Extract solution 
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return uh
# %%
def L2_norm(v: Expr):
    """Computes the L2-norm of v
    """
    return np.sqrt(MPI.COMM_WORLD.allreduce(
        assemble_scalar(form(inner(v, v) * dx)), op=MPI.SUM))


uex = exact(n)
diff = uh - uex
L2_norm(diff)

# %%
