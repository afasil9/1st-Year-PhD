#%%
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

def L2_norm(v: Expr):
    """Computes the L2-norm of v
    """
    return np.sqrt(MPI.COMM_WORLD.allreduce(
        assemble_scalar(form(inner(v, v) * dx)), op=MPI.SUM))

n = 10 # Mesh Size
domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
gdim = domain.geometry.dim


V = fem.FunctionSpace(domain, ("Lagrange", 1))

facets = mesh.locate_entities_boundary(domain, dim=(domain.topology.dim - 1),
                                       marker=lambda x: np.isclose(x[0], 0.0)|np.isclose(x[0], 1.0)|np.isclose(x[1], 1.0)|np.isclose(x[2], 1.0)|
                                                                      np.isclose(x[1], 0.0)|np.isclose(x[2], 0.0))
dofs = fem.locate_dofs_topological(V=V, entity_dim=2, entities=facets)

mt = mesh.meshtags(domain, 2, facets,3)
mt.name = "facets"

# xdmf = io.XDMFFile(domain.comm, "mesh.xdmf", "w")
# xdmf.write_mesh(domain)
# xdmf.write_meshtags(mt, domain.geometry)

alpha_in = 1
beta_in = 1


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = ufl.as_vector((1,1,1))


# Exact Solution stuff

# x = SpatialCoordinate(domain)
# uex = as_vector((sin(pi*x[0]), sin(pi*x[1]), sin(pi*x[2]))) 
# f = curl(alpha_in*curl(uex)) + beta_in*uex

alpha = fem.Constant(domain, default_scalar_type(alpha_in))
beta = fem.Constant(domain, default_scalar_type(beta_in))

a = inner(alpha * curl(u), curl(v)) * dx + inner(beta * u, v) * dx
L = inner(f,v) * dx

#%%
#Boundary Conditions

cond = fem.Function(V)
# cond.x.array[:]= 0
cond.interpolate(lambda x: np.vstack((np.sin(np.pi*x[0]), np.sin(np.pi*x[1]), np.sin(np.pi*x[2]))))
bc = fem.dirichletbc(cond, dofs)


# problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
problem = LinearProblem(a, L, bcs=[bc])
uh = problem.solve()

e = L2_norm(uex - uh)

print(e)

V0 = fem.FunctionSpace(domain, ("Discontinuous Lagrange", 1, (gdim,)))
vfun = fem.Function(V0, dtype=default_scalar_type)
vfun.interpolate(uh)


# with io.XDMFFile(domain.comm, "results.xdmf", "w") as file:
#     file.write_mesh(domain)
#     file.write_function(vfun)

# with io.VTXWriter(domain.comm,("results.bp"), [uh]) as vtx:
# #     vtx.write(0.0)

# from dolfinx.io import VTXWriter
# with VTXWriter(domain.comm, "output_nedelec.bp", vfun, "bp4") as f:
#     f.write(0.0)


# x = [10,20,30]
# y = [0.11092821571695327,0.05559946158233355,0.037509941698718824]

# plt.plot(x,y)
# plt.yscale('log')
# plt.xscale('log')
# plt.show()
# %%
