#-----------------------------------------------
#  Import libraries to get the code working.
#-----------------------------------------------
from dolfin import *
from mshr import *
import sys, os, sympy, shutil, math
import numpy as np
from matplotlib import pyplot as plt # ivan

#----------------------------------------------------------------
# this is a scale factor
#----------------------------------------------------------------
Scale = 1

#------------------------------------------------------
# Material constant. The values of parameters involved
# in the models are defined here.
#------------------------------------------------------
E  = 29e9
nu = 0.3

# In this case this quantity contains the density, so
# gravity =  rho * g, with g the gravity acceleration.

rho    = 2.7e3
g    = 9.8
gravity = rho*g

ell  = 0.1 * Scale;

ellv = ell/sqrt(2)
k_ell = Constant(1.e-6) # residual stiffness

# We use the relationship;  Gc = c_w *ell * w1 / sqrt(2)
# In the compression case " sigma_M^2 = w1 * E" is larger than
# "sigma_M" of the traction case.  
w1	=	Constant( 1500.0)

#------------------------------------------------------
# Numerical parameters of the alternate minimization.
# The algorithm stops when it reaches "maxiter" iterations,
# or alternatively it stops when the error between two
# sucessive iterations reaches the tolerance "toll".
#------------------------------------------------------
maxiter = 100 
toll    = 1e-5 

#-------------------------------------------------------------
# The output is stored in a folder with path especified
# in "savedir".
# The files are stored in a folder named "modelname".
#--------------------------------------------------------------
modelname = "Cavity_w1-1500_P1_kw-25e3"
#modelname = "Cavity_w1-1500_quad_quad"

# others
regenerate_mesh = True

savedir = "results/%s"%(modelname)
if os.path.isdir(savedir):
    shutil.rmtree(savedir)
    
#----------------------------------------------------------------
# Parameters to define the geometry if apply are defined here.
# Geometry
#----------------------------------------------------------------
ndim = 3 

#----------------------------------------------------------------------------
# In this block we define boundary sets for boundary conditions.
# This depends on particular tests. In this case the mesh is readed
# from external files.  
#----------------------------------------------------------------------------
# boundaries labels. These are availables in the file "*_faces.xml"
# Labels for cavities
CAVEUP      = 101
CAVEBOTTOM  = 102
CAVEMID     = 103

# Labels for external boundaries
BOXUP       = 201
BOXMIDX1    = 202
BOXMIDX2    = 203
BOXMIDY1    = 204
BOXMIDY2    = 205
BOXBOTTOM   = 206

#------------------------------------------------------
# Read mesh and boundaries from external files.
#------------------------------------------------------
meshname = "Socavacion_incrArea5000_maxArea200000_"

# read mesh
mesh = Mesh("Mesh/" + meshname + str(0) +".xml");
mesh.init()

# read boundaries
boundaries = MeshFunction('size_t', mesh, "Mesh/" + meshname + str(0)+"_faces.xml")

mesh_xmax,mesh_ymax,mesh_zmax = mesh.coordinates().max(axis=0)
print ("ZMAX:", mesh_zmax)

mesh_xmin,mesh_ymin,mesh_zmin = mesh.coordinates().min(axis=0)
print ("ZMIN:", mesh_zmin)

# normal vectors
normal_v = FacetNormal ( mesh)

#------------------------------------------------------
# Create function space for 3D elasticity + Damage
#------------------------------------------------------
V_vector     = VectorFunctionSpace ( mesh, "CG", 1)
V_scalar     = FunctionSpace       ( mesh, "CG", 1)
V_tensor     = TensorFunctionSpace ( mesh, "DG", 1)

#----------------------------------------------------------------------------
# Boudary conditions.
# Definiton of Dirichlet as well as homogeneous Neumann.
# Notice that homegeneous Neumann b.c. are equivalent to omite any infomation
# at boundaries.
#----------------------------------------------------------------------------
# Brief description about the set of boundary conditions for the displacement
# field.
#----------------------------------------------------------------------------
bc_boxmidx1  = DirichletBC ( V_vector.sub(1), Constant(0.0), boundaries, BOXMIDX1)
bc_boxmidx2  = DirichletBC ( V_vector.sub(1), Constant(0.0), boundaries, BOXMIDX2)

bc_boxmidy1  = DirichletBC ( V_vector.sub(0), Constant(0.0), boundaries, BOXMIDY1)
bc_boxmidy2  = DirichletBC ( V_vector.sub(0), Constant(0.0), boundaries, BOXMIDY2)

bc_boxbottom = DirichletBC ( V_vector.sub(2), Constant(0.0), boundaries, BOXBOTTOM)

bc_u = [ bc_boxbottom]

# Newmann boundary condition
kx = gravity
g_bc_zz =  Expression ( 'k * (mesh_zmax - x[2])', degree = 2, k = kx,mesh_zmax = mesh_zmax) 

#----------------------------------------------------------------------------
# Brief description about the set of boundary conditions for the damage.
#----------------------------------------------------------------------------
# For the left boundary the damage is set to zero, that means Dirichlet b.c.
bc_alpha_up = DirichletBC ( V_scalar, 0.0, boundaries, BOXUP)

bc_alpha = [ ]

#----------------------------------------------------------------------------
#  In this a prescribed stress is imposed. 
#----------------------------------------------------------------------------
sigma00 = np.zeros( ( 3,3) )

#----------------------------------------------------------------------------
sigma00 = Constant ( sigma00)

# The boundary condition " sigma00 * n " is impossed via Neumann boundary
# conditions.  We set the vector of 
g_bc =  sigma00 * normal_v   

