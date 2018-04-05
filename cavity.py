#-----------------------------------------------------------------------------
# Author        : Gino I. Montecinos and Javier Ramirez.
# Date and place: January 2018, CMM-Uchile. 
#-----------------------------------------------------------------------------
# Context       :
#
#  This code is generated in the context of the project
# "The influence of the hidraulic fracture on the seismic activity".
#---------------------------------------
# Purposes      :
# This script simulates fracture mechanics via a coupled linear
# elasticity - gradient damage model. The total energy is computed
# and thus a variational formulation is then derived.
#  
# This code follows the ideas of the references.
#
# [1] B. Bourdin, J.J. Marigo, C. Maurini, P. Sicsic,
#     Morphogenesis and propagation of complex cracks induced by thermal shocks,
#     Physical Review Letters 112, 014301 (2014). 
#     http://arxiv.org/pdf/1310.0501.pdf
#
# [2] P.Sicsic, J.J. Marigo, C. Maurini
#     Initiation of a periodic array of cracks in the thermal shock problem:
#     a gradient damage modeling,
#     Journal of the Mechanics and Physics of Solids 63, (2014).
#
#   
# [3] C. Maurini. Recipies of FEniCS codes. 2015.
#     https://hal.archives-ouvertes.fr/hal-00843625
#
# Further contributions have beend done by:
# Ivan Rojas.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# For each test problem we set parameters in the file TestParameters

# The new set of parameters meshes and folders where the data is stored
# is fixed into the file "Tesparameter"


#-----------------------------------------------
#  Import libraries to get the code working.
#-----------------------------------------------
#from TestNeumannW1_900 import *
#from TestNeumannW1_1100 import *

from TestNeumannW1_1500 import *

tic0=time() 
set_log_level(ERROR) # log level
set_log_level(PROGRESS)
#-----------------------------------------------

print ("-----------------------------------------")
print ("The Code Has Started.")
print ("-----------------------------------------")


# ------------------------------------------------------
# Parameters for an optimal compilation.
# ------------------------------------------------------
#parameters.parse()   # read paramaters from command line  #JSM
# set some dolfin specific parameters
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["allow_extrapolation"] = True


#----------------------------------------------------------------
# The minimization procedure requires parameters to get a suitable
# performance. The following is a suitable set of arrangements.
#----------------------------------------------------------------
solver_minimization_parameters =  {"method" : "gpcg", 
                            "linear_solver" : "gmres",
                            #--------------------------------
                            # These are parameters for optimization
                            #--------------------------------
                            "line_search": "armijo",
                            #"preconditioner" : "bjacobi", 
                            "preconditioner" : "jacobi", 
                            "maximum_iterations" :200,  
                            "error_on_nonconvergence": False,
                            #--------------------------------
                            # These are parameters for linear solver
                            #--------------------------------
                            "krylov_solver" : {
                                "maximum_iterations" : 200, 
                                "report" : True,
                                "monitor_convergence" : False,
                                "relative_tolerance" : 1e-8 
                            }
                            #--------------------------------
                           }


#----------------------------------------------------------------
# The linear solver requires parameters to get a suitable
# performance. The following is a suitable set of arrangements.
#----------------------------------------------------------------
solver_LS_parameters =  {"linear_solver" : "cg", 
                            "symmetric" : True, 
                            #"preconditioner" : "bjacobi", 
                            "preconditioner" : "jacobi", # ivan
                            "krylov_solver" : {
                                "report" : True,
                                "monitor_convergence" : False,
                                "relative_tolerance" : 1e-8 
                                }
                            }


#------------------------------------------------------
# In this block of code define the operators.  These is
# independent from the mesh.
#------------------------------------------------------

# Constitutive functions of the damage model. Here
# we define the operators acting on the damage system
# as well as the operator acting on the displacement
# field, which depends on the damage.
#------------------------------------------------------
Kw=Constant(9.e3)

mu    = E / ( 2.0 * ( 1.0 + nu))
lmbda = E * nu / ( 1.0 - nu**2)
ffD=lmbda/(lmbda+2*mu)


def Desviador(Tensor):
    return Tensor-Esferico(Tensor)
def Esferico(Tensor):
    Tensor2=as_matrix([[ffD,0,0],[0,ffD,0],[0,0,1]])
#    return (1./3)*tr(Tensor)*Identity(ndim)   
    return (inner(Tensor,Tensor2)/inner(Tensor2,Tensor2))*Tensor2
def w(alpha):
    return Kw*(1/(1-alpha)+ln(1-alpha))   #p=1
    #return Kw*(1/(1-alpha)+2*ln(1-alpha)+alpha) #p=2
    #return Kw*((alpha)**(1/2)/(1-alpha)-arctanh(alpha**(1/2)))
    #return Kw*((alpha)**(1/2)/(1-alpha)-(1/2)*ln((1+(alpha**(1/2))-(1-(alpha**(1/2))))))
    #return Kw*((alpha)**(1/2)/(1-alpha))
    #return w1*alpha**2
    #return w1*(1 - (1 - alpha)**(p/2))
    #return w1*(1 - exp(-p*alpha))

def A(alpha):
    return (1-alpha) #JSM
    #return (1-alpha)**2
    #return (1 - alpha)**p
    #return exp(-p*alpha)

#------------------------------------------------------
# Strain and stress in free damage regime.
#------------------------------------------------------
def eps ( v):
    return sym ( grad ( v) )

def sigma_0 ( eps):
    return 2.0 * mu * ( eps) + lmbda * tr (eps ) * Identity ( ndim)

#-------------------------------------------------------------------------------
# Fenics forms for the energies. This is affected by the damage in terms of the
# modification of the Young modulus. As this relationship is linear, we can write it
# as follows.
#-------------------------------------------------------------------------------
def sigma ( eps, alpha):
    return ( A ( alpha) + k_ell ) * sigma_0 ( eps)

#-------------------------------------------------------------------------------
#  Define the energies.  Total energy  div + deviatorica.
#-------------------------------------------------------------------------------
def energy_w ( u, alpha):
    mu    =  A ( alpha) * E / ( 2.0 * ( 1.0 + nu))
    lmbda =  A ( alpha) * E * nu / ( 1.0 - nu**2)


    Es = 0.5 * eps ( u) - tr ( eps ( u) ) / 3 * Identity ( ndim)
    Eploc = tr (eps (u) )
    return (  (lmbda / 2 + mu / 3) * Eploc**2 \
              + mu * inner (Es, Es) )

#-------------------------------------------------------------------------------
#  Define the energies. Deviatorica.
#-------------------------------------------------------------------------------

def energy_dev ( u, alpha):
    mu    =  A ( alpha) * E / ( 2.0 * ( 1.0 + nu))
    lmbda =  A ( alpha) * E * nu / ( 1.0 - nu**2)

    Es = 0.5 * eps ( u) - tr ( eps ( u) ) / 3 * Identity ( ndim)

    return (  mu * inner (Es, Es)   )

#-------------------------------------------------------------------------------
#  Define the energies. Div.
#-------------------------------------------------------------------------------

def energy_div ( u, alpha):
    mu    =  A ( alpha) * E / ( 2.0 * ( 1.0 + nu))
    lmbda =  A ( alpha) * E * nu / ( 1.0 - nu**2)
    
    Eploc = tr (eps (u) )
    
    return (  (lmbda / 2 + mu / 3) * Eploc**2 )


#--------------------------------------------------------------------------------
# The influence of body forces. IF no influence is availbale we set this
# constant as zero.
#------------------------------------------------------------------------------
#body_force = Constant( (0.0, 0.0, -gravity) )
body_force = Constant( (0.0, 0.0, -rhoG) )

print('[%d/%d] 209: t=%f s'%(mpi_comm_world().rank+1,mpi_comm_world().size,time()-tic0))


#------------------------------------------------------------------
#  Mesh without cavity.
#------------------------------------------------------------------
# The information concerning the mesh is filled in on the file
# "TestParameters.py".  This subroutine provides the following
# elements:
#
#   mesh       : name of the Computational domain
#   boundaries : physical boundaries and their labels.
#
#
#   bc_u       : Array of boundary conditions for "u".
#   bc_alpha   : Arrat of boundary conditions for "alpha". 
#


#----------------------------------------------------------------------    
#
# Energy dissipated in the creation of a single crack
#----------------------------------------------------------------------    


#----------------------------------------------------------------------------
# Variational formulation begins.
# This is done in TestParameters
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Define the function, test and trial fields. We have two types of operators
# "u, du, v" are vectorial expressions. "alpha, dalpha, beta" are scalar.
#----------------------------------------------------------------------------
u  = Function      ( V_vector, name = "u")
du = TrialFunction ( V_vector)
v  = TestFunction  ( V_vector)

u0  = Function(V_vector, name="u0")
du0 = TrialFunction(V_vector)
v0  = TestFunction(V_vector)

alpha  = Function      ( V_scalar, name = "alpha")
dalpha = TrialFunction ( V_scalar)
beta   = TestFunction  ( V_scalar)

u_aux       = Function ( V_scalar)

W_energy    	= Function ( V_scalar, name = "energy_w" )
dev_energy  	= Function ( V_scalar, name = "energy_dev" )
div_energy  	= Function ( V_scalar, name = "energy_div" )



alphaAux    = Function ( V_scalar)
#----------------------------------------------------------------------------
# Interpolate the initial condition for the damage variable "alpha."
# It uses interpolation with degree = 0, it is because the
# initial condition is a constant. In the case of more general
# initial conditions the degree have to be at least two times
# larger than the degree chosen for solving the variational problems.
#----------------------------------------------------------------------------
alpha_0 =	interpolate ( Expression("0.",degree = 2), V_scalar)

alpha.assign(alpha_0)



#-----------------------------------------s-------------------------------------
# Let us define ds and dx 
#------------------------------------------------------------------------------
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
dx = Measure('dx', domain=mesh)


#------------------------------------------------------------------------------
# Let us define the total energy of the system as the sum of elastic energy,
# dissipated energy due to the damage and external work due to body forces. 
#------------------------------------------------------------------------------
elastic_energy0    = 0.5*inner(sigma_0(eps(u)), eps(u))*dx
elastic_energy1    = 0.5*inner(sigma(eps(u), alpha), eps(u))*dx
elastic_energy2    = 0.5*inner(sigma(eps(u-u0), alpha), eps(u-u0))*dx

external_work     = dot ( body_force, u) * dx
#    			  + Include contributions from Neumann boundary conditions.


external_bc       = dot ( -g_bc_zz * normal_v, u) * ds ( CAVEUP) \
                  + dot ( -g_bc_zz * normal_v, u) * ds ( CAVEBOTTOM) \
                  + dot ( -g_bc_zz * normal_v, u) * ds ( CAVEMID) 


dissipated_energy = ( w ( alpha) +  ellv**2 * w1* \
                     dot ( grad ( alpha), grad ( alpha) ) ) * dx

total_energy0 = elastic_energy0 + dissipated_energy - external_work
total_energy1 = elastic_energy1 + dissipated_energy - external_work
total_energy2 = elastic_energy2 + dissipated_energy - external_work

#-----------------------------------------------------------------------
# Weak form of elasticity problem. This is the formal expression
# for the tangent problem which gives us the equilibrium equations.
#-----------------------------------------------------------------------
E_u0    = derivative ( total_energy0, u, v0)
E_u     = derivative ( total_energy1, u, v)
E_alpha = derivative ( total_energy2, alpha, beta)

# Hessian matrix
E_alpha_alpha = derivative ( E_alpha, alpha, dalpha)

# Writing tangent problems in term of test and trial functions for
# matrix assembly
E_du0     = replace ( E_u0, {u : du0} )
E_du     = replace ( E_u,{u:du} )
E_dalpha = replace ( E_alpha, { alpha : dalpha} )

#------------------------------------------------------------------------------
# Once the tangent problems are formulated in terms of trial and text functions,
# we define the variatonal problems.
#------------------------------------------------------------------------------
# Variational problem for the displacement.
problem_u0  = LinearVariationalProblem( lhs(E_du0), rhs(E_du0), u0, bc_u)
problem_u   = LinearVariationalProblem( lhs(E_du), rhs(E_du), u, bc_u)

# Define the classs Optimization Problem for then define the damage.
# Variational problem for the damage (non-linear to use variational
# inequality solvers of petsc)
class DamageProblem ( OptimisationProblem):
        
    def __init__(self): # ivan
        OptimisationProblem.__init__(self) # ivan

    # Objective vector    
    def f(self, x):
        alpha.vector()[:] = x
        return assemble(total_energy2)

    # Gradient of the objective function
    def F(self, b, x):
        alpha.vector()[:] = x
        assemble(E_alpha, tensor=b)        
        

    # Hessian of the objective function    
    def J(self, A, x):
        alpha.vector()[:] = x
        assemble(E_alpha_alpha, tensor=A)    

# define the minimization problem using the class.
problem_alpha = DamageProblem()
  

print('[%d/%d] 361: t=%f s'%(mpi_comm_world().rank+1,mpi_comm_world().size,time()-tic0))

  
#---------------------------------------------------------------------------- 
# Set up the solvers. Define the object for solving the displacement
# problem, "solver_u".
#----------------------------------------------------------------------------
solver_u0   = LinearVariationalSolver(problem_u0)
solver_u    = LinearVariationalSolver(problem_u)

# Get the set of paramters for the class "solver_u".
# This only requires the solution of linear system solver.
solver_u0.parameters.update(solver_LS_parameters)
solver_u.parameters.update(solver_LS_parameters)

#---------------------------------------------------------------
# we need to define the corresponding object, "solver_alpha".
#---------------------------------------------------------------
# The object associated to minimization is created. 
solver_alpha = PETScTAOSolver ( )

# Get the set of paramters for the class "solver_alpha".
# This requires the solution of a minimization problem.   
solver_alpha.parameters.update ( solver_minimization_parameters)

# As the optimization is a constrained type we need to provide the
# corresponding lower and upper  bounds.
lb = interpolate ( Expression ( "0.0", degree = 0), V_scalar) 
ub = interpolate ( Expression ( "1.0", degree = 0), V_scalar)
lb.vector()[:] = alpha.vector() 


#-----------------------------------------------------------------------
# Define extras.
#-----------------------------------------------------------------------
# Crete the files to store the solution of damage and displacements.
# use .pvd if .xdmf is not working

#file_alphaDiff = File ( savedir + "/alphaDiff.pvd")
file_alpha      = File ( savedir + "/alpha.pvd")
file_energW     = File ( savedir + "/energy_w.pvd")
file_energDev   = File ( savedir + "/energy_dev.pvd")
file_energDiv   = File ( savedir + "/energy_div.pvd")
file_u0         = File(savedir+"/u0.pvd") # use .pvd if .xdmf is not working
file_u          = File ( savedir + "/u.pvd")
file_sigma      = File(savedir+"/sigma.pvd") # use .pvd if .xdmf is not working
file_epsilon    = File(savedir+"/epsilon.pvd") # use .pvd if .xdmf is not working

# Define the function "alpha_error" to measure relative error, used in
# stop criterion.

alpha_error = Function ( V_scalar)

#strain  = eps(u)
#stressG = project(sigma(strain,alpha),V_tensor)
#strainG = project(strain,V_tensor)


#-----------------------------------------------------------------------
#   Main loop.
#-----------------------------------------------------------------------
# Apply BC to the lower and upper bounds
# As the constraints may change we have to
# apply the boundary condition at any interation loop.
for bc in bc_alpha :
    bc.apply ( lb.vector ( ) )
    bc.apply ( ub.vector ( ) )

# Solve elastic problem un time 0
if False :
    solver_u0.solve()

print('[%d/%d] 430: t=%f s ..... u0 solved'%(mpi_comm_world().rank+1,mpi_comm_world().size,time()-tic0))

file_u0     << ( u0, 0.0)

# Alternate mininimization
# Initialization
iter = 1; err_alpha = 1




print('[%d/%d] 444: t=%f s'%(mpi_comm_world().rank+1,mpi_comm_world().size,time()-tic0))




#-------------------------------------------------------------------    
# Iterations of the alternate minimization stop if an error limit is
# reached or a maximim number of iterations have been done.
#------------------------------------------------------------------- 
while False and err_alpha > toll and iter < maxiter :
    # solve elastic problem
    print('[%d/%d] 455: t=%f s. solver_u.solve ( )'%(mpi_comm_world().rank+1,mpi_comm_world().size,time()-tic0))
    solver_u.solve ( )
        
    # solve damage problem via a constrained minimization algorithm.
    solver_alpha.solve ( problem_alpha, alpha.vector ( ),
                        lb.vector ( ), ub.vector ( ) )
                            
    # Test the error, it is employed in one of the stoped criteria.
    # Compute the error vector.
    alpha_error.vector ( )[ :] = alpha.vector ( ) - alpha_0.vector ( )
        
    # Compute the norm of the the error vector.
    err_alpha = np.linalg.norm ( alpha_error.vector ( ).get_local ( ),
                                    ord = np.Inf)
          
    # monitor the results
    if mpi_comm_world().rank == 0:
        print ("Iteration:  %2d, Error: %2.8g, alpha_max: %.8g" \
            % ( iter, err_alpha, alpha.vector ( ).max ( ) ))

    # update the solution for the current alternate minimization iteration.
    alpha_0.assign(alpha)

    iter = iter + 1
	#End of   "while err_alpha > toll and iter < maxiter"

# updating the lower bound with the solution of the solution
# corresponding to the current global iteration, it is for accounting
# for the irreversibility.
lb.vector ( ) [ :] = alpha.vector ( )


if mpi_comm_world().rank == 0:
    print ("-----------------------------------------")
    print("End of the alternate minimization.")
    print ("-----------------------------------------")
        
#---------------------------------------------------------------------------
# End of Main loop.
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Post-processing after having calculated u and alpha
#---------------------------------------------------------------------------
# Store u,alpha, sigma and epsilon
#file_alpha     << ( alpha   , 0.)
#file_u         << ( u, 0.)
#strain      = eps(u)
#stress      = project(sigma(strain,alpha),V_tensor)
#stressG.assign(stress)
#strainG.assign(project(strain,V_tensor))
#file_sigma     << ( stressG, 0.)
#file_epsilon     << ( strainG, 0.)

print('[%d/%d] 507: t=%f s. alpha,u saved...'%(mpi_comm_world().rank+1,mpi_comm_world().size,time()-tic0))

if False :
    
    # Calculate the energies for the geometry without cavities
    elastic_energy_value    = assemble ( elastic_energy1)
    dissipated_energy_value = assemble ( dissipated_energy)
    
    #  Control if alpha is too small
    alpha.vector()[alpha.vector() < 1e-12] = 0.0
    
    # Eval the energy
    W_energy.assign( project ( energy_w ( u, alpha), V_scalar))
    dev_energy.assign( project ( energy_dev ( u, alpha), V_scalar))
    div_energy.assign( project ( energy_div ( u, alpha), V_scalar))
    
    #  Control if energies are too small
    W_energy.vector()[W_energy.vector() < 1e-12] = 0.0
    dev_energy.vector()[dev_energy.vector() < 1e-12 ] = 0.0
    div_energy.vector()[div_energy.vector() < 1e-12 ] = 0.0
    
    # store the energy
    file_energW  << ( W_energy, 0.)
    file_energDev  << ( dev_energy, 0.)
    file_energDiv  << ( div_energy, 0.)

# Store the damage for this geometry
alphaAux.assign ( alpha)

#------------------------------------------------------------------
# The main loop for mesh without cavity  has finished.
#------------------------------------------------------------------
print ("-----------------------------------------")
print ("Geometry without cavity is finished.")
print ("-----------------------------------------")

# Remove previous integrating factors "dx, ds"
del ds, dx


#------------------------------------------------------------------
#  Mesh with cavity.
#------------------------------------------------------------------
# Start loop over new geometries. These are obtained from a sequence
# of geometries which are obtained from an external folder. 
# The number of external files is "NstepW" and the call is driven
# by the counter "itmesh".
itmesh = 30 # JSM
NstepW = 30

# Before start we store the energy associated to the geometry without cavity.
energies   		= np.zeros( ( NstepW, 4) )



print('[%d/%d] 558: t=%f s'%(mpi_comm_world().rank+1,mpi_comm_world().size,time()-tic0))



#---------------------------------------------------------------------------------------
#  Starting the loop of the mesh sequence. It is driven by the index "itmesh".
#---------------------------------------------------------------------------------------

amax=0.0
a0 = Vector(mpi_comm_self())
a1 = Vector(mpi_comm_self())
a2 = Vector(mpi_comm_self())

while itmesh <= NstepW :
    # Read mesh from a sequence of meshes generated externally.
    # These are stored into the folder named "Mesh".
    mesh_new = Mesh ( "Mesh/" + meshname + str(itmesh) + ".xml");
    mesh_new.init()
    
    

    print('[%d/%d] 581: t=%f s'%(mpi_comm_world().rank+1,mpi_comm_world().size,time()-tic0))
    
    
    
    
    # read boundaries for new mesh
    boundaries_new = MeshFunction ('size_t', mesh_new,
                    "Mesh/" + meshname + str(itmesh) + "_faces.xml")

    dsN = Measure('ds', domain=mesh_new, subdomain_data=boundaries_new)
    dxN = Measure('dx', domain=mesh_new)
    
    normal_v_new = FacetNormal ( mesh_new)
    
    # Create new function space for elasticity + Damage
    V_vector_new   = VectorFunctionSpace ( mesh_new, "CG", 1)
    V_scalar_new   = FunctionSpace       ( mesh_new, "CG", 1)
    V_tensor_new   = TensorFunctionSpace ( mesh_new, "CG", 1)
    strainGN = Function(V_tensor_new, name="epsN")
    stressGN = Function(V_tensor_new, name="stressN")
	
	#new boundary conditions
    g_bc_new =  sigma00 * normal_v_new   
    
	#---!----------------------------------------------------------------------------------
    # Observation: To generate a sequence of plots in paraview the name
    # of the variable must be the same. It is achieved by including name="alpha".
    # at the moment of the definition of the structure "alpha".
    #
    #  << alphaN =  Function ( V_scalar_new, name="alpha") >>
    #
    # The same definition needs to be done for displacement "u" and
    # other arrays as the difference of damage without cavity and damage with cavity,
    # "alphaDiff".
    #-------------------------------------------------------------------------------------
    # Define the function, test and trial fields
    uN, duN, vN = Function ( V_vector_new, name="u"), \
                  TrialFunction ( V_vector_new), TestFunction ( V_vector_new)

    
    alphaN, dalphaN, betaN = Function ( V_scalar_new, name="alpha"), \
                             TrialFunction ( V_scalar_new), \
                             TestFunction(V_scalar_new)


    W_energyN    = Function ( V_scalar_new, name = "energy_w" )
    dev_energyN  = Function ( V_scalar_new, name = "energy_dev" )
    div_energyN  = Function ( V_scalar_new, name = "energy_div" )

    
    alpharef  = Function ( V_scalar_new)
      
    # name = "alpha_diff" forces the output format to rename the variables with this name.
    #alphaDiff = Function ( V_scalar_new, name = "alpha_diff")

    #-------------------------------------------------
    # Project the rerence solution into the new mesh.
    # Warning: In parallel implementations this projection
    # may fail.
    #-------------------------------------------------    
    alpharef = interpolate (  alpha, V_scalar_new)
    u0ref    = interpolate (u0, V_vector_new)
    #alpharef = alpha
    # Here use the damage obtained in the previous step, with the previous mesh
    alphaN_0   = interpolate ( alphaAux, V_scalar_new)
    alphaN.assign(alphaN_0)
    #alphaN   = alphaAux 
    #alphaN   = project ( alphaAux, V_scalar_new) 

    # Define the initial damage.
    alphaN_0 = interpolate ( Expression ( "0.0", degree = 0), V_scalar_new) 
    if(itmesh==1):
    	alphaN_0.assign(alpharef)
    else: 
    	alphaN_0.assign(interpolate ( alphaAux, V_scalar_new))
    	#alphaN_0.assign(alphaAux)
    	
    #----------------------------------------------------------------------------    
    # Boudary conditions. Definiton of Dirichlet as well as homogeneous Neumann.
    # Notice that homegeneous Neumann b.c. are equivalent to omite any infomation
    # at boundaries.
    #----------------------------------------------------------------------------
    # Brief description about the set of boundary conditions for the displacement
    # field (Optional).
    #----------------------------------------------------------------------------
    bc_boxmidx1N  = DirichletBC ( V_vector_new.sub(0), Constant(0.0), boundaries_new, BOXMIDX1)
    bc_boxmidx2N  = DirichletBC ( V_vector_new.sub(0), Constant(0.0), boundaries_new, BOXMIDX2)

    bc_boxmidy1N  = DirichletBC ( V_vector_new.sub(1), Constant(0.0), boundaries_new, BOXMIDY1)
    bc_boxmidy2N  = DirichletBC ( V_vector_new.sub(1), Constant(0.0), boundaries_new, BOXMIDY2)

    bc_boxbottomN = DirichletBC ( V_vector_new.sub(2), Constant(0.0), boundaries_new, BOXBOTTOM)

    bc_uN = [bc_boxbottomN,bc_boxmidx1N,bc_boxmidx2N,bc_boxmidy1N,bc_boxmidy2N]
#    bc_uN = [bc_boxbottomN]
        
    # Boundary condition for the damage is damped into the array "bc_alphaN".
    bc_alpha_upN = DirichletBC ( V_scalar_new, 0.0, boundaries_new, CAVEUP )
    bc_alpha_bottomN = DirichletBC ( V_scalar_new, 0.0, boundaries_new, CAVEBOTTOM )
    bc_alpha_midN = DirichletBC ( V_scalar_new, 0.0, boundaries_new, CAVEMID )
    # bc - alpha (Neumann boundary condition for damage on the left boundary)
    bc_alphaN = [ ]
    
    #--------------------------------------------------------------------------------        
    # Let us define the total energy of the system as the sum of elastic energy,
    # dissipated energy due to the damage and external work due to body forces. 
    #--------------------------------------------------------------------------------
    elastic_energy1_new    = 0.5 * inner ( sigma ( eps(uN), alphaN), eps ( uN) ) * dxN
    elastic_energy2_new    = 0.5 * inner ( sigma ( eps(uN-u0ref), alphaN), eps ( uN-u0ref) ) * dxN
    elastic_energy2_new    = 0.5 * inner ( Desviador(sigma ( eps(uN), alphaN)), Desviador(eps ( uN) )) * dxN \
                            - 0.5 * inner ( Esferico(sigma ( eps(uN), alphaN)), Esferico(eps ( uN) )) * dxN
    
    external_work_new     = dot (body_force, uN) * dxN
    # External work includes also the influence of Neumann boundary conditions. 
    #external_bc_new  = dot ( -g_bc_zz * normal_v_new, uN) * dsN ( CAVEUP)  \
    #                 + dot ( -g_bc_zz * normal_v_new, uN) * dsN ( CAVEBOTTOM)  \
    #                 + dot ( -g_bc_zz * normal_v_new, uN) * dsN ( CAVEMID)  

    external_bc_new  = dot ( -g_bc_zz *ffD* normal_v_new, uN) * dsN ( BOXMIDX1)  \
                     + dot ( -g_bc_zz *ffD* normal_v_new, uN) * dsN ( BOXMIDX2)  \
                     + dot ( -g_bc_zz *ffD* normal_v_new, uN) * dsN ( BOXMIDY1)  \
                     + dot ( -g_bc_zz *ffD* normal_v_new, uN) * dsN ( BOXMIDY2)  

            
    dissipated_energy_new = ( w ( alphaN) + ellv**2 *w1 * \
                              dot ( grad ( alphaN), grad ( alphaN) ) ) * dxN

    total_energy1_new = elastic_energy1_new + dissipated_energy_new-external_work_new-external_bc_new
    total_energy2_new = elastic_energy2_new + dissipated_energy_new-external_work_new-external_bc_new

    #--------------------------------------------------------------------------------            
    # Weak form of elasticity problem. This is the formal expression
    # for the tangent problem which gives us the equilibrium equations.
    #-------------------------------------------------------------------------------- 
    E_uN     = derivative ( total_energy1_new, uN, vN)
    E_alphaN = derivative ( total_energy2_new, alphaN, betaN)
    
    # Hessian matrix
    E_alpha_alphaN = derivative ( E_alphaN, alphaN, dalphaN)
    
    # Writing tangent problems in term of test and trial functions for matrix assembly
    E_duN     = replace ( E_uN, { uN : duN} )
    E_dalphaN = replace ( E_alphaN, { alphaN : dalphaN} )
    
    #---------------------------------------------------------------------------------        
    # Once the tangent problems are formulated in terms of trial and text functions,
    # we define the variatonal problems.
    #---------------------------------------------------------------------------------
    # Variational problem for the displacement.
    problem_uN     = LinearVariationalProblem ( lhs ( E_duN), rhs ( E_duN), uN, bc_uN)
    
    #------------------------------------------------------------------------------------
    # For solve the first order equilibrium, we need to create the variational formulation
    # from the tangent problem "E_alpha".
    #------------------------------------------------------------------------------------      
    class DamageProblemN ( OptimisationProblem) :
        def __init__ ( self): # ivan
            OptimisationProblem.__init__ ( self) # ivan
        # Objective vector    
        def f ( self, x):
            alphaN.vector ( ) [ :] = x
            return assemble ( total_energy2_new)
        # Gradient of the objective function
        def F ( self, b, x):
            alphaN.vector ( ) [ :] = x
            assemble ( E_alphaN, tensor = b)        
                       
        # Hessian of the objective function    
        def J ( self, A, x):
            alphaN.vector ( ) [ :] = x
            assemble ( E_alpha_alphaN, tensor = A)
               
    # The following line activates the optimization solver for the damage.    
    problem_alphaN = DamageProblemN ( )
    #-----------------------------------------------------------------------------
    # Set up the solvers       
    #-----------------------------------------------------------------------------                                 
    solver_uN = LinearVariationalSolver ( problem_uN)
    solver_uN.parameters.update ( solver_LS_parameters)
        
    solver_alphaN = PETScTAOSolver ( )
    solver_alphaN.parameters.update ( solver_minimization_parameters)
            
    #--------------------------------------------------------------------------------
    #  For the constraint minimization problem we require the lower and upper bound,
    # "lbN" and "ubN".  They are initialized though interpolations.
    #--------------------------------------------------------------------------------
    lbN	= alphaN_0

    ubN = interpolate ( Expression ( "1.0", degree = 0), V_scalar_new) 

	#---------------------------------------------------------------------------------                                
    # Define the function "alpha_error" to measure relative error, used in
    # stop criterion.
    #---------------------------------------------------------------------------------
    alphaN_error = Function ( V_scalar_new)
    
	#---------------------------------------------------------------------------------                                
    #   Main loop.
    #---------------------------------------------------------------------------------
	#-----------------------------------------------------------------------------                                
    # Apply BC to the lower and upper bounds
    # As the constraints may change we have to
    # apply the boundary condition at any interation loop.
    #-----------------------------------------------------------------------------
    #for bc in bc_alphaN :
    #    bc.apply ( lbN.vector ( ) )
    #    bc.apply ( ubN.vector ( ) )
    
    #-----------------------------------------------------------------------------                                        
    # Alternate mininimization
    # Initialization                
    #-----------------------------------------------------------------------------          
        
    #-------------------------------------------------------------------    
    # Iterations of the alternate minimization stop if an error limit is
    # reached or a maximim number of iterations have been done.
    #-------------------------------------------------------------------  
    iterKm=1
    maxiterKm=1
    while amax<0.99 and iterKm <= maxiterKm :     
        iterKm += 1                                                             
        if mpi_comm_world().rank == 0:
        	print ("=================================================")
        	print ("Comienza Calculo con Wk=%f ( t=%f s.)..."%(Kw,time()-tic0))
        	print ("--------------------------------------------------")
        
        iter = 1; err_alphaN = 1
        while err_alphaN > toll and iter < maxiter :   

            alphaN.vector().gather(a0, np.array(range(V_scalar_new.dim()), "intc"))                             
            # solve elastic problem
            print('[%d/%d] 799: t=%f s. BEGIN solver_uN.solve ( )...'%(mpi_comm_world().rank+1,mpi_comm_world().size,time()-tic0))
            solver_uN.solve ( )
            print('[%d/%d] 802: t=%f s. END solver_uN.solve ( )...'%(mpi_comm_world().rank+1,mpi_comm_world().size,time()-tic0))
                
            err_alpha2=1
            
            while err_alpha2>toll and iter<maxiter:
                
                alphaN.vector().gather(a1, np.array(range(V_scalar_new.dim()), "intc"))    
                print('[%d/%d] 803: t=%f s. solver_alphaN.solve ( ... )...'%(mpi_comm_world().rank+1,mpi_comm_world().size,time()-tic0))
                alphaN.vector ( )[alphaN.vector ( )>.99]=.99
                
                solver_alphaN.solve ( problem_alphaN, alphaN.vector ( ), lbN.vector ( ), ubN.vector ( ) )
                #niterTotal = niterTotal+1
                #-------------------------------------------------------------------
                alphaN.vector().gather(a2, np.array(range(V_scalar_new.dim()), "intc"))
                err_alpha2 = np.linalg.norm(a2 - a1, ord = np.Inf)
                if mpi_comm_world().rank >= 0:
                    print("Process %d: Iteration:  %2d,            aError: %2.8g, alpha_max: %.8g,   [%.8g,%.8g]" \
                        %(mpi_comm_world().rank,iter, err_alpha2, a2.max(), (a2-a0).min(), (a2-a0).max()))
    
                                     
    		# solve damage problem via a constrained minimization algorithm.
            err_alphaN = np.linalg.norm(a2 - a0, ord = np.Inf)
    
            amax=1.0* alphaN.vector ( ).max ( )             
            # monitor the results for the new mesh
            if mpi_comm_world ( ).rank == 0:
                print ("Remesh: %d, Iteration:  %2d, Error: %2.8g, alpha_max: %.8g" \
                    % ( itmesh, iter, err_alphaN, amax ))
            # update the solution for the current alternate minimization iteration.
            alphaN_0.assign ( alphaN)
                        
            iter = iter + 1
                        
                                                                                                
        if mpi_comm_world().rank == 0:
            print ("--------------------------------------------------")
            print ("End of the alternate minimization in Remesh: %d" \
            				%( itmesh ))
            print ("--------------------------------------------------")
    
    	#----------------------------------------------------------------------------------------
        # Once a new damage has been obtained, we store it into an auxiliary variable "alphaAux"
        #----------------------------------------------------------------------------------------
        alphaAux = Function ( V_scalar_new)
    
        #  Control if alpha is too small
        alphaN.vector()[alphaN.vector() < 1e-12] = 0.0
    
        # Assign the values of alpha for later.
        alphaAux.assign ( alphaN)
    
    	#-----------------------------------------------------------------------------------                                                                                            
        # Dump the solution into files.
        # In order to evidence the influence of the cavity we
        # take the difference between the reference solution (without holes)
        # and the solution corresponding to the augmented cylinder.
        #-----------------------------------------------------------------------------------
        #alphaDiff.vector()[:] = alphaN.vector() - alpharef.vector()
        #alphaDiff.vector()[alphaDiff.vector() < 0] = 0.0        
        #file_alphaDiff << ( alphaDiff, 1.0 * itmesh)        
        virTime=float(0.0 * itmesh-1.0*Kw)

        print('[%d/%d] 872: t=%f s. virTime=%f, amax=%f'%(mpi_comm_world().rank+1,mpi_comm_world().size,time()-tic0,virTime,amax))

        strainN      = eps(uN)

        print('[%d/%d] 876: t=%f s. virTime=%f, amax=%f'%(mpi_comm_world().rank+1,mpi_comm_world().size,time()-tic0,virTime,amax))


        stressN      = project(sigma(strainN,alpha),V_tensor_new, solver_type='cg')


        print('[%d/%d] 882: t=%f s. virTime=%f, amax=%f'%(mpi_comm_world().rank+1,mpi_comm_world().size,time()-tic0,virTime,amax))

        stressGN.assign(stressN)


        print('[%d/%d] 887: t=%f s. virTime=%f, amax=%f'%(mpi_comm_world().rank+1,mpi_comm_world().size,time()-tic0,virTime,amax))

        strainGN.assign(project(strainN,V_tensor_new,solver_type='cg'))
        
            
        

        print('[%d/%d] 895: t=%f s. virTime=%f, amax=%f'%(mpi_comm_world().rank+1,mpi_comm_world().size,time()-tic0,virTime,amax))
        
        file_alpha     << ( alphaN   , virTime)
        file_u         << ( uN, virTime)
        file_sigma     << ( stressGN, virTime)
        file_epsilon     << ( strainGN, virTime)
        Kw.assign(Constant(Kw*(10*amax+1)/11))


    # Eval the energy
    W_energyN.assign( project ( energy_w ( uN, alphaN), V_scalar_new))
    dev_energyN.assign( project ( energy_dev ( uN, alphaN), V_scalar_new))
    div_energyN.assign( project ( energy_div ( uN, alphaN), V_scalar_new))

    #  Control if energies are too small
    W_energyN.vector()[W_energyN.vector() < 1e-12] = 0.0
    dev_energyN.vector()[dev_energyN.vector() < 1e-12 ] = 0.0
    div_energyN.vector()[div_energyN.vector() < 1e-12 ] = 0.0
    
    # store the energy
    file_energW    << ( W_energyN,   1.0 * itmesh)
    file_energDev  << ( dev_energyN, 1.0 * itmesh)
    file_energDiv  << ( div_energyN, 1.0 * itmesh)
    
    
    itmesh = itmesh + 1
    
    
	#------------------------------------------------------------------------------------                                                                                
    # Calculate the energies 
    #------------------------------------------------------------------------------------                        
    elastic_energy_value    = assemble ( elastic_energy1_new)
    dissipated_energy_value = assemble ( dissipated_energy_new)

    # Compute the energy into the array enrgies.
    energies[itmesh-2] = np.array ( [ 1.0 * itmesh, elastic_energy_value,
                                            dissipated_energy_value,
                                      elastic_energy_value + dissipated_energy_value] )
    

	#------------------------------------------------------------------------------------   
    # Free memory for lists depending on the current mesh iteration
    #------------------------------------------------------------------------------------
    if ( True) :
        #---------------------------------------------------------------------------------              
        # Free memory for function, test functions and general arrays
        # involved into algebraic manipulations.
        #---------------------------------------------------------------------------------
        #del uN
        del duN
        del vN
        
        del alphaN
        del dalphaN
        del betaN
        
        del alphaN_0
        del alpharef
        del alphaN_error
        #del alphaDiff

        del W_energyN
        del div_energyN
        del dev_energyN
        
        del ubN
        del lbN
        
        #---------------------------------------------------------------------------------        
        # Free memory for boundary conditions
        # Please be carefull with the name of variables.
        #---------------------------------------------------------------------------------
        del bc_uN
        del bc_boxmidx1N, bc_boxmidx2N, bc_boxmidy1N, bc_boxmidy2N, bc_boxbottomN
        del bc_alpha_upN
        del bc_alphaN
        del g_bc_new
        del normal_v_new
        
        #---------------------------------------------------------------------------------
        # Free the memory for geometry
        #---------------------------------------------------------------------------------
        del boundaries_new
        del mesh_new
        
        del V_vector_new
        del V_scalar_new
        
        #---------------------------------------------------------------------------------      
        # Free the memory for solvers.
		#---------------------------------------------------------------------------------
        del total_energy1_new
        del total_energy2_new
        del elastic_energy1_new
        del external_work_new
        del dissipated_energy_new

        del E_uN
        del E_alphaN
        del E_alpha_alphaN

        del E_duN
        del E_dalphaN
        
        del solver_uN
        del problem_uN
        del problem_alphaN

        del dsN, dxN

        del DamageProblemN
             
    
    	#---------------------------------------------------------------------------------
    	# The main loop for remeshing has finished.
    	#---------------------------------------------------------------------------------
                                                                                               


# Save the energy for each configuration 
np.savetxt(savedir + '/energies.txt', energies)

print "-----------------------------------------"
print("Geometry with cavity is finished.")
print "-----------------------------------------"   





#----------------------------------------------------------------
# Define operator aimed to generate particular plots.
#----------------------------------------------------------------
import matplotlib.pyplot as plt

#----------------------------------------------------------------
# Plot energy, stresses, sigma, u, alpha and alpha dot
#----------------------------------------------------------------
import matplotlib.pyplot as plt


def plot_energy():
    p1, = plt.plot(energies[:,0], energies[:,1],'b-o',linewidth=2)
    p2, = plt.plot(energies[:,0], energies[:,2],'r-o',linewidth=2)
    p3, = plt.plot(energies[:,0], energies[:,3],'k--',linewidth=2)
    plt.legend([p1, p2, p3], ["Elastic","Dissipated","Total"])
    plt.xlabel('Mesh Iterations')
    plt.ylabel('Energies')
    plt.savefig(savedir + '/energies.png')
    plt.grid(True)


plot_energy()

# Free Memory
#del times
del energies

del u, du, v

del alpha, dalpha, beta, alpha_0
del alpha_error

del W_energy, div_energy, dev_energy
#del file_alphaDiff
del file_alpha
del file_u

del lb, ub

del solver_u
del solver_alpha
del alphaAux
del V_vector
del V_scalar

del bc_u
del bc_boxmidx1, bc_boxmidx2, bc_boxmidy1, bc_boxmidy2, bc_boxbottom
del bc_alpha_up
del bc_alpha      

#del iterations
del g_bc
#del ds
del normal_v
del mesh, boundaries




#----------------------------------------------------------------
#   End of the main program.
#----------------------------------------------------------------
