"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsors:
- U.S. National Science Foundation (NSF) EAGER Award CMMI-2127134
- U.S. Defense Advanced Research Projects Agency (DARPA) Young Faculty Award
  (N660012314013)
- NSF CAREER Award CMMI-2047692
- NSF Award CMMI-2245251

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7
"""

import time

import numpy as np
from mpi4py import MPI

from fenitop.fem import form_fem
from fenitop.parameterize import DensityFilter, Heaviside
from fenitop.sensitivity import Sensitivity
from fenitop.optimize import optimality_criteria, mma_optimizer
from fenitop.utility import Communicator, Plotter, save_xdmf


def topopt(x, fem, opt, beta):
    """Main function for topology optimization."""

    # Initialization
    comm = MPI.COMM_WORLD
    linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(fem, opt)

    rho_field.vector_array = x.copy()
    
    density_filter = DensityFilter(comm, rho_field, rho_phys_field,
                                   opt["filter_radius"], fem["petsc_options"])
    heaviside = Heaviside(rho_phys_field)
    sens_problem = Sensitivity(comm, opt, linear_problem, u_field, lambda_field, rho_phys_field)
    S_comm = Communicator(rho_phys_field.function_space, fem["mesh_serial"])
    if comm.rank == 0:
        plotter = Plotter(fem["mesh_serial"])
    num_consts = 1 if opt["opt_compliance"] else 2
    num_elems = rho_field.vector.array.size
    if not opt["use_oc"]:
        rho_old1, rho_old2 = np.zeros(num_elems), np.zeros(num_elems)
        low, upp = None, None

    # Apply passive zones
    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T
    solid, void = opt["solid_zone"](centers), opt["void_zone"](centers)
    rho_ini = np.full(num_elems, opt["vol_frac"])
    rho_ini[solid], rho_ini[void] = 0.995, 0.005
    rho_field.vector.array[:] = rho_ini
    rho_min, rho_max = np.zeros(num_elems), np.ones(num_elems)
    rho_min[solid], rho_max[void] = 0.99, 0.01

    # Density filter and Heaviside projection
    density_filter.forward()
    heaviside.forward(beta)

    # Solve FEM
    linear_problem.solve_fem()

    # Compute function values and sensitivities
    [C_value, V_value, U_value], sensitivities = sens_problem.evaluate()
    heaviside.backward(sensitivities)
    [dCdrho, dVdrho, dUdrho] = density_filter.backward(sensitivities)
    if opt["opt_compliance"]:
        g_vec = np.array([V_value-opt["vol_frac"]])
        dJdrho, dgdrho = dCdrho, np.vstack([dVdrho])
    else:
        g_vec = np.array([V_value-opt["vol_frac"], C_value-opt["compliance_bound"]])
        dJdrho, dgdrho = dUdrho, np.vstack([dVdrho, dCdrho])
    
    return C_value, V_value, U_value, dJdrho, dgdrho