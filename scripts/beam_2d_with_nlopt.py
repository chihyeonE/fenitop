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

import sys
sys.path.insert(0, '../')

import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType
import petsc4py
import nlopt
import matplotlib.pyplot as plt

from fenitop.topopt import topopt


mesh = create_rectangle(MPI.COMM_WORLD, [[0, 0], [60, 20]],
                        [60, 60], CellType.quadrilateral)
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(MPI.COMM_SELF, [[0, 0], [60, 20]],
                                   [60, 60], CellType.quadrilateral)
else:
    mesh_serial = None

fem = {  # FEM parameters
    "mesh": mesh,
    "mesh_serial": mesh_serial,
    "young's modulus": 100,
    "poisson's ratio": 0.25,
    "disp_bc": lambda x: np.isclose(x[0], 0),
    "traction_bcs": [[(0, -0.2),
                      lambda x: (np.isclose(x[0], 60) & np.greater(x[1], 8) & np.less(x[1], 12))]],
    "body_force": (0, 0),
    "quadrature_degree": 2,
    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
    },
}

opt = {  # Topology optimization parameters
    "max_iter": 400,
    "opt_tol": 1e-5,
    "vol_frac": 0.5,
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone": lambda x: np.full(x.shape[1], False),
    "penalty": 3.0,
    "epsilon": 1e-6,
    "filter_radius": 1.2,
    "beta_interval": 50,
    "beta_max": 128,
    "use_oc": True,
    "move": 0.02,
    "opt_compliance": True,
}

if __name__ == "__main__":

    
    evaluation_history = []
    numevl = 1

    def f(v, grad):
        global numevl

        f0, V, U, dJ_du, dGdu = topopt(v,fem,opt,cur_beta)

        print(np.shape(dJ_du))

        if grad.size > 0:
            grad[:] = dJ_du.flatten()
        evaluation_history.append(np.real(f0))

        np.savetxt("structure"+str(numevl), v)

        numevl+=1

        print(numevl)

        return np.real(f0)

    maxeval = 10
    num_betas = 10
    cur_beta = 2

    algorithm = nlopt.LD_MMA
    n = 60*60
    x = 0.5*np.ones((n,))

    for i in range(num_betas):
        solver = nlopt.opt(algorithm, n)
        solver.set_lower_bounds(0)
        solver.set_upper_bounds(1)
        solver.set_max_objective(f)
        solver.set_maxeval(maxeval)
        print(x)
        xopt = solver.optimize(x)
        print(xopt)
        cur_beta *= 2

    np.savetxt("evaluation_history", evaluation_history)

    # plt.figure()
    # plt.plot(evaluation_history)
    # plt.savefig("./result.png")


    

# Execute the code in parallel:
# mpirun -n 8 python3 scripts/beam_2d.py
