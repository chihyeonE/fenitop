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
from fenitop.fem import form_fem

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
    comm = MPI.COMM_WORLD
    evaluation_history = []
    numevl = 1
    cur_beta = 2

    def f(x, gradient):
            t = x[0]
            v = x[1:]
            if gradient.size > 0:
                gradient[0] = 1
                gradient[1:] = 0
            return t

    def c(result, v, grad, cur_beta):

        global numevl
        t = v[0]
        x = v[1:]

        print(f"Optimization iteration {numevl}, Current beta: {cur_beta}")

        # `topopt`로부터 `f0`, `dJ_du` 등 계산된 값들을 받아옵니다.
        C, V, U, dCdu, dGdu = topopt(v, fem, opt, cur_beta)
        

        f0 = [-C, (0.4-V)**2]

        print("Objective value (f0):", f0)
        dJ_du = np.zeros((60*60, 2))
        dJ_du[:,0] = -dCdu
        dJ_du[:,1] = 2*dGdu - 0.8

        # `dJ_du` 값을 확인하여 gradient가 올바른지 확인

        if grad.size > 0:
                grad[:,0] = -1
                grad[:, 1:] = dJ_du.T

        result[:] = np.real(f0) - t

        evaluation_history.append(np.real(f0))

        # 각 `v` 값을 개별 파일로 저장
        np.savetxt(f"structure{numevl}.txt", v)
        numevl += 1
        #return np.real(f0)

    maxeval = 10
    num_betas = 10
    n = 60 * 60
    x = 0.4 * np.ones((n,))

    linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(fem, opt)
    centers = rho_field.function_space.tabulate_dof_coordinates()[:n].T
    solid, void = opt["solid_zone"](centers), opt["void_zone"](centers)

    lb = np.zeros((n,))
    ub = np.ones((n,))
    lb[solid] = 1
    ub[void] = 0

    x = np.insert(x,0,1.2)
    lb = np.insert(lb, 0, -np.inf)
    ub = np.insert(ub, 0, +np.inf)

    tol_epi = tol_epi = np.array([1e-4] * (2))

    for i in range(num_betas):
        solver = nlopt.opt(nlopt.LD_MMA, n+1)
        solver.set_lower_bounds(lb)
        solver.set_upper_bounds(ub)
        solver.set_min_objective(f)
        solver.set_maxeval(maxeval)
        solver.add_inequality_mconstraint(
                    lambda rr, xx, gg: c(
                        rr,
                        xx,
                        gg,
                        cur_beta,
                    ),
                    tol_epi
                )

        print("Initial x:", x[:5])  # 처음 5개 요소만 출력
        x = np.copy(solver.optimize(x))
        print("Updated x after optimization:", x[:5])  # 업데이트 확인

        cur_beta *= 2

    if comm.rank == 0:
        np.savetxt("evaluation_history.txt", evaluation_history)
        plt.figure()
        plt.plot(evaluation_history)
        plt.savefig("result.png")
