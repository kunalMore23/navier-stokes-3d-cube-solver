import os
import warnings

from sympy import Symbol, Eq, Abs

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain

from modulus.sym.geometry.primitives_3d import Box
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)
import vtk
# import modulus.sym.geometry.primitives_1d as geoSym1D
from modulus.sym.utils.io.vtk import var_to_polyvtk

points = 1000000
room = Box((0, 0, 0), (10, 10, 5))
inlet = Box((0, 9, 2.25), (1, 10, 2.75))
outlet = Box((9, 0, 2.25), (10, 1, 2.75))
room_interior = room - inlet - outlet

room_interior_sample = room_interior.sample_interior(nr_points=points)
var_to_polyvtk(room_interior_sample, "room_interior_sample")

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)
    
    # Define the architecture directly here
    flow_net = instantiate_arch(
            input_keys=[Key("x"), Key("y")],
            output_keys=[Key("u"), Key("v"), Key("p")],
            cfg=cfg.arch.fully_connected
        )
    
    nodes = ns.make_nodes() + [flow_net.make_node(name="flow_network")]

    ns_domain = Domain()
    
    x, y = Symbol("x"), Symbol("y")

    top_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=room_interior,
        outvar={"u": 1.0, "v": 0},
        batch_size=20,
        lambda_weighting={"u": 1.0 - 20 * Abs(x), "v": 1.0},  # weight edges to be zero
        criteria=Eq(y, 10 / 2),
    )
    ns_domain.add_constraint(top_wall, "top_wall")

    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=room_interior,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=20,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
        },
    )
    ns_domain.add_constraint(interior, "interior")

    
    # make solver
    slv = Solver(cfg, ns_domain)
    
    # start solver
    slv.solve()


run()
