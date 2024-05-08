import os
import sys
import argparse
from pathlib import Path

import numpy as np
import taichi as ti

from fluid_simulator import DyeFluidSimulator, FluidSimulator


def main():
    # ---------------------------- PARSER ----------------------------
    parser = argparse.ArgumentParser(description="Fluid Simulator")
    
    # boundary condition argument
    parser.add_argument(
        "-bc",
        "--boundary_condition",
        help="Boundary condition number",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=1,
    )
    
    # reynolds number argument
    parser.add_argument(
        "-re", "--reynolds_num", help="Reynolds number", type=float, default=1000000.0
    )
    
    # grid resolution argument
    parser.add_argument("-res", "--resolution", help="Resolution of y-axis", type=int, default=128)
    
    # time step argument
    parser.add_argument("-dt", "--time_step", help="Time step", type=float, default=0.0)
    
    # flow visualization argument
    parser.add_argument(
        "-vis",
        "--visualization",
        help="Flow visualization type",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
    )
    
    # vorticity confinement argument
    parser.add_argument(
        "-vc",
        "--vorticity_confinement",
        help="Vorticity Confinement. 0.0 is disable.",
        type=float,
        default=5.0,
    )
    
    # advection scheme argument
    parser.add_argument(
        "-scheme",
        "--advection_scheme",
        help="Advection Scheme",
        type=str,
        choices=["upwind", "kk", "cip"],
        default="cip",
    )
    
    # dye calculation argument
    parser.add_argument(
        "-no_dye", "--no_dye", 
        help="No dye calculation", 
        action="store_true",
        default=True
    )
    
    # device argument
    parser.add_argument("-cpu", "--cpu", action="store_true")

    args = parser.parse_args()

    # ---------------------------- MAIN ----------------------------
    # load parameters
    n_bc = args.boundary_condition
    re = args.reynolds_num
    resolution = args.resolution
    # dt = args.time_step if args.time_step != 0.0 else 0.05 / resolution
    dt = args.time_step if args.time_step != 0.0 else 0.0005
    vis_num = args.visualization
    no_dye = args.no_dye
    scheme = args.advection_scheme
    vor_eps = args.vorticity_confinement if args.vorticity_confinement != 0.0 else None
    dx = 1 / resolution

    # load device
    if args.cpu:
        ti.init(arch=ti.cpu)
    else:
        device_memory_GB = 2.0 if resolution > 1000 else 1.0
        ti.init(arch=ti.gpu, device_memory_GB=device_memory_GB)

    # load simulator
    if no_dye:
        fluid_sim = FluidSimulator.create(n_bc, resolution, dt, dx, re, vor_eps, scheme)
    else:
        fluid_sim = DyeFluidSimulator.create(n_bc, resolution, dt, dx, re, vor_eps, scheme)

    # print parameters
    print(f"Boundary Condition: {n_bc}")
    print(f"dt: {dt}")
    print(f"Reynolds Number: {re}")
    print(f"Resolution: {resolution}")
    print(f"Scheme: {scheme}")
    print(f"Vorticity Confinement: {vor_eps}")
    print(f"dye calculation: {not no_dye}")

    # create output directory
    output_path = Path(__file__).parent.resolve() / "output" / \
                  f"bc={n_bc}_res={resolution}_dt={dt}_reynold={re}"
    os.makedirs(output_path, exist_ok=True)
    print(f"Output Path: {output_path}")
    
    img_path = output_path / "img"
    os.makedirs(img_path, exist_ok=True)
    
    data_path = output_path / "data"
    os.makedirs(data_path, exist_ok=True)

    step = 0
    for step in range(2000):
        # simulate one step
        fluid_sim.step()

        # save simulation every 10 steps
        if step % 100 == 0:
            # save norm img
            img_norm = fluid_sim.get_norm_field()
            ti.tools.imwrite(img_norm, str(img_path / f"{step:03}_norm.png"))
            
            # save pressure img
            img_pressure = fluid_sim.get_pressure_field()
            ti.tools.imwrite(img_pressure, str(img_path / f"{step:03}_pressure.png"))
            
            # save vorticity img
            img_vorticity = fluid_sim.get_vorticity_field()
            ti.tools.imwrite(img_vorticity, str(img_path / f"{step:03}_vorticity.png"))
            
            # # save x velocity img 
            # img_vx = fluid_sim._solver.vx.current.to_numpy()
            # ti.tools.imwrite(img_vx, str(img_path / f"{step:03}_vx.png"))
            
            # save pressure data
            pressure = fluid_sim._solver.p.current.to_numpy()
            np.save(str(data_path / f"{step:03}_pressure.npy"), pressure)
            
            # save x velocity data
            vx = fluid_sim._solver.v.current.to_numpy()[:, :, 0]
            vx = vx.reshape(-1, resolution)
            np.save(str(data_path / f"{step:03}_vx.npy"), vx)
            
            # save y velocity data
            vy = fluid_sim._solver.v.current.to_numpy()[:, :, 1]
            vy = vy.reshape(-1, resolution)
            np.save(str(data_path / f"{step:03}_vy.npy"), vy)
            
            

if __name__ == "__main__":
    main()
