import taichi as ti

from advection import advect_kk_scheme, advect_upwind
from boundary_condition import get_boundary_condition
from solver import CipMacSolver, DyeCipMacSolver, DyeMacSolver, MacSolver
from pressure_updater import RedBlackSorPressureUpdater
from vorticity_confinement import VorticityConfinement
from visualization import visualize_norm, visualize_pressure, visualize_vorticity


@ti.data_oriented
class FluidSimulator:
    def __init__(self, solver):
        self._solver = solver
        print('solver._resolution:', solver._resolution)
        self.rgb_buf = ti.Vector.field(3, ti.f32, shape=solver._resolution)  # image buffer
        # self._wall_color = ti.Vector([0.5, 0.7, 0.5])
        self._wall_color = ti.Vector([0.0, 1.0, 0.0])

    def step(self):
        self._solver.update()

    def get_norm_field(self):
        self._to_norm(self.rgb_buf, *self._solver.get_fields()[:2])
        return self.rgb_buf

    def get_pressure_field(self):
        self._to_pressure(self.rgb_buf, self._solver.get_fields()[1])
        return self.rgb_buf

    def get_vorticity_field(self):
        self._to_vorticity(self.rgb_buf, self._solver.get_fields()[0])
        return self.rgb_buf

    def field_to_numpy(self):
        fields = self._solver.get_fields()
        return {"v": fields[0].to_numpy(), "p": fields[1].to_numpy()}

    @ti.kernel
    def _to_norm(self, rgb_buf: ti.template(), vc: ti.template(), pc: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = 0.2 * visualize_norm(vc[i, j])
            rgb_buf[i, j] += 0.002 * visualize_pressure(pc[i, j])
            if self._solver.is_wall(i, j):
                rgb_buf[i, j] = self._wall_color

    @ti.kernel
    def _to_pressure(self, rgb_buf: ti.template(), pc: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = 0.04 * visualize_pressure(pc[i, j])
            if self._solver.is_wall(i, j):
                rgb_buf[i, j] = self._wall_color

    @ti.kernel
    def _to_vorticity(self, rgb_buf: ti.template(), vc: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = 0.005 * visualize_vorticity(vc, i, j, self._solver.dx)
            if self._solver.is_wall(i, j):
                rgb_buf[i, j] = self._wall_color

    @staticmethod
    def create(num, resolution, dt, dx, re, vor_eps, scheme):
        boundary_condition = get_boundary_condition(num, resolution, True)
        vorticity_confinement = (
            VorticityConfinement(boundary_condition, dt, dx, vor_eps)
            if vor_eps is not None
            else None
        )
        pressure_updater = RedBlackSorPressureUpdater(
            boundary_condition, dt, dx, relaxation_factor=1.3, n_iter=2
        )

        if scheme == "cip":
            solver = CipMacSolver(
                boundary_condition, pressure_updater, dt, dx, re, vorticity_confinement
            )
        elif scheme == "upwind":
            solver = MacSolver(
                boundary_condition,
                pressure_updater,
                advect_upwind,
                dt,
                dx,
                re,
                vorticity_confinement,
            )
        elif scheme == "kk":
            solver = MacSolver(
                boundary_condition,
                pressure_updater,
                advect_kk_scheme,
                dt,
                dx,
                re,
                vorticity_confinement,
            )

        return FluidSimulator(solver)


@ti.data_oriented
class DyeFluidSimulator(FluidSimulator):
    def get_dye_field(self):
        self._to_dye(self.rgb_buf, self._solver.get_fields()[2])
        return self.rgb_buf

    def field_to_numpy(self):
        fields = self._solver.get_fields()
        return {"v": fields[0].to_numpy(), "p": fields[1].to_numpy(), "dye": fields[2].to_numpy()}

    @ti.kernel
    def _to_dye(self, rgb_buf: ti.template(), dye: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = dye[i, j]
            if self._solver.is_wall(i, j):
                rgb_buf[i, j] = self._wall_color

    @staticmethod
    def create(num, resolution, dt, dx, re, vor_eps, scheme):
        boundary_condition = get_boundary_condition(num, resolution, False)
        vorticity_confinement = (
            VorticityConfinement(boundary_condition, dt, dx, vor_eps)
            if vor_eps is not None
            else None
        )
        pressure_updater = RedBlackSorPressureUpdater(
            boundary_condition, dt, dx, relaxation_factor=1.3, n_iter=2
        )

        if scheme == "cip":
            solver = DyeCipMacSolver(
                boundary_condition, pressure_updater, dt, dx, re, vorticity_confinement
            )
        elif scheme == "upwind":
            solver = DyeMacSolver(
                boundary_condition,
                pressure_updater,
                advect_upwind,
                dt,
                dx,
                re,
                vorticity_confinement,
            )
        elif scheme == "kk":
            solver = DyeMacSolver(
                boundary_condition,
                pressure_updater,
                advect_kk_scheme,
                dt,
                dx,
                re,
                vorticity_confinement,
            )

        return DyeFluidSimulator(solver)
