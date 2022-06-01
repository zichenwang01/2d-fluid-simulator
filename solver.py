from abc import ABCMeta, abstractmethod

import taichi as ti

from advection import vorticity_vec
from differentiation import diff2_x, diff2_y, diff_x, diff_y, sample, sign


class DoubleBuffers:
    def __init__(self, resolution, n_channel):
        if n_channel == 1:
            self.current = ti.field(float, shape=resolution)
            self.next = ti.field(float, shape=resolution)
        else:
            self.current = ti.Vector.field(n_channel, float, shape=resolution)
            self.next = ti.Vector.field(n_channel, float, shape=resolution)

    def swap(self):
        self.current, self.next = self.next, self.current

    def reset(self):
        self.current.fill(0)
        self.next.fill(0)


@ti.data_oriented
class Solver(metaclass=ABCMeta):
    def __init__(self, boundary_condition, vor_epsilon=None):
        self._bc = boundary_condition
        self._resolution = boundary_condition.get_resolution()

        # for vorticity confinement
        self.vor = ti.field(float, shape=self._resolution)  # vorticity
        self.vor_abs = ti.field(float, shape=self._resolution)
        self.vor_epsilon = vor_epsilon

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def get_fields(self):
        pass

    @ti.func
    def is_wall(self, i, j):
        return self._bc.is_wall(i, j)

    @ti.kernel
    def _clamp_field(self, field: ti.template(), low: float, high: float):
        for i, j in field:
            field[i, j] = ti.max(ti.min(field[i, j], high), low)

    @ti.kernel
    def _calc_vorticity(self, vor: ti.template(), vor_abs: ti.template(), vc: ti.template()):
        for i, j in vor:
            if not self._bc.is_wall(i, j):
                vor[i, j] = diff_x(vc, i, j).y - diff_y(vc, i, j).x
                vor_abs[i, j] = ti.abs(vor[i, j])

    @ti.kernel
    def _add_vorticity(
        self,
        vn: ti.template(),
        vc: ti.template(),
        vor: ti.template(),
        vor_abs: ti.template(),
    ):
        for i, j in vn:
            if not self._bc.is_wall(i, j):
                vn[i, j] = vc[i, j] + self.dt * vorticity_vec(vor, vor_abs, i, j) * self.vor_epsilon


@ti.data_oriented
class MacSolver(Solver):
    """Maker And Cell method"""

    def __init__(self, boundary_condition, advect_function, dt, Re, p_iter, vor_epsilon=None):
        super().__init__(boundary_condition, vor_epsilon)

        self._advect = advect_function

        self.dt = dt
        self.Re = Re

        self.v = DoubleBuffers(self._resolution, 2)  # velocity
        self.p = DoubleBuffers(self._resolution, 1)  # pressure
        self.p_iter = p_iter

        # initial condition
        self.v.current.fill(ti.Vector([0.4, 0.0]))

    def update(self):
        self._bc.set_boundary_condition(self.v.current, self.p.current)
        self._update_velocities(self.v.next, self.v.current, self.p.current)
        self._clamp_field(self.v.next, -40.0, 40.0)
        self.v.swap()

        if self.vor_epsilon is not None:
            self._calc_vorticity(self.vor, self.vor_abs, self.v.current)
            self._add_vorticity(self.v.next, self.v.current, self.vor, self.vor_abs)
            self.v.swap()

        self._bc.set_boundary_condition(self.v.current, self.p.current)
        for _ in range(self.p_iter):
            self._update_pressures(self.p.next, self.p.current, self.v.current)
            self.p.swap()

    def get_fields(self):
        return self.v.current, self.p.current

    @ti.kernel
    def _update_velocities(self, vn: ti.template(), vc: ti.template(), pc: ti.template()):
        for i, j in vn:
            if not self._bc.is_wall(i, j):
                vn[i, j] = vc[i, j] + self.dt * (
                    -self._advect(vc, vc, i, j)
                    - ti.Vector(
                        [
                            diff_x(pc, i, j),
                            diff_y(pc, i, j),
                        ]
                    )
                    + (diff2_x(vc, i, j) + diff2_y(vc, i, j)) / self.Re
                )

    @ti.kernel
    def _update_pressures(self, pn: ti.template(), pc: ti.template(), vc: ti.template()):
        for i, j in pn:
            if not self._bc.is_wall(i, j):
                dx = diff_x(vc, i, j)
                dy = diff_y(vc, i, j)
                pn[i, j] = (
                    (
                        sample(pc, i + 1, j)
                        + sample(pc, i - 1, j)
                        + sample(pc, i, j + 1)
                        + sample(pc, i, j - 1)
                    )
                    - (dx.x + dy.y) / self.dt
                    + dx.x ** 2
                    + dy.y ** 2
                    + 2 * dy.x * dx.y
                ) * 0.25


@ti.data_oriented
class DyeMacSolver(MacSolver):
    """Maker And Cell method"""

    def __init__(self, boundary_condition, advect_function, dt, Re, p_iter, vor_epsilon=None):
        super().__init__(boundary_condition, advect_function, dt, Re, p_iter, vor_epsilon)
        self.dye = DoubleBuffers(self._resolution, 3)  # dye

    def update(self):
        self._bc.set_boundary_condition(self.v.current, self.p.current, self.dye.current)
        self._update_velocities(self.v.next, self.v.current, self.p.current)
        self._clamp_field(self.v.next, -40.0, 40.0)  # 発散しないようにクランプする
        self.v.swap()

        if self.vor_epsilon is not None:
            self._calc_vorticity(self.vor, self.vor_abs, self.v.current)
            self._add_vorticity(self.v.next, self.v.current, self.vor, self.vor_abs)
            self.v.swap()

        self._bc.set_boundary_condition(self.v.current, self.p.current, self.dye.current)
        for _ in range(self.p_iter):
            self._update_pressures(self.p.next, self.p.current, self.v.current)
            self.p.swap()

        self._bc.set_boundary_condition(self.v.current, self.p.current, self.dye.current)
        self._update_dye(self.dye.next, self.dye.current, self.v.current)
        self._clamp_field(self.dye.next, 0.0, 1.0)  # 発散しないようにクランプする
        self.dye.swap()

    def get_fields(self):
        return self.v.current, self.p.current, self.dye.current

    @ti.kernel
    def _update_dye(self, dn: ti.template(), dc: ti.template(), vc: ti.template()):
        for i, j in dn:
            if not self._bc.is_wall(i, j):
                dn[i, j] = dc[i, j] - self.dt * self._advect(vc, dc, i, j)


@ti.data_oriented
class CipMacSolver(Solver):
    """Maker And Cell method"""

    def __init__(self, boundary_condition, dt, Re, p_iter, vor_epsilon=None):
        super().__init__(boundary_condition, vor_epsilon)
        self.dt = dt
        self.Re = Re

        self.v = DoubleBuffers(self._resolution, 2)  # velocity
        self.vx = DoubleBuffers(self._resolution, 2)  # velocity gradient x
        self.vy = DoubleBuffers(self._resolution, 2)  # velocity gradient y
        self.p = DoubleBuffers(self._resolution, 1)  # pressure

        self.p_iter = p_iter

        # initial condition
        self._initialize()

    def _initialize(self):
        self.v.current.fill(ti.Vector([0.4, 0.0]))
        self.vx.reset()
        self.vy.reset()

        self._bc.set_boundary_condition(self.v.current, self.p.current)

        self._calc_grad_x(self.vx.current, self.v.current)
        self._calc_grad_y(self.vy.current, self.v.current)

    def update(self):
        self._bc.set_boundary_condition(self.v.current, self.p.current)
        self._update_velocities(self.v, self.vx, self.vy, self.p)
        self._clamp_field(self.v.current, -40.0, 40.0)
        self._clamp_field(self.vx.current, -20.0, 20.0)
        self._clamp_field(self.vy.current, -20.0, 20.0)

        if self.vor_epsilon is not None:
            self._calc_vorticity(self.vor, self.vor_abs, self.v.current)
            self._add_vorticity(self.v.next, self.v.current, self.vor, self.vor_abs)
            self.v.swap()

        self._bc.set_boundary_condition(self.v.current, self.p.current)
        for _ in range(self.p_iter):
            self._update_pressures(self.p.next, self.p.current, self.v.current)
            self.p.swap()

    def get_fields(self):
        return self.v.current, self.p.current, self.vor_abs

    @ti.kernel
    def _calc_grad_x(self, fx: ti.template(), f: ti.template()):
        for i, j in fx:
            fx[i, j] = diff_x(f, i, j)

    @ti.kernel
    def _calc_grad_y(self, fy: ti.template(), f: ti.template()):
        for i, j in fy:
            fy[i, j] = diff_y(f, i, j)

    def _update_velocities(self, v, vx, vy, p):
        self._non_advection_phase(v.next, v.current, p.current)
        self._non_advection_phase_grad(vx.next, vy.next, vx.current, vy.current, v.current, v.next)
        v.swap()
        vx.swap()
        vy.swap()

        self._advection_phase(
            v.next, vx.next, vy.next, v.current, vx.current, vy.current, v.current
        )
        v.swap()
        vx.swap()
        vy.swap()

    @ti.kernel
    def _non_advection_phase(
        self,
        fn: ti.template(),
        fc: ti.template(),
        pc: ti.template(),
    ):
        """中間量の計算"""
        for i, j in fn:
            if not self._bc.is_wall(i, j):
                G = (
                    -ti.Vector(
                        [
                            diff_x(pc, i, j),
                            diff_y(pc, i, j),
                        ]
                    )
                    + self._calc_diffusion(fc, i, j)
                )
                fn[i, j] = fc[i, j] + G * self.dt

    @ti.kernel
    def _non_advection_phase_grad(
        self,
        fxn: ti.template(),
        fyn: ti.template(),
        fxc: ti.template(),
        fyc: ti.template(),
        fc: ti.template(),
        fn: ti.template(),
    ):
        """中間量の勾配の計算"""
        for i, j in fn:
            if not self._bc.is_wall(i, j):
                # 勾配の更新
                fxn[i, j] = (
                    fxc[i, j] + (fn[i + 1, j] - fc[i + 1, j] - fn[i - 1, j] + fc[i - 1, j]) / 2.0
                )
                fyn[i, j] = (
                    fyc[i, j] + (fn[i, j + 1] - fc[i, j + 1] - fn[i, j - 1] + fc[i, j - 1]) / 2.0
                )

    @ti.func
    def _calc_diffusion(self, fc, i, j):
        return (diff2_x(fc, i, j) + diff2_y(fc, i, j)) / self.Re

    @ti.kernel
    def _advection_phase(
        self,
        fn: ti.template(),
        fxn: ti.template(),
        fyn: ti.template(),
        fc: ti.template(),
        fxc: ti.template(),
        fyc: ti.template(),
        v: ti.template(),
    ):
        for i, j in fn:
            if not self._bc.is_wall(i, j):
                self._cip_advect(fn, fxn, fyn, fc, fxc, fyc, v, i, j)

    @ti.func
    def _cip_advect(self, fn, fxn, fyn, fc, fxc, fyc, v, i, j):
        i_s = int(sign(v[i, j].x))
        j_s = int(sign(v[i, j].y))
        i_m = i - i_s
        j_m = j - j_s

        tmp1 = fc[i, j] - fc[i, j_m] - fc[i_m, j] + fc[i_m, j_m]
        tmp2 = fc[i_m, j] - fc[i, j]
        tmp3 = fc[i, j_m] - fc[i, j]

        a = (i_s * (fxc[i_m, j] + fxc[i, j]) - 2.0 * (-tmp2)) / i_s
        b = (j_s * (fyc[i, j_m] + fyc[i, j]) - 2.0 * (-tmp3)) / j_s
        c = (-tmp1 - i_s * (fxc[i, j_m] - fxc[i, j])) / j_s
        d = (-tmp1 - j_s * (fyc[i_m, j] - fyc[i, j])) / i_s
        e = 3.0 * tmp2 + i_s * (fxc[i_m, j] + 2.0 * fxc[i, j])
        f = 3.0 * tmp3 + j_s * (fyc[i, j_m] + 2.0 * fyc[i, j])
        g = (-(fyc[i_m, j] - fyc[i, j]) + c) / i_s

        X = -v[i, j].x * self.dt
        Y = -v[i, j].y * self.dt

        # 移流量の更新
        fn[i, j] = (
            ((a * X + c * Y + e) * X + g * Y + fxc[i, j]) * X
            + ((b * Y + d * X + f) * Y + fyc[i, j]) * Y
            + fc[i, j]
        )

        # 勾配の更新
        Fx = (3.0 * a * X + 2.0 * c * Y + 2.0 * e) * X + (d * Y + g) * Y + fxc[i, j]
        Fy = (3.0 * b * Y + 2.0 * d * X + 2.0 * f) * Y + (c * X + g) * X + fyc[i, j]

        dx = diff_x(v, i, j)
        dy = diff_y(v, i, j)
        fxn[i, j] = Fx - self.dt * (Fx * dx.x + Fy * dx.y) / 2.0
        fyn[i, j] = Fy - self.dt * (Fx * dy.x + Fy * dy.y) / 2.0

    @ti.kernel
    def _update_pressures(self, pn: ti.template(), pc: ti.template(), fc: ti.template()):
        for i, j in pn:
            if not self._bc.is_wall(i, j):
                dx = diff_x(fc, i, j)
                dy = diff_y(fc, i, j)
                pn[i, j] = (
                    (
                        sample(pc, i + 1, j)
                        + sample(pc, i - 1, j)
                        + sample(pc, i, j + 1)
                        + sample(pc, i, j - 1)
                    )
                    - (dx.x + dy.y) / self.dt
                    + dx.x ** 2
                    + dy.y ** 2
                    + 2 * dy.x * dx.y
                ) * 0.25


@ti.data_oriented
class DyeCipMacSolver(CipMacSolver):
    """Maker And Cell method"""

    def __init__(self, boundary_condition, dt, Re, p_iter, vor_epsilon=None):
        self.dye = DoubleBuffers(boundary_condition.get_resolution(), 3)  # dye
        self.dyex = DoubleBuffers(boundary_condition.get_resolution(), 3)  # dye gradient x
        self.dyey = DoubleBuffers(boundary_condition.get_resolution(), 3)  # dye gradient y

        super().__init__(boundary_condition, dt, Re, p_iter, vor_epsilon)

    def _initialize(self):
        self.v.current.fill(ti.Vector([0.4, 0.0]))
        self.vx.reset()
        self.vy.reset()
        self.dye.reset()
        self.dyex.reset()
        self.dyey.reset()

        self.dye.current.copy_from(self._bc._bc_dye)
        self._bc.set_boundary_condition(self.v.current, self.p.current, self.dye.current)

        self._calc_grad_x(self.vx.current, self.v.current)
        self._calc_grad_y(self.vy.current, self.v.current)

        self._calc_grad_x(self.dyex.current, self.dye.current)
        self._calc_grad_y(self.dyey.current, self.dye.current)

    def update(self):
        self._bc.set_boundary_condition(self.v.current, self.p.current, self.dye.current)
        self._update_velocities(self.v, self.vx, self.vy, self.p)
        self._clamp_field(self.v.current, -40.0, 40.0)
        self._clamp_field(self.vx.current, -20.0, 20.0)
        self._clamp_field(self.vy.current, -20.0, 20.0)

        if self.vor_epsilon is not None:
            self._calc_vorticity(self.vor, self.vor_abs, self.v.current)
            self._add_vorticity(self.v.next, self.v.current, self.vor, self.vor_abs)
            self.v.swap()

        self._bc.set_boundary_condition(self.v.current, self.p.current, self.dye.current)
        for _ in range(self.p_iter):
            self._update_pressures(self.p.next, self.p.current, self.v.current)
            self.p.swap()

        self._bc.set_boundary_condition(self.v.current, self.p.current, self.dye.current)
        self._update_dye(
            self.dye,
            self.dyex,
            self.dyey,
            self.v,
        )
        self._clamp_field(self.dye.current, 0.0, 1.0)
        self._clamp_field(self.dyex.current, -1.0, 1.0)
        self._clamp_field(self.dyey.current, -1.0, 1.0)

    def get_fields(self):
        return self.v.current, self.p.current, self.dye.current

    @ti.kernel
    def _non_advection_phase_dye(
        self,
        dn: ti.template(),
        dc: ti.template(),
    ):
        """中間量の計算"""
        for i, j in dn:
            if not self._bc.is_wall(i, j):
                dn[i, j] = dc[i, j] + self._calc_diffusion(dc, i, j) * self.dt

    def _update_dye(self, dye, dyex, dyey, v):
        self._non_advection_phase_dye(dye.next, dye.current)
        self._non_advection_phase_grad(
            dyex.next, dyey.next, dyex.current, dyey.current, dye.current, dye.next
        )
        dye.swap()
        dyex.swap()
        dyey.swap()

        self._advection_phase(
            dye.next, dyex.next, dyey.next, dye.current, dyex.current, dyey.current, v.current
        )
        dye.swap()
        dyex.swap()
        dyey.swap()
