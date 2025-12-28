# solver_state_space.py
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple, Optional

from scipy.integrate import solve_ivp
from scipy.optimize import root


Array = np.ndarray


@dataclass
class Model:
    state_names: Sequence[str]
    input_names: Sequence[str]
    params: Dict[str, float]
    f: Callable[[float, Array, Array, Dict[str, float]], Array]   # dx/dt = f(t,x,u,p)


def finite_diff_jacobian_x(
    model: Model,
    t0: float,
    x0: Array,
    u0: Array,
    eps: float = 1e-6,
) -> Array:
    """Jacobiano A = df/dx por diferencias finitas centradas."""
    n = len(x0)
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = eps
        fp = model.f(t0, x0 + dx, u0, model.params)
        fm = model.f(t0, x0 - dx, u0, model.params)
        A[:, j] = (fp - fm) / (2.0 * eps)
    return A


def finite_diff_jacobian_u(
    model: Model,
    t0: float,
    x0: Array,
    u0: Array,
    eps: float = 1e-6,
) -> Array:
    """Jacobiano B = df/du por diferencias finitas centradas."""
    n = len(x0)
    m = len(u0)
    B = np.zeros((n, m), dtype=float)
    for j in range(m):
        du = np.zeros(m)
        du[j] = eps
        fp = model.f(t0, x0, u0 + du, model.params)
        fm = model.f(t0, x0, u0 - du, model.params)
        B[:, j] = (fp - fm) / (2.0 * eps)
    return B


def simulate_nonlinear(
    model: Model,
    x0: Array,
    u_of_t: Callable[[float, Array], Array],
    t_span: Tuple[float, float],
    t_eval: Array,
    method: str = "RK45",
) -> Tuple[Array, Array]:
    """Simula x_dot = f(t,x,u(t,x),p) con solve_ivp."""
    def rhs(t, x):
        u = u_of_t(t, x)
        return model.f(t, x, u, model.params)

    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method=method, rtol=1e-7, atol=1e-9)
    if not sol.success:
        raise RuntimeError(f"Integración falló: {sol.message}")
    return sol.t, sol.y.T  # (N,), (N,n)


def find_equilibrium(
    model: Model,
    x_guess: Array,
    u_star: Array,
    t0: float = 0.0,
) -> Array:
    """Encuentra x* tal que f(t0, x*, u*, p)=0 (si existe)."""
    def F(x):
        return model.f(t0, x, u_star, model.params)

    sol = root(F, x_guess, method="hybr")
    if not sol.success:
        raise RuntimeError(f"No convergió equilibrio: {sol.message}")
    return sol.x


def simulate_linearized(
    A: Array,
    B: Array,
    dx0: Array,
    du_of_t: Callable[[float, Array], Array],
    t_span: Tuple[float, float],
    t_eval: Array,
) -> Tuple[Array, Array]:
    """Simula delta_x_dot = A delta_x + B delta_u(t,delta_x)."""
    def rhs(t, dx):
        du = du_of_t(t, dx)
        return A @ dx + B @ du

    sol = solve_ivp(rhs, t_span, dx0, t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(f"Integración lineal falló: {sol.message}")
    return sol.t, sol.y.T


def plot_compare(t: Array, Xnl: Array, Xlin: Array, state_names: Sequence[str], title: str):
    n = Xnl.shape[1]
    fig, ax = plt.subplots(n, 1, figsize=(10, 2.6*n), sharex=True)
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].plot(t, Xnl[:, i], label="No lineal")
        ax[i].plot(t, Xlin[:, i], "--", label="Linealizado")
        ax[i].set_ylabel(state_names[i])
        ax[i].grid(True, alpha=0.3)
        ax[i].legend()
    ax[-1].set_xlabel("t [s]")
    plt.suptitle(title)
    plt.show()


# ============================================================
# EJEMPLO (tu modelo v-gamma)
# ============================================================
def main():
    # 1) Define el modelo (los alumnos editan esta parte)
    def f_vgam(t, x, u, p):
        v, gam = x
        (u_thrust,) = u  # u es vector
        g = p["g"]
        k = p["k"]

        # protección (singularidad 1/v)
        v_safe = v
        if abs(v_safe) < 1e-3:
            v_safe = 1e-3 * np.sign(v_safe if v_safe != 0 else 1.0)

        dv = u_thrust - g*np.sin(gam) - k*v*abs(v)
        dgam = -(g / v_safe) * np.cos(gam)
        return np.array([dv, dgam], dtype=float)

    model = Model(
        state_names=["v [m/s]", "gamma [rad]"],
        input_names=["u_thrust [m/s^2]"],
        params={"g": 9.81, "k": 0.0020},
        f=f_vgam,
    )

    # 2) Define nominal (puedes hacerlo por "diseño" o encontrar equilibrio)
    v_star = 60.0
    gam_star = 0.0
    u_star = np.array([model.params["k"] * v_star**2], dtype=float)  # u* = k v*^2

    x_star = np.array([v_star, gam_star], dtype=float)

    # 3) Simulación NL con un escalón en u
    du_step = 0.6
    t_eval = np.linspace(0.0, 8.0, 801)
    t_span = (t_eval[0], t_eval[-1])

    def u_of_t(t, x):
        return u_star + np.array([du_step if t >= 1.0 else 0.0], dtype=float)

    t, Xnl = simulate_nonlinear(model, x_star, u_of_t, t_span, t_eval)

    # 4) Linealiza automáticamente (numérico): A,B
    A = finite_diff_jacobian_x(model, t0=0.0, x0=x_star, u0=u_star)
    B = finite_diff_jacobian_u(model, t0=0.0, x0=x_star, u0=u_star)

    # 5) Simula el modelo linealizado en delta
    dx0 = np.zeros_like(x_star)

    def du_of_t(t, dx):
        return np.array([du_step if t >= 1.0 else 0.0], dtype=float)

    _, DX = simulate_linearized(A, B, dx0, du_of_t, t_span, t_eval)
    Xlin = x_star + DX

    # 6) Diagnóstico rápido: autovalores
    eigA = np.linalg.eigvals(A)
    print("A =\n", A)
    print("B =\n", B)
    print("eig(A) =", eigA)

    # 7) Gráficas
    plot_compare(t, Xnl, Xlin, model.state_names, "Comparación: No lineal vs Linealizado (v-gamma)")


if __name__ == "__main__":
    main()
