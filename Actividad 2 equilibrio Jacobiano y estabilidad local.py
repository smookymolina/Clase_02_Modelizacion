import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

from scipy.integrate import solve_ivp
from scipy.optimize import root

Array = np.ndarray


# ============================================================
# 1) Estructura del modelo
# ============================================================
@dataclass
class Model:
    state_names: Sequence[str]
    input_names: Sequence[str]
    params: Dict[str, float]
    f: Callable[[float, Array, Array, Dict[str, float]], Array]


# ============================================================
# 2) Jacobiano A = df/dx
# ============================================================
def finite_diff_jacobian_x(model, t0, x0, u0, eps=1e-6):
    n = len(x0)
    A = np.zeros((n, n))
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = eps
        fp = model.f(t0, x0 + dx, u0, model.params)
        fm = model.f(t0, x0 - dx, u0, model.params)
        A[:, j] = (fp - fm) / (2 * eps)
    return A


# ============================================================
# 3) Jacobiano B = df/du
# ============================================================
def finite_diff_jacobian_u(model, t0, x0, u0, eps=1e-6):
    n = len(x0)
    m = len(u0)
    B = np.zeros((n, m))
    for j in range(m):
        du = np.zeros(m)
        du[j] = eps
        fp = model.f(t0, x0, u0 + du, model.params)
        fm = model.f(t0, x0, u0 - du, model.params)
        B[:, j] = (fp - fm) / (2 * eps)
    return B


# ============================================================
# 4) Simulación no lineal
# ============================================================
def simulate_nonlinear(model, x0, u_of_t, t_span, t_eval):
    def rhs(t, x):
        return model.f(t, x, u_of_t(t, x), model.params)

    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval)
    return sol.t, sol.y.T


# ============================================================
# 5) Dinámica v–gamma (SISTEMA DE LA TAREA)
# ============================================================
def vgamma_dynamics(t, x, u, p):
    v, gamma = x
    g = p["g"]
    k = p["k"]

    v_eps = 1e-6  # protección división por cero

    vdot = u[0] - g * np.sin(gamma) - k * v * abs(v)
    gammadot = -(g / max(v, v_eps)) * np.cos(gamma)

    return np.array([vdot, gammadot])


# ============================================================
# 6) Comparación gráfica
# ============================================================
def plot_compare(t, Xnl, Xlin, state_names, title):
    n = Xnl.shape[1]
    fig, ax = plt.subplots(n, 1, figsize=(9, 3*n), sharex=True)

    if n == 1:
        ax = [ax]

    for i in range(n):
        ax[i].plot(t, Xnl[:, i], label="No lineal")
        ax[i].plot(t, Xlin[:, i], "--", label="Linealizado")
        ax[i].set_ylabel(state_names[i])
        ax[i].grid()
        ax[i].legend()

    ax[-1].set_xlabel("t [s]")
    plt.suptitle(title)
    plt.show()


# ============================================================
# 7) BLOQUE PRINCIPAL — RESUELVE LA TAREA
# ============================================================
if __name__ == "__main__":

    # Parámetros
    g = 9.81
    k = 0.002
    params = {"g": g, "k": k}

    model = Model(
        state_names=["v", "gamma"],
        input_names=["u"],
        params=params,
        f=vgamma_dynamics
    )

    # Punto de equilibrio propuesto
    v_star = 10.0
    gamma_star = 0.0
    u_star = np.array([k * v_star**2])  # vdot = 0

    x_star = np.array([v_star, gamma_star])

    print("=== Punto nominal ===")
    print(f"v* = {v_star}")
    print(f"gamma* = {gamma_star}")
    print(f"u* = {u_star[0]}")

    # Jacobianos
    A = finite_diff_jacobian_x(model, 0.0, x_star, u_star)
    B = finite_diff_jacobian_u(model, 0.0, x_star, u_star)

    print("\n=== Jacobiano A ===")
    print(A)

    print("\n=== Jacobiano B ===")
    print(B)

    # Autovalores
    eigvals = np.linalg.eigvals(A)
    print("\n=== Autovalores de A ===")
    for lam in eigvals:
        print(lam)

    # Simulación
    t_eval = np.linspace(0, 10, 400)

    x0 = x_star + np.array([0.5, 0.05])  # pequeña perturbación
    u_of_t = lambda t, x: u_star

    t, Xnl = simulate_nonlinear(model, x0, u_of_t, (0, 10), t_eval)

    # Modelo linealizado
    dx0 = x0 - x_star
    Xlin = x_star + (np.exp(np.outer(t, eigvals.real))[:, :2] * dx0)

    plot_compare(
        t,
        Xnl,
        Xlin,
        model.state_names,
        "Comparación: Modelo no lineal vs linealizado"
    )
