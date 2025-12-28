import numpy as np
import matplotlib.pyplot as plt

# Utilidades: Model como estructura limpia de datos
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

# SciPy: integracion ODE y solucion de ecuaciones no lineales
from scipy.integrate import solve_ivp
from scipy.optimize import root

Array = np.ndarray  # alias para legibilidad


# ============================================================
# 1) Estructura del modelo (la "ficha técnica")
# ============================================================
@dataclass
class Model:
    """
    Model = contenedor con:
      - nombres de estados e inputs (solo para etiquetas)
      - params: diccionario de constantes fisicas
      - f: funcion dinamica dx/dt = f(t, x, u, p)
    """
    state_names: Sequence[str]
    input_names: Sequence[str]
    params: Dict[str, float]
    f: Callable[[float, Array, Array, Dict[str, float]], Array]  # dx/dt = f(t,x,u,p)


# ============================================================
# 2) Jacobiano A = df/dx (diferencias finitas centradas)
# ============================================================
def finite_diff_jacobian_x(
    model: Model,
    t0: float,
    x0: Array,
    u0: Array,
    eps: float = 1e-6,
) -> Array:
    """
    Aproxima A = df/dx alrededor de (t0, x0, u0).
    Metodo: diferencias finitas centradas por columnas.

    Interpretacion:
      A[i, j] = "como cambia la derivada del estado i (dx_i/dt)
                si perturbo un poco el estado j (x_j)"
    """
    n = len(x0)  # numero de estados
    A = np.zeros((n, n), dtype=float)

    for j in range(n):
        # Perturbacion solo en el estado j: dx = eps * e_j
        dx = np.zeros(n, dtype=float)
        dx[j] = eps

        # Evaluaciones simetricas
        fp = model.f(t0, x0 + dx, u0, model.params)
        fm = model.f(t0, x0 - dx, u0, model.params)

        # Diferencia finita centrada -> columna j
        A[:, j] = (fp - fm) / (2.0 * eps)

    return A


# ============================================================
# 3) Jacobiano B = df/du (diferencias finitas centradas)
# ============================================================
def finite_diff_jacobian_u(
    model: Model,
    t0: float,
    x0: Array,
    u0: Array,
    eps: float = 1e-6,
) -> Array:
    """
    Aproxima B = df/du alrededor de (t0, x0, u0).

    Interpretacion:
      B[i, j] = "como cambia dx_i/dt si perturbo la entrada u_j"
    """
    n = len(x0)  # estados
    m = len(u0)  # entradas
    B = np.zeros((n, m), dtype=float)

    for j in range(m):
        # Perturbacion solo en la entrada j: du = eps * e_j
        du = np.zeros(m, dtype=float)
        du[j] = eps

        fp = model.f(t0, x0, u0 + du, model.params)
        fm = model.f(t0, x0, u0 - du, model.params)

        # Diferencia finita centrada -> columna j
        B[:, j] = (fp - fm) / (2.0 * eps)

    return B


# ============================================================
# 4) Simulacion no lineal con solve_ivp
# ============================================================
def simulate_nonlinear(
    model: Model,
    x0: Array,
    u_of_t: Callable[[float, Array], Array],
    t_span: Tuple[float, float],
    t_eval: Array,
    method: str = "RK45",
) -> Tuple[Array, Array]:
    """
    Simula el sistema NO lineal:
      x_dot = f(t, x, u(t,x), p)

    u_of_t(t, x) permite:
      - escalones / rampas
      - control en lazo cerrado (u depende de x)
    """
    def rhs(t, x):
        u = u_of_t(t, x)                 # entrada en ese instante
        return model.f(t, x, u, model.params)  # dinamica

    sol = solve_ivp(
        rhs, t_span, x0, t_eval=t_eval, method=method,
        rtol=1e-7, atol=1e-9
    )
    if not sol.success:
        raise RuntimeError(f"Integración falló: {sol.message}")

    # sol.y viene como (n_estados, N); lo transponemos a (N, n_estados)
    return sol.t, sol.y.T


# ============================================================
# 5) Encontrar equilibrio con root
# ============================================================
def find_equilibrium(
    model: Model,
    x_guess: Array,
    u_star: Array,
    t0: float = 0.0,
) -> Array:
    """
    Encuentra x* tal que:
      f(t0, x*, u*, p) = 0

    Nota ingenieril:
      - x_guess es CRITICO: el solver converge al equilibrio "cercano"
      - si hay varios equilibrios, distintos guesses -> distintos x*
    """
    def F(x):
        return model.f(t0, x, u_star, model.params)

    sol = root(F, x_guess, method="hybr")
    if not sol.success:
        raise RuntimeError(f"No convergió equilibrio: {sol.message}")
    return sol.x


# ============================================================
# 6) Simulacion del modelo linealizado
# ============================================================
def simulate_linearized(
    A: Array,
    B: Array,
    dx0: Array,
    du_of_t: Callable[[float, Array], Array],
    t_span: Tuple[float, float],
    t_eval: Array,
) -> Tuple[Array, Array]:
    """
    Simula el sistema linealizado en perturbaciones:
      delta_x_dot = A delta_x + B delta_u(t, delta_x)

    du_of_t(t, dx) permite:
      - escalones en delta_u
      - control lineal por retroalimentacion: delta_u = -K dx
    """
    def rhs(t, dx):
        du = du_of_t(t, dx)
        return A @ dx + B @ du

    sol = solve_ivp(
        rhs, t_span, dx0, t_eval=t_eval, method="RK45",
        rtol=1e-9, atol=1e-11
    )
    if not sol.success:
        raise RuntimeError(f"Integración lineal falló: {sol.message}")
    return sol.t, sol.y.T


# ============================================================
# 7) Comparacion grafica NL vs Linealizado
# ============================================================
def plot_compare(t: Array, Xnl: Array, Xlin: Array, state_names: Sequence[str], title: str):
    """
    Grafica cada estado en una fila:
      - Xnl: trayectoria del modelo no lineal
      - Xlin: trayectoria del modelo linealizado (x* + delta_x)
    """
    n = Xnl.shape[1]
    fig, ax = plt.subplots(n, 1, figsize=(10, 2.6*n), sharex=True)

    # Si n=1, matplotlib regresa un solo eje; lo normalizamos a lista
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
