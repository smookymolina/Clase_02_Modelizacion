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
    f: Callable[[float, Array, Array, Dict[str, float]], Array]


# ============================================================
# 2) Jacobiano A = df/dx (diferencias finitas centradas)
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
# 4) Simulacion no lineal
# ============================================================
def simulate_nonlinear(model, x0, u_of_t, t_span, t_eval):
    def rhs(t, x):
        u = u_of_t(t, x)
        return model.f(t, x, u, model.params)

    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval)
    return sol.t, sol.y.T


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # Parámetros del sistema
    a = 1.0        # a > 0
    b = 1.0        # b > 0
    x0 = np.array([0.0])

    # Definición de la dinámica
    def f_first_order(t, x, u, p):
        return np.array([
            -p["a"] * x[0] + p["b"] * u[0]
        ])

    # Crear modelo
    model = Model(
        state_names=["x"],
        input_names=["u"],
        params={"a": a, "b": b},
        f=f_first_order
    )

    # Entradas
    def u_step(t, x):
        return np.array([1.0])

    f_sin = 0.5  # Hz
    def u_sin(t, x):
        return np.array([np.sin(2*np.pi*f_sin*t)])

    # Tiempo de simulación
    t_span = (0.0, 10.0)
    t_eval_01  = np.arange(0, 10.01, 0.1)
    t_eval_001 = np.arange(0, 10.001, 0.01)

    # ---- Escalón ----
    t1, X_step_01 = simulate_nonlinear(
        model, x0, u_step, t_span, t_eval_01
    )
    t2, X_step_001 = simulate_nonlinear(
        model, x0, u_step, t_span, t_eval_001
    )

    plt.figure()
    plt.plot(t1, X_step_01[:,0], label="Δt = 0.1")
    plt.plot(t2, X_step_001[:,0], "--", label="Δt = 0.01")
    plt.xlabel("t [s]")
    plt.ylabel("x(t)")
    plt.title("Respuesta al escalón")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ---- Seno ----
    t3, X_sin_01 = simulate_nonlinear(
        model, x0, u_sin, t_span, t_eval_01
    )
    t4, X_sin_001 = simulate_nonlinear(
        model, x0, u_sin, t_span, t_eval_001
    )

    plt.figure()
    plt.plot(t3, X_sin_01[:,0], label="Δt = 0.1")
    plt.plot(t4, X_sin_001[:,0], "--", label="Δt = 0.01")
    plt.xlabel("t [s]")
    plt.ylabel("x(t)")
    plt.title("Respuesta senoidal")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ========================================================
    # VALORES PEDIDOS EN LA TAREA (SIN CAMBIAR EL MODELO)
    # ========================================================

    tau = 1 / a
    t_95 = 3 * tau

    print("=== Interpretación del sistema ===")
    print(f"Constante de tiempo τ = 1/a = {tau:.2f} s")
    print(f"Tiempo aproximado al 95% del estacionario ≈ {t_95:.2f} s")

