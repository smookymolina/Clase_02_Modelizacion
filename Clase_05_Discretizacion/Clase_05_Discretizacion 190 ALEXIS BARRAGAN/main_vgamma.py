# main_vgamma.py
"""
Ejemplo guiado (Clase 2): modelo v-gamma
- Simula el sistema no lineal con un escalón en u
- Linealiza numéricamente alrededor de (x*, u*)
- Simula el modelo linealizado en variables delta
- Compara No lineal vs Linealizado

Requisitos:
  - Tener en la misma carpeta: solver_state_space.py
  - Paquetes: numpy, matplotlib, scipy

Ejecución:
  python main_vgamma.py
"""

import numpy as np

from solver_state_space import (
    Model,
    simulate_nonlinear,
    finite_diff_jacobian_x,
    finite_diff_jacobian_u,
    simulate_linearized,
    plot_compare,
)


def f_vgam(t, x, u, p):
    """
    Dinámica del ejemplo v-gamma:
      x = [v, gam]
      u = [u_thrust]

      dv   = u - g sin(gamma) - k v|v|
      dgam = -(g/v) cos(gamma)   (con protección en v)
    """
    v, gam = x
    (u_thrust,) = u  # u es vector, extraemos el escalar

    g = p["g"]
    k = p["k"]

    # Protección numérica para evitar singularidad 1/v
    v_safe = v
    if abs(v_safe) < 1e-3:
        v_safe = 1e-3 * np.sign(v_safe if v_safe != 0 else 1.0)

    dv = u_thrust - g * np.sin(gam) - k * v * abs(v)
    dgam = -(g / v_safe) * np.cos(gam)

    return np.array([dv, dgam], dtype=float)


def main():
    # =========================================================
    # 1) Definir modelo
    # =========================================================
    model = Model(
        state_names=["v [m/s]", "gamma [rad]"],
        input_names=["u_thrust [m/s^2]"],
        params={"g": 9.81, "k": 0.0020},
        f=f_vgam,
    )

    # =========================================================
    # 2) Punto nominal (equilibrio por diseño)
    #    Elegimos v* y gamma*, y calculamos u* tal que dv=0
    # =========================================================
    v_star = 60.0
    gam_star = 0.0
    x_star = np.array([v_star, gam_star], dtype=float)

    # En gamma*=0: dv = u* - k v*^2  => u* = k v*^2
    u_star = np.array([model.params["k"] * v_star**2], dtype=float)

    # =========================================================
    # 3) Simulación NO LINEAL con escalón en entrada
    # =========================================================
    du_step = 1.0  # magnitud del escalón
    t_eval = np.linspace(0.0, 100.0, 801)
    t_span = (t_eval[0], t_eval[-1])

    def u_of_t(t, x):
        # Escalón en t = 1.0 s
        return u_star + np.array([du_step if t >= 1.0 else 0.0], dtype=float)

    t, Xnl = simulate_nonlinear(model, x_star, u_of_t, t_span, t_eval)

    # =========================================================
    # 4) Linealización numérica: A y B alrededor de (x*, u*)
    # =========================================================
    A = finite_diff_jacobian_x(model, t0=0.0, x0=x_star, u0=u_star)
    B = finite_diff_jacobian_u(model, t0=0.0, x0=x_star, u0=u_star)

    # =========================================================
    # 5) Simulación del modelo linealizado en variables delta
    #    delta_x_dot = A delta_x + B delta_u
    # =========================================================
    dx0 = np.zeros_like(x_star)

    def du_of_t(t, dx):
        return np.array([du_step if t >= 1.0 else 0.0], dtype=float)

    _, DX = simulate_linearized(A, B, dx0, du_of_t, t_span, t_eval)
    Xlin = x_star + DX

    # =========================================================
    # 6) Diagnóstico rápido: autovalores de A
    # =========================================================
    eigA = np.linalg.eigvals(A)
    print("=== Linealización en (x*, u*) ===")
    print("x* =", x_star)
    print("u* =", u_star)
    print("\nA =\n", A)
    print("\nB =\n", B)
    print("\neig(A) =", eigA)

    # =========================================================
    # 7) Comparación gráfica
    # =========================================================
    plot_compare(
        t,
        Xnl,
        Xlin,
        model.state_names,
        "Comparación: No lineal vs Linealizado (v-gamma)",
    )


if __name__ == "__main__":
    main()
