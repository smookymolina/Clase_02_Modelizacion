# v_gamma_equilibrio_solver.py
import numpy as np

def f_vgam(x, u, g=9.81, k=0.0020, v_eps=1e-3):
    v, gam = x
    v_safe = v if abs(v) > v_eps else (np.sign(v) * v_eps if v != 0 else v_eps)
    dv = u - g*np.sin(gam) - k*v*np.abs(v)
    dgam = -(g / v_safe) * np.cos(gam)
    return np.array([dv, dgam], dtype=float)

def newton_2d_for_equilibrium(x0, u, g=9.81, k=0.0020, tol=1e-10, itmax=40, eps=1e-6):
    """
    Encuentra x* tal que f(x*,u)=0 (u fijo) con Newton-Raphson (2D).
    """
    x = x0.astype(float).copy()
    for it in range(itmax):
        fx = f_vgam(x, u, g=g, k=k)
        if np.linalg.norm(fx) < tol:
            return x, True, it, fx

        # Jacobiano numérico
        J = np.zeros((2, 2), dtype=float)
        for i in range(2):
            dx = np.zeros(2)
            dx[i] = eps
            J[:, i] = (f_vgam(x + dx, u, g=g, k=k) - fx) / eps

        # Paso Newton
        try:
            dxn = np.linalg.solve(J, -fx)
        except np.linalg.LinAlgError:
            return x, False, it, fx

        x = x + dxn

    return x, False, itmax, f_vgam(x, u, g=g, k=k)

def main():
    # u fijo (tu ejemplo)
    u = 7.2
    x0 = np.array([60.0, 0.05])  # guess: v~60 m/s, gamma~0.05 rad

    x_star, ok, it, fx = newton_2d_for_equilibrium(x0, u)
    print("=== Equilibrio (Newton 2D) ===")
    print("u =", u)
    print("x* =", x_star)
    print("converge:", ok, "iter:", it)
    print("f(x*,u) =", fx)

if __name__ == "__main__":
    main()
