# v_gamma_jacobian_autovalores.py
import numpy as np

def f_vgam(t, x, u, g=9.81, k=0.0020, v_eps=1e-3):
    """
    Modelo v-gamma:
      dv   = u - g sin(gamma) - k v|v|
      dgam = -(g/v) cos(gamma)  (con protección para v~0)
    x = [v, gamma]
    """
    v, gam = x
    if abs(v) > v_eps:
        v_safe = v
    else:
        v_safe = (np.sign(v) * v_eps) if v != 0 else v_eps

    dv = u - g*np.sin(gam) - k*v*np.abs(v)
    dgam = -(g / v_safe) * np.cos(gam)
    return np.array([dv, dgam], dtype=float)

def jacobian_num(f, x, u, eps=1e-6):
    n = len(x)
    A = np.zeros((n, n), dtype=float)
    fx = f(0.0, x, u)
    for i in range(n):
        dx = np.zeros(n, dtype=float)
        dx[i] = eps
        A[:, i] = (f(0.0, x + dx, u) - fx) / eps
    return A

def is_equilibrium(x, u, tol=1e-6):
    fx = f_vgam(0.0, x, u)
    return np.linalg.norm(fx, ord=2) < tol, fx

def main():
    # Punto de operación (ejemplo de tu slide)
    x_star = np.array([60.0, 0.0])  # [v*, gamma*]  gamma en rad
    u_star = 7.2                    # u*

    eq, fx = is_equilibrium(x_star, u_star, tol=1e-6)
    print("=== Punto de operación ===")
    print("x* =", x_star, "  u* =", u_star)
    print("f(x*,u*) =", fx, " => equilibrio:", eq)

    A = jacobian_num(f_vgam, x_star, u_star, eps=1e-6)
    eigA = np.linalg.eigvals(A)

    print("\n=== Linealización ===")
    print("A =\n", A)
    print("\n=== Autovalores ===")
    print("eig(A) =", eigA)

    # Interpretación breve
    re = np.real(eigA)
    im = np.imag(eigA)
    print("\n=== Lectura rápida ===")
    for i, lam in enumerate(eigA):
        st = "estable (Re<0)" if re[i] < 0 else ("inestable (Re>0)" if re[i] > 0 else "marginal (Re=0)")
        osc = "oscilatorio (Im!=0)" if abs(im[i]) > 1e-10 else "no oscilatorio"
        print(f"lambda[{i}] = {lam:.6g} -> {st}, {osc}")

if __name__ == "__main__":
    main()
