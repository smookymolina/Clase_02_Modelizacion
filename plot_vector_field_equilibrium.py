import numpy as np
import matplotlib.pyplot as plt

def f_vgam(v, gam, u_thrust, g=9.81, k=0.0020, v_eps=1e-3):
    """
    Dinámica del ejemplo v-gamma: ejemplo
      dv   = u - g sin(gamma) - k v|v|
      dgam = -(g/v) cos(gamma)   (con protección en v)
    """
    v_safe = np.where(np.abs(v) < v_eps, np.sign(v) * v_eps, v)
    v_safe = np.where(v_safe == 0.0, v_eps, v_safe)

    dv = u_thrust - g*np.sin(gam) - k*v*np.abs(v)
    dgam = -(g / v_safe) * np.cos(gam)
    return dv, dgam

def main():
    # --- Punto nominal (equilibrio por diseño)
    g = 9.81
    k = 0.0020
    v_star = 60.0
    gam_star = 0.0
    u_star = k * v_star**2  # equilibrio en dv=0 cuando gamma=0

    # --- Mallado para el campo vectorial (elige rangos "bonitos" para la diapositiva)
    v_min, v_max = 20.0, 100.0
    gam_min, gam_max = -0.6, 0.6  # rad

    Nv, Ng = 23, 23
    V = np.linspace(v_min, v_max, Nv)
    G = np.linspace(gam_min, gam_max, Ng)
    VV, GG = np.meshgrid(V, G)

    dV, dG = f_vgam(VV, GG, u_star, g=g, k=k)

    # --- Normalización opcional para que las flechas se vean uniformes
    # Evita que flechas gigantes dominen la figura.
    mag = np.sqrt(dV**2 + dG**2)
    mag_safe = np.where(mag < 1e-9, 1.0, mag)
    dVn = dV / mag_safe
    dGn = dG / mag_safe

    # --- Perturbaciones delta x alrededor del equilibrio
    # (puedes ajustar magnitudes según el "zoom" que quieras)
    perturbations = [
        np.array([+8.0,  0.00]),
        np.array([-8.0,  0.00]),
        np.array([ 0.0, +0.15]),
        np.array([ 0.0, -0.15]),
        np.array([+6.0, +0.10]),
        np.array([-6.0, -0.10]),
    ]

    # --- Figura
    plt.figure(figsize=(10, 5.6))

    # Campo vectorial (normalizado)
    plt.quiver(
        VV, GG, dVn, dGn,
        angles="xy", scale=35, width=0.0022
    )

    # Equilibrio
    plt.plot(v_star, gam_star, marker="o", markersize=10)
    plt.annotate(
        r"$(x^\star)$",
        (v_star, gam_star),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=12
    )

    # Flechas de perturbación δx
    for dx in perturbations:
        v0 = v_star + dx[0]
        g0 = gam_star + dx[1]
        plt.plot(v0, g0, marker="x", markersize=8)
        # flecha desde equilibrio hacia el punto perturbado
        plt.arrow(
            v_star, gam_star,
            dx[0], dx[1],
            length_includes_head=True,
            head_width=0.03,
            head_length=2.2,
            linewidth=1.2
        )
        plt.annotate(
            r"$\delta x$",
            (v0, g0),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=10
        )

    # Etiquetas y formato
    plt.title("Campo vectorial y equilibrio con perturbaciones $\\delta x$ (modelo v-$\\gamma$)")
    plt.xlabel(r"$v$ [m/s]")
    plt.ylabel(r"$\gamma$ [rad]")
    plt.grid(True, alpha=0.25)
    plt.xlim(v_min, v_max)
    plt.ylim(gam_min, gam_max)

    plt.tight_layout()
    
    # Muestra la figura en tiempo real (sin guardar automáticamente)
    plt.show()

if __name__ == "__main__":
    main()