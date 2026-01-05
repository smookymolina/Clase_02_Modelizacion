# caida_drag_equilibrio.py
import numpy as np
import matplotlib.pyplot as plt

def dynamics(t, x, g=9.81, k=0.02):
    """
    x = [h, v]
    hdot = v
    vdot = -g - k*v|v|
    """
    h, v = x
    dh = v
    dv = -g - k*v*np.abs(v)
    return np.array([dh, dv], dtype=float)

def rk4_step(f, t, x, dt, **kwargs):
    k1 = f(t, x, **kwargs)
    k2 = f(t + dt/2, x + dt*k1/2, **kwargs)
    k3 = f(t + dt/2, x + dt*k2/2, **kwargs)
    k4 = f(t + dt,   x + dt*k3,   **kwargs)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def simulate_rk4(t0=0.0, tf=20.0, dt=0.01, h0=1000.0, v0=0.0, g=9.81, k=0.02, stop_at_ground=True):
    n = int(np.floor((tf - t0)/dt)) + 1
    t = np.linspace(t0, t0 + dt*(n-1), n)
    x = np.zeros((n, 2), dtype=float)
    x[0] = [h0, v0]

    for i in range(n-1):
        x[i+1] = rk4_step(dynamics, t[i], x[i], dt, g=g, k=k)
        if stop_at_ground and x[i+1, 0] <= 0.0:
            # recorte simple al tocar el suelo
            x = x[:i+2]
            t = t[:i+2]
            break
    return t, x

def terminal_velocity(g=9.81, k=0.02, descending=True):
    # Para descenso, v* negativa
    vt = np.sqrt(g/k)
    return -vt if descending else vt

def local_stability_vstar(g=9.81, k=0.02):
    # Para v<0: vdot = -g + k v^2  -> f'(v)=2kv
    vstar = -np.sqrt(g/k)
    fp = 2*k*vstar  # = -2*sqrt(gk)
    tau = 1.0/abs(fp)
    return vstar, fp, tau

def main():
    g = 9.81
    k = 0.02
    h0 = 1000.0
    v0 = 0.0
    t0 = 0.0
    tf = 60.0
    dt = 0.02

    vstar, fp, tau = local_stability_vstar(g=g, k=k)
    print("=== Equilibrio (velocidad terminal, descenso) ===")
    print(f"v* = {vstar:.6f} m/s")
    print("=== Estabilidad local ===")
    print(f"f'(v*) = {fp:.6f} 1/s  (negativo => estable)")
    print(f"tau ~ 1/|f'(v*)| = {tau:.6f} s")

    t, x = simulate_rk4(t0=t0, tf=tf, dt=dt, h0=h0, v0=v0, g=g, k=k, stop_at_ground=True)
    h = x[:, 0]
    v = x[:, 1]

    # Gráficas
    plt.figure()
    plt.plot(t, h)
    plt.xlabel("t [s]")
    plt.ylabel("h [m]")
    plt.title("Caída con drag: altura vs tiempo")
    plt.grid(True)

    plt.figure()
    plt.plot(t, v, label="v(t)")
    plt.axhline(vstar, linestyle="--", label="v* (terminal)")
    plt.xlabel("t [s]")
    plt.ylabel("v [m/s]")
    plt.title("Caída con drag: velocidad vs tiempo (convergencia a v*)")
    plt.grid(True)
    plt.legend()

    # plt.show()
    plt.savefig('plot_caida.png')
    print("Gráfica guardada en 'plot_caida.png'")

if __name__ == "__main__":
    main()
