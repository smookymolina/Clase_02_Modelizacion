# respuesta_libre_forzada_metricas.py
import numpy as np
import matplotlib.pyplot as plt

def step_response_second_order(wn=3.0, zeta=0.25, t_end=8.0, n=2000):
    """
    Respuesta al escalón unitario de un 2º orden estándar:
      G(s) = wn^2 / (s^2 + 2 zeta wn s + wn^2)
    Fórmula cerrada para zeta<1 (subamortiguado).
    """
    t = np.linspace(0, t_end, n)
    if zeta < 1.0:
        wd = wn*np.sqrt(1 - zeta**2)
        phi = np.arctan(np.sqrt(1 - zeta**2)/zeta)
        y = 1 - (1/np.sqrt(1 - zeta**2))*np.exp(-zeta*wn*t)*np.sin(wd*t + phi)
    elif np.isclose(zeta, 1.0):
        y = 1 - np.exp(-wn*t)*(1 + wn*t)
    else:
        r1 = -wn*(zeta - np.sqrt(zeta**2 - 1))
        r2 = -wn*(zeta + np.sqrt(zeta**2 - 1))
        C1 = r2/(r2 - r1)
        C2 = -r1/(r2 - r1)
        y = 1 + C1*np.exp(r1*t) + C2*np.exp(r2*t)
    return t, y

def settling_time(t, y, band=0.05, y_final=1.0):
    """
    Ts: primer tiempo a partir del cual y se queda dentro de ±band*y_final.
    """
    lo = y_final*(1 - band)
    hi = y_final*(1 + band)
    inside = (y >= lo) & (y <= hi)
    for i in range(len(t)):
        if inside[i] and np.all(inside[i:]):
            return t[i]
    return np.nan

def overshoot(y, y_final=1.0):
    ymax = np.max(y)
    return max(0.0, (ymax - y_final)/y_final) * 100.0

def main():
    wn = 2.5
    zeta = 1.0
    t, y = step_response_second_order(wn=wn, zeta=zeta, t_end=8.0, n=3000)

    Ts = settling_time(t, y, band=0.05, y_final=1.0)
    Mp = overshoot(y, y_final=1.0)

    print("=== Métricas ===")
    print(f"wn = {wn:.3f} rad/s, zeta = {zeta:.3f}")
    print(f"Ts (±5%) = {Ts:.4f} s")
    print(f"Mp = {Mp:.2f} %")

    plt.figure()
    plt.plot(t, y, label="y(t) escalón")
    plt.axhline(0.95, linestyle="--")
    plt.axhline(1.05, linestyle="--")
    if np.isfinite(Ts):
        plt.axvline(Ts, linestyle="--", label=f"Ts={Ts:.2f}s")
    plt.xlabel("t [s]")
    plt.ylabel("y(t)")
    plt.title("Respuesta forzada (escalón) y métricas: Ts, Mp")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
