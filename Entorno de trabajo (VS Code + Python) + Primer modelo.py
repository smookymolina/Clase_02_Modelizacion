import numpy as np
import matplotlib.pyplot as plt
# ============================================================
# Modelización en Ingeniería Aeroespacial - Clase 1
# Ejemplo: Cohete vertical 1D + Calibración del parámetro k = CdA/m
# ============================================================
def rmse(y: np.ndarray, yhat: np.ndarray)-> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def format_si(x: float, unit: str = "") -> str:
    """Formato simple para imprimir valores con unidades."""
    return f"{x:.6g} {unit}".strip()


# ----------------------------
# Modelo del cohete (1D vertical)
# ----------------------------
def simulate_rocket_euler(
    t: np.ndarray,
    params: dict,
    h0: float = 0.0,
    v0: float = 0.0,
    clamp_to_ground: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simula el cohete con integración explícita de Euler (didáctico).

    Estados:
        h(t) = altura [m]
        v(t) = velocidad [m/s]

    Ecuaciones:
        dh/dt = v
        dv/dt = (T/m) - g - 0.5*rho*k*v*|v|
        donde k = CdA/m  [m^2/kg]

    Nota importante:
    En este script T ya se interpreta como "aceleración por empuje" (T/m) [m/s^2]
    para evitar meter masa explícita en Clase 1.
    """
    # Parámetros
    g: float = params["g"]           # [m/s^2]
    rho: float = params["rho"]       # [kg/m^3]
    T_acc: float = params["T_acc"]   # [m/s^2]  (equivale a T/m)
    t_burn: float = params["t_burn"] # [s]
    k: float = params["k"]           # [m^2/kg]
    h_min: float = params.get("h_min", 0.0)

    # Preparación
    dt = float(t[1] - t[0])
    n = len(t)

    h = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)
    a = np.zeros(n, dtype=float)

    h[0] = h0
    v[0] = v0

    for i in range(n - 1):
        ti = float(t[i])

        # Empuje tipo ON/OFF
        thrust_acc = T_acc if ti <= t_burn else 0.0

        # Arrastre cuadrático: D/m = 0.5*rho*k*v|v|
        drag_acc = 0.5 * rho * k * v[i] * abs(v[i])

        # Aceleración total
        a[i] = thrust_acc - g - drag_acc

        # Euler explícito
        v[i + 1] = v[i] + a[i] * dt
        h[i + 1] = h[i] + v[i] * dt

        # Opcional: evitar alturas negativas (suelo)
        if clamp_to_ground and h[i + 1] < h_min:
            h[i + 1] = h_min
            v[i + 1] = 0.0

    # Último punto de aceleración (aprox. igual al penúltimo)
    a[-1] = a[-2]

    return h, v, a


# ----------------------------
# Generación de "datos" sintéticos
# ----------------------------
def generate_synthetic_measurements(
    t: np.ndarray,
    true_params: dict,
    noise_std_h: float = 2.0,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera datos sintéticos de altura (como si fueran mediciones de sensor).
    - h_true: altura "real" (del modelo)
    - h_meas: altura con ruido gaussiano
    """
    rng = np.random.default_rng(seed)
    h_true, _, _ = simulate_rocket_euler(t, true_params)
    h_meas = h_true + rng.normal(0.0, noise_std_h, size=h_true.shape)
    return h_true, h_meas


# ----------------------------
# Calibración del parámetro k (grid search)
# ----------------------------
def fit_k_grid_search(
    t: np.ndarray,
    base_params: dict,
    h_meas: np.ndarray,
    k_candidates: np.ndarray,
) -> dict:
    """
    Ajusta k por búsqueda en rejilla:
    - Simula el modelo para cada k
    - Evalúa RMSE entre h_meas y h_hat
    - Devuelve el mejor k y métricas
    """
    best = {
        "k": None,
        "rmse": np.inf,
        "h_hat": None,
    }
    errors = np.zeros_like(k_candidates, dtype=float)

    for idx, k in enumerate(k_candidates):
        params = dict(base_params)
        params["k"] = float(k)

        h_hat, _, _ = simulate_rocket_euler(t, params)
        e = rmse(h_meas, h_hat)
        errors[idx] = e

        if e < best["rmse"]:
            best["k"] = float(k)
            best["rmse"] = float(e)
            best["h_hat"] = h_hat

    best["k_candidates"] = k_candidates
    best["errors"] = errors
    return best


# ----------------------------
# Main (lo que correrás en clase)
# ----------------------------
def main():
    # ============
    # 1) Tiempo
    # ============
    t_end = 12.0
    dt = 0.01
    t = np.arange(0.0, t_end + dt, dt)

    # Tiempo usado SOLO para el ajuste (0 a 6 s)
    t_fit = t[t <= 6.0]

    # Tiempo total para validación (0 a 12 s)
    t_val = t

    # ============
    # 2) Parámetros base del modelo
    # ============
    base_params = {
        "g": 9.81,
        "rho": 1.225,
        "T_acc": 35.0,
        "t_burn": 3.0,
        "k": 0.0,
        "h_min": 0.0,
    }

    # ============
    # 3) Parámetros verdaderos (experimento)
    # ============
    true_params = dict(base_params)
    true_params["k"] = 0.015

    noise_std_h = 8.0

    h_true_fit, h_meas_fit = generate_synthetic_measurements(
        t_fit, true_params, noise_std_h=noise_std_h, seed=7
    )

    # ============
    # 4) Ajuste de k usando SOLO datos hasta 6 s
    # ============
    k_candidates = np.linspace(0.0, 0.05, 251)
    fit = fit_k_grid_search(t_fit, base_params, h_meas_fit, k_candidates)

    k_hat = fit["k"]
    rmse_hat = fit["rmse"]
    errors = fit["errors"]

    # ============
    # 5) Predicción hasta 12 s (validación)
    # ============
    params_val = dict(base_params)
    params_val["k"] = k_hat

    h_hat_val, _, _ = simulate_rocket_euler(t_val, params_val)

    # ============
    # 6) Reporte
    # ============
    print("\n================= CLASE 1: VALIDACIÓN =================")
    print(f"Ruido del sensor: sigma = {noise_std_h} m")
    print(f"k verdadero: {true_params['k']} m^2/kg")
    print(f"k estimado:  {k_hat} m^2/kg")
    print(f"RMSE ajuste (0–6 s): {rmse_hat} m")
    print("=======================================================\n")

    # ============
    # 7) Gráfica de validación
    # ============
    plt.figure()
    plt.plot(t_val, h_hat_val, label="Predicción del modelo (0–12 s)", linewidth=2)
    plt.plot(t_fit, h_meas_fit, "o", label="Datos usados para ajuste (0–6 s)", markersize=3)
    plt.axvline(6.0, linestyle="--", label="Fin del ajuste (6 s)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Altura [m]")
    plt.title("Validación simple: ajuste hasta 6 s, predicción hasta 12 s")
    plt.grid(True)
    plt.legend()

    plt.show()
if __name__ == "__main__":
    main()
