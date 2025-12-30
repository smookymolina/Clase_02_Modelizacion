import sys
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# OPCIONAL: SciPy (para RK45/solve_ivp)
# ============================================================
try:
    from scipy.integrate import solve_ivp
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ============================================================
# UTILIDADES
# ============================================================
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78 + "\n")


def print_step(msg: str) -> None:
    print(f"\n--- {msg} ---")


def prompt_float(msg: str, default: float, *, min_value=None, max_value=None) -> float:
    """
    Prompt robusto para float:
    - Enter => default
    - Repite si hay error de formato
    - Valida rango si se especifica
    """
    while True:
        s = input(f"{msg} (Enter = {default}): ").strip()
        if s == "":
            val = float(default)
            print(f"   → Usando: {val}")
        else:
            try:
                val = float(s)
                print(f"   → Usando: {val}")
            except ValueError:
                print("   ✗ Entrada inválida. Escribe un número (ej: 9.81) o Enter.")
                continue

        if (min_value is not None) and (val < min_value):
            print(f"   ✗ Debe ser >= {min_value}. Intenta de nuevo.")
            continue
        if (max_value is not None) and (val > max_value):
            print(f"   ✗ Debe ser <= {max_value}. Intenta de nuevo.")
            continue

        return val


def prompt_int(msg: str, default: int, *, min_value=None, max_value=None) -> int:
    """
    Prompt robusto para int:
    - Enter => default
    - Repite si hay error de formato
    - Valida rango si se especifica
    """
    while True:
        s = input(f"{msg} (Enter = {default}): ").strip()
        if s == "":
            val = int(default)
            print(f"   → Usando: {val}")
        else:
            try:
                val = int(s)
                print(f"   → Usando: {val}")
            except ValueError:
                print("   ✗ Entrada inválida. Escribe un entero (ej: 401) o Enter.")
                continue

        if (min_value is not None) and (val < min_value):
            print(f"   ✗ Debe ser >= {min_value}. Intenta de nuevo.")
            continue
        if (max_value is not None) and (val > max_value):
            print(f"   ✗ Debe ser <= {max_value}. Intenta de nuevo.")
            continue

        return val


def prompt_yes_no(msg: str, default_yes: bool = True) -> bool:
    """
    Pregunta Sí/No intuitiva:
    - Enter => default
    - Acepta: s/si/y/yes  y  n/no
    """
    default_str = "S" if default_yes else "N"
    while True:
        s = input(f"{msg} [S/N] (Enter = {default_str}): ").strip().lower()
        if s == "":
            ans = default_yes
            print(f"   → Usando: {'SÍ' if ans else 'NO'}")
            return ans
        if s in ("s", "si", "sí", "y", "yes"):
            print("   → Usando: SÍ")
            return True
        if s in ("n", "no"):
            print("   → Usando: NO")
            return False
        print("   ✗ Responde S o N (o Enter).")


# ============================================================
# DINÁMICA: caída con arrastre cuadrático
# x = [h, v]
# dh/dt = v
# dv/dt = -g - k*v*|v|
# ============================================================
def make_dynamics(g: float, k: float):
    def f(t: float, x: np.ndarray) -> np.ndarray:
        h, v = x
        dh = v
        dv = -g - k * v * abs(v)
        return np.array([dh, dv], dtype=float)
    return f


# ============================================================
# MALLA DE TIEMPO ROBUSTA
# ============================================================
def make_time_grid(t_span: tuple, dt: float) -> np.ndarray:
    t0, tf = t_span
    if dt <= 0:
        raise ValueError("dt debe ser > 0")
    N = int(np.ceil((tf - t0) / dt)) + 1
    t = t0 + np.arange(N, dtype=float) * dt
    t[-1] = tf  # cierre exacto
    return t


# ============================================================
# MÉTODOS NUMÉRICOS (Euler / RK4) con stop opcional en h=0
# ============================================================
def euler_method(f, x0: np.ndarray, t_span: tuple, dt: float, stop_at_ground: bool = True):
    t = make_time_grid(t_span, dt)
    x = np.zeros((len(t), len(x0)), dtype=float)
    x[0] = x0.astype(float)

    for n in range(len(t) - 1):
        dtn = t[n + 1] - t[n]
        x[n + 1] = x[n] + dtn * f(t[n], x[n])

        if stop_at_ground and x[n + 1, 0] <= 0.0:
            x[n + 1, 0] = 0.0
            return t[: n + 2], x[: n + 2]

    return t, x


def rk4_method(f, x0: np.ndarray, t_span: tuple, dt: float, stop_at_ground: bool = True):
    t = make_time_grid(t_span, dt)
    x = np.zeros((len(t), len(x0)), dtype=float)
    x[0] = x0.astype(float)

    for n in range(len(t) - 1):
        tn = t[n]
        xn = x[n]
        dtn = t[n + 1] - t[n]

        k1 = f(tn, xn)
        k2 = f(tn + dtn / 2.0, xn + dtn * k1 / 2.0)
        k3 = f(tn + dtn / 2.0, xn + dtn * k2 / 2.0)
        k4 = f(tn + dtn, xn + dtn * k3)

        x[n + 1] = xn + (dtn / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if stop_at_ground and x[n + 1, 0] <= 0.0:
            x[n + 1, 0] = 0.0
            return t[: n + 2], x[: n + 2]

    return t, x


def rk45_scipy(f, x0: np.ndarray, t_span: tuple, t_eval: np.ndarray,
               rtol: float = 1e-7, atol: float = 1e-9, stop_at_ground: bool = True):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy no está disponible. Instala con: pip install scipy")

    def rhs(t, y):
        return f(t, y)

    events = None
    if stop_at_ground:
        def hit_ground(t, y):
            return y[0]
        hit_ground.terminal = True
        hit_ground.direction = -1
        events = hit_ground

    sol = solve_ivp(
        rhs,
        t_span=t_span,
        y0=x0.tolist(),
        method="RK45",
        t_eval=t_eval,
        dense_output=False,
        rtol=rtol,
        atol=atol,
        events=events
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.t, sol.y.T


# ============================================================
# REFERENCIA (error)
# ============================================================
def compute_reference(f, x0, t_span, t_eval, prefer_scipy=True, stop_at_ground=True):
    if prefer_scipy and SCIPY_AVAILABLE:
        t_ref, X_ref = rk45_scipy(
            f, x0, t_span, t_eval,
            rtol=1e-9, atol=1e-11,
            stop_at_ground=stop_at_ground
        )
        ref_label = "Referencia: SciPy RK45 (rtol=1e-9, atol=1e-11)"
        return t_ref, X_ref, ref_label

    tf = t_span[1]
    dt_ref = min(1e-3, (tf - t_span[0]) / 20000.0)
    t_ref, X_ref = rk4_method(f, x0, t_span, dt_ref, stop_at_ground=stop_at_ground)

    t_end = t_ref[-1]
    t_eval_clip = t_eval[t_eval <= t_end + 1e-12]
    X_interp = np.zeros((len(t_eval_clip), X_ref.shape[1]), dtype=float)
    for i in range(X_ref.shape[1]):
        X_interp[:, i] = np.interp(t_eval_clip, t_ref, X_ref[:, i])

    ref_label = f"Referencia: RK4 (dt_ref={dt_ref:g}) interpolada"
    return t_eval_clip, X_interp, ref_label


# ============================================================
# PLOTEO
# ============================================================
def plot_results(t, X, t_ref, X_ref, method_name, ref_label):
    h = X[:, 0]
    v = X[:, 1]
    href = X_ref[:, 0]
    vref = X_ref[:, 1]

    plt.figure()
    plt.plot(t, h, label=method_name)
    plt.plot(t_ref, href, "--", label=ref_label)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Altura h [m]")
    plt.title("Altura vs Tiempo")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(t, v, label=method_name)
    plt.plot(t_ref, vref, "--", label=ref_label)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Velocidad v [m/s]")
    plt.title("Velocidad vs Tiempo")
    plt.grid(True)
    plt.legend()

    h_on_ref = np.interp(t_ref, t, h)
    v_on_ref = np.interp(t_ref, t, v)
    eh = np.abs(h_on_ref - href)
    ev = np.abs(v_on_ref - vref)

    plt.figure()
    plt.semilogy(t_ref, eh + 1e-15, label="|e_h|")
    plt.semilogy(t_ref, ev + 1e-15, label="|e_v|")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Error absoluto (log)")
    plt.title("Errores absolutos vs Tiempo")
    plt.grid(True, which="both")
    plt.legend()

    plt.show()


# ============================================================
# INTERPRETACIÓN
# ============================================================
def interpret(method: str, dt: float, g: float, k: float, x0: np.ndarray, t_span: tuple):
    h0, v0 = x0
    t0, tf = t_span
    print("\nInterpretación (ingeniería):")
    print(f"- Modelo: dv/dt = -g - k*v|v|   (g={g}, k={k})")
    print(f"- Condiciones: h0={h0} m, v0={v0} m/s, tiempo: [{t0}, {tf}] s")
    print(f"- Método: {method.upper()}")

    if method in ("euler", "rk4"):
        print(f"- Paso fijo: dt={dt} s")
        if method == "euler":
            print("  * Euler (orden 1) => error global ~O(dt).")
        else:
            print("  * RK4 (orden 4) => error global ~O(dt^4) (si dt es suficientemente pequeño).")
    else:
        print("- RK45 adaptativo: ajusta dt internamente según tolerancias.")

    if k > 0:
        vt = np.sqrt(g / k)
        print(f"- Validación física: |v_t| = sqrt(g/k) ≈ {vt:.4g} m/s")
        print("  * Esperas que v(t) tienda a un valor negativo cercano a -|v_t|.")


# ============================================================
# MAIN
# ============================================================
def main():
    print_header("CLASE 3 - SOLVER NUMÉRICO (Euler / RK4 / SciPy RK45)")
    print("Instrucciones rápidas:")
    print("  - Si presionas Enter, se usa el valor por defecto mostrado.")
    print("  - Si escribes un número, se usará ese valor.")
    print("  - Si te equivocas (letras), te lo vuelve a pedir.\n")

    print_step("1) Parámetros físicos")
    g = prompt_float("Gravedad g [m/s^2]", 9.81, min_value=1e-12)
    k = prompt_float("Coeficiente de arrastre k [1/m]", 0.02, min_value=0.0)

    print_step("2) Condiciones iniciales")
    h0 = prompt_float("Altura inicial h0 [m]", 1000.0)
    v0 = prompt_float("Velocidad inicial v0 [m/s] (negativa = hacia abajo)", 0.0)

    print_step("3) Horizonte de simulación")
    t0 = prompt_float("Tiempo inicial t0 [s]", 0.0)
    tf = prompt_float("Tiempo final tf [s]", 20.0, min_value=t0 + 1e-12)

    stop_at_ground = prompt_yes_no("¿Detener al tocar suelo (h=0)?", default_yes=True)

    print_step("4) Selección de método numérico")
    print("  1) Euler (paso fijo)")
    print("  2) RK4   (paso fijo)")
    if SCIPY_AVAILABLE:
        print("  3) SciPy RK45 (solve_ivp, paso adaptativo)")
    else:
        print("  3) SciPy RK45 (NO DISPONIBLE: instala scipy)")

    choice = input("Elige opción (Enter = 2): ").strip()
    if choice == "":
        choice = "2"
        print("   → Usando: 2 (RK4)")
    if choice not in ("1", "2", "3"):
        raise ValueError("Opción inválida")

    print_step("5) Configuración de salida (gráficas)")
    N_eval = prompt_int("Número de puntos para graficar (t_eval)", 401, min_value=2)
    t_eval = np.linspace(t0, tf, N_eval)

    f = make_dynamics(g, k)
    x0 = np.array([h0, v0], dtype=float)
    t_span = (t0, tf)

    dt = None
    if choice == "1":
        dt = prompt_float("Paso dt para Euler [s]", 0.5, min_value=1e-12)
        t, X = euler_method(f, x0, t_span, dt, stop_at_ground=stop_at_ground)
        method_name = f"Euler (dt={dt})"
        method_key = "euler"
    elif choice == "2":
        dt = prompt_float("Paso dt para RK4 [s]", 0.5, min_value=1e-12)
        t, X = rk4_method(f, x0, t_span, dt, stop_at_ground=stop_at_ground)
        method_name = f"RK4 (dt={dt})"
        method_key = "rk4"
    else:
        if not SCIPY_AVAILABLE:
            raise RuntimeError("SciPy no disponible. Instala con: pip install scipy")
        rtol = prompt_float("rtol (tolerancia relativa)", 1e-6, min_value=1e-15)
        atol = prompt_float("atol (tolerancia absoluta)", 1e-9, min_value=1e-15)
        t, X = rk45_scipy(f, x0, t_span, t_eval, rtol=rtol, atol=atol, stop_at_ground=stop_at_ground)
        method_name = f"SciPy RK45 (rtol={rtol:g}, atol={atol:g})"
        method_key = "rk45"

    # Referencia
    t_ref, X_ref, ref_label = compute_reference(
        f, x0, t_span, t_eval, prefer_scipy=True, stop_at_ground=stop_at_ground
    )

    # Errores en t_ref
    h = X[:, 0]
    v = X[:, 1]
    h_on_ref = np.interp(t_ref, t, h)
    v_on_ref = np.interp(t_ref, t, v)

    rmse_h = rmse(h_on_ref, X_ref[:, 0])
    rmse_v = rmse(v_on_ref, X_ref[:, 1])
    max_h = float(np.max(np.abs(h_on_ref - X_ref[:, 0])))
    max_v = float(np.max(np.abs(v_on_ref - X_ref[:, 1])))

    print_header("RESULTADOS")
    print(f"Método: {method_name}")
    print(ref_label)
    print(f"RMSE altura   : {rmse_h:.6g} m")
    print(f"RMSE velocidad: {rmse_v:.6g} m/s")
    print(f"Max |e_h|     : {max_h:.6g} m")
    print(f"Max |e_v|     : {max_v:.6g} m/s")

    interpret(method_key, dt if dt is not None else float("nan"), g, k, x0, t_span)
    plot_results(t, X, t_ref, X_ref, method_name, ref_label)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
