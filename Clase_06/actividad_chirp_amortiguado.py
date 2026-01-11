"""
Clase 6: Discretización de Modelos y Estabilidad Numérica
Actividad: Chirp Lineal Amortiguado

Objetivo:
    Cada alumno debe discretizar la señal chirp lineal amortiguado,
    graficar continuo vs discreto, y justificar el Ts elegido.

    DESAFÍO: Experimentar con diferentes valores de los parámetros
    y observar el efecto del aliasing vs fidelidad de la señal.

Señal:
    x(t) = A * exp(-alpha*t) * sin(2*pi*(f0*t + 0.5*beta*t²) + phi)

Parámetros:
    A     : Amplitud inicial
    alpha : Coeficiente de amortiguamiento [1/s]
    f0    : Frecuencia inicial [Hz]
    beta  : Tasa de barrido de frecuencia [Hz/s]
    phi   : Fase inicial [rad]
    Ts    : Periodo de muestreo [s]

Nota:
    La frecuencia instantánea es: f(t) = f0 + beta*t
    A medida que t aumenta, la frecuencia también aumenta (o disminuye si beta < 0).
"""

import numpy as np
import matplotlib.pyplot as plt


def chirp_signal(t, A=1.0, alpha=0.25, f0=1.0, beta=2.0, phi=0.0):
    """
    Calcula la señal chirp lineal amortiguado en tiempo continuo.

    Parámetros:
        t     : array de tiempo [s]
        A     : amplitud inicial
        alpha : amortiguamiento [1/s]
        f0    : frecuencia inicial [Hz]
        beta  : tasa de barrido [Hz/s]
        phi   : fase inicial [rad]

    Retorna:
        x(t) = A * exp(-alpha*t) * sin(2*pi*(f0*t + 0.5*beta*t²) + phi)
    """
    phase = 2 * np.pi * (f0 * t + 0.5 * beta * t**2) + phi
    return A * np.exp(-alpha * t) * np.sin(phase)


def instantaneous_frequency(t, f0, beta):
    """
    Calcula la frecuencia instantánea del chirp.

    Parámetros:
        t    : tiempo [s]
        f0   : frecuencia inicial [Hz]
        beta : tasa de barrido [Hz/s]

    Retorna:
        f(t) = f0 + beta*t
    """
    return f0 + beta * t


def sample_times(Ts, t0=0.0, tf=5.0):
    """
    Genera los instantes de tiempo discretos.

    Parámetros:
        Ts : periodo de muestreo [s]
        t0 : tiempo inicial [s]
        tf : tiempo final [s]

    Retorna:
        Array de instantes discretos: t_k = t0 + k*Ts
    """
    k_max = int(np.floor((tf - t0) / Ts))
    k = np.arange(0, k_max + 1)
    return t0 + k * Ts


def main():
    """
    Función principal: genera y compara señales continua y discretizada.
    """
    # ========================================================================
    # PARÁMETROS DE LA ACTIVIDAD
    # ========================================================================
    # TODO: Los estudiantes deben EXPERIMENTAR modificando estos valores

    A = 1.0          # Amplitud inicial
    alpha = 0.25     # Amortiguamiento [1/s] - prueba: 0.1, 0.5, 1.0
    f0 = 1.0         # Frecuencia inicial [Hz] - prueba: 0.5, 1.0, 2.0
    beta = 2.0       # Tasa de barrido [Hz/s] - prueba: 1.0, 2.0, 3.0, -1.0
    phi = 0.0        # Fase inicial [rad]
    Ts = 0.08        # Periodo de muestreo [s] - ¡IMPORTANTE! prueba: 0.05, 0.1, 0.2
    tf = 5.0         # Tiempo final [s]

    print("=" * 70)
    print("ACTIVIDAD: Chirp Lineal Amortiguado - Continuo vs Discreto")
    print("=" * 70)
    print(f"\nParámetros de la señal:")
    print(f"  Amplitud (A)        : {A}")
    print(f"  Amortiguamiento (α) : {alpha} [1/s]")
    print(f"  Frecuencia inicial (f0) : {f0} [Hz]")
    print(f"  Tasa de barrido (β)     : {beta} [Hz/s]")
    print(f"  Fase (φ)            : {phi} [rad]")
    print(f"  Periodo muestreo Ts : {Ts} [s]")
    print(f"  Frecuencia muestreo : {1/Ts:.2f} [Hz]")
    print(f"  Tiempo final        : {tf} [s]")

    # ========================================================================
    # ANÁLISIS DE LA SEÑAL
    # ========================================================================
    f_inicial = f0
    f_final = f0 + beta * tf
    f_maxima = max(abs(f_inicial), abs(f_final))
    frecuencia_nyquist = 1 / (2 * Ts)
    constante_tiempo = 1.0 / alpha

    print(f"\nAnálisis de frecuencias:")
    print(f"  Frecuencia inicial f(0)  : {f_inicial:.2f} [Hz]")
    print(f"  Frecuencia final f({tf})   : {f_final:.2f} [Hz]")
    print(f"  Frecuencia máxima        : {f_maxima:.2f} [Hz]")
    print(f"  Frecuencia de Nyquist    : {frecuencia_nyquist:.2f} [Hz]")
    print(f"  Constante de tiempo (τ)  : {constante_tiempo:.3f} [s]")

    # Verificar criterio de Nyquist
    print(f"\nCriterio de Nyquist:")
    if f_maxima < frecuencia_nyquist:
        print(f"  ✓  CUMPLE: f_max ({f_maxima:.2f} Hz) < f_Nyquist ({frecuencia_nyquist:.2f} Hz)")
        print(f"     No debería haber aliasing significativo.")
    else:
        print(f"  ⚠️  NO CUMPLE: f_max ({f_maxima:.2f} Hz) > f_Nyquist ({frecuencia_nyquist:.2f} Hz)")
        print(f"     ¡ADVERTENCIA! Se esperan efectos de aliasing.")
        print(f"     Para evitar aliasing, Ts debería ser < {1/(2*f_maxima):.3f} s")

    # ========================================================================
    # GENERAR SEÑAL CONTINUA (alta resolución)
    # ========================================================================
    t_continuo = np.linspace(0.0, tf, 5000)
    x_continuo = chirp_signal(t_continuo, A, alpha, f0, beta, phi)
    f_instantanea = instantaneous_frequency(t_continuo, f0, beta)

    # ========================================================================
    # GENERAR SEÑAL DISCRETIZADA (muestreo)
    # ========================================================================
    t_discreto = sample_times(Ts, 0.0, tf)
    x_discreto = chirp_signal(t_discreto, A, alpha, f0, beta, phi)

    print(f"\nPuntos generados:")
    print(f"  Señal continua  : {len(t_continuo)} puntos")
    print(f"  Señal discreta  : {len(t_discreto)} puntos")
    print(f"  Razón           : {len(t_continuo)/len(t_discreto):.1f}:1")

    # ========================================================================
    # VISUALIZACIÓN (Dos subplots)
    # ========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # -------- Subplot 1: Señal en el tiempo --------
    ax1.plot(t_continuo, x_continuo,
             'b-', linewidth=2, label='Continuo x(t)', alpha=0.7)
    ax1.stem(t_discreto, x_discreto,
             linefmt='r-', markerfmt='ro', basefmt=' ',
             label='Discreto x[k]')

    # Envolvente exponencial
    envolvente_pos = A * np.exp(-alpha * t_continuo)
    envolvente_neg = -A * np.exp(-alpha * t_continuo)
    ax1.plot(t_continuo, envolvente_pos, 'g--',
             linewidth=1, alpha=0.5, label='Envolvente ±A·e^(-αt)')
    ax1.plot(t_continuo, envolvente_neg, 'g--',
             linewidth=1, alpha=0.5)

    ax1.set_xlabel('Tiempo t [s]', fontsize=12)
    ax1.set_ylabel('Amplitud x', fontsize=12)
    ax1.set_title(f'Chirp Lineal Amortiguado: Continuo vs Discreto\n' +
                  f'A={A}, α={alpha} 1/s, f0={f0} Hz, β={beta} Hz/s, Ts={Ts} s',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim([0, tf])

    # -------- Subplot 2: Frecuencia instantánea --------
    ax2.plot(t_continuo, f_instantanea,
             'b-', linewidth=2, label='f(t) = f0 + βt')
    ax2.axhline(y=frecuencia_nyquist, color='r', linestyle='--',
                linewidth=2, label=f'f_Nyquist = {frecuencia_nyquist:.2f} Hz')

    # Marcar zona de aliasing
    if f_maxima > frecuencia_nyquist:
        ax2.axhspan(frecuencia_nyquist, f_maxima, alpha=0.2, color='red',
                    label='Zona de aliasing')

    ax2.set_xlabel('Tiempo t [s]', fontsize=12)
    ax2.set_ylabel('Frecuencia f(t) [Hz]', fontsize=12)
    ax2.set_title('Frecuencia Instantánea vs Frecuencia de Nyquist',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    ax2.set_xlim([0, tf])

    plt.tight_layout()
    plt.show()

    # ========================================================================
    # PREGUNTAS GUÍA PARA EL ALUMNO
    # ========================================================================
    print("\n" + "=" * 70)
    print("PREGUNTAS PARA RESPONDER EN TU REPORTE:")
    print("=" * 70)
    print("\n1. JUSTIFICACIÓN DEL Ts ELEGIDO:")
    print("   - ¿Por qué elegiste este valor de Ts?")
    print("   - ¿Cumple con el criterio de Nyquist para todo el rango de frecuencias?")
    print("   - ¿Qué compromiso hay entre resolución temporal y aliasing?")

    print("\n2. EXPERIMENTACIÓN CON PARÁMETROS:")
    print("   a) Aumenta Ts a 0.15 s. ¿Qué observas? ¿Hay aliasing?")
    print("   b) Disminuye Ts a 0.05 s. ¿Mejora la fidelidad?")
    print("   c) Aumenta beta a 5.0 Hz/s. ¿Qué debe pasar con Ts?")
    print("   d) Prueba beta negativo (-2.0). ¿Qué ocurre con la frecuencia?")

    print("\n3. ANÁLISIS DE ALIASING:")
    print("   - ¿En qué instante de tiempo comienza a aparecer aliasing?")
    print("   - ¿Cómo se manifiesta el aliasing en la señal discreta?")
    print("   - ¿Qué relación tiene con la frecuencia instantánea?")

    print("\n4. EFECTOS DEL AMORTIGUAMIENTO:")
    print("   - Prueba alpha = 0.1 (poco amortiguamiento)")
    print("   - Prueba alpha = 1.0 (mucho amortiguamiento)")
    print("   - ¿Cómo afecta alpha a la duración efectiva de la señal?")

    print("\n5. APLICACIONES PRÁCTICAS:")
    print("   - ¿Dónde se usan señales chirp en ingeniería aeroespacial?")
    print("   - (Pista: radar, sonar, telecomunicaciones, análisis estructural)")

    print("=" * 70)
    print("\n💡 SUGERENCIA: Ejecuta este script varias veces modificando")
    print("   los parámetros en la sección 'PARÁMETROS DE LA ACTIVIDAD'")
    print("   y documenta tus observaciones para el reporte.")
    print("=" * 70)


if __name__ == "__main__":
    main()
