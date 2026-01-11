"""
Clase 6: Discretización de Modelos y Estabilidad Numérica
Práctica: Señal Amortiguada (Continua vs Discretizada)

Objetivo:
    Comparar una señal amortiguada en tiempo continuo vs su versión discretizada
    y entender el efecto del periodo de muestreo Ts.

Señal:
    x(t) = A * exp(-alpha*t) * sin(2*pi*f*t + phi)

Parámetros:
    A     : Amplitud inicial
    alpha : Coeficiente de amortiguamiento [1/s]
    f     : Frecuencia [Hz]
    phi   : Fase inicial [rad]
    Ts    : Periodo de muestreo [s]
"""

import numpy as np
import matplotlib.pyplot as plt


def damped_signal(t, A=1.0, alpha=0.7, f=2.0, phi=0.0):
    """
    Calcula la señal amortiguada en tiempo continuo.

    Parámetros:
        t     : array de tiempo [s]
        A     : amplitud inicial
        alpha : amortiguamiento [1/s]
        f     : frecuencia [Hz]
        phi   : fase inicial [rad]

    Retorna:
        x(t) = A * exp(-alpha*t) * sin(2*pi*f*t + phi)
    """
    return A * np.exp(-alpha * t) * np.sin(2*np.pi*f*t + phi)


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
    # PARÁMETROS DE LA PRÁCTICA (valores de la presentación)
    # ========================================================================
    A = 1.0          # Amplitud inicial
    alpha = 0.7      # Amortiguamiento [1/s]
    f = 2.0          # Frecuencia [Hz]
    phi = 0.0        # Fase inicial [rad]
    Ts = 0.12        # Periodo de muestreo [s]
    tf = 5.0         # Tiempo final [s]

    print("=" * 70)
    print("PRÁCTICA: Señal Amortiguada - Continuo vs Discreto")
    print("=" * 70)
    print(f"\nParámetros de la señal:")
    print(f"  Amplitud (A)        : {A}")
    print(f"  Amortiguamiento (α) : {alpha} [1/s]")
    print(f"  Frecuencia (f)      : {f} [Hz]")
    print(f"  Fase (φ)            : {phi} [rad]")
    print(f"  Periodo muestreo Ts : {Ts} [s]")
    print(f"  Frecuencia muestreo : {1/Ts:.2f} [Hz]")
    print(f"  Tiempo final        : {tf} [s]")

    # ========================================================================
    # GENERAR SEÑAL CONTINUA (alta resolución)
    # ========================================================================
    t_continuo = np.linspace(0.0, tf, 4000)
    x_continuo = damped_signal(t_continuo, A, alpha, f, phi)

    # ========================================================================
    # GENERAR SEÑAL DISCRETIZADA (muestreo)
    # ========================================================================
    t_discreto = sample_times(Ts, 0.0, tf)
    x_discreto = damped_signal(t_discreto, A, alpha, f, phi)

    print(f"\nPuntos generados:")
    print(f"  Señal continua  : {len(t_continuo)} puntos")
    print(f"  Señal discreta  : {len(t_discreto)} puntos")
    print(f"  Razón           : {len(t_continuo)/len(t_discreto):.1f}:1")

    # ========================================================================
    # ANÁLISIS DE LA SEÑAL
    # ========================================================================
    periodo_oscilacion = 1.0 / f
    constante_tiempo = 1.0 / alpha
    muestras_por_periodo = periodo_oscilacion / Ts

    print(f"\nAnálisis:")
    print(f"  Periodo de oscilación       : {periodo_oscilacion:.3f} [s]")
    print(f"  Constante de tiempo (τ=1/α) : {constante_tiempo:.3f} [s]")
    print(f"  Muestras por periodo        : {muestras_por_periodo:.2f}")

    if muestras_por_periodo < 2:
        print("  ⚠️  ADVERTENCIA: Ts es muy grande! Violación del teorema de Nyquist.")
        print("     Se requiere fs > 2*f, es decir Ts < 0.25 s")
    elif muestras_por_periodo < 10:
        print("  ⚠️  ADVERTENCIA: Ts está cerca del límite. Se recomienda Ts más pequeño.")
    else:
        print("  ✓  Ts adecuado para capturar la señal sin aliasing.")

    # ========================================================================
    # VISUALIZACIÓN
    # ========================================================================
    plt.figure(figsize=(12, 6))

    # Graficar señal continua
    plt.plot(t_continuo, x_continuo,
             'b-', linewidth=2, label='Continuo x(t)', alpha=0.7)

    # Graficar señal discretizada
    plt.stem(t_discreto, x_discreto,
             linefmt='r-', markerfmt='ro', basefmt=' ',
             label='Discreto x[k]')

    # Graficar envolvente exponencial
    envolvente_pos = A * np.exp(-alpha * t_continuo)
    envolvente_neg = -A * np.exp(-alpha * t_continuo)
    plt.plot(t_continuo, envolvente_pos, 'g--',
             linewidth=1, alpha=0.5, label='Envolvente ±A·e^(-αt)')
    plt.plot(t_continuo, envolvente_neg, 'g--',
             linewidth=1, alpha=0.5)

    # Configuración de la gráfica
    plt.xlabel('Tiempo t [s]', fontsize=12)
    plt.ylabel('Amplitud x', fontsize=12)
    plt.title(f'Señal Amortiguada: Continuo vs Discreto\n' +
              f'A={A}, α={alpha} 1/s, f={f} Hz, Ts={Ts} s',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.xlim([0, tf])

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("PREGUNTAS PARA REFLEXIONAR:")
    print("=" * 70)
    print("1. ¿Qué sucede si aumentas Ts a 0.3 s? ¿Se conserva la forma de la señal?")
    print("2. ¿Qué sucede si disminuyes Ts a 0.05 s? ¿Mejora la representación?")
    print("3. ¿Cuántas muestras por periodo son necesarias para capturar bien la señal?")
    print("4. ¿Cómo afecta el valor de α (amortiguamiento) a la duración de la señal?")
    print("5. Prueba aumentar f a 5 Hz. ¿Qué debe pasar con Ts para evitar aliasing?")
    print("=" * 70)


if __name__ == "__main__":
    main()
