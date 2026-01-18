"""
Clase 6: Discretización de Modelos y Estabilidad Numérica
Ejemplos Básicos: Señales Continuas vs Discretizadas

Este archivo contiene ejemplos simples de los conceptos presentados en clase:
    - Ejemplo 1: Señal senoidal
    - Ejemplo 2: Señal exponencial (decaimiento)
    - Ejemplo 3: Comparación visual

Objetivo:
    Entender cómo el muestreo afecta la representación de señales continuas.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# EJEMPLO 1: SEÑAL SENOIDAL
# ============================================================================
def ejemplo_seno():
    """
    Ejemplo 1: Seno continuo vs muestreado
    x(t) = A * sin(2*pi*f*t)
    """
    print("=" * 70)
    print("EJEMPLO 1: Señal Senoidal")
    print("=" * 70)

    # Parámetros
    A = 1.0      # Amplitud
    f = 2.0      # Frecuencia [Hz]
    Ts = 0.15    # Periodo de muestreo [s]
    tf = 2.0     # Tiempo final [s]

    print(f"\nParámetros:")
    print(f"  Amplitud (A)          : {A}")
    print(f"  Frecuencia (f)        : {f} [Hz]")
    print(f"  Periodo de oscilación : {1/f} [s]")
    print(f"  Periodo de muestreo   : {Ts} [s]")
    print(f"  Frecuencia de muestreo: {1/Ts:.2f} [Hz]")

    # Señal continua
    t_cont = np.linspace(0, tf, 1000)
    x_cont = A * np.sin(2 * np.pi * f * t_cont)

    # Señal discreta
    k_max = int(np.floor(tf / Ts))
    k = np.arange(0, k_max + 1)
    t_disc = k * Ts
    x_disc = A * np.sin(2 * np.pi * f * t_disc)

    print(f"\nMuestras por periodo: {(1/f) / Ts:.2f}")
    if (1/f) / Ts < 2:
        print("⚠️  ADVERTENCIA: Menos de 2 muestras por periodo!")
        print("   Violación del teorema de Nyquist -> Aliasing")
    else:
        print("✓  Suficientes muestras para capturar la señal")

    # Graficar
    plt.figure(figsize=(12, 5))
    plt.plot(t_cont, x_cont, 'b-', linewidth=2, label='Continuo x(t)', alpha=0.7)
    plt.stem(t_disc, x_disc, linefmt='r-', markerfmt='ro',
             basefmt=' ', label='Discreto x[k]')
    plt.xlabel('Tiempo t [s]', fontsize=12)
    plt.ylabel('Amplitud x', fontsize=12)
    plt.title(f'Ejemplo 1: Seno - A={A}, f={f} Hz, Ts={Ts} s',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()


# ============================================================================
# EJEMPLO 2: SEÑAL EXPONENCIAL (DECAIMIENTO)
# ============================================================================
def ejemplo_exponencial():
    """
    Ejemplo 2: Exponencial continuo vs muestreado
    x(t) = A * exp(-a*t)
    """
    print("\n" + "=" * 70)
    print("EJEMPLO 2: Señal Exponencial (Decaimiento)")
    print("=" * 70)

    # Parámetros
    A = 1.0      # Amplitud inicial
    a = 1.5      # Tasa de decaimiento [1/s]
    Ts = 0.25    # Periodo de muestreo [s]
    tf = 3.0     # Tiempo final [s]

    print(f"\nParámetros:")
    print(f"  Amplitud inicial (A)  : {A}")
    print(f"  Tasa de decaimiento (a): {a} [1/s]")
    print(f"  Constante de tiempo τ : {1/a:.3f} [s]")
    print(f"  Periodo de muestreo   : {Ts} [s]")

    # Señal continua
    t_cont = np.linspace(0, tf, 1000)
    x_cont = A * np.exp(-a * t_cont)

    # Señal discreta
    k_max = int(np.floor(tf / Ts))
    k = np.arange(0, k_max + 1)
    t_disc = k * Ts
    x_disc = A * np.exp(-a * t_disc)

    # Análisis
    tiempo_5_tau = 5 / a  # Tiempo para alcanzar ~99% del decaimiento
    muestras_en_5_tau = tiempo_5_tau / Ts

    print(f"\nAnálisis:")
    print(f"  Tiempo a 5τ (99% decaimiento): {tiempo_5_tau:.3f} [s]")
    print(f"  Muestras en 5τ               : {muestras_en_5_tau:.1f}")

    # Graficar
    plt.figure(figsize=(12, 5))
    plt.plot(t_cont, x_cont, 'b-', linewidth=2, label='Continuo x(t)', alpha=0.7)
    plt.stem(t_disc, x_disc, linefmt='r-', markerfmt='ro',
             basefmt=' ', label='Discreto x[k]')

    # Marcar constante de tiempo
    plt.axvline(x=1/a, color='g', linestyle='--', alpha=0.5,
                label=f'τ = {1/a:.3f} s')
    plt.axhline(y=A*np.exp(-1), color='g', linestyle='--', alpha=0.5)

    plt.xlabel('Tiempo t [s]', fontsize=12)
    plt.ylabel('Amplitud x', fontsize=12)
    plt.title(f'Ejemplo 2: Exponencial - A={A}, a={a} 1/s, Ts={Ts} s',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()


# ============================================================================
# EJEMPLO 3: COMPARACIÓN CON DIFERENTES Ts
# ============================================================================
def ejemplo_comparacion_Ts():
    """
    Ejemplo 3: Efecto de diferentes periodos de muestreo
    """
    print("\n" + "=" * 70)
    print("EJEMPLO 3: Efecto del Periodo de Muestreo Ts")
    print("=" * 70)

    # Parámetros fijos
    A = 1.0
    f = 3.0      # Frecuencia [Hz]
    tf = 1.5     # Tiempo final [s]

    # Diferentes periodos de muestreo
    Ts_valores = [0.05, 0.15, 0.3]  # [s]

    print(f"\nSeñal: x(t) = {A} * sin(2π * {f} * t)")
    print(f"Frecuencia de Nyquist mínima: {2*f} Hz (Ts_max = {1/(2*f):.3f} s)")

    # Señal continua (referencia)
    t_cont = np.linspace(0, tf, 2000)
    x_cont = A * np.sin(2 * np.pi * f * t_cont)

    # Crear subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for idx, Ts in enumerate(Ts_valores):
        ax = axes[idx]

        # Señal discreta
        k_max = int(np.floor(tf / Ts))
        k = np.arange(0, k_max + 1)
        t_disc = k * Ts
        x_disc = A * np.sin(2 * np.pi * f * t_disc)

        # Análisis
        fs = 1 / Ts
        muestras_por_periodo = (1/f) / Ts
        cumple_nyquist = fs > 2 * f

        # Graficar
        ax.plot(t_cont, x_cont, 'b-', linewidth=2,
                label='Continuo x(t)', alpha=0.5)
        ax.stem(t_disc, x_disc, linefmt='r-', markerfmt='ro',
                basefmt=' ', label='Discreto x[k]')

        # Título con información
        estado = "✓ Cumple Nyquist" if cumple_nyquist else "⚠️ NO cumple Nyquist"
        ax.set_title(f'Ts = {Ts} s  |  fs = {fs:.2f} Hz  |  ' +
                     f'{muestras_por_periodo:.1f} muestras/periodo  |  {estado}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Tiempo t [s]', fontsize=10)
        ax.set_ylabel('Amplitud', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim([0, tf])

        print(f"\nTs = {Ts} s:")
        print(f"  fs = {fs:.2f} Hz")
        print(f"  Muestras por periodo: {muestras_por_periodo:.2f}")
        print(f"  Estado: {estado}")

    plt.tight_layout()
    plt.show()


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================
def main():
    """
    Ejecuta todos los ejemplos.
    """
    print("\n" + "=" * 70)
    print("EJEMPLOS BÁSICOS DE DISCRETIZACIÓN")
    print("Clase 6: Discretización y Estabilidad Numérica")
    print("=" * 70)

    # Ejecutar ejemplos
    ejemplo_seno()

    input("\nPresiona Enter para continuar al Ejemplo 2...")
    ejemplo_exponencial()

    input("\nPresiona Enter para continuar al Ejemplo 3...")
    ejemplo_comparacion_Ts()

    print("\n" + "=" * 70)
    print("CONCEPTOS CLAVE:")
    print("=" * 70)
    print("1. Teorema de Nyquist: fs > 2 * f_max")
    print("   donde fs = 1/Ts es la frecuencia de muestreo")
    print("")
    print("2. Aliasing: Distorsión cuando fs < 2*f_max")
    print("   La señal discreta no representa correctamente la continua")
    print("")
    print("3. Compromiso: Ts pequeño = más muestras = mejor fidelidad")
    print("   pero también = más costo computacional")
    print("")
    print("4. Regla práctica: Usar 10-20 muestras por periodo")
    print("   para una buena representación visual")
    print("=" * 70)


if __name__ == "__main__":
    main()
