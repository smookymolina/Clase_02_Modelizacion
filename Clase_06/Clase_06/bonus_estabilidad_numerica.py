"""
Clase 6: Discretización de Modelos y Estabilidad Numérica
BONUS: Estabilidad Numérica de Métodos de Integración

Este archivo es OPCIONAL y muestra cómo diferentes métodos de discretización
pueden afectar la estabilidad de un sistema, incluso si el sistema continuo es estable.

Conceptos:
    - Euler explícito puede volver inestable un sistema estable
    - RK4 tiene mejor región de estabilidad
    - La elección de Ts (paso de tiempo) es crítica

Sistema de ejemplo:
    dx/dt = -2*x  (sistema estable en continuo, λ = -2)
"""

import numpy as np
import matplotlib.pyplot as plt


def euler_explicito(x, f, dt):
    """
    Método de Euler explícito: x_{k+1} = x_k + dt * f(x_k)
    """
    return x + dt * f(x)


def rk4(x, f, dt):
    """
    Método Runge-Kutta de 4to orden (RK4)
    Más preciso y estable que Euler
    """
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def solucion_analitica(t, x0, lam):
    """
    Solución analítica para dx/dt = λ*x
    x(t) = x0 * exp(λ*t)
    """
    return x0 * np.exp(lam * t)


def ejemplo_estabilidad_euler():
    """
    Demuestra cómo Euler puede volver inestable un sistema estable.
    Sistema: dx/dt = -2*x (estable, λ = -2)
    """
    print("=" * 70)
    print("EJEMPLO: Estabilidad Numérica - Euler Explícito")
    print("=" * 70)

    # Parámetros del sistema
    lam = -2.0       # Autovalor (negativo -> sistema estable)
    x0 = 1.0         # Condición inicial
    tf = 3.0         # Tiempo final

    # Función del sistema: dx/dt = λ*x
    f = lambda x: lam * x

    # Diferentes pasos de tiempo
    dt_valores = [0.5, 1.0, 1.2]  # [s]

    print(f"\nSistema: dx/dt = {lam}*x")
    print(f"Autovalor λ = {lam} (negativo -> estable en continuo)")
    print(f"Condición inicial x0 = {x0}")

    # Condición de estabilidad de Euler: |1 + dt*λ| < 1
    dt_critico = 2 / abs(lam)
    print(f"\n⚠️  Euler es estable solo si: dt < {dt_critico} s")
    print(f"   Condición: |1 + dt*λ| < 1")

    # Solución analítica (referencia)
    t_analitico = np.linspace(0, tf, 1000)
    x_analitico = solucion_analitica(t_analitico, x0, lam)

    # Crear figura
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for idx, dt in enumerate(dt_valores):
        ax = axes[idx]

        # Simular con Euler
        N = int(tf / dt)
        t_euler = np.zeros(N + 1)
        x_euler = np.zeros(N + 1)
        x_euler[0] = x0

        for i in range(N):
            t_euler[i+1] = (i + 1) * dt
            x_euler[i+1] = euler_explicito(x_euler[i], f, dt)

        # Verificar estabilidad
        factor_amplificacion = 1 + dt * lam
        es_estable = abs(factor_amplificacion) < 1

        # Graficar
        ax.plot(t_analitico, x_analitico, 'b-', linewidth=2,
                label='Solución analítica', alpha=0.7)
        ax.plot(t_euler, x_euler, 'ro-', linewidth=1.5,
                markersize=6, label='Euler explícito')

        # Información
        estado = "✓ ESTABLE" if es_estable else "✗ INESTABLE"
        color_titulo = 'green' if es_estable else 'red'

        ax.set_title(f'dt = {dt} s  |  Factor: 1+dt·λ = {factor_amplificacion:.2f}  |  {estado}',
                     fontsize=12, fontweight='bold', color=color_titulo)
        ax.set_xlabel('Tiempo t [s]', fontsize=10)
        ax.set_ylabel('x(t)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim([0, tf])

        # Ajustar límites del eje y según estabilidad
        if es_estable:
            ax.set_ylim([-0.2, 1.2])
        else:
            # Si es inestable, puede explotar
            y_min = min(-0.5, min(x_euler) * 1.1)
            y_max = max(1.5, max(x_euler) * 1.1)
            ax.set_ylim([y_min, y_max])
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        print(f"\ndt = {dt} s:")
        print(f"  Factor de amplificación: {factor_amplificacion:.3f}")
        print(f"  |1 + dt·λ| = {abs(factor_amplificacion):.3f}")
        print(f"  Estado: {estado}")

    plt.tight_layout()
    plt.show()


def ejemplo_comparacion_metodos():
    """
    Compara Euler vs RK4 para el mismo sistema.
    """
    print("\n" + "=" * 70)
    print("EJEMPLO: Comparación Euler vs RK4")
    print("=" * 70)

    # Parámetros
    lam = -2.0
    x0 = 1.0
    tf = 3.0
    dt = 0.5  # Paso que hace Euler inestable

    f = lambda x: lam * x

    print(f"\nSistema: dx/dt = {lam}*x")
    print(f"Paso de tiempo: dt = {dt} s")
    print(f"Este dt viola la condición de estabilidad de Euler ({dt} > {2/abs(lam)})")

    # Solución analítica
    t_analitico = np.linspace(0, tf, 1000)
    x_analitico = solucion_analitica(t_analitico, x0, lam)

    # Euler explícito
    N = int(tf / dt)
    t_num = np.zeros(N + 1)
    x_euler = np.zeros(N + 1)
    x_rk4 = np.zeros(N + 1)
    x_euler[0] = x0
    x_rk4[0] = x0

    for i in range(N):
        t_num[i+1] = (i + 1) * dt
        x_euler[i+1] = euler_explicito(x_euler[i], f, dt)
        x_rk4[i+1] = rk4(x_rk4[i], f, dt)

    # Graficar
    plt.figure(figsize=(12, 6))

    plt.plot(t_analitico, x_analitico, 'b-', linewidth=3,
             label='Solución analítica', alpha=0.7)
    plt.plot(t_num, x_euler, 'r^--', linewidth=1.5, markersize=8,
             label='Euler explícito (INESTABLE)')
    plt.plot(t_num, x_rk4, 'go-', linewidth=1.5, markersize=6,
             label='RK4 (estable)')

    plt.xlabel('Tiempo t [s]', fontsize=12)
    plt.ylabel('x(t)', fontsize=12)
    plt.title(f'Comparación de Métodos - dt = {dt} s\n' +
              'Sistema estable (λ=-2) pero Euler lo hace inestable',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=11)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

    print("\nResultados:")
    print(f"  Solución analítica en t={tf}: {x_analitico[-1]:.6f}")
    print(f"  Euler en t={tf}:              {x_euler[-1]:.6f}")
    print(f"  RK4 en t={tf}:                {x_rk4[-1]:.6f}")
    print(f"\n✓ RK4 se mantiene estable mientras que Euler diverge!")


def ejemplo_sistema_stiff():
    """
    Ejemplo de sistema stiff (rígido) con dos escalas de tiempo.
    """
    print("\n" + "=" * 70)
    print("EJEMPLO AVANZADO: Sistema Stiff (Rígido)")
    print("=" * 70)

    print("\nSistema con dos modos:")
    print("  dx1/dt = -10*x1        (modo rápido)")
    print("  dx2/dt = -0.1*x2       (modo lento)")

    # Parámetros
    lam_rapido = -10.0
    lam_lento = -0.1
    x0 = [1.0, 1.0]
    tf = 5.0
    dt_valores = [0.05, 0.15, 0.25]

    # Funciones
    f1 = lambda x: lam_rapido * x
    f2 = lambda x: lam_lento * x

    print(f"\nCaracterísticas:")
    print(f"  Modo rápido: τ_rápido = {-1/lam_rapido:.2f} s")
    print(f"  Modo lento:  τ_lento  = {-1/lam_lento:.2f} s")
    print(f"  Razón de stiffness: {abs(lam_rapido/lam_lento):.1f}:1")
    print(f"\n⚠️  Para capturar el modo rápido, dt debe ser << {-2/lam_rapido} s")

    fig, axes = plt.subplots(len(dt_valores), 2, figsize=(14, 10))

    for idx, dt in enumerate(dt_valores):
        N = int(tf / dt)
        t = np.zeros(N + 1)
        x1_euler = np.zeros(N + 1)
        x2_euler = np.zeros(N + 1)
        x1_euler[0] = x0[0]
        x2_euler[0] = x0[1]

        for i in range(N):
            t[i+1] = (i + 1) * dt
            x1_euler[i+1] = euler_explicito(x1_euler[i], f1, dt)
            x2_euler[i+1] = euler_explicito(x2_euler[i], f2, dt)

        # Solución analítica
        t_anal = np.linspace(0, tf, 1000)
        x1_anal = np.exp(lam_rapido * t_anal)
        x2_anal = np.exp(lam_lento * t_anal)

        # Modo rápido
        ax1 = axes[idx, 0]
        ax1.plot(t_anal, x1_anal, 'b-', linewidth=2, label='Analítica')
        ax1.plot(t, x1_euler, 'ro-', markersize=4, label='Euler')
        ax1.set_ylabel('x₁ (modo rápido)', fontsize=10)
        ax1.set_title(f'dt = {dt} s', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=8)
        ax1.set_ylim([-0.5, 1.5])

        # Modo lento
        ax2 = axes[idx, 1]
        ax2.plot(t_anal, x2_anal, 'b-', linewidth=2, label='Analítica')
        ax2.plot(t, x2_euler, 'ro-', markersize=4, label='Euler')
        ax2.set_ylabel('x₂ (modo lento)', fontsize=10)
        ax2.set_title(f'dt = {dt} s', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=8)

        if idx == len(dt_valores) - 1:
            ax1.set_xlabel('Tiempo t [s]', fontsize=10)
            ax2.set_xlabel('Tiempo t [s]', fontsize=10)

        print(f"\ndt = {dt} s:")
        print(f"  x1 en t={tf}: Euler = {x1_euler[-1]:.6f}, Exacto = {x1_anal[-1]:.6f}")
        print(f"  x2 en t={tf}: Euler = {x2_euler[-1]:.6f}, Exacto = {x2_anal[-1]:.6f}")

    plt.tight_layout()
    plt.show()


def main():
    """
    Función principal: ejecuta todos los ejemplos de estabilidad.
    """
    print("\n" + "=" * 70)
    print("BONUS: ESTABILIDAD NUMÉRICA DE MÉTODOS DE DISCRETIZACIÓN")
    print("=" * 70)
    print("\nEste material es OPCIONAL pero muy importante para entender")
    print("cómo la discretización puede afectar la estabilidad de sistemas.")

    # Ejemplo 1
    ejemplo_estabilidad_euler()

    input("\nPresiona Enter para ver la comparación Euler vs RK4...")
    ejemplo_comparacion_metodos()

    input("\nPresiona Enter para ver el ejemplo de sistema stiff...")
    ejemplo_sistema_stiff()

    print("\n" + "=" * 70)
    print("LECCIONES CLAVE:")
    print("=" * 70)
    print("\n1. ESTABILIDAD vs PRECISIÓN son conceptos diferentes:")
    print("   - Un método puede ser preciso pero numéricamente inestable")
    print("   - La estabilidad limita el tamaño máximo de dt")

    print("\n2. EULER EXPLÍCITO:")
    print("   - Simple pero con región de estabilidad pequeña")
    print("   - Condición: |1 + dt·λ| < 1 para cada autovalor λ")
    print("   - Puede volver inestable un sistema estable")

    print("\n3. RK4 (Runge-Kutta 4to orden):")
    print("   - Más complejo pero más robusto")
    print("   - Mayor región de estabilidad")
    print("   - Mejor precisión para mismo dt")

    print("\n4. SISTEMAS STIFF (RÍGIDOS):")
    print("   - Múltiples escalas de tiempo (rápidas y lentas)")
    print("   - Requieren dt muy pequeño con métodos explícitos")
    print("   - Solución: métodos implícitos (Backward Euler, BDF)")
    print("   - En Python: scipy.integrate.solve_ivp con method='Radau' o 'BDF'")

    print("\n5. REGLA PRÁCTICA:")
    print("   - Para sistemas con autovalor λ más negativo:")
    print("     dt < 2/|λ| (Euler)")
    print("     dt < 2.8/|λ| (RK4)")
    print("   - Para sistemas stiff: usar métodos implícitos")

    print("\n" + "=" * 70)
    print("APLICACIONES EN INGENIERÍA AEROESPACIAL:")
    print("=" * 70)
    print("- Control digital de actitud (diferentes tiempos de respuesta)")
    print("- Dinámica orbital (fuerzas gravitacionales vs perturbaciones)")
    print("- Aerodinámica (flujo rápido vs dinámica estructural lenta)")
    print("- Sistemas térmicos (convección rápida vs conducción lenta)")
    print("=" * 70)


if __name__ == "__main__":
    main()
