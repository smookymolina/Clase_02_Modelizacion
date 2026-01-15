import numpy as np
import matplotlib.pyplot as plt

def simular_sistema(dt, tipo_entrada):
    """
    Simula el sistema dx = -a*x + b*u usando el método de Euler.
    """
    # 1. Definir parámetros (a > 0, b > 0, x(0))
    a = 2.5
    b = 3.0
    tiempo_total = 10.0  # Simular de 0 a 10s
    
    # Crear vector de tiempo
    t = np.arange(0, tiempo_total, dt)
    n = len(t)
    
    # Inicializar vectores de estado (x) y entrada (u)
    x = np.zeros(n)
    u = np.zeros(n)
    
    # Condición inicial x(0)
    x[0] = 0.0 

    # 2. Bucle de Simulación (Integrador Básico / Euler)
    for i in range(n - 1):
        tiempo_actual = t[i]
        
        # Definir la entrada u(t) 
        if tipo_entrada == "escalon":
            # u(t) = 1 (Escalón unitario)
            u_val = 1.0
        elif tipo_entrada == "seno":
            # u(t) = sin(2*pi*f*t)
            f = 0.25 # frecuencia en Hz
            u_val = np.sin(2 * np.pi * f * tiempo_actual)
        
        u[i] = u_val 
        
        # --- Ecuación Diferencial: dx = -ax + bu ---
        dx = -a * x[i] + b * u_val
        
        # --- Método de Euler: x_nuevo = x_viejo + dx * dt ---
        x[i + 1] = x[i] + dx * dt
        
    # Ajuste final para el último valor de u
    if tipo_entrada == "escalon": u[-1] = 1.0
    else: u[-1] = np.sin(2 * np.pi * 0.25 * t[-1])

    return t, x, u

# ==========================================
# EJECUCIÓN Y GRÁFICAS 
# ==========================================

plt.figure(figsize=(12, 6))

# CASO A: Entrada Escalón con dt = 0.1
t1, x1, u1 = simular_sistema(dt=0.1, tipo_entrada="escalon")
plt.subplot(1, 2, 1)
plt.plot(t1, u1, 'k--', alpha=0.5, label='Entrada u(t)')
plt.plot(t1, x1, 'b-', linewidth=2, label='Salida x(t) (dt=0.1)')
plt.title("Respuesta al Escalón (dt=0.1)")
plt.xlabel("Tiempo (s)")
plt.legend()
plt.grid()

# CASO B: Entrada Senoidal con dt = 0.01 y dt = 0.1
t2, x2, u2 = simular_sistema(dt=0.01, tipo_entrada="seno") 
t3, x3, u3 = simular_sistema(dt=0.5,  tipo_entrada="seno") 

plt.subplot(1, 2, 2)
plt.plot(t2, u2, 'k--', alpha=0.3, label='Entrada u(t)')
plt.plot(t2, x2, 'g-', label='x(t) Preciso (dt=0.01)')
plt.plot(t3, x3, 'r-o', markersize=4, label='x(t) Con Error (dt=0.01)')
plt.title("Respuesta al Seno: Comparación de dt")
plt.xlabel("Tiempo (s)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Interpretacion:
# En la grafica de escalon vemos con la linea punteada salta de 0 a 1 de inmediato, mientras que la curva azul representa la inercia, 
# se puede ver que la linea azul llega hasta el valor 1.2, con una ganancia en el sistema debido a que b tiene un valor mayor que a.
# En la grafica de seno se observa la linea verde que es una simulación con tiempo de cambio "dt" muy pequeño, por lo que la simulación tiene muchas mediciones para predecir el comportamiento del sistema.
#La linea roja por otra parte tiene un "dt" mayor de 0.5, en ese tiempo la velocidad del modelo cambia y al momento de realizar la medición no es tan preciso, por eso se ve quebrado en varios
#puntos y se realiza la correción de una manera mas brusca.
# La constante de tiempo tau sería de 0.4 segundos, lo que nos dice que el sistema responde rapidomente a los cambios en la entrada. 
# El sistema alcanza aproximadamente el 63.2% de su valor final en 0.4 segundos después de un cambio en la entrada.