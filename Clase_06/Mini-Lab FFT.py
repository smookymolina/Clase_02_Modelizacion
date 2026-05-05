import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# MINI-LAB: Análisis Espectral de Vibración de Ala
# ==============================================================================
# INSTRUCCIONES:
# 1. Observa la ecuación de la señal de vibración de ala dada en la presentación:
#    x(t) = 1.2*sin(2*pi*8*t) + 0.5*sin(2*pi*22*t) + 0.3*cos(2*pi*35*t)
# 2. Reemplaza los valores "None" con los parámetros correspondientes
#    para reconstruir la señal y analizar su espectro.
# 3. Ejecuta el script.
# ==============================================================================

# ==========================================
# 1. PARÁMETROS DE ADQUISICIÓN
# ==========================================
# Define la frecuencia de muestreo (fs) y el número de muestras (N) según la clase
fs = 200  # TODO: Frecuencia de muestreo en Hz
N = 512  # TODO: Número total de muestras

if fs is None or N is None:
    raise ValueError("Debes definir 'fs' y 'N' antes de continuar.")

T = N / fs
t = np.linspace(0, T, N, endpoint=False)

# ==========================================
# 2. DEFINICIÓN DE LA SEÑAL
# ==========================================
# Construye la señal x(t) sustituyendo las amplitudes (A) y frecuencias (f)
# para las 3 componentes (flexión, torsión, vibración estructural).

# Componente 1: Modo de flexión del ala
A1 = 1.2  # Amplitud
f1 = 8  # Frecuencia [Hz]

# Componente 2: Modo de torsión
A2 = 0.5  # Amplitud
f2 = 22  # Frecuencia [Hz]

# Componente 3: Vibración estructural alta
A3 = 0.3  # Amplitud
f3 = 35  # Frecuencia [Hz]

if any(v is None for v in [A1, f1, A2, f2, A3, f3]):
    raise ValueError("Debes completar todas las amplitudes y frecuencias de la señal.")

# Ecuación de la señal compuesta:
# NOTA: Observa si las componentes usan seno o coseno
x = A1 * np.sin(2 * np.pi * f1 * t) \
  + A2 * np.sin(2 * np.pi * f2 * t) \
  + A3 * np.cos(2 * np.pi * f3 * t)

# ==========================================
# 3. FFT (Transformada Rápida de Fourier)
# ==========================================
X = np.fft.fft(x)
freq = np.fft.fftfreq(N, d=1/fs)

# Nos quedamos solo con la mitad positiva del espectro
freq_pos = freq[:N//2]
X_pos = X[:N//2]

# Calculamos la magnitud escalada
# Multiplicar por 2/N recupera las amplitudes reales de las sinusoides
mag = (2 / N) * np.abs(X_pos)

# ==========================================
# 4. GRÁFICAS
# ==========================================
plt.figure(figsize=(12, 5))

# Dominio del Tiempo
plt.subplot(1, 2, 1)
# Graficamos solo una porción del tiempo para ver bien la forma de onda
# (Por ejemplo, los primeros 0.4 segundos como en la presentación)
num_muestras_plot = int(0.4 * fs) 
plt.plot(t[:num_muestras_plot], x[:num_muestras_plot], color='#1f77b4', linewidth=1.5)
plt.title("Señal en el tiempo (primeros 0.4 s)")
plt.xlabel("Tiempo t [s]")
plt.ylabel("x(t)")
plt.grid(True, linestyle='--', alpha=0.7)

# Dominio de la Frecuencia (Espectro)
plt.subplot(1, 2, 2)
# Usamos stem para resaltar los picos discretos de frecuencia
markerline, stemlines, baseline = plt.stem(freq_pos, mag)
plt.setp(markerline, color='#004c99', markersize=8)
plt.setp(stemlines, color='#004c99', linewidth=2.5)
plt.title("Espectro de magnitud")
plt.xlabel("Frecuencia f [Hz]")
plt.ylabel("|X(f)| [Amplitud]")
plt.xlim(0, 50)  # Limitamos el eje X a 50 Hz para ver mejor los modos
plt.ylim(0, 1.5) # Escala Y ajustada a las amplitudes del problema
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()