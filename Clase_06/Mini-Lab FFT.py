import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARÁMETROS DE ADQUISICIÓN (INPUT USUARIO)
# ==========================================
fs = float(input("Frecuencia de muestreo fs [Hz]: "))
N = int(input("Número de muestras N: "))

T = N / fs
t = np.linspace(0, T, N, endpoint=False)

# ==========================================
# 2. DEFINICIÓN DE LA SEÑAL
# ==========================================
num_componentes = int(input("Número de componentes en la señal: "))

x = np.zeros_like(t)

print("\n--- Definir cada componente ---")
for i in range(num_componentes):
    print(f"\nComponente {i+1}:")
    
    A = float(input("Amplitud: "))
    f = float(input("Frecuencia [Hz]: "))
    tipo = input("Tipo (sin/cos): ").lower()
    
    if tipo == "sin":
        x += A * np.sin(2*np.pi*f*t)
    elif tipo == "cos":
        x += A * np.cos(2*np.pi*f*t)
    else:
        print("Tipo no válido, usando seno por defecto")
        x += A * np.sin(2*np.pi*f*t)

# ==========================================
# 3. FFT
# ==========================================
X = np.fft.fft(x)
freq = np.fft.fftfreq(N, d=1/fs)

# Solo parte positiva
freq_pos = freq[:N//2]
X_pos = X[:N//2]

# Magnitud
mag = (2 / N) * np.abs(X_pos)

# ==========================================
# 4. GRÁFICAS
# ==========================================
plt.figure(figsize=(12, 5))

# Tiempo
plt.subplot(1, 2, 1)
plt.plot(t, x)
plt.title("Señal en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("x(t)")
plt.grid()

# Frecuencia
plt.subplot(1, 2, 2)
plt.stem(freq_pos, mag)
plt.title("Espectro de magnitud")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X(f)|")
plt.grid()

plt.tight_layout()
plt.show()