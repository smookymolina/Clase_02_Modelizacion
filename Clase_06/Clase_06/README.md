# Clase 6: Discretización de Modelos y Estabilidad Numérica

## 📚 Contenido del Material

Este directorio contiene el material práctico para la Clase 6 del curso de **Modelización en Ingeniería Aeroespacial**.

### Archivos Disponibles

1. **`ejemplo_discretizacion_basica.py`** - Ejemplos introductorios
2. **`practica_senal_amortiguada.py`** - Práctica guiada
3. **`actividad_chirp_amortiguado.py`** - Actividad para entregar

---

## 🎯 Objetivos de Aprendizaje

Al completar esta clase, serás capaz de:

- ✅ Entender la diferencia entre modelos continuos y discretos
- ✅ Aplicar el teorema de muestreo de Nyquist
- ✅ Identificar y evitar el fenómeno de aliasing
- ✅ Seleccionar adecuadamente el periodo de muestreo `Ts`
- ✅ Implementar discretización de señales en Python
- ✅ Analizar la estabilidad numérica de métodos de discretización

---

## 🚀 Instrucciones de Uso

### Requisitos Previos

Asegúrate de tener instaladas las siguientes librerías de Python:

```bash
pip install numpy matplotlib
```

### Orden de Ejecución

#### 1️⃣ **Paso 1: Ejemplos Básicos** (15-20 minutos)

Ejecuta primero el archivo de ejemplos para familiarizarte con los conceptos:

```bash
python ejemplo_discretizacion_basica.py
```

Este archivo contiene:
- **Ejemplo 1**: Señal senoidal continua vs muestreada
- **Ejemplo 2**: Señal exponencial (decaimiento)
- **Ejemplo 3**: Comparación de diferentes periodos de muestreo

**Objetivo**: Entender visualmente cómo el muestreo afecta la representación de señales.

---

#### 2️⃣ **Paso 2: Práctica Guiada** (20-30 minutos)

Ejecuta la práctica de señal amortiguada:

```bash
python practica_senal_amortiguada.py
```

Esta práctica implementa:
```
x(t) = A * exp(-α*t) * sin(2πft + φ)
```

**Tareas**:
- Ejecuta el código con los parámetros por defecto
- Observa la gráfica y el análisis de la consola
- Responde las preguntas de reflexión al final del script
- Experimenta modificando los parámetros sugeridos

**Parámetros para experimentar**:
- `A` (amplitud): 0.5, 1.0, 2.0
- `alpha` (amortiguamiento): 0.3, 0.7, 1.5
- `f` (frecuencia): 1.0, 2.0, 5.0
- `Ts` (periodo muestreo): 0.05, 0.12, 0.3

---

#### 3️⃣ **Paso 3: Actividad Evaluable** (40-60 minutos)

Ejecuta y completa la actividad del chirp lineal amortiguado:

```bash
python actividad_chirp_amortiguado.py
```

Esta actividad implementa un **chirp lineal amortiguado**:
```
x(t) = A * exp(-α*t) * sin(2π(f₀*t + ½β*t²) + φ)
```

donde la frecuencia instantánea es: `f(t) = f₀ + β*t`

---

## 📝 Entregable de la Actividad

### Qué Debes Entregar

Un **reporte breve** (2-3 páginas) que incluya:

#### 1. **Parámetros Seleccionados**
- Valores de A, α, f₀, β, φ, Ts elegidos
- **Justificación** de por qué elegiste ese `Ts` específico

#### 2. **Gráficas**
- Señal continua vs discretizada (subplot 1)
- Frecuencia instantánea vs frecuencia de Nyquist (subplot 2)
- Incluye títulos, etiquetas y leyendas claras

#### 3. **Respuestas a las Preguntas**

Responde todas las preguntas que aparecen al final del script `actividad_chirp_amortiguado.py`:

1. **Justificación del Ts elegido**
2. **Experimentación con parámetros**
3. **Análisis de aliasing**
4. **Efectos del amortiguamiento**
5. **Aplicaciones prácticas**

#### 4. **Análisis Crítico**
- ¿Qué compromiso existe entre resolución temporal y aliasing?
- ¿Qué sucede cuando aumentas β (tasa de barrido)?
- ¿Cómo afecta el amortiguamiento α a la duración efectiva de la señal?

#### 5. **Código Modificado**
- Incluye el script `.py` con tus parámetros elegidos
- Comenta las secciones donde experimentaste

---

## 📊 Conceptos Clave

### Teorema de Muestreo de Nyquist

Para evitar **aliasing**, la frecuencia de muestreo debe cumplir:

```
fs > 2 * f_max
```

donde:
- `fs = 1/Ts` es la frecuencia de muestreo
- `f_max` es la frecuencia máxima de la señal

**En la práctica**: Usa `fs ≥ 10 * f_max` para buena fidelidad visual.

### Aliasing

Fenómeno que ocurre cuando `Ts` es muy grande (o `fs` muy pequeño). La señal discreta no representa correctamente la señal continua y aparecen frecuencias falsas.

### Señal Chirp

Una señal cuya frecuencia **varía con el tiempo**. Útil en:
- **Radar**: Detección de objetos y medición de distancias
- **Sonar**: Exploración submarina
- **Telecomunicaciones**: Sincronización y ecualización
- **Análisis estructural**: Detección de resonancias

---

## 🔍 Preguntas Frecuentes

### ¿Qué `Ts` debo elegir?

Depende de la frecuencia máxima de tu señal:

```python
f_max = max(abs(f0), abs(f0 + beta * tf))
Ts < 1 / (2 * f_max)  # Mínimo teórico (Nyquist)
Ts < 1 / (10 * f_max) # Recomendado en práctica
```

### ¿Cómo sé si hay aliasing?

Observa estos síntomas:
- La señal discreta no sigue la forma de la continua
- Aparecen oscilaciones de frecuencia incorrecta
- La frecuencia instantánea supera la frecuencia de Nyquist

### ¿Puedo usar beta negativo?

¡Sí! Un `β < 0` hace que la frecuencia **disminuya** con el tiempo (chirp descendente). Es útil en aplicaciones como compresión de pulso en radar.

---

## 💡 Sugerencias para Experimentar

### Experimento 1: Violación de Nyquist
```python
f0 = 2.0
beta = 3.0
Ts = 0.2  # Muy grande -> aliasing
```

### Experimento 2: Chirp Rápido
```python
beta = 5.0   # Barrido rápido
Ts = 0.03    # Necesitas Ts pequeño
```

### Experimento 3: Chirp Descendente
```python
f0 = 5.0
beta = -2.0  # Frecuencia disminuye
```

### Experimento 4: Poco Amortiguamiento
```python
alpha = 0.1  # Señal dura más tiempo
tf = 10.0    # Aumenta tiempo para ver efecto
```

---

## 🎓 Criterios de Evaluación

Tu actividad será evaluada según:

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| **Justificación del Ts** | 25% | Explicación clara y técnica del Ts elegido |
| **Gráficas** | 20% | Calidad, claridad y presentación |
| **Respuestas a preguntas** | 30% | Profundidad y precisión técnica |
| **Experimentación** | 15% | Pruebas con diferentes parámetros |
| **Conclusiones** | 10% | Síntesis y aprendizajes clave |

---

## 📚 Recursos Adicionales

### Teoría
- Presentación de la Clase 6 (PDF)
- Apuntes sobre teorema de Nyquist
- Referencias sobre métodos de discretización

### Lecturas Recomendadas
- "Digital Signal Processing" - Oppenheim & Schafer
- "Signals and Systems" - Alan V. Oppenheim
- "Numerical Methods for Engineers" - Chapra & Canale

### Aplicaciones en Ingeniería Aeroespacial
- Sistemas de radar Doppler
- Telemetría de satélites
- Control digital de actitud
- Análisis de vibraciones estructurales

---

## 📧 Contacto

Si tienes dudas sobre las actividades, contacta al instructor:

- **Email**: jair.molina@unii.edu.mx
- **Horario de consulta**: (Consultar calendario del curso)

---

## 📅 Fecha de Entrega

Consulta la plataforma del curso para la fecha límite de entrega.

**Formato de entrega**:
- Reporte en PDF
- Código Python (.py)
- Nombre del archivo: `Clase06_ApellidoNombre.zip`

---

## 🌟 ¡Éxito en tu Actividad!

Recuerda: El objetivo no es solo completar la actividad, sino **entender** cómo la discretización afecta los modelos y sistemas reales. Este conocimiento es fundamental para control digital, procesamiento de señales y simulación de sistemas dinámicos.

**¡Experimenta, observa y aprende!** 🚀
