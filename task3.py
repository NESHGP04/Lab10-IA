import numpy as np
import matplotlib.pyplot as plt

CARRILES = 20
K = 5

def transicion(h):
    delta = np.random.choice([-1, 0, 1])
    return int(np.clip(h + delta, 0, CARRILES - 1))

def emision(sensor, h):
    dist = abs(sensor - h)
    if dist == 0:
        return 0.6
    elif dist == 1:
        return 0.2
    else:
        return 0.2 / (CARRILES - 2)

def simular_vehiculo(pasos):
    estado_real = [np.random.randint(0, CARRILES)]
    observaciones = []

    for _ in range(pasos):
        nuevo_estado = transicion(estado_real[-1])
        estado_real.append(nuevo_estado)

        # generar sensor
        probs = []
        for h in range(CARRILES):
            probs.append(emision(nuevo_estado, h))

        probs = np.array(probs)
        probs /= probs.sum()

        sensor = np.random.choice(range(CARRILES), p=probs)
        observaciones.append(sensor)

    return estado_real[1:], observaciones

# Particle Filter
def filtrado_particulas(observaciones):

    particulas = np.random.randint(0, CARRILES, K)
    estimaciones = []

    for t, sensor in enumerate(observaciones):

        # PASO 1: Propagar
        propuestas = np.array([transicion(h) for h in particulas])

        # PASO 2: Ponderar
        pesos = np.array([emision(sensor, h) for h in propuestas])
        pesos = pesos / pesos.sum()

        # PASO 3: Remuestreo correcto
        idx = np.random.choice(len(propuestas), size=K, p=pesos)
        particulas = propuestas[idx]

        # Estimación (media)
        estimaciones.append(np.mean(particulas))

    return estimaciones, particulas

# Visualización
def visualizar(trayectoria_real, estimaciones, observaciones):

    pasos = len(trayectoria_real)

    plt.figure(figsize=(10,5))

    plt.plot(range(pasos), trayectoria_real, label="Real", linewidth=2)
    plt.plot(range(pasos), estimaciones, label="Estimación (media partículas)", linestyle="--")

    plt.scatter(range(pasos), trayectoria_real, color='blue')
    plt.scatter(range(pasos), estimaciones, color='orange')

    plt.title("Seguimiento con Particle Filter (K=5)")
    plt.xlabel("Tiempo")
    plt.ylabel("Carril")
    plt.legend()

    plt.show()

# MAIN
if __name__ == "__main__":

    np.random.seed(42)

    pasos = 30
    trayectoria_real, observaciones = simular_vehiculo(pasos)

    estimaciones, particulas_finales = filtrado_particulas(observaciones)

    visualizar(trayectoria_real, estimaciones, observaciones)