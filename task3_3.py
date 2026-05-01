import numpy as np
import matplotlib.pyplot as plt
from task3_1 import (
    simular_vehiculo,
    transicion,
    emision,
    CARRILES
)

def diversidad(particulas):
    return np.var(particulas) #usa varianza porque mide la dispersión directamente

def alerta_colapso(particulas, umbral=0.5):
    return diversidad(particulas) < umbral

def hay_colapso_real(alertas, min_consecutivos=5):
    contador = 0

    for a in alertas:
        if a:
            contador += 1
            if contador >= min_consecutivos:
                return True
        else:
            contador = 0

    return False

def filtrado_particulas_con_colapso(observaciones, K, umbral):

    particulas = np.random.randint(0, CARRILES, K)
    estimaciones = []
    alertas = []
    varianzas = []

    for sensor in observaciones:

        # Propagación
        propuestas = np.array([transicion(h) for h in particulas])

        # Ponderación
        pesos = np.array([emision(sensor, h) for h in propuestas])
        pesos = pesos / pesos.sum()

        # Remuestreo
        idx = np.random.choice(len(propuestas), size=K, p=pesos)
        particulas = propuestas[idx]

        # Estimación
        estimaciones.append(np.mean(particulas))

        # Métrica de diversidad
        v = diversidad(particulas)
        varianzas.append(v)

        # Alerta
        alertas.append(v < umbral)

    return estimaciones, particulas, alertas, varianzas

def evaluar_umbral(umbral, num_sim=30, pasos=30, K=5):
    colapsos_detectados = 0

    for _ in range(num_sim):
        _, obs = simular_vehiculo(pasos)
        _, _, alertas, _ = filtrado_particulas_con_colapso(obs, K, umbral)

        if hay_colapso_real(alertas):
            colapsos_detectados += 1

    return colapsos_detectados / num_sim

if __name__ == "__main__":

    np.random.seed(42)

    pasos = 30
    K = 5

    print("\n=== Evaluación de umbrales ===\n")

    umbrales = [0.1, 0.3, 0.5, 1.0, 2.0]

    for u in umbrales:
        tasa = evaluar_umbral(u, K=K)
        print(f"Umbral {u}: colapso detectado en {tasa*100:.1f}% de simulaciones")

    umbral = 0.5

    print("\n=== Escenario 1: Sin colapso ===\n")

    real, obs = simular_vehiculo(pasos)
    est, part, alertas, varianzas = filtrado_particulas_con_colapso(obs, K, umbral)

    print("¿Colapso real?:", hay_colapso_real(alertas))

    plt.figure()
    plt.plot(real, label="Real")
    plt.plot(est, label="Estimación")
    plt.title("Sin colapso")
    plt.legend()
    plt.show()

    print("\n=== Escenario 2: Colapso recuperable ===\n")

    real, obs = simular_vehiculo(pasos)

    # ruido temporal
    obs[5:10] = [np.random.randint(0, CARRILES) for _ in range(5)]

    est, part, alertas, varianzas = filtrado_particulas_con_colapso(obs, K, umbral)

    print("Pasos con alerta:", [i for i, a in enumerate(alertas) if a])
    print("¿Colapso real?:", hay_colapso_real(alertas))

    plt.figure()
    plt.plot(real, label="Real")
    plt.plot(est, label="Estimación")
    plt.title("Colapso recuperable")
    plt.legend()
    plt.show()

    print("\n=== Escenario 3: Colapso irrecuperable ===\n")

    real, obs = simular_vehiculo(pasos)

    # ruido prolongado fuerte
    obs[3:20] = [np.random.randint(0, CARRILES) for _ in range(17)]

    est, part, alertas, varianzas = filtrado_particulas_con_colapso(obs, K, umbral)

    print("Pasos con alerta:", [i for i, a in enumerate(alertas) if a])
    print("¿Colapso real?:", hay_colapso_real(alertas))

    plt.figure()
    plt.plot(real, label="Real")
    plt.plot(est, label="Estimación")
    plt.title("Colapso irrecuperable")
    plt.legend()
    plt.show()