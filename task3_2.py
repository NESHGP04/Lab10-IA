import numpy as np
import matplotlib.pyplot as plt
from task3_1 import (
    simular_vehiculo,
    transicion,
    emision,
    CARRILES
)

def calcular_error(trayectoria_real, estimaciones):
    return np.mean(np.abs(np.array(trayectoria_real) - np.array(estimaciones)))

#simulación
def experimento(K, num_sim=50, pasos=30):
    errores_por_tiempo = []
    errores_totales = []

    for _ in range(num_sim):
        trayectoria_real, observaciones = simular_vehiculo(pasos)

        estimaciones, _ = filtrado_particulas_param(observaciones, K)

        errores = np.abs(np.array(trayectoria_real) - np.array(estimaciones))
        errores_por_tiempo.append(errores)

        errores_totales.append(np.mean(errores))

    errores_por_tiempo = np.array(errores_por_tiempo)
    error_promedio_t = errores_por_tiempo.mean(axis=0)

    return error_promedio_t, errores_totales

#función para estimar la posición del vehículo en cada instante
def filtrado_particulas_param(observaciones, K):

    particulas = np.random.randint(0, CARRILES, K)
    estimaciones = []

    for sensor in observaciones:
        propuestas = np.array([transicion(h) for h in particulas])

        pesos = np.array([emision(sensor, h) for h in propuestas])
        pesos = pesos / pesos.sum()

        idx = np.random.choice(len(propuestas), size=K, p=pesos)
        particulas = propuestas[idx]

        estimaciones.append(np.mean(particulas))

    return estimaciones, particulas

#métricas
def metricas(errores_totales):
    return {
        "error_promedio": np.mean(errores_totales),
        "error_max": np.max(errores_totales),
        "porcentaje_error_alto": np.mean(np.array(errores_totales) > 5) * 100
    }

def peores_escenarios(K, num_sim=50, pasos=30):
    resultados = []

    for i in range(num_sim):
        real, obs = simular_vehiculo(pasos)
        est, _ = filtrado_particulas_param(obs, K)

        error = np.mean(np.abs(np.array(real) - np.array(est)))

        resultados.append((error, real, obs, est))

    resultados.sort(key=lambda x: x[0], reverse=True)

    return resultados[:5]

def tabla_resultados(err_K5, err_K20):

    m5 = metricas(err_K5)
    m20 = metricas(err_K20)

    print("\nComparación de desempeño:\n")
    print(f"{'Métrica':<30}{'K=5':<15}{'K=20':<15}")
    print("-"*60)
    print(f"{'Error promedio':<30}{m5['error_promedio']:<15.3f}{m20['error_promedio']:<15.3f}")
    print(f"{'Error máximo':<30}{m5['error_max']:<15.3f}{m20['error_max']:<15.3f}")
    print(f"{'% error > 5 carriles':<30}{m5['porcentaje_error_alto']:<15.2f}{m20['porcentaje_error_alto']:<15.2f}")

if __name__ == "__main__":

    np.random.seed(42)

    # Ejecutar experimentos
    error_K5, errores_totales_K5 = experimento(K=5)
    error_K20, errores_totales_K20 = experimento(K=20)

    # Gráfica
    plt.plot(error_K5, label="K=5")
    plt.plot(error_K20, label="K=20")
    plt.xlabel("Tiempo")
    plt.ylabel("Error promedio")
    plt.legend()
    plt.title("Error de seguimiento")
    plt.show()

    # Métricas
    print("K=5:", metricas(errores_totales_K5))
    print("K=20:", metricas(errores_totales_K20))

    #peores escenarios, donde k=5 falla más
    peores = peores_escenarios(5)

    for i, (error, _, _, _) in enumerate(peores):
        print(f"Escenario {i+1}: error = {error}")

    tabla_resultados(errores_totales_K5, errores_totales_K20)