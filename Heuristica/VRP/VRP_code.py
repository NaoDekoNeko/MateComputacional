from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary, value, PULP_CBC_CMD
import numpy as np

def solve_vrp(N, V, Q, d, q):
    # Crear el problema de optimización
    prob = LpProblem("VRP", LpMinimize)

    # Crear variables de decisión
    x = {(i, j): LpVariable(name=f"x_{i}_{j}", cat=LpBinary) for i in range(N) for j in range(N)}
    u = {i: LpVariable(name=f"u_{i}", cat='Continuous') for i in range(N)}

    # Función objetivo
    prob += lpSum(d[i][j] * x[i, j] for i in range(N) for j in range(N))

    # Restricciones
    for j in range(1, N):
        prob += lpSum(x[i, j] for i in range(N) if i != j) == 1

    for i in range(1, N):
        prob += lpSum(x[i, j] for j in range(N) if i != j) == 1

    prob += lpSum(x[i, 0] for i in range(1, N)) == V
    prob += lpSum(x[0, j] for j in range(1, N)) == V

    for i in range(1, N):
        for j in range(1, N):
            if i != j:
                prob += u[i] - u[j] + Q * x[i, j] <= Q - q[j]

    for j in range(1, N):
        prob += lpSum(q[i] * x[i, j] for i in range(N) if i != j) <= Q

    # Resolver el problema
    prob.solve(PULP_CBC_CMD(msg=0))

    # Mostrar resultados
    print("Status:", prob.status)
    print("Costo total:", round(value(prob.objective), 2))

    for v in prob.variables():
        print(v.name, "=", value(v))

def generate_random_distance_matrix(num_cities):
    # Generar una matriz de distancias aleatorias (simétricas)
    distances = np.random.randint(10, 100, size=(num_cities, num_cities))
    distances = (distances + distances.T) // 2  # Hacer simétrica la matriz
    np.fill_diagonal(distances, 0)  # Asegurar que la distancia de una ciudad a sí misma sea 0

    # Generar un vector de capacidades aleatorias
    capacity = np.random.randint(1, 5, size=(num_cities,))
    capacity[0] = 0

    return distances, capacity

# Ejemplo de uso con 20 ciudades
num_cities = 5
distance_matrix, capacity_vector = generate_random_distance_matrix(num_cities)
V = 2  # Número de vehículos
Q = 10  # Capacidad de los vehículos

# Mostrar la matriz de distancias
print("Matriz de distancias:")
print(distance_matrix)

# Mostrar el vector de capacidades
print("\nVector de capacidades:")
print(capacity_vector)

solve_vrp(num_cities, V, Q, distance_matrix, capacity_vector)