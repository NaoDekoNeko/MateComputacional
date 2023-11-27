import time

def knapsack_dynamic_programming(weights, values, W):
    n = len(values)
    M = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    print("Tabla de Programación Dinámica:")

    # Inicia el temporizador
    start_time = time.time()

    # Llena la tabla con los valores óptimos
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                M[i][w] = 0
            elif weights[i - 1] <= w:
                M[i][w] = max(values[i - 1] + M[i - 1][w - weights[i - 1]], M[i - 1][w])
            else:
                M[i][w] = M[i - 1][w]
        # Imprime la fila de la tabla actual
        print(M[i])

    # Imprime el tiempo de ejecución
    print(f"El algoritmo de Programación Dinámica tardó {time.time() - start_time:.6f} segundos en ejecutarse")

    # Reconstruir la solución
    selected_items = []
    i, w = n, W
    while i > 0 and w > 0:
        if M[i][w] != M[i - 1][w]:
            selected_items.append(i - 1)
            w -= weights[i - 1]
        i -= 1

    return M[n][W], selected_items[::-1]

if __name__ == "__main__":
    # Datos proporcionados
    values = [10, 40, 30, 20]
    weights = [4, 3, 5, 2]
    W = 8
    n = len(values)

    result, selected_items = knapsack_dynamic_programming(weights, values, W)
    print(f"El valor máximo que se puede obtener es: {result}")
    print("Se seleccionaron los elementos con índices:", selected_items)