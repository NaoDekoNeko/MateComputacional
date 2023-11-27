import time

def knapsack_greedy(weights, values, W):
    n = len(values)

    inicio = time.time()

    # Calcula la relación valor-peso para cada elemento
    value_per_weight = [(v / w, w, v) for w, v in zip(weights, values)]  # zip combina las tuplas de weights y values
    # Ordena los elementos en orden decreciente de la relación valor-peso
    value_per_weight.sort(reverse=True)

    total_value = 0
    selected_items = []

    print("Tabla de Greedy:")
    print("Iteración - Peso Restante - Valor Acumulado - Elementos Seleccionados")

    for i in range(n):
        if value_per_weight[i][1] <= W:
            W -= value_per_weight[i][1]
            total_value += value_per_weight[i][2]
            selected_items.append(value_per_weight[i])

        # Imprime el estado en cada iteración
        print(f"Iteración {i + 1} - Peso Restante: {W} - Valor Acumulado: {total_value} - Elementos Seleccionados: {selected_items}")

    # Imprime el tiempo de ejecución
    print(f"El algoritmo Greedy tardó {time.time() - inicio:.6f} segundos en ejecutarse")

    return total_value, selected_items

if __name__ == "__main__":

    # Datos proporcionados
    values = [10, 40, 30, 20]
    weights = [4, 3, 5, 2]
    W = 8

    result, selected_items = knapsack_greedy(weights, values, W)
    print(f"El valor máximo que se puede obtener es: {result}")
    print("Se seleccionaron los elementos con relación valor-peso:", selected_items)
