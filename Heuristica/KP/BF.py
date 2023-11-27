from itertools import combinations
from prettytable import PrettyTable
import time

def knapsack_bruteforce(weights, values, W):
    n = len(values)
    best_value = 0
    best_combination = None

    print("Tabla de Fuerza Bruta:")
    # Genera todas las combinaciones posibles de 0 y 1 para indicar si el elemento está incluido o no
    table = PrettyTable()
    table.field_names = ["Combinación", "Peso", "Valor"]
    
    #print("Combinación, Peso, Valor")
    
    inicio = time.time()

    for r in range(n + 1):
        for combination in combinations(range(n), r):
            total_weight = sum(weights[i] for i in combination)
            total_value = sum(values[i] for i in combination)

            # Agrega la fila a la tabla
            table.add_row([combination, total_weight, total_value])

            # Imprime la combinación actual
            #print(combination, total_weight, total_value)

            if total_weight <= W and total_value > best_value:
                best_value = total_value
                best_combination = combination
    
    # Imprime el tiempo de ejecución
    print(f"El algoritmo de Fuerza Bruta tardó {time.time() - inicio:.6f} segundos en ejecutarse")

    # Imprime la tabla
    print(table)

    return best_value, best_combination

if __name__ == "__main__":

    # Datos proporcionados
    values = [10, 40, 30, 20]
    weights = [4, 3, 5, 2]
    W = 8

    # Realiza la búsqueda por fuerza bruta y muestra los resultados
    result, selected_items = knapsack_bruteforce(weights, values, W)
    print(f"\nEl valor máximo que se puede obtener por fuerza bruta es: {result}")
    print("Se seleccionaron los elementos con la siguiente combinación:", selected_items)