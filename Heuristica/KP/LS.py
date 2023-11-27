import time

def evaluate_solution(solution, weights, values):
    # Función auxiliar para evaluar el valor y el peso total de una solución
    total_value = sum(value for value, included in zip(values, solution) if included == 1)
    total_weight = sum(weight for weight, included in zip(weights, solution) if included == 1)
    return total_value, total_weight

def knapsack_local_search(initial_solution, weights, values, W, iterations):
    inicio = time.time()
    # Inicializa la solución actual y su valor y peso asociados
    current_solution = initial_solution.copy()
    current_value, current_weight = evaluate_solution(current_solution, weights, values)

    print("Tabla de Búsqueda Local:")
    
    # Imprime el estado inicial
    print(f"Iteración 0 - Valor: {current_value}, Peso: {current_weight}, Solución: {current_solution}")

    # Realiza la búsqueda local durante el número especificado de iteraciones
    for iteration in range(1, iterations + 1):
        improved = False  # Indica si se ha mejorado la solución en la iteración actual
        for i in range(len(initial_solution)):
            if current_solution[i] == 0 and weights[i] <= W:
                # Intenta agregar un elemento a la solución
                new_solution = current_solution.copy()
                new_solution[i] = 1
                new_value, new_weight = evaluate_solution(new_solution, weights, values)

                # Comprueba si la nueva solución mejora el valor y sigue siendo factible
                if new_value > current_value and new_weight <= W:
                    current_solution = new_solution
                    current_value = new_value
                    current_weight = new_weight
                    W -= weights[i]
                    improved = True

        # Imprime el estado en cada iteración
        print(f"Iteración {iteration} - Valor: {current_value}, Peso: {current_weight}, Solución: {current_solution}")

        if not improved:
            # Si no se ha mejorado en esta iteración, termina la búsqueda
            break
    
    # Imprime el tiempo de ejecución
    print(f"El algoritmo Local Search tardó {time.time() - inicio:.6f} segundos en ejecutarse")

    return current_value, current_solution

if __name__ == "__main__":
    # Datos proporcionados
    values = [10, 40, 30, 20]
    weights = [4, 3, 5, 2]
    W = 8
    initial_solution = [0, 0, 1, 0]  # Puedes proporcionar tu propia solución inicial
    iterations = 100

    # Realiza la búsqueda local y muestra los resultados
    result, selected_items = knapsack_local_search(initial_solution, weights, values, W, iterations)
    print(f"El valor máximo que se puede obtener es: {result}")
    print("Se seleccionaron los elementos con la búsqueda local:", selected_items)
