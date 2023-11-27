from DP import knapsack_dynamic_programming
from Greedy import knapsack_greedy
from LS import knapsack_local_search
from BF import knapsack_bruteforce

# Datos para el caso con más elementos
values_large = [10, 40, 30, 20, 15, 25, 35, 45, 50, 60]
weights_large = [4, 3, 5, 2, 7, 6, 8, 9, 10, 1]
W_large = 20

# Ejecutar los algoritmos con el caso con más elementos
result_dp, selected_items_dp = knapsack_dynamic_programming(weights_large, values_large, W_large)
result_greedy, selected_items_greedy = knapsack_greedy(weights_large, values_large, W_large)
result_local_search, selected_items_local_search = knapsack_local_search([0] * len(values_large), weights_large, values_large, W_large, 100)
result_bruteforce, selected_items_bruteforce = knapsack_bruteforce(weights_large, values_large, W_large)

# Imprimir los resultados
print("Resultados para el caso con más elementos:")
print(f"Programación Dinámica: Valor máximo = {result_dp}, Elementos seleccionados = {selected_items_dp}")
print(f"Greedy: Valor máximo = {result_greedy}, Elementos seleccionados = {selected_items_greedy}")
print(f"Búsqueda Local: Valor máximo = {result_local_search}, Elementos seleccionados = {selected_items_local_search}")
print(f"Fuerza Bruta: Valor máximo = {result_bruteforce}, Elementos seleccionados = {selected_items_bruteforce}")
