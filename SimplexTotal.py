import numpy as np

class SimplexBigM:
    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b
        self.num_vars = len(c)
        self.num_constraints = len(A)
        self.BIG_M = 999999999999.000000
        self.pivots = [self.BIG_M for i in range(self.num_constraints)]
        self.basic_vars = [self.num_vars + i for i in range(self.num_constraints)]
        self.non_basic_vars = [i for i in range(self.num_vars)]
        self.table = []
        
    def solve(self):
        # Crear la tabla inicial
        for i in range(self.num_constraints):
            row = []
            for j in range(self.num_vars):
                row.append(self.A[i][j])
            for j in range(self.num_constraints):
                if i == j:
                    row.append(-1) # Coeficiente para las variables de holgura
                    row.append(1)  # Coeficiente para las variables artificiales
                else:
                    row.append(0)
                    row.append(0)
            row.append(self.b[i])  # Término independiente
            self.table.append(row)

        # Añadir la función objetivo a la tabla
        z_row = []
        for j in range(self.num_vars):
            z_row.append(-self.c[j])
        for j in range(self.num_constraints):
            z_row.append(0)  # Coeficiente para las variables de holgura en la función objetivo
            z_row.append(-self.BIG_M)  # Coeficiente para las variables artificiales en la función objetivo
        z_row.append(0)  # Término independiente de la función objetivo
        self.table.append(z_row)

        print("Paso 1\n",np.array(self.table))
        print("Pivotes: ", np.array(self.pivots))

        # Implementar el método simplex...
        while True:
            pivot_column_index = self.find_pivot_column()
            if pivot_column_index < 0:
                break
            pivot_row_index = self.find_pivot_row(pivot_column_index)
            if pivot_row_index < 0:
                raise Exception('El problema no tiene solución óptima.')
        
            # Actualizar las listas de variables básicas y no básicas
            self.basic_vars[pivot_row_index], self.non_basic_vars[pivot_column_index] = self.non_basic_vars[pivot_column_index], self.basic_vars[pivot_row_index]
        
            self.pivot(pivot_row_index, pivot_column_index)

    def printResult(self):
        # Obtener los valores de las variables básicas
        x_values = [0] * self.num_vars
        for i in range(self.num_constraints):
            if self.basic_vars[i] < self.num_vars:
                x_values[self.basic_vars[i]] = self.table[i][-1]

        # Imprimir los valores de las x
        for i, x_value in enumerate(x_values):
            print(f"x{i+1} = {x_value}")
    
        z = 0
        # Imprimir el valor de z
        for i in range (self.num_vars):
            z += self.c[i] * x_values[i]
    
        print(f"z = {z}")

    def find_pivot_column(self):
        aux = self.table[-1][:self.num_vars]  # Ignorar los elementos correspondientes a las variables de holgura y artificiales
        last_row = np.array(aux)
        
        for i in range(len(self.pivots)):
            last_row -= self.pivots[i] * np.array(self.table[i][:self.num_vars])
        
        print("Restar la fila objetivo\n",np.array(last_row))
        
        min_value = min(last_row)
        
        if min_value >= 0:  # Si todos los valores son mayores o iguales a cero, no hay una columna pivote válida
            return -1
        
        return list(last_row).index(min_value)

    def find_pivot_row(self, pivot_column_index):
        min_ratio = float('inf')
        
        pivot_row_index = -1
        
        for i, row in enumerate(self.table[:-1]):  # Ignorar la última fila (función objetivo)
            if row[pivot_column_index] > 0:  # Solo considerar filas donde el valor en la columna del pivote es positivo.
                ratio = row[-1] / row[pivot_column_index]
                
                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_row_index = i
        
        return pivot_row_index    

    def pivot(self, pivot_row_index, pivot_column_index):
        pivot_element = self.table[pivot_row_index][pivot_column_index]

        self.pivots[pivot_row_index] = self.table[-1][pivot_column_index]

        print("Pivote: ", pivot_element, "\n Posicion: ", pivot_row_index + 1 , pivot_column_index + 1)
        print("Pivotes: ", np.array(self.pivots))

        # Dividir toda la fila del pivote por el elemento del pivote.
        self.table[pivot_row_index] = [element / pivot_element for element in self.table[pivot_row_index]]

        print("Dividir la fila del pivote por el elemento pivote\n",np.array(self.table))

        # Para cada fila, menos la ultima, restarle un múltiplo de la fila del pivote para que los otros elementos en la columna del pivote se conviertan en cero.
        for i in range(len(self.table[:-1])):  # Ignorar la última fila
            if i != pivot_row_index:
                multiplier = self.table[i][pivot_column_index]
                self.table[i] -= multiplier * np.array(self.table[pivot_row_index])
        print("Hacer ceros\n",np.array(self.table))

class SimplexSolverMax:
    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b
        self.num_vars = len(c)
        self.num_constraints = len(A)
        self.basic_vars = [self.num_vars + i for i in range(self.num_constraints)]
        self.non_basic_vars = [i for i in range(self.num_vars)]

    def solve(self):
        # Crear la tabla inicial
        table = []
        for i in range(self.num_constraints):
            row = []
            for j in range(self.num_vars):
                row.append(self.A[i][j])
            for j in range(self.num_constraints):
                if i == j:
                    row.append(1)  # Coeficiente para las variables de holgura
                else:
                    row.append(0)
            row.append(self.b[i])  # Término independiente
            table.append(row)

        # Añadir la función objetivo a la tabla
        z_row = []
        for j in range(self.num_vars):
            z_row.append(-self.c[j])
        for j in range(self.num_constraints):
            z_row.append(0)  # Coeficiente para las variables artificiales en la función objetivo
        z_row.append(0)  # Término independiente de la función objetivo
        table.append(z_row)

        print("Paso 1\n",np.array(table))

        # Implementar el método simplex...
        while True:
            pivot_column_index = self.find_pivot_column(table)
            if pivot_column_index < 0:
                break
            pivot_row_index = self.find_pivot_row(table, pivot_column_index)
            if pivot_row_index < 0:
                raise Exception('El problema no tiene solución óptima.')
            self.pivot(table, pivot_row_index, pivot_column_index)
        return table

    def printResult(self, table):
        # Obtener los valores de las variables básicas
        x_values = [0] * self.num_vars
        for i in range(self.num_constraints):
            if table[i][-1] != 0:
                x_values[i] = table[i][-1]

        # Imprimir los valores de las x
        for i, x_value in enumerate(x_values):
            print(f"x{i+1} = {x_value}")
        
        z = 0
        # Imprimir el valor de z
        for i in range (self.num_vars):
            z += self.c[i] * x_values[i]
        
        print(f"z = {z}")

    def find_pivot_column(self, table):
        last_row = table[-1]
        min_value = min(last_row[0:self.num_vars])
        if min_value >= 0:
            return -1  # No hay coeficientes negativos en la última fila, por lo que hemos alcanzado la solución óptima.
        else:
            return list(last_row).index(min_value)

    def find_pivot_row(self, table, pivot_column_index):
        min_ratio = float('inf')
        pivot_row_index = -1
        for i, row in enumerate(table[:-1]):  # Ignorar la última fila (función objetivo)
            if row[pivot_column_index] > 0:  # Solo considerar filas donde el valor en la columna del pivote es positivo.
                ratio = row[-1] / row[pivot_column_index]
                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_row_index = i
        return pivot_row_index

    def pivot(self, table, pivot_row_index, pivot_column_index):
        pivot_element = table[pivot_row_index][pivot_column_index]

        print("Pivote: ", pivot_element, "\n Posicion: ", pivot_row_index + 1 , pivot_column_index + 1)


        # Dividir toda la fila del pivote por el elemento del pivote.
        table[pivot_row_index] = [element / pivot_element for element in table[pivot_row_index]]

        print("Dividir la fila del pivote por el elemento pivote\n",np.array(table))
        # Para cada fila, restarle un múltiplo de la fila del pivote para que los otros elementos en la columna del pivote se conviertan en cero.
        for i, row in enumerate(table):
            if i != pivot_row_index:
                multiplier = np.array(table[pivot_row_index]) * table[i][pivot_column_index]
                table[i] = np.array(row) - multiplier

        print("Hacer ceros\n",np.array(table))

class DualSimplexSolver:
    def __init__(self, c, A, b):
        self.c, self.A, self.b = self.transformar_matrices(c, A, b)

    def transformar_matrices(self, c, A, b):
        # Intercambiar los valores de c y b
        c, b = b, c
        # Intercambiar los elementos de A en el orden indicado
        A = np.transpose(A)
        return c, A, b

    def solveMax(self):
        simplex = SimplexSolverMax(self.c, self.A, self.b)
        table = simplex.solve()
        x_dual = [0] * len(self.b)
        for i in range(len(self.b)):
            x_dual[i] = table[-1][i-len(self.b)-1]
        print("x: ", x_dual)
        print("z: ", table[-1][-1])
        #print("z: ", np.dot(self.b, x_dual))
    
if __name__ == "__main__":
    """
    A = [[0.0048, 0.0484, 0.0051],
         [0.1364, 1.3636, 0.1364],
         [0.1364, 1.3636, 0.1364],
         [0.2045, 2.0455, 0.1705],
         [0.1461, 1.4610, 0.1461],
         [0.1000, 0.6666, 0.1000],
         [0.0667, 0.2000, 0.0500],
         [0.1500, 0.9999, 0.1500],
         [0.0500, 0.1000, 0.0500],
         [0.0528, 0.3333, 0.0333],
         [0.1000, 0.1666, 0.1000],
         [0.1000, 0.1000, 0.1000],
         [0.0667, 0.1333, 0.0667],
         [1.0000, 0.0000, 0.0000]]
    b = [300., 2400, 1200, 4800, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 21714]
    c = [15.59, 143.23, 17.82]
    """
    A = [[1,-4],
         [-2,1],
         [-3,4],
         [2,1]]
    b = [4,2,12,8]
    c = [1,2]

    simplex_dual = SimplexSolverMax(c, A, b)
    table = simplex_dual.solve()
    matriz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    for fila in table:
        for elemento in fila:
            print(elemento, end=' ')
        print()
