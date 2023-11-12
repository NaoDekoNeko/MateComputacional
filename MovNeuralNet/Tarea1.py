import numpy as np
import matplotlib.pyplot as plt

# Datos de entrada
X = np.array([[1, 1, 1, 1],
              [1, -1, -1, 1],
              [-1, 1, 1, 1],
              [-1, -1, -1, 1]])

# Datos de salida esperados
T = np.array([[1, -1],
              [-1, 1],
              [1, -1],
              [-1, 1]])

# Pesos de la red neuronal
W = np.array([0.5, 0.3, 0.7, 0.1, 0.4,
              0.9, 0.8, 0.9, 0.5, 0.4,
              0.7, 0.9, 0.4, 0.7, 0.9,
              0.8, 0.7, 0.6])

# Umbrales
U = np.array([0.8, 0.2, 0.6, 0.9, 0.8])

# Tasa de aprendizaje
n = 0.65

# Número máximo de iteraciones y error mínimo
maxIter = 2500
minError = 0.1
Emc = 1000000

# Inicialización de variables para el ploteo
axisX = np.arange(0, maxIter + 1)
axis = np.zeros((len(W), maxIter + 1))
axisEmc = np.array([])

# Función de activación sigmoide
def f(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide
def df(x):
    return f(x) * (1 - f(x))

# Variables para almacenar valores intermedios
neta = np.zeros(5)
fneta = np.zeros(5)
delta3 = np.zeros(4)
delta4 = np.zeros(4)

# Contador de iteraciones
j = 0
errors = []

# Bucle principal de entrenamiento
while (j < maxIter) and (Emc > minError):
    j += 1
    Error = np.zeros((len(X), 2))

    # Bucle para cada muestra de entrada
    for i in range(len(X)):
        # Cálculos para la capa oculta
        neta[0] = X[i][0] * W[0] + X[i][1] * W[3] + X[i][2] * W[6] + X[i][3] * W[9] - U[0]
        neta[1] = X[i][0] * W[1] + X[i][1] * W[4] + X[i][2] * W[7] + X[i][3] * W[10] - U[1]
        neta[2] = X[i][0] * W[2] + X[i][1] * W[5] + X[i][2] * W[8] + X[i][3] * W[11] - U[2]
        fneta[0] = f(neta[0])
        fneta[1] = f(neta[1])
        fneta[2] = f(neta[2])

        # Cálculos para la capa de salida
        neta[3] = fneta[0] * W[12] + fneta[1] * W[14] + fneta[2] * W[16] - U[3]
        neta[4] = fneta[0] * W[13] + fneta[1] * W[15] + fneta[2] * W[17] - U[4]
        fneta[3] = f(neta[3])
        fneta[4] = f(neta[4])

        # Cálculos de los errores y deltas
        delta3[0] = (T[i][0] - fneta[3]) * df(neta[3])
        delta3[1] = df(neta[0]) * delta3[0] * W[12]
        delta3[2] = df(neta[1]) * delta3[0] * W[14]
        delta3[3] = df(neta[2]) * delta3[0] * W[16]

        delta4[0] = (T[i][1] - fneta[4]) * df(neta[4])
        delta4[1] = df(neta[0]) * delta4[0] * W[13]
        delta4[2] = df(neta[1]) * delta4[0] * W[15]
        delta4[3] = df(neta[2]) * delta4[0] * W[17]

        # Actualización de pesos y umbrales
        W[12] += n * delta3[0] * fneta[0]
        W[14] += n * delta3[0] * fneta[1]
        W[16] += n * delta3[0] * fneta[2]

        W[13] += n * delta4[0] * fneta[0]
        W[15] += n * delta4[0] * fneta[1]
        W[17] += n * delta4[0] * fneta[2]

        W[0] += n * delta3[1] * X[i][0] + n * delta4[1] * X[i][0]
        W[3] += n * delta3[1] * X[i][1] + n * delta4[1] * X[i][1]
        W[6] += n * delta3[1] * X[i][2] + n * delta4[1] * X[i][2]
        W[9] += n * delta3[1] * X[i][3] + n * delta4[1] * X[i][3]

        W[1] += n * delta3[2] * X[i][0] + n * delta4[2] * X[i][0]
        W[4] += n * delta3[2] * X[i][1] + n * delta4[2] * X[i][1]
        W[7] += n * delta3[2] * X[i][2] + n * delta4[2] * X[i][2]
        W[10] += n * delta3[2] * X[i][3] + n * delta4[2] * X[i][3]

        W[2] += n * delta3[3] * X[i][0] + n * delta4[3] * X[i][0]
        W[5] += n * delta3[3] * X[i][1] + n * delta4[3] * X[i][1]
        W[8] += n * delta3[3] * X[i][2] + n * delta4[3] * X[i][2]
        W[11] += n * delta3[3] * X[i][3] + n * delta4[3] * X[i][3]

        U[4] += n * delta4[0] * -1
        U[3] += n * delta3[0] * -1
        U[2] += n * delta3[3] * -1 + n * delta4[3] * -1
        U[1] += n * delta3[2] * -1 + n * delta4[2] * -1
        U[0] += n * delta3[1] * -1 + n * delta4[1] * -1

        # Cálculo del error para la muestra actual
        Error[i][0] = 0.5 * (T[i][0] - fneta[3]) ** 2
        Error[i][1] = 0.5 * (T[i][1] - fneta[4]) ** 2

    # Almacenamiento de los pesos en cada iteración para el ploteo
    for i in range(len(W)):
        axis[i, j] = W[i]

    # Cálculo del error medio cuadrático
    sum_squared_errors = np.sum(Error ** 2)
    num_samples = len(X)
    Emc = np.sqrt(sum_squared_errors / (2*num_samples))
    errors.append(Emc)

    # Impresión del progreso
    print("Iteracion: ", j, " Error: ", Emc)

# Impresión de los pesos finales
print("W: ", W)

# Función para asignar etiquetas de clase (1 o -1) basadas en el valor de la salida
def valor(x):
    if x > 0.5:
        return 1
    else:
        return -1

# Mapeo o comprobación del aprendizaje
print("Mapeo o comprobación del aprendizaje")

for i in range(len(X)):
    neta[0] = X[i][0] * W[0] + X[i][1] * W[3] + X[i][2] * W[6] + X[i][3] * W[9] - U[0]
    neta[1] = X[i][0] * W[1] + X[i][1] * W[4] + X[i][2] * W[7] + X[i][3] * W[10] - U[1]
    neta[2] = X[i][0] * W[2] + X[i][1] * W[5] + X[i][2] * W[8] + X[i][3] * W[11] - U[2]
    fneta[0] = f(neta[0])
    fneta[1] = f(neta[1])
    fneta[2] = f(neta[2])
    neta[3] = fneta[0] * W[12] + fneta[1] * W[14] + fneta[2] * W[16] - U[3]
    neta[4] = fneta[0] * W[13] + fneta[1] * W[15] + fneta[2] * W[17] - U[4]
    fneta[3] = f(neta[3])
    fneta[4] = f(neta[4])

    # Impresión de la entrada, salida esperada y salida obtenida
    print("Entrada: ", X[i], " Salida: ", T[i], " Salida Obtenida: ", valor(fneta[3]), valor(fneta[4]))

# Ploteo de la curva de aprendizaje
plt.figure()
plt.plot(np.arange(1, j+1), errors, label='Error de Entrenamiento')
plt.xlabel('Iteración')
plt.ylabel('Error')
plt.title('Curva de Aprendizaje')
plt.legend()
plt.show()

# Ploteo de los pesos a lo largo de las iteraciones
fig, (pesos, error) = plt.subplots(2)
pesos.set_title("Pesos")
for axi in axis:
    pesos.plot(axisX, axi)

plt.show()
