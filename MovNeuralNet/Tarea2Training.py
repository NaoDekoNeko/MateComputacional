import numpy as np
import matplotlib.pyplot as plt
import pickle

# Tasa de aprendizaje
n = 0.65

# Número máximo de iteraciones y error mínimo
maxIter = 3000
minError = 0.1

# Función de activación sigmoide
def f(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide
def df(x):
    return f(x) * (1 - f(x))

# Función para asignar etiquetas de clase (1 o -1) basadas en el valor de la salida
def valor(x):
    if x > 0.5:
        return 1
    else:
        return 0
    
# Función para asignar etiqueta de enfermedad (0-3) basada en la salida
def enfermedad(x,y):
    enfermedad = x*2 + y
    return f"enfermedad {enfermedad}"

def loadData():
    # Establecer una semilla para reproducibilidad
    np.random.seed(42)

    # Número de casos de entrenamiento
    num_cases = 20

    # Número de síntomas
    num_symptoms = 8

    # Generar valores iniciales para W y U dentro del rango (0.4, 0.9)
    initial_values_W = np.random.uniform(0.4, 0.9, size=(80,))
    initial_values_U = np.random.uniform(0.4, 0.9, size=(10,))

    # Definir el cambio máximo permitido (0.1)
    max_change = 0.1

    # Generar datos de entrada aleatorios (0 o 1) para cada síntoma
    X = np.random.randint(2, size=(num_cases, num_symptoms))

    # Definir clases de enfermedades (0, 1, 2, 3)
    # Se asigna una clase para cada combinación única de síntomas
    # Esto es solo un ejemplo, puedes ajustar las reglas según sea necesario
    T = np.zeros((num_cases,2), dtype=int)

    # Asignar clases basadas en ciertos síntomas
    for i in range(num_cases):
        if X[i, 0] == 1 and X[i, 2] == 1:
            T[i][0] = 0
            T[i][1] = 1
        elif X[i, 4] == 1 and X[i, 6] == 1:
            T[i][0] = 1
            T[i][1] = 0
        elif X[i, 1] == 1 and X[i, 3] == 1 and X[i, 5] == 1:
            T[i][0] = 1
            T[i][1] = 1

    # Pesos de la red neuronal
    W = np.clip(initial_values_W + np.random.uniform(-max_change, max_change, size=(80,)), 0.4, 0.9)

    # Umbrales
    U = np.clip(initial_values_U + np.random.uniform(-max_change, max_change, size=(10,)), 0.4, 0.9)

    return X,T,W,U

def train():
    X,T,W,U = loadData()
    # Inicialización de variables para el ploteo
    axisX = np.arange(0, maxIter + 1)
    axis = np.zeros((len(W), maxIter + 1))
    axisEmc = np.array([])

    # Variables para almacenar valores intermedios
    neta = np.zeros(10)
    fneta = np.zeros(10)
    delta8 = np.zeros(9)
    delta9 = np.zeros(9)

    # Contador de iteraciones
    j = 0
    errors = []

    Emc = 1000000

    # Bucle principal de entrenamiento
    while (j < maxIter) and (Emc > minError):
        j += 1
        Error = np.zeros((len(X), 2))

        # Bucle para cada muestra de entrada
        for i in range(len(X)):
            # Cálculos para la capa oculta
            neta[0] = X[i][0] * W[0] + X[i][1] * W[8] + X[i][2] * W[16] + X[i][3] * W[24] + X[i][4] * W[32] + X[i][5] * W[40] + X[i][6] * W[48] + X[i][7] * W[56] - U[0]
            neta[1] = X[i][0] * W[1] + X[i][1] * W[9] + X[i][2] * W[17] + X[i][3] * W[25] + X[i][4] * W[33] + X[i][5] * W[41] + X[i][6] * W[49] + X[i][7] * W[57] - U[1]
            neta[2] = X[i][0] * W[2] + X[i][1] * W[10] + X[i][2] * W[18] + X[i][3] * W[26] + X[i][4] * W[34] + X[i][5] * W[42] + X[i][6] * W[50] + X[i][7] * W[58] - U[2]
            neta[3] = X[i][0] * W[3] + X[i][1] * W[11] + X[i][2] * W[19] + X[i][3] * W[27] + X[i][4] * W[35] + X[i][5] * W[43] + X[i][6] * W[51] + X[i][7] * W[59] - U[3]
            neta[4] = X[i][0] * W[4] + X[i][1] * W[12] + X[i][2] * W[20] + X[i][3] * W[28] + X[i][4] * W[36] + X[i][5] * W[44] + X[i][6] * W[52] + X[i][7] * W[60] - U[4]
            neta[5] = X[i][0] * W[5] + X[i][1] * W[13] + X[i][2] * W[21] + X[i][3] * W[29] + X[i][4] * W[37] + X[i][5] * W[45] + X[i][6] * W[53] + X[i][7] * W[61] - U[5]
            neta[6] = X[i][0] * W[6] + X[i][1] * W[14] + X[i][2] * W[22] + X[i][3] * W[30] + X[i][4] * W[38] + X[i][5] * W[46] + X[i][6] * W[54] + X[i][7] * W[62] - U[6]
            neta[7] = X[i][0] * W[7] + X[i][1] * W[15] + X[i][2] * W[23] + X[i][3] * W[31] + X[i][4] * W[39] + X[i][5] * W[47] + X[i][6] * W[55] + X[i][7] * W[63] - U[7]

            fneta[0] = f(neta[0])
            fneta[1] = f(neta[1])
            fneta[2] = f(neta[2])
            fneta[3] = f(neta[3])
            fneta[4] = f(neta[4])
            fneta[5] = f(neta[5])
            fneta[6] = f(neta[6])
            fneta[7] = f(neta[7])

            # Cálculos para la capa de salida
            neta[8] = fneta[0] * W[64] + fneta[1] * W[66] + fneta[2] * W[68] + fneta[3] * W[70] + fneta[4] * W[72] + fneta[5] * W[74] + fneta[6] * W[76] + fneta[7] * W[78] - U[8]
            neta[9] = fneta[0] * W[65] + fneta[1] * W[67] + fneta[2] * W[69] + fneta[3] * W[71] + fneta[4] * W[73] + fneta[5] * W[75] + fneta[6] * W[77] + fneta[7] * W[79] - U[9]
            fneta[8] = f(neta[8])
            fneta[9] = f(neta[9])

            # Cálculos de los errores y deltas
            delta8[0] = (T[i][0] - fneta[8]) * df(neta[8])
            delta8[1] = df(neta[0]) * delta8[0] * W[64]
            delta8[2] = df(neta[1]) * delta8[0] * W[66]
            delta8[3] = df(neta[2]) * delta8[0] * W[68]
            delta8[4] = df(neta[3]) * delta8[0] * W[70]
            delta8[5] = df(neta[4]) * delta8[0] * W[72]
            delta8[6] = df(neta[5]) * delta8[0] * W[74]
            delta8[7] = df(neta[6]) * delta8[0] * W[76]
            delta8[8] = df(neta[7]) * delta8[0] * W[78]

            delta9[0] = (T[i][1] - fneta[9]) * df(neta[9])
            delta9[1] = df(neta[0]) * delta9[0] * W[65]
            delta9[2] = df(neta[1]) * delta9[0] * W[67]
            delta9[3] = df(neta[2]) * delta9[0] * W[69]
            delta9[4] = df(neta[3]) * delta9[0] * W[71]
            delta9[5] = df(neta[4]) * delta9[0] * W[73]
            delta9[6] = df(neta[5]) * delta9[0] * W[75]
            delta9[7] = df(neta[6]) * delta9[0] * W[77]
            delta9[8] = df(neta[7]) * delta9[0] * W[79]

            # Actualización de pesos y umbrales
            W[64] += n * delta8[0] * fneta[0]
            W[66] += n * delta8[0] * fneta[1]
            W[68] += n * delta8[0] * fneta[2]
            W[70] += n * delta8[0] * fneta[3]
            W[72] += n * delta8[0] * fneta[4]
            W[74] += n * delta8[0] * fneta[5]
            W[76] += n * delta8[0] * fneta[6]
            W[78] += n * delta8[0] * fneta[7]

            W[65] += n * delta9[0] * fneta[0]
            W[67] += n * delta9[0] * fneta[1]
            W[69] += n * delta9[0] * fneta[2]
            W[71] += n * delta9[0] * fneta[3]
            W[73] += n * delta9[0] * fneta[4]
            W[75] += n * delta9[0] * fneta[5]
            W[77] += n * delta9[0] * fneta[6]
            W[79] += n * delta9[0] * fneta[7]

            W[0] += n * delta8[1] * X[i][0] + n * delta9[1] * X[i][0]
            W[8] += n * delta8[1] * X[i][1] + n * delta9[1] * X[i][1]
            W[16] += n * delta8[1] * X[i][2] + n * delta9[1] * X[i][2]
            W[24] += n * delta8[1] * X[i][3] + n * delta9[1] * X[i][3]
            W[32] += n * delta8[1] * X[i][4] + n * delta9[1] * X[i][4]
            W[40] += n * delta8[1] * X[i][5] + n * delta9[1] * X[i][5]
            W[48] += n * delta8[1] * X[i][6] + n * delta9[1] * X[i][6]
            W[56] += n * delta8[1] * X[i][7] + n * delta9[1] * X[i][7]

            W[1] += n * delta8[2] * X[i][0] + n * delta9[2] * X[i][0]
            W[9] += n * delta8[2] * X[i][1] + n * delta9[2] * X[i][1]
            W[17] += n * delta8[2] * X[i][2] + n * delta9[2] * X[i][2]
            W[25] += n * delta8[2] * X[i][3] + n * delta9[2] * X[i][3]
            W[33] += n * delta8[2] * X[i][4] + n * delta9[2] * X[i][4]
            W[41] += n * delta8[2] * X[i][5] + n * delta9[2] * X[i][5]
            W[49] += n * delta8[2] * X[i][6] + n * delta9[2] * X[i][6]
            W[57] += n * delta8[2] * X[i][7] + n * delta9[2] * X[i][7]

            W[2] += n * delta8[3] * X[i][0] + n * delta9[3] * X[i][0]
            W[10] += n * delta8[3] * X[i][1] + n * delta9[3] * X[i][1]
            W[18] += n * delta8[3] * X[i][2] + n * delta9[3] * X[i][2]
            W[26] += n * delta8[3] * X[i][3] + n * delta9[3] * X[i][3]
            W[34] += n * delta8[3] * X[i][4] + n * delta9[3] * X[i][4]
            W[42] += n * delta8[3] * X[i][5] + n * delta9[3] * X[i][5]
            W[50] += n * delta8[3] * X[i][6] + n * delta9[3] * X[i][6]
            W[58] += n * delta8[3] * X[i][7] + n * delta9[3] * X[i][7]

            W[3] += n * delta8[4] * X[i][0] + n * delta9[4] * X[i][0]
            W[11] += n * delta8[4] * X[i][1] + n * delta9[4] * X[i][1]
            W[19] += n * delta8[4] * X[i][2] + n * delta9[4] * X[i][2]
            W[27] += n * delta8[4] * X[i][3] + n * delta9[4] * X[i][3]
            W[35] += n * delta8[4] * X[i][4] + n * delta9[4] * X[i][4]
            W[43] += n * delta8[4] * X[i][5] + n * delta9[4] * X[i][5]
            W[51] += n * delta8[4] * X[i][6] + n * delta9[4] * X[i][6]
            W[59] += n * delta8[4] * X[i][7] + n * delta9[4] * X[i][7]

            W[4] += n * delta8[5] * X[i][0] + n * delta9[5] * X[i][0]
            W[12] += n * delta8[5] * X[i][1] + n * delta9[5] * X[i][1]
            W[20] += n * delta8[5] * X[i][2] + n * delta9[5] * X[i][2]
            W[28] += n * delta8[5] * X[i][3] + n * delta9[5] * X[i][3]
            W[36] += n * delta8[5] * X[i][4] + n * delta9[5] * X[i][4]
            W[44] += n * delta8[5] * X[i][5] + n * delta9[5] * X[i][5]
            W[52] += n * delta8[5] * X[i][6] + n * delta9[5] * X[i][6]
            W[60] += n * delta8[5] * X[i][7] + n * delta9[5] * X[i][7]

            W[5] += n * delta8[6] * X[i][0] + n * delta9[6] * X[i][0]
            W[13] += n * delta8[6] * X[i][1] + n * delta9[6] * X[i][1]
            W[21] += n * delta8[6] * X[i][2] + n * delta9[6] * X[i][2]
            W[29] += n * delta8[6] * X[i][3] + n * delta9[6] * X[i][3]
            W[37] += n * delta8[6] * X[i][4] + n * delta9[6] * X[i][4]
            W[45] += n * delta8[6] * X[i][5] + n * delta9[6] * X[i][5]
            W[53] += n * delta8[6] * X[i][6] + n * delta9[6] * X[i][6]
            W[61] += n * delta8[6] * X[i][7] + n * delta9[6] * X[i][7]

            W[6] += n * delta8[7] * X[i][0] + n * delta9[7] * X[i][0]
            W[14] += n * delta8[7] * X[i][1] + n * delta9[7] * X[i][1]
            W[22] += n * delta8[7] * X[i][2] + n * delta9[7] * X[i][2]
            W[30] += n * delta8[7] * X[i][3] + n * delta9[7] * X[i][3]
            W[38] += n * delta8[7] * X[i][4] + n * delta9[7] * X[i][4]
            W[46] += n * delta8[7] * X[i][5] + n * delta9[7] * X[i][5]
            W[54] += n * delta8[7] * X[i][6] + n * delta9[7] * X[i][6]
            W[62] += n * delta8[7] * X[i][7] + n * delta9[7] * X[i][7]

            W[7] += n * delta8[8] * X[i][0] + n * delta9[8] * X[i][0]
            W[15] += n * delta8[8] * X[i][1] + n * delta9[8] * X[i][1]
            W[23] += n * delta8[8] * X[i][2] + n * delta9[8] * X[i][2]
            W[31] += n * delta8[8] * X[i][3] + n * delta9[8] * X[i][3]
            W[39] += n * delta8[8] * X[i][4] + n * delta9[8] * X[i][4]
            W[47] += n * delta8[8] * X[i][5] + n * delta9[8] * X[i][5]
            W[55] += n * delta8[8] * X[i][6] + n * delta9[8] * X[i][6]
            W[63] += n * delta8[8] * X[i][7] + n * delta9[8] * X[i][7]

            U[9] += n * delta9[0] * -1
            U[8] += n * delta8[0] * -1
            U[7] += n * delta8[8] * -1 + n * delta9[8] * -1
            U[6] += n * delta8[7] * -1 + n * delta9[7] * -1
            U[5] += n * delta8[6] * -1 + n * delta9[6] * -1
            U[4] += n * delta8[5] * -1 + n * delta9[5] * -1
            U[3] += n * delta8[4] * -1 + n * delta9[4] * -1
            U[2] += n * delta8[3] * -1 + n * delta9[3] * -1
            U[1] += n * delta8[2] * -1 + n * delta9[2] * -1
            U[0] += n * delta8[1] * -1 + n * delta9[1] * -1

            # Cálculo del error para la muestra actual
            Error[i][0] = 0.5 * (T[i][0] - fneta[9]) ** 2
            Error[i][1] = 0.5 * (T[i][1] - fneta[8]) ** 2

        # Almacenamiento de los pesos en cada iteración para el ploteo
        for i in range(len(W)):
            axis[i, j] = W[i]

        # Cálculo del error medio cuadrático
        sum_squared_errors = np.sum(Error)
        num_samples = len(X)
        Emc = np.sqrt(sum_squared_errors / (2*num_samples))
        errors.append(Emc)

        # Impresión del progreso
        print("Iteracion: ", j, " Error: ", Emc)

    # Guardar valores iniciales
    initial_values = {
        'W': W,
        'U': U,
        'maxIter': maxIter,
        'minError': minError,
        'n': n,
        # Otros valores que desees guardar
    }

    with open('initial_values.pkl', 'wb') as file:
        pickle.dump(initial_values, file)

    # Impresión de los pesos  y umbrales finales
    print("Pesos: ", W)
    print("Umbrales: ", U)

    return W,neta,fneta,T,X,U,errors,axis,axisX

def learningComp(W, neta, fneta, T, X, U):
    # Mapeo o comprobación del aprendizaje
    print("Mapeo o comprobación del aprendizaje")

    for i in range(len(X)):
        neta[0] = X[i][0] * W[0] + X[i][1] * W[8] + X[i][2] * W[16] + X[i][3] * W[24] + X[i][4] * W[32] + X[i][5] * W[40] + X[i][6] * W[48] + X[i][7] * W[56] - U[0]
        neta[1] = X[i][0] * W[1] + X[i][1] * W[9] + X[i][2] * W[17] + X[i][3] * W[25] + X[i][4] * W[33] + X[i][5] * W[41] + X[i][6] * W[49] + X[i][7] * W[57] - U[1]
        neta[2] = X[i][0] * W[2] + X[i][1] * W[10] + X[i][2] * W[18] + X[i][3] * W[26] + X[i][4] * W[34] + X[i][5] * W[42] + X[i][6] * W[50] + X[i][7] * W[58] - U[2]
        neta[3] = X[i][0] * W[3] + X[i][1] * W[11] + X[i][2] * W[19] + X[i][3] * W[27] + X[i][4] * W[35] + X[i][5] * W[43] + X[i][6] * W[51] + X[i][7] * W[59] - U[3]
        neta[4] = X[i][0] * W[4] + X[i][1] * W[12] + X[i][2] * W[20] + X[i][3] * W[28] + X[i][4] * W[36] + X[i][5] * W[44] + X[i][6] * W[52] + X[i][7] * W[60] - U[4]
        neta[5] = X[i][0] * W[5] + X[i][1] * W[13] + X[i][2] * W[21] + X[i][3] * W[29] + X[i][4] * W[37] + X[i][5] * W[45] + X[i][6] * W[53] + X[i][7] * W[61] - U[5]
        neta[6] = X[i][0] * W[6] + X[i][1] * W[14] + X[i][2] * W[22] + X[i][3] * W[30] + X[i][4] * W[38] + X[i][5] * W[46] + X[i][6] * W[54] + X[i][7] * W[62] - U[6]
        neta[7] = X[i][0] * W[7] + X[i][1] * W[15] + X[i][2] * W[23] + X[i][3] * W[31] + X[i][4] * W[39] + X[i][5] * W[47] + X[i][6] * W[55] + X[i][7] * W[63] - U[7]

        fneta[0] = f(neta[0])
        fneta[1] = f(neta[1])
        fneta[2] = f(neta[2])
        fneta[3] = f(neta[3])
        fneta[4] = f(neta[4])
        fneta[5] = f(neta[5])
        fneta[6] = f(neta[6])
        fneta[7] = f(neta[7])

        neta[8] = fneta[0] * W[64] + fneta[1] * W[66] + fneta[2] * W[68] + fneta[3] * W[70] + fneta[4] * W[72] + fneta[5] * W[74] + fneta[6] * W[76] + fneta[7] * W[78] - U[8]
        neta[9] = fneta[0] * W[65] + fneta[1] * W[67] + fneta[2] * W[69] + fneta[3] * W[71] + fneta[4] * W[73] + fneta[5] * W[75] + fneta[6] * W[77] + fneta[7] * W[79] - U[9]
    
        fneta[8] = f(neta[8])
        fneta[9] = f(neta[9])

        # Impresión de la entrada, salida esperada y salida obtenida
        print("Entrada: ", X[i], " Salida: ", enfermedad(T[i][0], T[i][1]), " Salida Obtenida: ", enfermedad(valor(fneta[8]),valor(fneta[9])))

def plotErrors(errors):
    # Ploteo de la curva de aprendizaje
    plt.figure()
    plt.plot(np.arange(1, maxIter+1), errors, label='Error de Entrenamiento')
    plt.xlabel('Iteración')
    plt.ylabel('Error')
    plt.title('Curva de Aprendizaje')
    plt.legend()
    plt.show()

def plotWeights(axis, axisX):
    # Ploteo de los pesos a lo largo de las iteraciones
    fig, (pesos, error) = plt.subplots(2)
    pesos.set_title("Pesos")
    for axi in axis:
        pesos.plot(axisX, axi)

    plt.show()

def test(prueba):
    # Cargar valores iniciales
    with open('initial_values.pkl', 'rb') as file:
        initial_values = pickle.load(file)

    # Obtener los pesos finales
    W = initial_values['W']
    U = initial_values['U']

    print("Pesos: ", W)
    print("Umbrales: ", U)

    # Realizar los cálculos para la capa oculta
    neta_oculta = np.dot(prueba, W[:64].reshape(8, 8)) - U[:8]
    fneta_oculta = f(neta_oculta)

    # Realizar los cálculos para la capa de salida
    neta_salida = np.dot(fneta_oculta, W[64:].reshape(8, 2)) - U[8:]
    fneta_salida = f(neta_salida)

    # Obtener la salida final
    output = [valor(fneta_salida[1]), valor(fneta_salida[0])]

    # Obtener la etiqueta de enfermedad
    enfermedad_label = enfermedad(output[0], output[1])

    # Imprimir el resultado
    print("Vector de prueba: ", prueba)
    print("Salida Obtenida: ", output)
    print("Etiqueta de Enfermedad: ", enfermedad_label)

def beginTrain():
    W,neta,fneta,T,X,U,errors,axis,axisX = train()
    learningComp(W, neta, fneta, T, X, U)
    plotErrors(errors)
    plotWeights(axis, axisX)

def beginTest():
    # Vector de prueba
    prueba = np.array([1,1,1,1,0,1,0,1])
    test(prueba)

if __name__ == "__main__":
    #beginTrain()
    
    beginTest()