import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 1, 1, 1],
              [1, -1, -1, 1],
              [-1, 1, 1, 1],
              [-1, -1, -1, 1]])

T = np.array([[1, -1],
              [-1, 1],
              [1, -1],
              [-1, 1]])

W = np.array([0.5, 0.3, 0.7, 0.1, 0.4,
              0.9, 0.8, 0.9, 0.5, 0.4,
              0.7, 0.9, 0.4, 0.7, 0.9,
              0.8, 0.7, 0.6])

U = np.array([0.8, 0.2, 0.6, 0.9, 0.8])

n = 0.65

maxIter = 2500
minError = 0.1
Emc = 1000000

axisX = np.arange(0, maxIter + 1)
axis = np.zeros((len(W), maxIter + 1))
axisEmc = np.array([])

def f(x):
    return 1 / (1 + np.exp(-x))

def df(x):
    return f(x) * (1 - f(x))

neta = np.zeros(5)
fneta = np.zeros(5)
delta3 = np.zeros(4)
delta4 = np.zeros(4)

j = 0
errors = []

while (j < maxIter) and (Emc > minError):
    j += 1
    Error = np.zeros((len(X), 2))
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
        delta3[0] = (T[i][0] - fneta[3]) * df(neta[3])
        delta3[1] = fneta[0] * delta3[0] * W[12]
        delta3[2] = fneta[1] * delta3[0] * W[14]
        delta3[3] = fneta[2] * delta3[0] * W[16]
        delta4[0] = (T[i][1] - fneta[4]) * df(neta[4])
        delta4[1] = fneta[0] * delta4[0] * W[13]
        delta4[2] = fneta[1] * delta4[0] * W[15]
        delta4[3] = fneta[2] * delta4[0] * W[17]

        W[12] += n * delta3[0] * fneta[0]
        W[14] += n * delta3[0] * fneta[1]
        W[16] += n * delta3[0] * fneta[2]

        W[13] += n * delta4[0] * fneta[0]
        W[15] += n * delta4[0] * fneta[1]
        W[17] += n * delta4[0] * fneta[2]

        W[0] += n * delta3[1] * X[i][0]
        W[3] += n * delta3[1] * X[i][1]
        W[6] += n * delta3[1] * X[i][2]
        W[9] += n * delta3[1] * X[i][3]

        W[1] += n * delta3[2] * X[i][0]
        W[4] += n * delta3[2] * X[i][1]
        W[7] += n * delta3[2] * X[i][2]
        W[10] += n * delta3[2] * X[i][3]

        W[2] += n * delta3[3] * X[i][0]
        W[5] += n * delta3[3] * X[i][1]
        W[8] += n * delta3[3] * X[i][2]
        W[11] += n * delta3[3] * X[i][3]

        U[3] += n * delta3[0] * -1
        U[2] += n * delta3[3] * -1
        U[1] += n * delta3[2] * -1
        U[0] += n * delta3[1] * -1

        Error[i][0] = 0.5 * (T[i][0] - fneta[3]) ** 2
        Error[i][1] = 0.5 * (T[i][1] - fneta[4]) ** 2

    for i in range(len(W)):
        axis[i, j] = W[i]

    sum_squared_errors = np.sum(Error ** 2)
    num_samples = len(X)
    Emc = np.sqrt(sum_squared_errors / (2*num_samples))
    errors.append(Emc)
    print("Iteracion: ", j, " Error: ", Emc)

print("W: ", W)

def valor(x):
    if x > minError:
        return 1
    else:
        return -1

print("Mapeo o comprobacion del aprendizaje")

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
    print("Entrada: ", X[i], " Salida: ", T[i], " Salida Obtenida: ", valor(fneta[3]), valor(fneta[4]))
    print(fneta[3], "==>", valor(fneta[3]), " | ", fneta[4], "==>", valor(fneta[4]))

# Plotting the learning curve
plt.figure()
plt.plot(np.arange(1, j+1), errors, label='Training Error')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Learning Curve')
plt.legend()
plt.show()

fig, (pesos, error) = plt.subplots(2)
pesos.set_title("Pesos")
for axi in axis:
    pesos.plot(axisX, axi)

plt.show()
