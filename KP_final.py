import numpy as np
import pulp
import pandas as pd
import time

def GeneradorKp(N,V):
    A = [list(np.random.randint(6,24,size=N))]
    c = list(np.random.randint(3,12,size=N))
    return A,[V],c

def CreateDictionary():
    dic = {
        "Algoritmo":[],
        "N Tablas":[],
        "tiempo":[],
        "Area Maxima":[],
        "Volumen Solucion":[] ,
        "Solucion": []
    }
    return dic

def Area(X,c):
    return sum(np.array(X)*np.array(c))

def Volumen(X0,A):
    sV=0
    for i,s in enumerate(X0):
        if s == 1:
          sV += A[0][i]
    return sV

def GetVecino(X0):
    N =len(X0)
    Xv = X0.copy()
    i = np.random.randint(N)
    if Xv[i] == 1: Xv[i] = 0
    else: Xv[i]  = 1
    return Xv

def Perturbar(X):
    return list(np.random.randint(0,2,size=N))

def ILS_Exploracion(X0,c,b,maxArea):
    i=0
    while (i < 200):
      Xv = GetVecino(X0)
      Av = Area(Xv,c)
      Vv = Volumen(Xv,A)
      if ( Av > maxArea and Vv <= b[0] ):
        X0 = Xv
      i+=1
    return X0

def FillResults(dic,tiempo,Area,Solucion,V,N):
    dic["tiempo"].append(tiempo)
    dic["Area Maxima"].append(Area)
    dic["N Tablas"].append(N)
    dic["Solucion"].append(Solucion)
    dic["Volumen Solucion"].append(V)

def MetaheuristicaKP(c, A, b):
    dic = {
        "Algoritmo":["MetaHeuristica Kp"],
        "N Tablas":[],
        "tiempo":[],
        "Area Maxima":[],
        "Volumen Solucion":[] ,
        "Solucion": []

    }
    inicio = time.time()
    N = len(c)
    while True:
      X0 = list(np.random.randint(0,2,size=N))
      A0 = Area(X0,c)
      V0 = Volumen(X0,A)
      if(V0 <= b[0]) : break

    i=0
    while(i<500):
      xP = Perturbar(X0)
      Ap = Area(xP,c)
      Vp = Volumen(xP,A)
      if Ap >  A0 and Vp <= b[0]:
        X0 = xP
        A0 = Area(X0,c)
        X0 = ILS_Exploracion(X0,c,b,A0)
      i+=1
    sV = Volumen(X0,A)
    fin = time.time()
    T = fin-inicio
    FillResults(dic,T,Area(X0,c),X0,sV,N)
    df_resultado = pd.DataFrame(dic)
    return df_resultado

def AlgoritmoSimplex_KP(c, A, b,objetivo="max"):
    dic = {
        "Algoritmo":["Simplex-KP Camion"],
        "N Tablas": [],
        "tiempo":[],
        "Area Maxima":[],
        "Volumen Solucion":[] ,
        "Solucion": []

    }

    inicio = time.time()
    # Verificar el objetivo y crear un problema de PuLP correspondiente
    if objetivo.lower() == "max": 
       prob = pulp.LpProblem("MaximizationProblem", pulp.LpMaximize)
    elif objetivo.lower() == "min": 
       prob = pulp.LpProblem("MinimizationProblem", pulp.LpMinimize)
    else: 
       raise ValueError("El parámetro 'objetivo' debe ser 'max' o 'min'")
    # Crear variables
    n = len(c)
    x = [pulp.LpVariable(f"x{i}", cat=pulp.LpBinary) for i in range(n)]
    # Definir la función objetivo
    if objetivo.lower() == "max": 
       prob += pulp.lpSum(c[i] * x[i] for i in range(n)), "ObjectiveFunction"
    elif objetivo.lower() == "min": 
       prob += pulp.lpSum(-c[i] * x[i] for i in range(n)), "ObjectiveFunction"
    # Definir las restricciones
    for i in range(len(A)): 
       prob += pulp.lpSum(A[i][j] * x[j] for j in range(n)) <= b[i], f"Constraint{i}"
    # Resolver el problema con el solucionador CBC
    prob.solve(pulp.PULP_CBC_CMD())
    # Obtener el estado, el valor óptimo y las soluciones de las variables
    if objetivo.lower() == "max":
       valor_optimo = pulp.value(prob.objective)
    elif objetivo.lower() == "min":
       valor_optimo = -pulp.value(prob.objective)
    soluciones = [int(x[i].varValue) for i in range(n)]
    fin = time.time()
    T = fin-inicio

    sV = Volumen(soluciones,A)
    FillResults(dic,T,valor_optimo,soluciones,sV,n)
    df_resultado = pd.DataFrame(dic)
    return df_resultado

def FillResults_Comparation(dic,list_df):
    for df in list_df:
      dic["Algoritmo"].append(df["Algoritmo"][0])
      dic["N Tablas"].append(df["N Tablas"][0])
      dic["tiempo"].append(df["tiempo"][0])
      dic["Area Maxima"].append(df["Area Maxima"][0])
      dic["Solucion"].append(df["Solucion"][0])
      dic["Volumen Solucion"].append(df["Volumen Solucion"][0])

dic = CreateDictionary()

def Comparacion_Simplex_Heuristica(c,A,b):
    df_Simplex = AlgoritmoSimplex_KP(c,A,b, objetivo="max")
    df_MH = MetaheuristicaKP(c,A,b)
    FillResults_Comparation(dic,[df_Simplex,df_MH])

for i in range(4,24):
  N = i
  V = 56
  A,b,c = GeneradorKp(N,V)
  Comparacion_Simplex_Heuristica(c,A,b)

pd.DataFrame(dic)