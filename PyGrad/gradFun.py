import numpy as np
import sympy as sp
# Creamos un diccionario para una mejor grafica
values = {
    "x": [],
    "y": [],
    "thetha":[]
}


def AgregarValores(dic,x,y,thetha):
  dic["x"].append(x)
  dic["y"].append(y)
  dic["thetha"].append(thetha)


def Min_G(f,G0,G1,x_min,y_min):
  thetha = sp.var("thetha")
  g = f.subs([ (x,x_min + thetha*G0), (y,y_min + thetha*G1)])
  df=sp.diff(g,thetha)
  thetha = sp.solve(sp.Eq(df, 0))
  if thetha:
    return thetha[0]
  else:
    return 0

def Metodo_gradiente(f,x0,y0,n_iter=1000,epsilon=1e-6):

  fx = sp.diff(f,x)
  fy = sp.diff(f,y)
  G = sp.Matrix([fx,fy])

  x_min = sp.Matrix([x0])[0]
  y_min = sp.Matrix([y0])[0]
  i=0


  while(i<=n_iter):
    x_current = x_min
    y_current = y_min

    G0 = G.subs([(x,x_min),(y,y_min)])[0]
    G1 = G.subs([(x,x_min),(y,y_min)])[1]

    thetha = Min_G(f,G0,G1,x_min,y_min)

    AgregarValores(values,float(x_min.evalf()),float(y_min.evalf()),float(thetha))

    x_min = x_min + thetha*G0
    y_min = y_min + thetha*G1
  
    if(np.abs(x_current-x_min) < epsilon and np.abs(y_current-y_min) < epsilon):
      return x_min.evalf(),y_min.evalf()

    print(f"Iter : {i}")
    print(f"x: {x_min:.6f} - y: {y_min:.6f}")
    i+=1

  print("No converge")
  return x_min.evalf(),y_min.evalf()


#Definimos las variables
x,y = sp.var("x y")

#Definimos la funcion a minimizar
f =  x**2 + y**2 + x*y - 3*x

Metodo_gradiente(f,2,2)