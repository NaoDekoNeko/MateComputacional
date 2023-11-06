import numpy as np
import sympy as sp

# Función para encontrar el tamaño de paso óptimo para el método de descenso de gradiente
def Min_G(f,G0,G1,G2,x_min,y_min, z_min):
    # Definir la variable de tamaño de paso
    thetha = sp.var("thetha")
    # Sustituir el punto actual y el tamaño de paso en la función
    g = f.subs([ (x,x_min + thetha*G0), (y,y_min + thetha*G1), (z,z_min + thetha*G2)])
    print(f"g(thetha) = {g.evalf(6)}")
    # Calcular la derivada de la función con respecto al tamaño de paso
    dg=sp.diff(g,thetha)
    print(f"dg(thetha) = {dg.evalf(6)}")
    # Resolver la ecuación donde la derivada es igual a cero para encontrar el tamaño de paso óptimo
    thetha = sp.solve(sp.Eq(dg, 0))
    if thetha:
        return thetha[0]
    else:
        return 0

# Función para realizar el método de descenso de gradiente
def Metodo_gradiente(f,x0,y0,z0,max_n_iter=1000,TOL=1e-6):
    # Calcular el gradiente de la función
    fx = sp.diff(f,x)
    fy = sp.diff(f,y)
    fz = sp.diff(f,z)
    G = sp.Matrix([fx,fy,fz])

    print(f"Gradiente: ({fx},{fy},{fz})")

    # Inicializar el punto actual
    x_min = sp.Matrix([x0])[0]
    y_min = sp.Matrix([y0])[0]
    z_min = sp.Matrix([z0])[0]
    x_actual = 0.1
    y_actual = 0.1
    z_actual = 0.1
    i=0

    # Iterar hasta que el cambio en el valor de la función sea menor que TOL
    while(np.abs(x_actual-x_min) > TOL and np.abs(y_actual-y_min) > TOL and np.abs(z_actual-z_min) > TOL):
        x_actual = x_min
        y_actual = y_min
        z_actual = z_min

        G0 = G.subs([(x,x_min),(y,y_min), (z,z_min)])[0]
        G1 = G.subs([(x,x_min),(y,y_min), (z,z_min)])[1]
        G2 = G.subs([(x,x_min),(y,y_min), (z,z_min)])[2]

        i+=1
        print(f"Iteracion : {i}")
        print(f"min g(theta) = F(({x_min:.6f},{y_min:.6f},{z_min:.6f}) + theta*gradiente F({G0:.6f},{G1:.6f},{G2:.6f})")   
        print(f"Gradiente F({x_actual:.6f},{y_actual:.6f},{z_actual:.6f}) = ({G0:.6f},{G1:.6f},{G2:.6f})")
        thetha = Min_G(f,G0,G1,G2,x_min,y_min, z_min)
        print(f"thetha = {thetha:.6f}")

        x_min = x_min + thetha*G0
        y_min = y_min + thetha*G1
        z_min = z_min + thetha*G2
        
        print(f"x: {x_min:.6f} - y: {y_min:.6f} - z: {z_min:.6f}")
        print(f"F({x_min:.6f},{y_min:.6f},{z_min:.6f}) = {f.subs([(x,x_min),(y,y_min),(z,z_min)]):.6f}")

        if (i == max_n_iter):
            print("No converge")
            return

    

#Definir las variables
x,y,z = sp.var("x y z")

#Definir la funcion a minimizar
f =  x**2 + y**2 - z**2 + 2*x*y - 2*x*z + 2*y*z

Metodo_gradiente(f,10,10,10)