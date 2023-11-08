import numpy as np
import sympy as sp

def MetodoNewton(f,x0,y0,z0,max_n_iter=1000,TOL=1e-6):
    # Calcular el gradiente de la función
    fx = sp.diff(f,x)
    fy = sp.diff(f,y)
    fz = sp.diff(f,z)
    G = sp.Matrix([fx,fy,fz])
    # Calcular la hessiana de la función
    H = sp.Matrix([[sp.diff(fx,x),sp.diff(fx,y),sp.diff(fx,z)],
                   [sp.diff(fy,x),sp.diff(fy,y),sp.diff(fy,z)],
                   [sp.diff(fz,x),sp.diff(fz,y),sp.diff(fz,z)]])

    print(f"Gradiente:\n {G}")
    print(f"Hessiana: \n {H}")
    # Inicializar el punto actual

    x_min = sp.Matrix([x0])[0]
    y_min = sp.Matrix([y0])[0]
    z_min = sp.Matrix([z0])[0]

    print(f"Punto inicial: ({x_min},{y_min},{z_min})")
    print(f"F(x,y,z) = {f.subs([(x,x_min),(y,y_min),(z,z_min)]):.6f}")
    x_actual = 0
    y_actual = 0
    z_actual = 0

    i=0

    # Iterar hasta que el cambio en el valor de la función sea menor que TOL
    while(np.linalg.norm(np.array([float(x_actual-x_min), float(y_actual-y_min), float(z_actual-z_min)])) > TOL):
        x_actual = x_min
        y_actual = y_min
        z_actual = z_min

        G_values = G.subs([(x,x_min),(y,y_min),(z,z_min)])
        H_values = H.subs([(x,x_min),(y,y_min),(z,z_min)])
    
        H_inv = np.linalg.inv(np.array(H_values).astype(np.float64))

        print("--------------------------------------------------")
        print(f"Iteracion : {i+1}")
        print(f"Hessiana: \n{H_values}")  
        print(f"Inverso de la Hessiana: \n{H_inv}")
        print(f"Gradiente F({x_actual:.6f},{y_actual:.6f},{z_actual:.6f}) = ({G_values[0]},{G_values[1]}, {G_values[2]})")

        x_min = x_min - H_inv[0][0]*G_values[0] - H_inv[0][1]*G_values[1] - H_inv[0][2]*G_values[2]
        y_min = y_min - H_inv[1][0]*G_values[0] - H_inv[1][1]*G_values[1] - H_inv[1][2]*G_values[2]
        z_min = z_min - H_inv[2][0]*G_values[0] - H_inv[2][1]*G_values[1] - H_inv[2][2]*G_values[2]
    
        print(f"x: {x_min:.6f} - y: {y_min:.6f} - z: {z_min:.6f}")
        print(f"F(x,y,z) = {f.subs([(x,x_min),(y,y_min),(z,z_min)]):.6f}")

        i += 1

        if (i == max_n_iter):
            print("No converge")
            return


if __name__ == "__main__":
    #Definir las variables
    x,y,z = sp.var("x y z")

    f = 40/(x*y*z) + 40*y*z + 10*x*y + 20*x*z

    MetodoNewton(f,6,4,5)
