import numpy as np
import sympy as sp

def Min_G(f, G0, G1, G2, x_min, y_min, z_min):
    thetha = sp.symbols("thetha")
    g = f.subs([(x, x_min + thetha * G0), (y, y_min + thetha * G1)])
    if G2 is not None:
        g = g.subs(z, z_min + thetha * G2)
    df = sp.diff(g, thetha)
    thetha_solutions = sp.solve(df, thetha)
    if thetha_solutions:
        return thetha_solutions[0]
    else:
        return 0

def Metodo_gradiente(f, x0, y0, z0=None, max_n_iter=1000, epsilon=1e-6):
    x, y, z = sp.symbols('x y z')
    G = sp.Matrix([sp.diff(f, x), sp.diff(f, y), sp.diff(f, z) if z0 is not None else 0])
    x_min = x0
    y_min = y0
    z_min = z0
    i = 0
    x_current = 0
    y_current = 0
    z_current = 0 if z0 is not None else None

    while i < max_n_iter and (np.abs(x_current - x_min) >= epsilon or np.abs(y_current - y_min) >= epsilon or (z0 is not None and np.abs(z_current - z_min) >= epsilon)):
        x_current = x_min
        y_current = y_min
        z_current = z_min

        G_val = G.subs({x: x_min, y: y_min, z: z_min if z_min is not None else 0})
        thetha = Min_G(f, G_val[0], G_val[1], G_val[2] if z0 is not None else None, x_min, y_min, z_min)

        x_min = x_min - thetha * G_val[0]
        y_min = y_min - thetha * G_val[1]
        if z_min is not None:
            z_min = z_min - thetha * G_val[2]

        print(f"Iter : {i}")
        print(f"x: {x_min:.6f} - y: {y_min:.6f}")
        if z_min is not None:
            print(f"z: {z_min:.6f}")

        i += 1

    if i == max_n_iter:
        print("No converge")

    return x_min, y_min, z_min if z_min is not None else None
