import numpy as np

from typing import Tuple
import math

from scipy.optimize import linprog
from scipy.sparse import csr_matrix

import math
from typing import NamedTuple, Any, Tuple, Union

from collections import deque

def is_integer(x, TOL=1e-8):
    return abs(x - round(x)) <= TOL


class IP(NamedTuple):
    """
    Entradas para un solucionador de programación lineal
    """

    cT: Any  # vector que contiene la función objetivo
    Aub: csr_matrix  # matriz de restricciones de desigualdad
    bub: Any  # vector de restricciones de desigualdad
    Aeq: csr_matrix  # matriz de restricciones de igualdad
    beq: Any  # vector de restricciones de igualdad
    bounds: list[Tuple[Union[int, None], Union[int, None]]]  # límites de las variables


class Solution(NamedTuple):
    x: Any
    fun: float


def first(soln: Solution, prob: IP) -> Tuple[IP, IP]:
    """
    Ramifica en la primera variable no entera en la solución
    """
    for ix, x in enumerate(soln.x):
        if not is_integer(x):
            print(f"variable {ix} = {x} no es integral")
            # subproblema 1: añade var[ix] <= floor(x)
            bounds = list(prob.bounds)
            lb, ub = bounds[ix][0], bounds[ix][1]

            if ub is None:
                ub = float("inf")
            bounds[ix] = lb, min(ub, math.floor(x))
            print(
                f"subproblema izquierdo con nuevos límites para var {ix}: {bounds[ix]}"
            )
            s1 = IP(prob.cT, prob.Aub, prob.bub, prob.Aeq, prob.beq, bounds)

            # subproblema 2: añade var[ix] >= ceil(x)
            bounds = list(prob.bounds)
            lb, ub = bounds[ix][0], bounds[ix][1]
            bounds[ix] = max(lb, math.ceil(x)), ub
            print(f"subproblema derecho con nuevos límites para var {ix}: {bounds[ix]}")
            s2 = IP(prob.cT, prob.Aub, prob.bub, prob.Aeq, prob.beq, bounds)

            return s1, s2

    raise Exception("no se pudo ramificar en solución entera")


def most_infeasible(soln: Solution, prob: IP) -> Tuple[IP, IP]:
    """
    Ramifica en la variable que es la "menos entera", es decir, la variable con el máximo de abs(x - round(x))
    """

    dist = np.abs(soln.x - np.round(soln.x))
    ix = np.argmax(dist)
    x = soln.x[ix]

    if not is_integer(x):
        print(f"variable {ix} = {x} es la más inviable")
        # subproblema 1: añade var[ix] <= floor(x)
        bounds = list(prob.bounds)
        lb, ub = bounds[ix][0], bounds[ix][1]

        if ub is None:
            ub = float("inf")
        bounds[ix] = lb, min(ub, math.floor(x))
        print(f"subproblema izquierdo con nuevos límites para var {ix}: {bounds[ix]}")
        s1 = IP(prob.cT, prob.Aub, prob.bub, prob.Aeq, prob.beq, bounds)

        # subproblema 2: añade var[ix] >= ceil(x)
        bounds = list(prob.bounds)
        lb, ub = bounds[ix][0], bounds[ix][1]
        bounds[ix] = max(lb, math.ceil(x)), ub
        print(f"subproblema derecho con nuevos límites para var {ix}: {bounds[ix]}")
        s2 = IP(prob.cT, prob.Aub, prob.bub, prob.Aeq, prob.beq, bounds)

        return s1, s2

    raise Exception("no se pudo ramificar en solución entera")


def solve_lp_relaxation(ip: IP) -> Solution:
    result = linprog(
        ip.cT,
        ip.Aub,
        ip.bub,
        ip.Aeq,
        ip.beq,
        ip.bounds,
        method="highs",
        options={},
    )
    if not result.success:
        return Solution(None, float("inf"))
    else:
        return Solution(result.x, result.fun)


def noop(soln: Solution, prob: IP) -> Tuple[IP, bool]:
    return prob, False


def branch_and_bound(
    ip: IP,
    best=Solution(None, float("inf")),  # mejor solución hasta ahora
    brancher=first,  # estrategia de ramificación
    cutter=noop,  # estrategia de corte
) -> Solution:
    print(f"resolviendo... (la mejor hasta ahora es {best})")

    TOL = 1e-8

    soln = solve_lp_relaxation(ip)

    # no hay solución factible
    if soln.x is None:
        print(f"no hay solución PL factible")
        return best
    else:  # solución factible, pero no es lo suficientemente buena
        if soln.fun >= best.fun:
            print(
                f"solución PL factible {soln} no es mejor que la mejor solución entera hasta ahora {best}"
            )
            return best
        else:
            print(f"solución PL factible {soln}")

    # si es una solución completamente entera, devuélvela si es mejor que la mejor hasta ahora,
    # de lo contrario devuelve la mejor hasta ahora
    all_integer = True
    for ix, x in enumerate(soln.x):
        if not is_integer(x, TOL):
            all_integer = False
            break
    if all_integer:
        if soln.fun < best.fun:
            print(f"solución entera {soln} es la mejor hasta ahora (mejor que {best})")
            return soln
        else:
            print(f"solución entera {soln} es peor que la mejor hasta ahora: {best}")
            return best

    # no hay solución entera, veamos si podemos encontrar algunos cortes
    sc, found = cutter(soln, ip)
    if found:
        print(f"se añadieron cortes, resolviendo en comparación con {best}")
        return branch_and_bound(sc, best, brancher=brancher, cutter=cutter)

    # de lo contrario, crea dos subproblemas para cada solución no entera
    s1, s2 = brancher(soln, ip)
    r1 = branch_and_bound(s1, best, brancher=brancher, cutter=cutter)

    if r1.fun < best.fun:
        print(f"s1 produjo una nueva mejor: {r1} (mejor que {best})")
        best = r1

    r2 = branch_and_bound(s2, best, brancher=brancher, cutter=cutter)
    if r2.fun < best.fun:
        print(f"s2 produjo una nueva mejor: {r2} (mejor que {best})")
        best = r2

    return best

def branch_and_bound_bfs(
    ip: IP,
    best=Solution(None, float("inf")),
    brancher=first,
    cutter=noop,
) -> Solution:
    print(f"resolviendo... (la mejor hasta ahora es {best})")

    TOL = 1e-8

    queue = deque()
    queue.append(ip)

    while queue:
        current_ip = queue.popleft()
        soln = solve_lp_relaxation(current_ip)

        if soln.x is None:
            print(f"no hay solución PL factible")
            continue
        elif soln.fun >= best.fun:
            print(f"solución PL factible {soln} no es mejor que la mejor solución entera hasta ahora {best}")
            continue
        else:
            print(f"solución PL factible {soln}")

        all_integer = True
        for ix, x in enumerate(soln.x):
            if not is_integer(x, TOL):
                all_integer = False
                break

        if all_integer:
            if soln.fun < best.fun:
                print(f"solución entera {soln} es la mejor hasta ahora (mejor que {best})")
                best = soln
            else:
                print(f"solución entera {soln} es peor que la mejor hasta ahora: {best}")
        else:
            sc, found = cutter(soln, current_ip)
            if found:
                print(f"se añadieron cortes, resolviendo en comparación con {best}")
                queue.append(sc)
            else:
                s1, s2 = brancher(soln, current_ip)
                queue.append(s1)
                queue.append(s2)

    return best

if "__main__" == __name__:
    # Problema de clase
    """
    cT = np.array([1, 5])
    cT *= -1  # para maximizar
    Aub = np.array(
        [
            [11, 6],
            [5, 50],
        ],
        dtype=np.float64,
    )
    bub = np.array(
        [66, 225],
        dtype=np.float64,
    )
    Aeq = None
    beq = None
    bounds = [
        (0, 20),
        (0, 20),
    ]
    """

    #Problema 1
    """"
    cT = np.array([5.5, 2.1])
    cT *= -1  # para maximizar
    Aub = np.array(
        [
            [-1, 1],
            [8, 2],
        ],
        dtype=np.float64,
    )
    bub = np.array(
        [2, 17],
        dtype=np.float64,
    )
    Aeq = None
    beq = None
    bounds = [
        (0, 20),
        (0, 20),
    ]
    """
    # Problema 2
    
    cT = np.array([2, 3])
    cT *= -1  # para maximizar
    Aub = np.array(
        [
            [4, 12],
            [10, 4],
        ],
        dtype=np.float64,
    )
    bub = np.array(
        [33, 35],
        dtype=np.float64,
    )
    Aeq = None
    beq = None
    bounds = [
        (0, 20),
        (0, 20),
    ]

    

    ip = IP(cT, Aub, bub, Aeq, beq, bounds)

    print("\n------------------DFS------------------\n")
    result = branch_and_bound(ip)

    print(f"check result {result.x}")
    print(f"check result {-1*result.fun}")


    print("\n------------------BFS------------------\n")
    # Llama a la función con la búsqueda en anchura
    best_solution = branch_and_bound_bfs(ip)
    print("Mejor solución encontrada:", best_solution.x, "con valor:", -1*best_solution.fun)