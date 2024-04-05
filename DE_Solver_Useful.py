import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import sympy as sym
import matplotlib.pyplot as plt
import re
import seaborn as sns
import pandas as pd


def solve_ivp_text(f,period,x0,dense_output=True,atol=1e-200,rtol=1e-13,max_step=np.inf,args=None,method="RK45"):
    """solve_ivp_text: Just scipy solve_ivp, but here the input function f (remember, dx/dt=f(x,t),can be written without meention of time t, and written in the form f="x0+x1-x2,x2*(x0-x1),(x1-x0)+x2" or f="[x0+x1-x2,x2*(x0-x1),(x1-x0)+x2]", via strings. 
    
    Period is the same as before, i.e,period=[t0,5], so is the case of the initial value

    x0 is just x0=[1,1,1]


    Returns:
    Bunch object with the following fields defined:
        t : ndarray, shape (n_points,)
            Time points.
        y : ndarray, shape (n, n_points)
            Values of the solution at `t`.
        sol : `OdeSolution` or None
            Found solution as `OdeSolution` instance; None if `dense_output` was
            set to False.
        t_events : list of ndarray or None
            Contains for each event type a list of arrays at which an event of
            that type event was detected. None if `events` was None.
        y_events : list of ndarray or None
            For each value of `t_events`, the corresponding value of the solution.
            None if `events` was None.
        nfev : int
            Number of evaluations of the right-hand side.
        njev : int
            Number of evaluations of the Jacobian.
        nlu : int
            Number of LU decompositions.
        status : int
            Reason for algorithm termination:

                * -1: Integration step failed.
                *  0: The solver successfully reached the end of `tspan`.
                *  1: A termination event occurred.

        message : string
            Human-readable description of the termination reason.
        success : bool
            True if the solver reached the interval end or a termination event
            occurred (``status >= 0``).

    """
    #Regulate the System Expresion
    def normalizador(f):
        if f[0] != "[":
            pass
        else:
            f="["+f+"]"
        f=f.replace(";",",").replace("^","**").replace("#", "").replace("sen","sin").replace("[","(").replace("]",")").replace("i","I").replace("X","x").replace("sIn","sin")
        return f
    f=normalizador(f)

    #Define the system components
    n_var = len(set(re.findall(r'x\d+', f)))
    n_eq = len(f.split(","))
    x=sym.symbols(f"x:{n_var}")
    f_sym=sym.sympify(f)
    f_num=sym.lambdify(x,f,"numpy")
    def system(x):
        def f_numeric(t, x):
            return f_num(*x)
        return f_numeric
    system=system(x)
    #Solve system
    
    approx_sol=solve_ivp(system,period,x0,dense_output=dense_output,atol=atol,rtol=rtol,max_step=max_step,args=args,method=method)
    return approx_sol
    
f="[x0+x1-x2,x2*(x0-x1),(x1-x0)+x2]"
x0=[1,1,1]
period=[0,5]
sol=solve_ivp_text(f,period,x0)
print(sol)