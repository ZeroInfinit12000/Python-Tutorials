import sympy as sym
import numpy as np
from scipy.optimize import root,curve_fit
from scipy.integrate import solve_ivp
import warnings
import re


def string_to_fun(f):
    """f: Transform string to numpy function

    Args:
        f (string): List of strings, say "[x0+x1,x0*x1]". This shall be written as such even if you only got 1 function!
        x (string): List of variables. If we have more variables than in f.
        the order in which we input them in x will be the order in which the.
        generated numpy function will considered them.

    Returns:
        Numpy function. Note that it work like this, if f is takes two arguments like x0,x1, then you can input
        functions like np.array([[2,2],[2,5],[3,4]]). Note that always the input is 1 numpy array. 
    
    Example:
        f=string_to_fun("x1+x0-x2+x3+x4","x0 x1 x2 x3 x4")
        s=np.array([[2,2,4,5,5],[2,5,4,4,3],[2,3,5,31,2],[2,2,4,3,4],[4,4,2,5,4]])
        print(f(s))

        f=string_to_fun("exp(x1)","x1")
        print(f([2,32,32,3,32]))

        f=string_to_fun("exp(x1),x1+x2","x1 x2")
        print(f([[2,4],[4,3]]))
    """

    def extract_variables(f):
        # Define a regular expression pattern to match variables
        variable_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')

        # Find all matches in the expression
        matches = variable_pattern.findall(f)

        # Remove known functions from the matches
        known_functions = [
        'factorial', 'factorial2', 'rf', 'ff', 'binomial', 'RisingFactorial',
        'FallingFactorial', 'subfactorial',

        'carmichael', 'fibonacci', 'lucas', 'motzkin', 'tribonacci', 'harmonic',
        'bernoulli', 'bell', 'euler', 'catalan', 'genocchi', 'andre', 'partition',

        'sqrt', 'root', 'Min', 'Max', 'Id', 'real_root', 'cbrt', 'Rem',

        're', 'im', 'sign', 'Abs', 'conjugate', 'arg', 'polar_lift',
        'periodic_argument', 'unbranched_argument', 'principal_branch',
        'transpose', 'adjoint', 'polarify', 'unpolarify',

        'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'sinc', 'asin', 'acos', 'atan',
        'asec', 'acsc', 'acot', 'atan2',

        'exp_polar', 'exp', 'ln', 'log', 'LambertW',

        'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch', 'asinh', 'acosh', 'atanh',
        'acoth', 'asech', 'acsch',

        'floor', 'ceiling', 'frac',

        'Piecewise', 'piecewise_fold', 'piecewise_exclusive',

        'erf', 'erfc', 'erfi', 'erf2', 'erfinv', 'erfcinv', 'erf2inv', 'Ei',
        'expint', 'E1', 'li', 'Li', 'Si', 'Ci', 'Shi', 'Chi', 'fresnels',
        'fresnelc',

        'gamma', 'lowergamma', 'uppergamma', 'polygamma', 'loggamma', 'digamma',
        'trigamma', 'multigamma',

        'dirichlet_eta', 'zeta', 'lerchphi', 'polylog', 'stieltjes', 'riemann_xi',

        'Eijk', 'LeviCivita', 'KroneckerDelta',

        'SingularityFunction',

        'DiracDelta', 'Heaviside',

        'bspline_basis', 'bspline_basis_set', 'interpolating_spline',

        'besselj', 'bessely', 'besseli', 'besselk', 'hankel1', 'hankel2', 'jn',
        'yn', 'jn_zeros', 'hn1', 'hn2', 'airyai', 'airybi', 'airyaiprime',
        'airybiprime', 'marcumq',

        'hyper', 'meijerg', 'appellf1',

        'legendre', 'assoc_legendre', 'hermite', 'hermite_prob', 'chebyshevt',
        'chebyshevu', 'chebyshevu_root', 'chebyshevt_root', 'laguerre',
        'assoc_laguerre', 'gegenbauer', 'jacobi', 'jacobi_normalized',

        'Ynm', 'Ynm_c', 'Znm',

        'elliptic_k', 'elliptic_f', 'elliptic_e', 'elliptic_pi',

        'beta', 'betainc', 'betainc_regularized',

        'mathieus', 'mathieuc', 'mathieusprime', 'mathieucprime',]
        
        known_functions=[funct.lower() for funct in known_functions]
        matches = [match for match in matches if match not in known_functions]

        # Remove duplicates
        unique_variables = tuple(set(matches))
        unique_variables = f"({','.join(unique_variables)})".replace("(","").replace(")","")

        return unique_variables

    f_string=f
    x_variables=sym.symbols(extract_variables(f))
    def normalizador(f):
        f=f.replace(";",",").replace("^","**").replace("#", "").replace("sen","sin").replace("i","I").replace("X","x").replace("sIn","sin")
        f="["+f+"]"
        f=f.replace("[[","[").replace("]]","]")
        return f
    
    f=sym.lambdify(x_variables,normalizador(f),"numpy")
        
    def fun(x):
        if len(f_string.split(","))==1 and len(str(x_variables).split(","))==1: 
            return np.array(*f(np.array(x).T)).T
        elif len(f_string.split(","))==1:
            return np.array(*f(*np.array(x).T)).T 
        else:            
            return np.array(f(*np.array(x).T)).T
    return fun


def system_solve_ivp(f,x):
    """f: Transform string to numpy function that can be used in solve_ivp

    Args:
        f (string): List of strings, say "[x0+x1,x0*x1]" or x0+x1,x0*x1, that represent the system of the form dx/dt=f(t,x)
        x (string): List of variables. Say "x0 x1 x2" or "x0,x1,x2"... You should not input t in this.
    Returns:
        Function of the form f(t,x) in numpy.
    Example:
        f="[x1+x0,x1*x0]"
        x="x0 x1"
        system=system_solve_ivp(f,x)
        approx_sol=solve_ivp(system,[0,3],[3,4],dense_output=True,atol=1e-200,rtol=1e-13)
        print(approx_sol)
    """
    x=sym.symbols(x)
    def normalizador(f):
        f=f.replace(";",",").replace("^","**").replace("#", "").replace("sen","sin").replace("i","I").replace("X","x").replace("sIn","sin")
        f="["+f+"]"
        f=f.replace("[[","[").replace("]]","]")
        return f

    f=sym.lambdify(x,normalizador(f),"numpy")
    def fun(t,x):
        return np.array(f(*np.array(x)))
    return fun


def find_root(f,x,method="hybr",xtol=1e-13,maxiter=1000,ftol=1e-13,tol=1e-13):
    """root_system Find the root

    Args:
        f: Basically, the system you want to be equal to 0
        x: The variables that said system has, written in the form "x y z" or "x,y,z".
        method: Choose method "hybr","broyden1","lm","anderson", more details scipy.optimize.root()
    Returns:
        Solution of the problem (if exists). If it does not converge, it will return None. See scipy.optimize.root() 
    Example:
        f="y*z-2,y-z+3"
        x="y z"
        sol=find_root(f,x)
        print(sol)

    """

    def string_to_fun(f,x):
        """f: Transform string to numpy function

        Args:
            f (string): List of strings, say "[x0+x1,x0*x1]". This shall be written as such even if you only got 1 function!
            x (string): List of variables. If we have more variables than in f.
            the order in which we input them in x will be the order in which the.
            generated numpy function will considered them.

        Returns:
            Numpy function. Note that it work like this, if f is takes two arguments like x0,x1, then you can input
            functions like np.array([[2,2,4],[2,5,4]]). Note that always the input is 1 numpy array. 
        
        Example:
            f=string_to_fun("x1+x0-x2+x3+x4","x0 x1 x2 x3 x4")
            s=np.array([[2,2,4],[2,5,4],[2,3,5],[2,2,4],[2,5,4]])
            print(f(s))

            f=string_to_fun("exp(x1)","x1")
            print(f([2,32,32,3,32]))

            f=string_to_fun("exp(x1),x1+x2","x1 x2")
            print(f([[2,4,3],[4,2,3]]))
        """
        f_string=f
        x_variables=sym.symbols(x)
        def normalizador(f):
            f=f.replace(";",",").replace("^","**").replace("#", "").replace("sen","sin").replace("i","I").replace("X","x").replace("sIn","sin")
            f="["+f+"]"
            f=f.replace("[[","[").replace("]]","]")
            return f
        
        f=sym.lambdify(x_variables,normalizador(f),"numpy")
            
        def fun(x):
            if len(f_string.split(","))==1 and len(str(x_variables).split(","))==1: 
                return np.array(*f(np.array(x)))
            elif len(f_string.split(","))==1:
                return np.array(*f(*np.array(x)))  
            else:
                return np.array(f(*np.array(x)))
        return fun
    f=string_to_fun(f,x)
    for i in range(1000):
        try:
            if method=="hybr":
                sol=root(f,np.random.uniform(-100,100,size=max(len(x.split(" ")),len(x.split(",")))), tol=tol,method=method,options={"xtol":xtol})
                warnings.filterwarnings("ignore", category=RuntimeWarning)
            if method=="broyden1":
                sol=root(f,np.random.uniform(-100,100,size=max(len(x.split(" ")),len(x.split(",")))),tol=tol,
                         method=method,options={"ftol":ftol,"xtol":xtol,"maxiter":maxiter})
            if method=="lm":
                sol=root(f,np.random.uniform(-100,100,size=max(len(x.split(" ")),len(x.split(",")))),tol=tol,
                         method=method,options={"ftol":ftol,"xtol":xtol,"maxiter":maxiter})
                warnings.filterwarnings("ignore", category=RuntimeWarning)
            if method=="anderson":
                sol=root(f,np.random.uniform(-100,100,size=max(len(x.split(" ")),len(x.split(",")))),tol=tol,
                         method=method,options={"ftol":ftol,"xtol":xtol,"maxiter":maxiter})
                warnings.filterwarnings("ignore", category=RuntimeWarning)
            if sol.success is True:
                return sol
            if i==1000:
                return print("No solution find in 1000 repetitions of the algorithm.")
                break

        except Exception:
            pass
 

def find_fit_curve(f,x,b,xdata,ydata,p0=None,bounds=(-np.inf,np.inf)):
    """find_fit_curve: Basically, the same stuff as fit_curve from scipy, but with a few less options, but with
    the posibility to input into in f (the function), x(the variables) and b(the coefficients to estimate)

    Args:
        f (string): x*a+y*b+z*c"
        x (string): "x y z" or "x,y,z". Note, always if you only got 1 variable use x="x,", not x="x"! 
        b (string ): "a,b,c".  Note, always if you only got 1 coefficient use b="b,", not b="b"
        xdata (np.array): must be same as number of variables,x, and must be in the form [[2,3,4],[3,2,3],[43,4,3]]. Note that if 
        it is just 1 variable, it must be [[2,3,4,5,6,7 ]] rather than [3,2,1,2]
        ydata (np.array): same as before, but here 1 dimensional array of same size as xdata
        p0 (tuple): Default None. This is basically the initial value of algirthm. Must be tuple or list same size b.

    Returns:
        Return Dictionary with coefficients,coefficiens covariance matrix, and function with the coefficients changed into it. This function
        will take values in the same format as the other one, i.e., [[2,3,4],[3,2,3],[43,4,3]], and will have as many arguments as variables.
    Example:
        f = "b*x**a+c+y"
        x = "x,y"
        b = "a,b,c"
        xdata=np.random.uniform(0,1,size=(300,2))
        ydata=np.random.uniform(0,1,size=300)
        sol=find_fit_curve(f,x,b,xdata,ydata)

        print(sol["fitted_fun"]([3,3]))
    """
    if p0 is None:
        p0=(1,)*max(len(b.split(" ")),len(b.split(",")))

    def fun_fit_curve(f, x, b):
        f_string = f
        x_variables = sym.symbols(x)
        b_variables = sym.symbols(b)
        incognitas = (*x_variables, *b_variables)

        def normalizador(f):
            f = f.replace(";", ",").replace("^", "**").replace("#", "").replace("sen", "sin").replace("i", "I").replace(
                "X", "x").replace("sIn", "sin")
            f = "[" + f + "]"
            f = f.replace("[[", "[").replace("]]", "]")
            return f

        f = sym.lambdify(incognitas, normalizador(f), "numpy")

        def fun(x, *b):
            return np.array(*f(*x.T, *b))

        return fun
    def fun_fitted_curve(f, x, b,b_estimated):
        f_string = f
        x_variables = sym.symbols(x)
        b_variables = sym.symbols(b)
        swap=dict(zip(b_variables,b_estimated))
        def normalizador(f):
            f = f.replace(";", ",").replace("^", "**").replace("#", "").replace("sen", "sin").replace("i", "I").replace(
                "X", "x").replace("sIn", "sin")
            return f
        f=sym.sympify(f).subs(swap)
        f = sym.lambdify(x_variables, f, "numpy")

        def fun(x):
            return np.array(f(*np.array(x).T))

        return fun

    fun = fun_fit_curve(f, x, b)

    #a(xdata,3)
    sol = curve_fit(fun, xdata=xdata, ydata=ydata, p0=p0,bounds=bounds)
    fitted_fun=fun_fitted_curve(f,x,b,sol[0])
    output={"coeffs":sol[0],"coeffs_cov":sol[1],"fitted_fun":fitted_fun}
    return output
