#======================================================================
# Github: https://github.com/thjsimmons
#======================================================================

# Reflects 

import numpy as np
from scipy.integrate import odeint
from numpy import sin, cos, pi, array
import matplotlib.pyplot as plt

def refl(x): # # Enter curve y(x) to reflect:
    return 0*x

def curv(x): # Enter curve y(x) to reflect about:
    return x**2

def normline_params(xp): # Parameters of normal line to curve:
    h = 0.001
    return (xp, curv(xp), (curv(xp + h) - curv(xp)) / h)

def normline(params, x): # Equation of normal line to curve
    xp, yp, m = params
    return yp - 1.0/m * (x - xp) # point slope form 

def roots(params, x): # Bisection method on this equation: 
    return refl(x) - normline(params, x)

def bisect(params, tol): # Bisection method for finding intersection of normal line and refl(x)
    error = 9999
    x_left = -100.0
    x_right = 100.0
    count = 0
    
    while error > tol and count < 20:
        x_avg = (x_left + x_right)/2.0
        y_avg = roots(params, x_avg)
        y_left = roots(params, x_left)
        y_right = roots(params, x_right)

        if y_left * y_avg < 0:
            x_right = x_avg
        elif y_right * y_avg < 0:
            x_left = x_avg
        elif y_avg == 0:
            print("Exact solution")

        error = abs(y_avg)
        count += 1
    
    return x_avg

def reflect_point(xc, xr): # reflects point over another point 
    return (2*xc-xr, 2*curv(xc)-refl(xr))

def reflect_curve(xs, bounds): # reflect y1(x) about y2(x)
    
    rxs = np.zeros(xs.shape[0])
    rys = np.zeros(xs.shape[0])
   
    for i in range(len(xs)):
        (rxs[i], rys[i]) = reflect_point(xs[i], bisect(normline_params(xs[i]), 0.001))
    
    return (rxs, rys)

fig = plt.figure()       
plt.gca().set_aspect('equal', adjustable='box')
plt.draw()
plt.xlim([-5, 5])
plt.ylim([-5, 5])
bounds = (-5, 5)

xs = np.linspace(bounds[0], bounds[1], 10000)
plt.plot(xs, refl(xs), label='hi')
plt.plot(xs, curv(xs), label='hi')
plt.title("Y=0 reflected over Y=x^2")
rs = reflect_curve(xs, bounds)
plt.plot(rs[0], rs[1], label='hi')
#plt.title("Y = 0 reflected over Y = x^2")
fig.savefig("reflect.jpg")
plt.show()

