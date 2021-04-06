import numpy as np
from scipy.integrate import odeint
from numpy import sin, cos, pi, array
import matplotlib.pyplot as plt

#
# Steps Needed:
#   
#   1) Define curve curv(x)
#   2) 
#   3) 
#   4) 
#


def curv(x):
    return np.sin(x)

def refl(x):
    return x

def normline_params(xp): # y = mx + b passing through (xp,yp) on curv(x)
    h = 0.01
    m = (curv(xp + h) - curv(xp)) / h # instantaneous derivative 
    yp = curv(xp)
    return (xp, yp, m)

def normline(params, x):
    xp, yp, m = params
    return yp + -1/m * (x - xp) # point slope form 

def roots(params, x):
    return refl(x) - normline(params, x)

def bisect(params, tol):
    error = 9999
    
    #params = normline_params(3) #####
   
    x_left = -100
    x_right = 100

    count = 0
    
    while error > tol and count < 20:
        x_avg = (x_left + x_right)/2
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
        #print("finished")
        count += 1
    
    return x_avg

def reflect_point(xc, xr):
    yr = refl(xr)
    yc = curv(xc)
    return (2*xc-xr, 2*yc-yr)

def reflect_curv(xvals, bounds):
    
    curv_vals = curv(x_vals)
    refl_vals = refl(x_vals)
    res_x_vals = np.zeros(xvals.shape[0])
    res_y_vals = np.zeros(xvals.shape[0])
   
    for i in range(len(x_vals)):
        
        x = x_vals[i]
        params = normline_params(x)
        x_sect = bisect(params, 0.01)
        (x_res, y_res) = reflect_point(x, x_sect)
        res_x_vals[i] = x_res
        res_y_vals[i] = y_res
        
    
    return (res_x_vals, res_y_vals)
        
    
plt.gca().set_aspect('equal', adjustable='box')
plt.draw()
plt.xlim([-5, 5])
plt.ylim([-5, 5])

bounds = (-20, 20)
x_vals = np.linspace(bounds[0], bounds[1], 1000)

# Norm Line Test:
#x_test = 0.4
#params = normline_params(x_test)
#x_sect = bisect(params, 0.05)
#plt.plot([x_sect], [refl(x_sect)], 'ro')
#plt.plot(x_vals, normline(params, x_vals), label='hi')


############

plt.plot(x_vals, refl(x_vals), label='hi')
plt.plot(x_vals, curv(x_vals), label='hi')

res = reflect_curv(x_vals, bounds)
plt.plot(res[0], res[1], label='hi')

###########
plt.show()
