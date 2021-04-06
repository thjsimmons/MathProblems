
import numpy as np
from scipy.integrate import odeint
from numpy import sin, cos, pi, array
import matplotlib.pyplot as plt

def path(theta):
    # return x, y
    # ends when x final
    a = 15
    b = 25
    u = 20
    m = 1
    f = 10
    g = 10

    ta = a /(u * cos(theta))
    tb = b /(u * cos(theta))

    ya = u*sin(theta)*ta - 0.5*g*ta**2
    ya_dot = u*sin(theta) - g*ta

    yb = ya + ya_dot*tb + 0.5*(f-g)*tb**2
    yb_dot = ya_dot + (f-g)*tb
    
    time = np.linspace(0.0, 10.0, 1000)
    
    x = np.zeros(time.shape[0])
    y = np.zeros(time.shape[0])
   
    for i in range(time.shape[0]):
        t = time[i]
        x_t = u * cos(theta) * t
        x[i] = x_t

        if x_t < a:
            y_t = u*sin(theta)*t - 0.5*g*t**2
            y[i] = y_t
        elif a < x_t < b:
            y_t = ya + ya_dot*(t-ta) + 0.5*(f-g)*(t-ta)**2
            y[i] = y_t
            
        else:
            y_t = yb + yb_dot*(t-tb-ta) - 0.5*g*(t-tb)**2
            y[i] = y_t
            if y_t < 0:
                y[i]  = 0
        
    return (x,y)

def x_resample(path):
    # Resample so that # of points is a function of the x range
    (x_init, y_init) = path
    x_range = np.max(x_init)
    x_new = np.linspace(0.0, 60, 1000)
    y_new = np.zeros(x_new.shape[0])

    count = 1

    y_new[0] = y_init[0]
    y_new[-1] = y_init[-1]
    
    for i in range(1000-1):
        slope = (y_init[i+1]-y_init[i])/(x_init[i+1]-x_init[i])
        if count == 1000:
            break
        while x_new[count]< x_init[i+1]:
            y_new[count] = y_init[i] + slope * (x_new[count]-x_init[i])
            count += 1
            if count == 1000:
                break
            
    return (x_new, y_new)

(x,y) = path(0.5)
#t = np.linspace(0.0, 10, 1000)
#x = t
#y = np.sin(t) * np.cos(5*t)

(x_res, y_res) = x_resample((x,y))

print("x length = ", x.shape[0])
#print("x res length = ", x_res.shape[0] )
      
plt.xlim([0, 60])
plt.ylim([-20, 20])

plt.plot(x, y, 'r', label='hi')
plt.plot(x_res, y_res, 'k', label='hi')

plt.show()
