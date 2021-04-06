import numpy as np
from scipy.integrate import odeint
from numpy import sin, cos, pi, array
import matplotlib.pyplot as plt


# for time in on line
# get vector, reflect point to new curve

def dist2(A, B):
    return np.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)

def reflectPoint(P, curve):
    curve_x, curve_y= curve
    
    minval = 999999
    mindex = -1
    
    for index in range(curve_x.shape[0]):
        val = dist2(P, (curve_x[index], curve_y[index]))
            
        if val < minval:
            minval = val
            mindex = index

    Q = (curve_x[mindex], curve_y[mindex]) # (x,y)

    PQ = (P[0] - Q[0], P[1] - Q[1])
    return (Q[0] - PQ[0], Q[1] - PQ[1])
        
            
            
def reflectCurve(line, para):
    line_x, line_y = line
    para_x, para_y = para
    
    ref_x = np.zeros(line_x.shape[0])
    ref_y = np.zeros(line_y.shape[0])

    for i in range(line_x.shape[0]):
        P = (line_x[i], line_y[i])
        R = reflectPoint(P, para)
        ref_x[i] = R[0]
        ref_y[i] = R[1]
    
    return (ref_x, ref_y)


t = np.linspace(-20, 20, 1200)
angle = 0 #np.pi / 4

#############
x_of_t = t
y_of_t = t

#############
line_x = x_of_t*np.cos(angle) - y_of_t*np.sin(angle)
line_y = x_of_t*np.sin(angle) + y_of_t*np.cos(angle)
#############
x_of_t = t
y_of_t = np.sin(t)

#y_of_t = np.power(time,2)
#############
para_x = x_of_t*np.cos(angle) - y_of_t*np.sin(angle)
para_y = x_of_t*np.sin(angle) + y_of_t*np.cos(angle)

  
plt.gca().set_aspect('equal', adjustable='box')
plt.draw()
print(np.power(3,2))
plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.plot(line_x, line_y, label='hi')
plt.plot(para_x, para_y, label='hi')

ref_x, ref_y = reflectCurve((line_x, line_y), (para_x, para_y))

plt.plot(ref_x, ref_y, label='hi')
plt.show()
