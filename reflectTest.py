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
    print(Q)
    PQ = (P[0] - Q[0], P[1] - Q[1])
    print(PQ)
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


time = np.linspace(-1, 1, 100)
angle = np.pi / 4
line_x = time*np.cos(angle)
line_y = time*np.sin(angle)
para_x = time*np.cos(angle) + np.power(time,2)*np.sin(angle)
para_y = time*np.sin(angle) - np.power(time,2)*np.cos(angle)

plt.gca().set_aspect('equal', adjustable='box')
plt.draw()
print(np.power(3,2))
plt.xlim([-2, 2])
plt.ylim([-2, 2])

P = (-1, 1)
Q = reflectPoint(P, (line_x, line_y))
print(Q)
plt.plot(line_x, line_y, label='hi')
plt.plot([P[0]], [P[1]], 'ro')
plt.plot([Q[0]], [Q[1]], 'ro')
#plt.plot(para_x, para_y, label='hi')

#ref_x, ref_y = reflectCurve((line_x, line_y), (para_x, para_y))

#plt.plot(ref_x, ref_y, label='hi')
plt.show()
