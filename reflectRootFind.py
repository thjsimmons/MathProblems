import numpy as np
from scipy.integrate import odeint
from numpy import sin, cos, pi, array
import matplotlib.pyplot as plt


# for time in on line
# get vector, reflect point to new curve

def dist2(A, B):
    return np.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)

# get equation of line perpindicular to curve then root find with the other curve, get
# the separation vector and reflect it

def deriv(time, curve):
    t = time 
    x, y = curve # not evenly
    
    x_dot = np.zeros(t.shape[0])
    y_dot = np.zeros(t.shape[0])
    
    for i in range(1, t.shape[0]):
        x_dot[i] = (x[i] - x[i-1]) / (t[i] - t[i-1])
        y_dot[i] = (y[i] - y[i-1]) / (t[i] - t[i-1])
        
    return (x_dot, y_dot) # negative reciprocal is perp (-ydot, x_dot)

def lineParams(index, N, time, dcurve, curve):
    tp = time[index]
    x, y = curve
    xdot, ydot = dcurve
    
    xp = x[index]
    yp = y[index]
    xp_dot = xdot[index] 
    yp_dot = ydot[index] 

    return tp, xp, yp, xp_dot, yp_dot

def perpLine(params, s):
    tp, xp, yp, xp_dot, yp_dot = params
    return (xp - yp_dot*(s - tp), yp + xp_dot*(s - tp))

def lineRoots(params, curve,  s): # vector bisection to (0, 0)
    # curve will be parametric equation of the curve
    lx, ly = perpLine(params, s) # find t on line that passes through curve 
    cx, cy = curve
    return (lx-cx, ly-cy)

def vector_bisect(func, tolerance): # tolerance is 0.01, different thing 
    # returns intersection point, need just function of line 
    error = 9999
    v_start = func(params, s_start) # (vx, vy)
    
    # (a,b) range
    s_back = -100
    s_front = 100
    
    # Bisection method:
    while error > tolerance:
        s_avg = (s_back + s_front)/2
        v_avg = func(s_avg) # lhRoots(x_avg, tclock, xp, yp) 
        v_back = func(s_back)
        v_front = func(s_front)

        if v_back[0] * v_avg[0] < 0 or v_back[1] * v_avg[1] < 0:
            s_front = s_avg
        elif v_front[0] * v_avg[0] < 0 or v_front[1] * v_avg[1] < 0:
            s_back = s_avg
        elif v_avg == (0,0):
            print "Exact solution"

        error = np.sqrt(v_avg[0]**2 + v_avg[1]**2) # switch to numpy, can just use np.norm()
    return s_avg             

def curve_eq(t):
    return t**2

def reflected_eq(t):
    return 0

# Use global variables in myRoots ? 
time = np.linspace(-5, 5, 1000) # time array
reflected = (time, 0)    # Curve to reflect through x^2 this m
curve = (time, time**2) # Curve x^2
dcurve = deriv(time, curve) # Derivative of x^2

index = 450 # index position on curve x^2
N = 1000
params = lineParams(index, N, time, dcurve, curve) # t value along the line itself) # 

def myRoots(s):
    # define curve here
    lx, ly = perpLine(params, s) # find t on line that passes through curve 
    cx, cy = reflected
    return (lx-cx, ly-cy)

s_avg = vector_bisect(p# (x,y) intersection point on reflected & pline 

plt.gca().set_aspect('equal', adjustable='box')
plt.draw()

plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.plot(curve[0], curve[1], label='hi')
plt.plot(pline[0], pline[1], label='hi')


plt.show()
