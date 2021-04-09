#======================================================================
# Github: https: //github.com/thjsimmons
#======================================================================

# Reflects a parametric curve A(t) = (x(t), y(t)) about another parametric curve B(t)

'''
    Program may be missing full complexity of reflection function, e.g. when there are multiple reflections

'''
import numpy as np
from scipy.integrate import odeint
from numpy import sin, cos, pi, array
import matplotlib.pyplot as plt

def RM(angle): # Rotation Matrix 
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

t = np.linspace(-20.0, 20.0, 1000)

# Enter Parametric Equation to reflect:

def eq_line(t):
    return np.array([t, np.cos(2*t)])

Lt = RM(0).dot(eq_line(t)).T

# Enter Parametric equation to reflect about:

def eq_para(t):
    return np.array([np.cos(t), np.sin(t)])

Pt = RM(0).dot(eq_para(t)).T

def dist2(A, B):
    return np.linalg.norm(A-B)

def reflectPoint(P, Ct): # minimize distance between curves to find point to reflect:

    minval = 99999
    mindex = -1

    t = np.arange(-20, 20, 0.01)

    for index in range(t.shape[0]):
        val = dist2(P, eq_para(t[index])) # should be indexed in reverse 
            
        if val < minval:
            minval = val
            mindex = index # find minimum on equations of curves 
   
    return 2*eq_para(t[mindex])-P

         
def reflectCurve(line, para): 
    rs = np.zeros(line.shape) 
    
    for i in range(rs.shape[0]):
        rs[i] = reflectPoint(line[i], para)
    
    return rs

fig = plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
plt.draw()
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.title("(t, cos(2t)) reflected over circle of radius 1")
plt.plot(Lt[:,0], Lt[:,1], label='hi')
plt.plot(Pt[:,0], Pt[:,1], label='hi')

Rt = reflectCurve(Lt, Pt) # here

plt.plot(Rt[:,0], Rt[:,1], label='hi')
fig.savefig("reflectParametric.jpg")
plt.show()
