import matplotlib.animation as animation
from scipy.integrate import odeint
from numpy import arange
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sqrt, sin, pi


## Numerical solution parameter
span = 20               # length of the time period over which solution is computed
N = 1000                     # number of time steps in 1 second
t_step = 1/N

## Physical properties of the system
m = 1                       # mass of each link in kg (both links have the same mass)
l = 1                       # length of each link in meters (both links have the same length)
g = 9.81                    # acceleration due to gravity - m/s*s 
I = (1/12)*m*l*l
gamma = I/m                 # Just a value created for easier calculations

## Initial conditions of the system
theta1 = pi/10              # the angle the first link makes with a vertical - radians
theta2 = 0                  # the angle the second link makes with a vertical - radians
q = 0                       # initial position of the cart
theta1_dt = 0               # acceleration of the first angle
theta2_dt = 0               # acceleration of the second angle
q_dt = 0.01                    # initial acceleration of the cart 

## Potential well modification coefficients
A_base = (3/2)*g*l          # coefficient responsible for canceling out term with cos(theta1)
B_base = (1/2)*g*l          # coefficient responsible for canceling out term with cos(theta2)
A_s = -5                     # strength of the imposed potential for A
B_s = -5                     # strength of the imposed potential for B
## Desired potential cancelation terms to be applied to the system
A = A_base*A_s
B = B_base*B_s


## Function to describe behaviur of the system
def double_pendulum_system(state, t):
    theta1, theta2, q, theta1_dt, theta2_dt,  q_dt = state
    ## Create empty 6x6 matrix of differential equation coefficieants
    deq_mat = np.eye(6,6)
    ## Fill in the matrix
    deq_mat[0,0] = cos(theta1-theta2)*l*l
    deq_mat[0,1] = 1/2*l*l+2*gamma
    deq_mat[0,2] = l*cos(theta2)
    deq_mat[0,3] = -l*l*sin(theta1-theta2)
    deq_mat[1,0] = 5/2*l*l +2*gamma
    deq_mat[1,1] = l*l*cos(theta1-theta2)
    deq_mat[1,2] = 3*l*cos(theta1)
    deq_mat[1,4] = -l*l*sin(theta2-theta1)
    deq_mat[2,0] = 3*l/4*cos(theta1)
    deq_mat[2,1] = l/4*cos(theta2)
    deq_mat[2,3] = -theta1*(3/4)*l
    deq_mat[2,4] = -theta2*(1/4)*l
    ## Fill in a resulting vector
    b = np.zeros(6)
    b[0] = sin(theta2)*(l*g-2*B)
    b[1] = sin(theta1)*(3*l*g-2*A)
    b[3] = theta1_dt*theta1_dt
    b[4] = theta2_dt*theta2_dt
    b[5] = q_dt
    ## Solve the matrix
    deq_mat_inv = np.linalg.inv(deq_mat)
    dxdt = np.matmul(deq_mat_inv, b)
    dxdt[4] = sqrt(dxdt[4])
    dxdt[3] = sqrt(dxdt[3])
    return dxdt


## Solve ODEs numerically
t = arange(0,span,t_step)
init_state = [theta1, theta2, q, theta1_dt, theta2_dt, q_dt]
state = odeint(double_pendulum_system, init_state, t)
print(state)
## Plot the solution
#Plot theta1
fig1 = plt.figure()
plt.xlabel('Time')
plt.ylabel('Theta 1')
plt.plot(t, state[:, 0])
#Plot theta2
fig1 = plt.figure()
plt.xlabel('Time')
plt.ylabel('Theta 2')
plt.plot(t, state[:, 1])
#Plot q
fig1 = plt.figure()
plt.xlabel('Time')
plt.ylabel('Cart Position')
plt.plot(t, state[:, 2])
#plt.show()


## Animating behaviour
x1 = l*sin(state[:, 0])
y1 = l*cos(state[:, 0])

x2 = l*sin(state[:, 1]) + x1
y2 = l*cos(state[:, 1]) + y1

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*t_step))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(state)),
                              interval=25, blit=True, init_func=init)

# ani.save('double_pendulum.mp4', fps=15)
plt.show()