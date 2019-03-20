import matplotlib.animation as animation
from scipy.integrate import odeint
from numpy import arange
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sqrt, sin, pi


## Numerical solution parameter
span = 10                       # length of the time period over which solution is computed
N = 60                         # number of time steps in 1 second
t_step = 1/N    
plot_flag = 0                   # 0 dont plot angles and position vs time
animation_flag = 1              # 0 don't show animation
## Physical properties of the system
m = 1                           # mass of each link in kg (both links have the same mass)
l = 1                           # length of each link in meters (both links have the same length)
g = 9.81                        # acceleration due to gravity - m/s*s 
I = (1/12)*m*l*l    
gamma = I/m                     # Just a value created for easier calculations

## Initial conditions of the system
theta1 = pi/20                  # the angle the first link makes with a vertical - radians
theta2 = pi/20                  # the angle the second link makes with a vertical - radians
q = 0                           # initial position of the cart
theta1_dt = 0                   # acceleration of the first angle
theta2_dt = 0                   # acceleration of the second angle
q_dt = 0                        # initial acceleration of the cart 

## Potential well modification coefficients
A_base = (3/2)*g*l              # coefficient responsible for canceling out term with cos(theta1)
B_base = (1/2)*g*l              # coefficient responsible for canceling out term with cos(theta2)
A_s = 10                         # strength of the imposed potential for A
B_s = 10                         # strength of the imposed potential for B
## Desired potential cancelation terms to be applied to the system
A = A_base*A_s
B = B_base*B_s


## Function to describe behaviur of the system
def double_pendulum_system(state, t):
    theta1, theta2, q, p_theta1, p_theta2,  p_q = state
    ## Create empty 3x63 matrix of differential equation coefficieants for generalized momentum
    deq_mat = np.eye(3,3)
    ## Fill in the matrix for the generalized momentum
    deq_mat[0,0] = (5/2)*l*l+2*gamma
    deq_mat[0,1] = l*l*cos(theta1-theta2)
    deq_mat[0,2] = 3*l*cos(theta1)
    deq_mat[1,0] = l*l*cos(theta1-theta2)
    deq_mat[1,1] = (1/2)*l*l+2*gamma
    deq_mat[1,2] = l*cos(theta2)
    deq_mat[2,0] = 3*l*cos(theta1)
    deq_mat[2,1] = l*cos(theta2)
    deq_mat[2,2] = 4
    ## Fill in a resulting vector
    b = np.zeros(3)
    b[0] = p_theta1
    b[1] = p_theta2
    b[2] = p_q
    ## Solve the matrix for theta1 dot, theta 2 dot and q dot
    # To get coefficients lets invert the matrix
    deq_mat_inv = np.linalg.inv(deq_mat)
    ## Differential equation vector for the theta1 dot, theta2 dot, and q dot
    th1_th2_q_dt = (2/m)*np.matmul(deq_mat_inv, b)
    ## Differential equations for the generalized momentum
    p_theta1_dt = (1/2)*m*(-l*l*th1_th2_q_dt[0]*th1_th2_q_dt[1]*sin(theta1-theta2) -\
                 3*l*th1_th2_q_dt[2]*th1_th2_q_dt[0]*sin(theta1))+(3/2)*m*g*l*sin(theta1)-A*m*sin(theta1)
    p_theta2_dt = (1/2)*m*(l*l*th1_th2_q_dt[0]*th1_th2_q_dt[1]*sin(theta1-theta2) - \
                 l*th1_th2_q_dt[2]*th1_th2_q_dt[1]*sin(theta2))+(1/2)*m*g*l*sin(theta2)-B*m*sin(theta2)
    p_q_dt = 0
    return np.append(th1_th2_q_dt, [p_theta1_dt, p_theta2_dt, p_q_dt])


## Solve ODEs numerically
t = arange(0,span,t_step)
## Initial generalized momentum calculation
p_theta1 = (1/2)*m*((5/2)*l*l*theta1_dt + l*l*theta2_dt*cos(theta1-theta2)) + gamma*m*theta1_dt
p_theta2 = (1/2)*m*((1/2)*l*l*theta2_dt + l*l*theta1_dt*cos(theta1-theta2)) + gamma*m*theta2_dt
p_q = (1/2)*m*(4*q_dt + 3*l*theta1_dt*cos(theta1) + l*cos(theta2)*theta2_dt)
## initial state summary
init_state = [theta1, theta2, q, p_theta1, p_theta2, p_q]
state = odeint(double_pendulum_system, init_state, t)


## Calculate energy
E = np.ones(len(state))

## Plot the solution
if (plot_flag == 1):
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
    plt.show()


## Animating behaviour
if (animation_flag == 1):
    x1 = l*sin(state[:, 0])+state[:,2]
    y1 = l*cos(state[:, 0])

    x2 = l*sin(state[:, 1]) + x1
    y2 = l*cos(state[:, 1]) + y1

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-3, 3), ylim=(-3, 3))
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    energy_template = 'Energy = %.1fJ'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    energy_text = ax.text(0.05, 0.85, '', transform=ax.transAxes)


    def init():
        line.set_data([], [])
        time_text.set_text('')
        energy_text.set_text('')
        return line, time_text, energy_text


    def animate(i):
        thisx = [state[i,2], x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*t_step))
        energy_text.set_text(energy_template % E[i])
        return line, time_text, energy_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(state)),
                                interval=25, blit=True, init_func=init)
    # ani.save('double_pendulum.mp4', fps=15)
    plt.show()