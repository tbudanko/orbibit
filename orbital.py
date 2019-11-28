"""
===========================
Orbital Mechanics Simulator
===========================

"""

import numpy as np
import scipy.integrate as integrate
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


class particle:
    def __init__(self, space, m):
        """ Mass particle constructor
            Attributes: mass, velocity vector, position vector, acceleration vector."""

        # Mass attribute
        self.m = m

        # Add particle object to space object.
        space.sysP = np.append(space.sysP, self)

        # Position, velocity, acceleration vectors as f(t)
        # r, v, a logs.
        self.r = np.array([[0, 0, 0]])
        self.v = np.array([[0, 0, 0]])
        self.a = np.array([[0, 0, 0]])

    def position(self, rP = [0, 0, 0]):
        """Modifies particle position vector.
            Used to set initial conditions."""
        self.r[-1] = np.array(rP)

    def velocity(self, vP = [0, 0, 0]):
        """Modifies particle velocity vector.
            Used to set initial conditions."""
        self.v[-1] = np.array(vP)

class space:
    def __init__(self):
        """Space constructor
            Computational grid, time-stepper, integrator and plotter.
            sysP - list of system mass particle objects"""

        # Particle objects list. Particle constructor adds object to list.
        self.sysP = np.array([])

        # System time vector initialization
        self.t = np.array([0])

        # Universal gravitational constant
        self.G = 6.674e-11

    def gravitate(self, t, stateR):
        """Receives current state of system, return system derivatives.
            Called by integrator.
            state vector - [r1, v1, r2, v2, ...]
            stateDel vector - [dr/dt, dv/dt] = [v, Fg(r)]"""

        # State derivatives of system particles
        # [dr/dt_1, dv/dt_1, dr/dt_2, dv/dt_2, ...]
        # Initialized as state to get the dimensions right.
        state = stateR.reshape(int(len(stateR) / 3), 3) #Reshaped for easier vector algebra
        stateDel = np.zeros((int(len(stateR) / 3), 3))


        for indP in range(0, int(len(state) / 2)):
            p = self.sysP[indP]

            # Velocity vector of particle P
            stateDel[2*indP] = state[2*indP + 1]

            for indQ in range(0, int(len(state) / 2)):
                q = self.sysP[indQ]

                if indQ != indP:
                    # Acceleration vector of particle P
                    stateDel[2*indP + 1] += - self.G * q.m / \
                                            (np.linalg.norm(state[2*indP] - state[2*indQ]) + 0.00001)**3 * \
                                            (state[2*indP] - state[2*indQ])

                else:
                    pass

            # Record acceleration to a log.
            if t==0:
                p.a[-1] = stateDel[2*indP + 1]
            else:
                p.a = np.append(p.a, np.array([stateDel[2*indP + 1]]), axis = 0)

        #print(stateDel)
        return stateDel.reshape(1, len(stateR))[0]


    def runt(self, tDur = 5000):
        """Integrator
            Receives required simulation duration.
            Integrates trajectories and records data to particle logs r, v, a logs."""

        # Initial value state vector
        state0 = np.zeros((2 * len(self.sysP), 3))
        for indP in range(0, len(self.sysP)):
            state0[2*indP] = self.sysP[indP].r[-1]
            state0[2*indP + 1] = self.sysP[indP].v[-1]
        state0 = state0.reshape(1, 6 * len(self.sysP))[0]


        # Trajectory integration solution object.
        data = integrate.solve_ivp(self.gravitate, [self.t[-1], self.t[-1] + tDur], state0, vectorized = True, max_step = 1)

        self.t = np.append(self.t, data.t[1:])

        for indP in range(0, len(self.sysP)):
            self.sysP[indP].r = np.append(self.sysP[indP].r, np.transpose(data.y[6 * indP : 6 * indP + 3, 1:]), axis = 0)
            self.sysP[indP].v = np.append(self.sysP[indP].v, np.transpose(data.y[6 * indP + 3: 6 * indP + 6, 1:]), axis = 0)

    def run(self, tDur = 5000, max_step = 1):
        """Integrator
            Receives required simulation duration.
            Integrates trajectories and records data to particle logs r, v, a logs."""

        # Initial value state vector
        state0 = np.zeros((2 * len(self.sysP), 3))
        for indP in range(0, len(self.sysP)):
            state0[2*indP] = self.sysP[indP].r[-1]
            state0[2*indP + 1] = self.sysP[indP].v[-1]
        state0 = state0.reshape(1, 6 * len(self.sysP))[0]


        # Trajectory integration solution object.
        integrator = integrate.RK45(self.gravitate, self.t[-1], state0, self.t[-1] + 500000, max_step = max_step, vectorized = True)

        integrator.step()

        t = self.t[-1]
        while t < tDur:
            integrator.step()
            for indP in range(0, len(self.sysP)):
                self.sysP[indP].r = np.append(self.sysP[indP].r, np.array([integrator.y[6 * indP: 6 * indP + 3]]), axis = 0)
                self.sysP[indP].v = np.append(self.sysP[indP].v, np.array([integrator.y[6 * indP + 3: 6 * indP + 6]]), axis = 0)
            t += integrator.step_size
            self.t = np.append(self.t, self.t[-1] + integrator.step_size)



    def tplot(self):
        """Plotting method."""

        figure = plt.figure()

        ax = plt.axes(projection = '3d')
        for p in self.sysP:
            ax.plot(p.r[:, 0], p.r[:, 1], p.r[:, 2])
        plt.show()






s = space()

p1 = particle(s, 1e24)
p1.position([0, 0, 0])
p1.velocity([0, 0, 0])

p2 = particle(s, 1e24)
p2.position([1e7, 0 ,3e6])
p2.velocity([0, 4000, 0])

p3 = particle(s, 1e24)
p3.position([5e6, -2e6 ,-3e6])
p3.velocity([1000, -1000, 0])


s.run(10000, 10)
s.tplot()
