from particleEngine.src.particle_box import ParticleBox
from matplotlib import animation, cm
import numpy as np
import matplotlib.pyplot as plt

def run():
    f=ParticleBox()
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0)

    ax.set_aspect('equal','box')
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    ax.grid()
    ax.set_xlabel("x ")
    ax.set_ylabel("y ")

    markersize = int(fig.dpi * f.psize * fig.get_figwidth()/ np.diff(ax.get_xbound())[0])
    # Note: I have adjusted the axis scale using subplot_adjust()function
    point1, = ax.plot([], [], 'ro',ms=markersize)
    point2, = ax.plot([], [], 'ro',ms=markersize)
    point3, = ax.plot([], [], 'ro',ms=markersize)

    def init():
        point1.set_data([], [])
        point2.set_data([], [])
        point3.set_data([], [])
        return point1, point2, point3

    def run(i):

        point1.set_xdata(f.particles[0].state[0])
        point1.set_ydata(f.particles[0].state[1])

        point2.set_xdata(f.particles[1].state[0])
        point2.set_ydata(f.particles[1].state[1])

        point3.set_xdata(f.particles[2].state[0])
        point3.set_ydata(f.particles[2].state[1])

        f.step(0.02)
        return point1, point2, point3,

    ani = animation.FuncAnimation(fig, run,interval=20,  blit=True,init_func=init)
    plt.show()

if __name__ == '__main__':
    run()