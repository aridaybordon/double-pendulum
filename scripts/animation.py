import matplotlib.pyplot as plt
from scripts.definitions import convert

from scripts.simulation import TIME, run_simulation
from matplotlib import animation

import json


def animate(z: list=[], from_simulation: bool=True, filename: str="double_pendulum.gif") -> None:
    # Animate a double pendulum simulation
    if from_simulation:
        # Obtain data from simulation
        pend1, pend2 = run_simulation(z0 = [0, 0, 3, 4], convert_data=True)
    else:
        # Obtain data from input
        pend1, pend2 = convert(z)

        pend1 = [[val[0] for val in pend1], [val[1] for val in pend1]]
        pend2 = [[val[0] for val in pend2], [val[1] for val in pend2]]


    x1, y1 = pend1
    x2, y2 = pend2

    FRAMES = len(x1)

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax  = plt.axes(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2))

    line1, = ax.plot([], [], lw=2, marker='o')
    line2, = ax.plot([], [], lw=2, marker='o')

    # Initialization function: plot the background of each frame
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2,


    # Animation function. This is called sequentially
    def animate(i):
        line1.set_data([0, x1[i]], [0, y1[i]])
        line2.set_data([x1[i], x2[i]], [y1[i], y2[i]])
        return line1, line2,


    # Call the animator.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=FRAMES, interval=33, blit=True)


    writergif = animation.PillowWriter(fps=30)
    anim.save(filename, writer=writergif)


def create_animation_from_prediction(filename) -> None:
    with open('data/nn_simulation.json', 'r') as f:
        z = list(json.load(f).values())    

    animate(z, from_simulation=False, filename=filename)