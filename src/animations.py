import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def update_plot(frame_number, ax, x, y, z, plot):
    """https://pythonmatplotlibtips.blogspot.com/2018/11/animation-3d-surface-plot-funcanimation-matplotlib.html"""
    print(frame_number)
    plot[0].remove()
    plot[0] = ax.plot_surface(x, y, z[frame_number], cmap="magma", rcount=100, ccount=100)


def animate(X, Y, Z_lst, speedup=10, save=None, repeat=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_zlim(np.min(Z_lst[0]), np.max(Z_lst[0]))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("$\parallel \psi \parallel^2$")

    fps = 30
    frn = len(Z_lst) // speedup

    saved_psi = Z_lst[::speedup]

    plot = [ax.plot_surface(X, Y, saved_psi[0], cmap="magma")]
    ani = animation.FuncAnimation(
        fig, update_plot, frn, fargs=(ax, X, Y, saved_psi, plot), interval=1000 / fps, repeat=repeat
    )

    if save:
        ani.save(save + ".mp4", writer="ffmpeg", fps=fps, dpi=300)
    else:
        plt.show()
