import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

def _radar_factory(num_vars):
    theta = 2*np.pi * np.linspace(0, 1-1./num_vars, num_vars)
    theta += np.pi/2

    def unit_poly_verts(theta):
        x0, y0, r = [0.5] * 3
        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
        return verts

    class RadarAxes(PolarAxes):
        name = 'radar'
        RESOLUTION = 1

        def fill(self, *args, **kwargs):
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(theta * 180/np.pi, labels)

        def _gen_axes_patch(self):
            verts = unit_poly_verts(theta)
            return plt.Polygon(verts, closed=True, edgecolor='k')

        def _gen_axes_spines(self):
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            verts.append(verts[0])
            path = Path(verts)
            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta

def radar_graph(labels = [], values = [], optimum = [], data = []):
    N = len(labels)
    theta = _radar_factory(N)
    max_val = max(max(optimum), max(values))
    fig= plt.figure(figsize=(9, 8), dpi= 80, facecolor='w', edgecolor='k') 
    ax = fig.add_subplot(1, 1, 1, projection='radar')
    ax.plot(theta, values, color='b')
    ax.plot(theta, optimum, color='r')
    ax.set_rlabel_position(90)
    ax.set_varlabels(labels)
    legend = ax.legend(data, loc=(0.9, .95),
                       labelspacing=0.1, fontsize='small')
    #tlt = title[0]
    #fig.text(0.5, 0.965, tlt,
    #         horizontalalignment='center', color='black', weight='bold',
    #         size='large', linespacing=2.0)
    plt.show()
    #plt.savefig("radar.png", dpi=100)}
