import numpy as np


class SValues:
    def __init__(self, views, frequency):
        self.views = views
        self.frequency = frequency
        self.values = np.zeros((views, views - 1), dtype=np.complex_)
        self.coordinates = np.zeros((views, views - 1, 2))

    def parse_file(self, filename):

        with open(filename) as f:
            content = f.readlines()
            view = 0
            for line in content:
                if line[0] == "#":
                    view += 1
            if view != self.views:
                raise ValueError(
                    "Number of views parsed from file {} ({}) differs from the value this object was initialized with ({})! Malformed or wrong file?".format(
                        filename, self.views, view))

        with open(filename) as f:
            content = f.readlines()
            view = -1
            view_count = 0

            for line in content:
                if line[0] == "#":
                    view += 1
                    view_count = 0
                else:
                    values = line.split()
                    self.coordinates[view, view_count, 0], self.coordinates[view, view_count, 1] = values[0], values[1]
                    self.values[view, view_count] = float(values[2]) + float(values[3]) * 1j
                    view_count += 1
