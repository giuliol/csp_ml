import numpy as np


class DatParser:
    def __init__(self, views, frequency):
        self.views = views
        self.frequency = frequency
        self.values = np.zeros((views, views - 1), dtype=np.complex_)
        self.coordinates = np.zeros((views, views - 1, 2))

    @staticmethod
    def count_views(filename):
        """
        Static method for counting the number of views in a .dat file.
        Counts the occurrences of # (hasthag) symbol in the file.
        :param self:
        :param filename: file to parse to count the views
        :return: the number of views
        """
        with open(filename) as f:
            content = f.readlines()
            views = 0
            for line in content:
                if line[0] == "#":
                    views += 1
        return views

    @staticmethod
    def parse_file(filename):
        """
        Parse a specific file and stores the features (electromagnetic S values) in a ndarray.
        :param filename: the file to parse
        :return: a real valued, (views * (views-1) * 2, ) sized ndarray.
        """
        with open(filename) as f:
            content = f.readlines()
            views = DatParser.count_views(filename)

        values = np.zeros((views, views - 1), dtype=np.complex_)

        with open(filename) as f:
            content = f.readlines()
            _views = -1
            view_count = 0

            for line in content:
                if line[0] == "#":
                    _views += 1
                    view_count = 0
                else:
                    readvalues = line.split()
                    # coordinates[views, view_count, 0], coordinates[views, view_count, 1] = values[0], values[1]
                    #
                    # Reads magnitude(3) and phase(4)
                    values[_views, view_count] = float(readvalues[4]) + float(readvalues[5]) * 1j
                    view_count += 1

        return DatParser.__flatten(views, values)

    @staticmethod
    def __flatten(views, values):

        out = np.zeros(views * (views - 1) * 2)


        i = 0
        for view in range(views):
            for value in values[view, :]:
                out[i] = value.real
                out[i + 1] = value.imag
                i += 2
                # out[start:2:start + (self.views - 1)] = self.values[view, :].Real

        return out
