from src.tools import simulator_data_parser
from src.ml import tentativo_ml, tflearn_autoencoder


def tools_test():
    print("------- TEST BEGIN -------")

    sv = simulator_data_parser.DatParser(10, 1000)
    sv.parse_file("res/ETotExt.SNR15.dat")

    vals = sv.values
    print(vals)
    print(sv.flatten())
    print("------- TEST   END -------")
    return 0


def ml_test():
    tflearn_autoencoder.tflearn_tutorial()
    return 0


def giulio_test():
    tentativo_ml.ml_sandbox()
    return 0
