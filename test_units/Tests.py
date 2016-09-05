from common import Tools


def toolstest():
    print("------- TEST BEGIN -------")

    sv = Tools.SValues(10, 1000)
    sv.parse_file("res/ETotExt.SNR15.dat")

    vals = sv.values
    print(vals.shape)


    print("------- TEST   END -------")
