def to_boolean(s):
    # """converts -1 to 0 and 1 to 1"""
    return (1 + s) / 2
    # return (1 - inp) / 2 # opposite mapping


def to_spin(b):
    # """converts 0 to -1 and 1 to 1"""
    return 2 * b - 1
    # return -2 * inp + 1 # opposite mapping
