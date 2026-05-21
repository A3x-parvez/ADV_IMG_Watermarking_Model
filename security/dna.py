import torch


DNA_RULES = {

    "00": "A",
    "01": "T",
    "10": "C",
    "11": "G"
}

REVERSE_RULES = {
    v: k for k, v in DNA_RULES.items()
}


def dna_encode(x):

    x = (x > 0.5).float()

    return x


def dna_decode(x):

    x = (x > 0.5).float()

    return x