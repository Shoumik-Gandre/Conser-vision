import enum


@enum.unique
class Architectures(enum.Enum):
    RESNET50 = 'resnet50'
    RESNET152 = 'resnet152'
    EFFICIENT_NET = 'efficient-net'

    def __str__(self):
        return self.value


@enum.unique
class TransferTechnique(enum.Enum):
    BASE = 'base'
    FREEZE = 'freeze'
    L2_SP = 'l2-sp'
    DELTA = 'delta'
    BSS = 'bss'
    CO_TUNING = 'co-tuning'

    def __str__(self):
        return self.value
