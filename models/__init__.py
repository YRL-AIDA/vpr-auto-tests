from .vpr_interface import VPRModel
from .boq_vpr import BOQMainVPR
from .mix_vpr import MIXMainVPR
from .vladbuff_vpr import VLADBuffMainVPR
from .pair_vpr import PairVPR

MODELS_PULL = {
    'boq': BOQMainVPR,
    'mix' : MIXMainVPR,
    'vladbuff' : VLADBuffMainVPR,
    'pair' : PairVPR
}