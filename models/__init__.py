from .vpr_interface import VPRModel
from .boq_vpr import BOQMainVPR
from .mix_vpr import MIXMainVPR

MODELS_PULL = {
    'boq': BOQMainVPR,
    'mix' : MIXMainVPR,
}