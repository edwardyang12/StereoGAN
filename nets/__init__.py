from nets.gwcnet import GwcNet
from nets.loss import model_psmnet_loss, stereo_psmnet_loss
from nets.loss import model_gwcnet_loss
from nets.psmnet import PSMNet

__models__ = {
    "gwcnet": GwcNet,
    "gwcnet-c": PSMNet
}

__loss__ = {
    "gwcnet": stereo_psmnet_loss,
    "gwcnet-c": stereo_psmnet_loss
}