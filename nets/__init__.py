from nets.gwcnet import GwcNet
from nets.loss import model_psmnet_loss, stereo_psmnet_loss
from nets.loss import model_gwcnet_loss
from nets.psmnet import PSMNet


import importlib
from nets.base_model import BaseModel

__models__ = {
    "gwcnet": GwcNet,
    "gwcnet-c": PSMNet
}

__loss__ = {
    "gwcnet": stereo_psmnet_loss,
    "gwcnet-c": stereo_psmnet_loss
}



def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "nets." + model_name + "_model"
    #print(model_filename)
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt, cmodel):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    #opt.model = "cycle_gan"
    
    model = find_model_using_name(opt.model)
    instance = model(opt, cmodel)
    print("model [%s] was created" % type(instance).__name__)
    return instance
