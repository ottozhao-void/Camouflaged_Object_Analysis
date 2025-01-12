from .camouflage import CamouflageDataset

def get_dataset(name):
    return {
        "camouflage": CamouflageDataset,
    }[name]
