import numpy as np
import random
import torch
import os

def check_cuda():
    """
    Checks if GPU is available.
    """    
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    return cuda_available, device

def set_seed(seed=1000):
    """
    Sets the seed for reproducability
    Args:
        seed (int, optional): Input seed. Defaults to 1000.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    cuda_available, _ = check_cuda()
    
    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def assert_square(bbox_xyxy_4: np.ndarray,
                  theshold:int=3) -> bool:
    """Checks if a bbox is square

    Args:
        bbox_xyxy_4 (np.ndarray): Bbox as np array of [xmin,ymin,xmax,ymax]
        theshold (int): Margin of difference between the height and width

    Returns:
        None
    """

    xmin, ymin, xmax, ymax = bbox_xyxy_4

    # Find the width and height of the bbox
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    assert abs(bbox_width - bbox_height) <= theshold, f'Target bbox is not square: bbox_width={bbox_width}, bbox_height={bbox_height}'
