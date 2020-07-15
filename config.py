import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

VIS_PORT = 8097

AGGR_MEAN = 'mean'
AGGR_GEO_MED = 'geom_median'
AGGR_FOOLSGOLD = 'foolsgold'
MAX_UPDATE_NORM = 100  # reject all updates larger than this amount
patience_iter = 20

TYPE_LOAN = 'loan'
TYPE_CIFAR = 'cifar'
TYPE_MNIST = 'mnist'
# TYPE_TINYIMAGENET = 'tiny-imagenet-200'
TYPE_TINYIMAGENET = 'tiny'