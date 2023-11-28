import os
import sys
import torch
import logging
import argparse
import pickle
from torch.utils.data import DataLoader

from lra_config import config
from lra_train import NetDual, TrainModel

sys.path.append('../')
from Models.net_conv import CONV
from Models.net_conv_rf import receptive_field
from Models.net_rnn import RNN
from Models.utils import seed_everything, DeFungiDataset

parser = argparse.ArgumentParser(description='experiment')
parser.add_argument('--model', type=str, default='CDIL')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()


# Configure device (GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set arguments
MODEL = args.model
SEED = args.seed

# Config parameters
cfg_training = config['training']
cfg_model = config['models']

BATCH = cfg_training['batch_size']
CLASS = cfg_model['n_class']
SEQ_LEN = cfg_model['n_length']
FIX_length = cfg_model['fix_length']
USE_EMBED = cfg_model['use_embedding']
CHAR_COCAB = cfg_model['vocab_size']
INPUT_SIZE = cfg_model['dim']


# Model selection with configs 
if MODEL == 'CDIL' or MODEL == 'DIL' or MODEL == 'TCN' or MODEL == 'CNN':
    LAYER = cfg_model['cnn_layer']
    NHID = cfg_model['cnn_hidden']
    KERNEL_SIZE = cfg_model['cnn_ks']
    net = CONV(MODEL, INPUT_SIZE, CLASS, [NHID] * LAYER, KERNEL_SIZE, False, False, USE_EMBED, CHAR_COCAB, FIX_length)
    receptive_field(seq_length=SEQ_LEN, model=MODEL, kernel_size=KERNEL_SIZE, layer=LAYER)
elif MODEL == 'Deformable':
    LAYER = cfg_model['cnn_layer']
    NHID = cfg_model['cnn_hidden']
    KERNEL_SIZE = cfg_model['cnn_ks']
    net = CONV('CNN', INPUT_SIZE, CLASS, [NHID] * LAYER, KERNEL_SIZE, True, False, USE_EMBED, CHAR_COCAB, FIX_length)
    receptive_field(seq_length=SEQ_LEN, model=MODEL, kernel_size=KERNEL_SIZE, layer=LAYER)
elif MODEL == 'LSTM' or MODEL == 'GRU':
    LAYER = cfg_model['rnn_layer']
    NHID = cfg_model['rnn_hidden']
    net = RNN(MODEL, INPUT_SIZE, CLASS, NHID, LAYER, USE_EMBED, CHAR_COCAB, FIX_length)
else:
    print('no model specified.')
    sys.exit()

net = net.to(device)
para_num = sum(p.numel() for p in net.parameters() if p.requires_grad)


# Log
file_name = 'P' + str(para_num) + '_' + MODEL + '_S' + str(SEED) + '_L' + str(LAYER) + '_H' + str(NHID)

os.makedirs('DeFungi_log', exist_ok=True)
os.makedirs('DeFungi_model', exist_ok=True)
log_file_name = './DeFungi_log/' + file_name + '.txt'
model_name = './DeFungi_model/' + file_name + '.ph'
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
loginf = logging.info

loginf(torch.cuda.get_device_name(device))
loginf(file_name)


# Optimize
optimizer = torch.optim.Adam(net.parameters())
loss = torch.nn.CrossEntropyLoss(reduction='sum')


# Data
trainloader = DataLoader(DeFungiDataset(f'./defungi_datasets/train.pickle', True), batch_size=BATCH, shuffle=True, drop_last=False)
valloader = DataLoader(DeFungiDataset(f'./defungi_datasets/dev.pickle', True), batch_size=BATCH, shuffle=False, drop_last=False)
testloader = DataLoader(DeFungiDataset(f'./defungi_datasets/test.pickle', False), batch_size=BATCH, shuffle=False, drop_last=False)


# train
TrainModel(
    fix_length=FIX_length,
    net=net,
    device=device,
    trainloader=trainloader,
    valloader=valloader,
    testloader=testloader,
    n_epochs=cfg_training['epoch'],
    optimizer=optimizer,
    loss=loss,
    loginf=loginf,
    file_name=model_name
)

# Save Training Results for front-end
filename = 'fungi_classifier.sav'
pickle.dump('DeFungi CDIL-CNN', open(filename, 'wb'))
