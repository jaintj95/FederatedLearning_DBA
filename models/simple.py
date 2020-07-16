import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np


class SimpleNet(nn.Module):

    def __init__(self, name=None, created_time=None):
        super(SimpleNet, self).__init__()
        self.created_time = created_time
        self.name = name

    def save_stats(self, epoch, loss, acc):
        self.stats['epoch'].append(epoch)
        self.stats['loss'].append(loss)
        self.stats['acc'].append(acc)

    def copy_params(self, state_dict, coefficient_transfer=100):

        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                shape = param.shape
                # random_tensor = (torch.cuda.FloatTensor(shape).random_(0, 100) <= coefficient_transfer).type(torch.cuda.FloatTensor)
                # negative_tensor = (random_tensor*-1)+1
                # own_state[name].copy_(param)
                own_state[name].copy_(param.clone())

    def plot_line(self, vis, eid, name, window, win_title, x_vals, y_vals):
        width_ = 700
        height_ = 400
        vis.line(X=x_vals, Y=y_vals, env=eid, name=name, win=window,
                 update='append' if vis.win_exists(window, env=eid) else None,
                 opts=dict(showlegend=True, title=win_title, width=width_, height=height_))

    def train_vis(self, vis, epoch, acc, loss=None, eid='main', is_poisoned=False, name=None):
        if name is None:
            name = self.name + '_poisoned' if is_poisoned else self.name

        window = 'train_acc_{0}'.format(self.created_time)
        win_title = 'Train Accuracy_{0}'.format(self.created_time)
        self.plot_line(vis, eid, name, window, win_title,
                       x_vals=np.array([epoch]), y_vals=np.array([acc]))

        if loss is not None:
            window = 'train_loss_{0}'.format(self.created_time)
            win_title = 'Train Loss_{0}'.format(self.created_time)
            self.plot_line(vis, eid, name, window, win_title,
                           x_vals=np.array([epoch]), y_vals=np.array([loss.cpu().data.numpy()]))

        return

    def train_batch_vis(self, vis, epoch, data_len, batch, loss, eid='main', name=None, is_poisoned=False):
        if name is None:
            name = self.name + '_poisoned' if is_poisoned else self.name
        else:
            name = name + '_poisoned' if is_poisoned else name

        window = 'train_batch_loss_{0}'.format(self.created_time)
        win_title = 'Train Batch loss_{0}'.format(self.created_time)
        self.plot_line(vis, eid, name, window, win_title,
                       x_vals=np.array([(epoch - 1) * data_len + batch]), y_vals=np.array([loss.cpu().data.numpy()]))

    def track_distance_batch_vis(self, vis, epoch, data_len, batch, distance_to_global_model, eid, name=None,
                                 is_poisoned=False):

        x = (epoch - 1) * data_len + batch + 1

        if name is None:
            name = self.name + '_poisoned' if is_poisoned else self.name
        else:
            name = name + '_poisoned' if is_poisoned else name

        window = 'global_dist_{0}'.format(self.created_time)
        win_title = 'Distance to Global {0}'.format(self.created_time)
        self.plot_line(vis, eid, name, window, win_title,
                       x_vals=np.array([x]), y_vals=np.array([distance_to_global_model]))

    def weight_vis(self, vis, epoch, weight, eid, name, is_poisoned=False):
        name = str(name) + '_poisoned' if is_poisoned else name
        window = f"Aggregation_Weight_{self.created_time}"
        win_title = f"Aggregation Weight {self.created_time}"
        self.plot_line(vis, eid, name, window, win_title,
                       x_vals=np.array([epoch]), y_vals=np.array([weight]))

    def alpha_vis(self, vis, epoch, alpha, eid, name, is_poisoned=False):
        name = str(name) + '_poisoned' if is_poisoned else name
        window = f"FG_Alpha_{self.created_time}"
        win_title = f"FG Alpha {self.created_time}"
        self.plot_line(vis, eid, name, window, win_title,
                       x_vals=np.array([epoch]), y_vals=np.array([alpha]))

    def trigger_test_vis(self, vis, epoch, acc, loss, eid, agent_name_key, trigger_name, trigger_value):
        name = f'{agent_name_key}_[{trigger_name}]_{trigger_value}'
        window = f"poison_triggerweight_vis_acc_{self.created_time}"
        win_title = f"Backdoor Trigger Test Accuracy_{self.created_time}"
        self.plot_line(vis, eid, name, window, win_title,
                       x_vals=np.array([epoch]), y_vals=np.array([acc]))

        if loss is not None:
            window = f"poison_trigger_loss_{self.created_time}"
            win_title = f"Backdoor Trigger Test Loss_{self.created_time}"
            self.plot_line(vis, eid, name, window, win_title,
                           x_vals=np.array([epoch]), y_vals=np.array([loss]))

    def trigger_agent_test_vis(self, vis, epoch, acc, loss, eid, name):
        window = f"poison_state_trigger_acc_{self.created_time}"
        win_title = f"Backdoor State Trigger Test Accuracy_{self.created_time}"
        self.plot_line(vis, eid, name, window, win_title,
                       x_vals=np.array([epoch]), y_vals=np.array([acc]))

        if loss is not None:
            window = f"poison_state_trigger_loss_{self.created_time}"
            win_title = f"Backdoor State Trigger Test Loss_{self.created_time}"
            self.plot_line(vis, eid, name, window, win_title,
                           x_vals=np.array([epoch]), y_vals=np.array([loss]))

    def poison_test_vis(self, vis, epoch, acc, loss, eid, agent_name_key):
        name = agent_name_key  # name= f'Model_{name}'
        window = f"poison_test_acc_{self.created_time}"
        win_title = f"Backdoor Task Accuracy_{self.created_time}"
        self.plot_line(vis, eid, name, window, win_title,
                       x_vals=np.array([epoch]), y_vals=np.array([acc]))

        if loss is not None:
            window = f"poison_loss_acc_{self.created_time}"
            win_title = f"Backdoor Task Test Loss_{self.created_time}"
            self.plot_line(vis, eid, name, window, win_title,
                           x_vals=np.array([epoch]), y_vals=np.array([loss]))

    def additional_test_vis(self, vis, epoch, acc, loss, eid, agent_name_key):
        name = agent_name_key
        window = f"additional_test_acc_{self.created_time}"
        win_title = f"Additional Test Accuracy_{self.created_time}"
        self.plot_line(vis, eid, name, window, win_title,
                       x_vals=np.array([epoch]), y_vals=np.array([acc]))

        if loss is not None:
            window = f"additional_test_loss_{self.created_time}"
            win_title = f"Additional Test Loss_{self.created_time}"
            self.plot_line(vis, eid, name, window, win_title,
                           x_vals=np.array([epoch]), y_vals=np.array([loss]))

    def test_vis(self, vis, epoch, acc, loss, eid, agent_name_key):
        name = agent_name_key  # name= f'Model_{name}'
        window = f"test_acc_{self.created_time}"
        win_title = f"Main Task Test Accuracy_{self.created_time}"
        self.plot_line(vis, eid, name, window, win_title,
                       x_vals=np.array([epoch]), y_vals=np.array([acc]))

        if loss is not None:
            window = f"test_loss_{self.created_time}"
            win_title = f"Main Task Test Loss_{self.created_time}"
            self.plot_line(vis, eid, name, window, win_title,
                           x_vals=np.array([epoch]), y_vals=np.array([loss]))


class SimpleMnist(SimpleNet):
    def __init__(self, name=None, created_time=None):
        super(SimpleMnist, self).__init__(name, created_time)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
