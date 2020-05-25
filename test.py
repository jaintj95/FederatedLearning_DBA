import torch
from torch import nn
import torch.nn.functional as F

import config
import main

"""
Module that defines various kinds of tests 
"""

def Mytest(helper, epoch, model, is_poison=False, visualize=True, agent_name_key=""):
    """
    Fetches data for testing
    """
    
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = 0
    
    if helper.params['type'] == config.TYPE_LOAN:
        # for i in range(len(helper.allStateHelperList)):
        #     state_helper = helper.allStateHelperList[i]
        for state_helper in helper.allStateHelperList:
            data_iterator = state_helper.get_testloader()
            for batch_idx, batch in enumerate(data_iterator):
                data, targets = state_helper.get_batch(data_iterator, batch, eval=True)
                dataset_size += len(data)
                output = model(data)
                total_loss += F.cross_entropy(output, targets, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(dim=1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    # elif helper.params['type'] == config.TYPE_CIFAR \
    #         or helper.params['type'] == config.TYPE_MNIST \
    #         or helper.params['type'] == config.TYPE_TINYIMAGENET:
    elif helper.params['type'] in [config.TYPE_CIFAR, config.TYPE_MNIST, config.TYPE_TINYIMAGENET]:
        data_iterator = helper.test_data
        for batch_idx, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_iterator, batch, eval=True)
            dataset_size += len(data)
            output = model(data)
            total_loss += F.cross_entropy(output, targets, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(dim=1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))  if dataset_size!=0 else 0
    avg_loss = total_loss / dataset_size if dataset_size!=0 else 0

    main.logger.info('___Test {} poisoned: {}, epoch: {}: Avg loss: {:.4f}, ''Accuracy: {}/{} ({:.4f}%)'
                .format(model.name, is_poison, epoch, avg_loss, correct, dataset_size, acc))

    if visualize: # loss =avg_loss
        model.test_vis(vis=main.vis, epoch=epoch, acc=acc, loss=None,
                       eid=helper.params['environment_name'], agent_name_key=str(agent_name_key))
    model.train()
    return (avg_loss, acc, correct, dataset_size)


def Mytest_poison(helper, epoch, model, is_poison=False, visualize=True, agent_name_key=""):
    """
    As the name implies, this func probably returns poisoned data.
    Will add more details later.
    """

    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    
    if helper.params['type'] == config.TYPE_LOAN:
        trigger_names = []
        trigger_values = []
    
        for j in range(0, helper.params['trigger_num']):
            for name in helper.params[str(j) + '_poison_trigger_names']:
                trigger_names.append(name)
            for value in helper.params[str(j) + '_poison_trigger_values']:
                trigger_values.append(value)
    
        # for i in range(0, len(helper.allStateHelperList)):
        #     state_helper = helper.allStateHelperList[i]
        for state_helper in helper.allStateHelperList:
            data_iterator = state_helper.get_testloader()
            for batch_idx, batch in enumerate(data_iterator):
                for index in range(len(batch[0])):
                    batch[1][index] = helper.params['poison_label_swap']
                    for j in range(0, len(trigger_names)):
                        name = trigger_names[j]
                        value = trigger_values[j]
                        batch[0][index][helper.feature_dict[name]] = value
                    poison_data_count += 1

                data, targets = state_helper.get_batch(data_iterator, batch, eval=True)
                dataset_size += len(data)
                output = model(data)
                total_loss += F.cross_entropy(output, targets, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(dim=1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
    
    # elif helper.params['type'] == config.TYPE_CIFAR \
    #         or helper.params['type'] == config.TYPE_MNIST \
    #         or helper.params['type'] == config.TYPE_TINYIMAGENET:
    elif helper.params['type'] in [config.TYPE_CIFAR, config.TYPE_MNIST, config.TYPE_TINYIMAGENET]:
        data_iterator = helper.test_data_poison
        for batch_idx, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=-1, eval=True)
            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += F.cross_entropy(output, targets, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(dim=1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count))  if poison_data_count!=0 else 0
    avg_loss = total_loss / poison_data_count if poison_data_count!=0 else 0
    main.logger.info('___Test {} poisoned: {}, epoch: {}: Avg loss: {:.4f}, ''Accuracy: {}/{} ({:.4f}%)'
                .format(model.name, is_poison, epoch, avg_loss, correct, poison_data_count, acc))
    if visualize: #loss = avg_loss
        model.poison_test_vis(vis=main.vis, epoch=epoch, acc=acc, loss=None, 
                            eid=helper.params['environment_name'], agent_name_key=str(agent_name_key))

    model.train()
    return avg_loss, acc, correct, poison_data_count


def Mytest_poison_trigger(helper, model, adver_trigger_index):
    """
    This just feels like Mytest_poison with a trigger at the end?
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    if helper.params['type'] == config.TYPE_LOAN:
        trigger_names = []
        trigger_values = []
        
        if adver_trigger_index == -1:
            for j in range(0, helper.params['trigger_num']):
                for name in helper.params[str(j) + '_poison_trigger_names']:
                    trigger_names.append(name)
                for value in helper.params[str(j) + '_poison_trigger_values']:
                    trigger_values.append(value)
        else:
            trigger_names = helper.params[str(adver_trigger_index) + '_poison_trigger_names']
            trigger_values = helper.params[str(adver_trigger_index) + '_poison_trigger_values']
        
        for i in range(0, len(helper.allStateHelperList)):
            state_helper = helper.allStateHelperList[i]
            data_iterator = state_helper.get_testloader()
            
            for batch_idx, batch in enumerate(data_iterator):
                for index in range(len(batch[0])):
                    batch[1][index] = helper.params['poison_label_swap']
                    for j in range(0, len(trigger_names)):
                        name = trigger_names[j]
                        value = trigger_values[j]
                        batch[0][index][helper.feature_dict[name]] = value
                    poison_data_count += 1

                data, targets = state_helper.get_batch(data_iterator, batch, eval=True)
                dataset_size += len(data)
                output = model(data)
                total_loss += F.cross_entropy(output, targets, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(dim=1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    # elif helper.params['type'] == config.TYPE_CIFAR \
    #         or helper.params['type'] == config.TYPE_MNIST \
    #         or helper.params['type'] == config.TYPE_TINYIMAGENET:
    elif helper.params['type'] in [config.TYPE_CIFAR, config.TYPE_MNIST, config.TYPE_TINYIMAGENET]:
        data_iterator = helper.test_data_poison
        adv_index = adver_trigger_index
        for batch_idx, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adv_index, eval=True)
            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += F.cross_entropy(output, targets, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(dim=1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    avg_loss = total_loss / poison_data_count if poison_data_count!=0 else 0

    model.train()
    return avg_loss, acc, correct, poison_data_count


def Mytest_poison_agent_trigger(helper, model, agent_name_key):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    if helper.params['type'] == config.TYPE_LOAN:
        adv_index = -1
        # for temp_index in range(0, len(helper.params['adversary_list'])):
        #     if agent_name_key == helper.params['adversary_list'][temp_index]:
        #         adv_index = temp_index
        #         break
        for temp_index in helper.params['adversary_list']:
            if int(agent_name_key) == helper.params['adversary_list'][temp_index]:
                adv_index = temp_index
                break
        
        trigger_names = helper.params[str(adv_index) + '_poison_trigger_names']
        trigger_values = helper.params[str(adv_index) + '_poison_trigger_values']

        for i in range(0, len(helper.allStateHelperList)):
            state_helper = helper.allStateHelperList[i]
            data_iterator = state_helper.get_testloader()
            for batch_idx, batch in enumerate(data_iterator):
                for index in range(len(batch[0])):
                    batch[1][index] = helper.params['poison_label_swap']
                    for j in range(0, len(trigger_names)):
                        name = trigger_names[j]
                        value = trigger_values[j]
                        batch[0][index][helper.feature_dict[name]] = value
                    poison_data_count += 1
                data, targets = state_helper.get_batch(data_iterator, batch, eval=True)
                dataset_size += len(data)
                output = model(data)
                total_loss += F.cross_entropy(output, targets, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(dim=1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    # elif helper.params['type'] == config.TYPE_CIFAR \
    #         or helper.params['type'] == config.TYPE_MNIST \
    #         or helper.params['type'] == config.TYPE_TINYIMAGENET:
    elif helper.params['type'] in [config.TYPE_CIFAR, config.TYPE_MNIST, config.TYPE_TINYIMAGENET]:
        data_iterator = helper.test_data_poison
        adv_index = -1

        # This whole loop seems redudant
        # If agent_name_key is in adversary_list then we replace adv_index with it.
        # Why complicate it so much then?
        # Maybe check for membership and if present then replace adv_index       
        # for temp_index in range(0, len(helper.params['adversary_list'])):
        #     if int(agent_name_key) == helper.params['adversary_list'][temp_index]:
        #         adv_index = temp_index
        #         break

        for temp_index in helper.params['adversary_list']:
            if int(agent_name_key) == helper.params['adversary_list'][temp_index]:
                adv_index = temp_index
                break
        

        for batch_idx, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adv_index, eval=True)
            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += F.cross_entropy(output, targets, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(dim=1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    avg_loss = total_loss / poison_data_count if poison_data_count!=0 else 0

    model.train()
    return avg_loss, acc, correct, poison_data_count
