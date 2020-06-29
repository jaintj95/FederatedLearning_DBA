import copy

import torch
from torch import nn, optim
import torch.nn.functional as F

import main
import test
import config
import utils.csv_record as csv_record


def LoanTrain(helper, start_epoch, local_model, target_model, is_poison, state_keys):

    epochs_submit_upd_dict = dict()
    num_samples_dict = dict()
    curr_adversary_count = len(helper.params['adversary_list'])

    for model_id in range(helper.params['no_models']):
        epochs_local_upd_list = []
        last_params_vars = dict()
        client_grad = []  # fg  only works for aggr_epoch_interval=1

        for name, param in target_model.named_parameters():
            last_params_vars[name] = target_model.state_dict()[name].clone()

        state_key = state_keys[model_id]
        # Synchronize LR and models
        model = local_model
        model.copy_params(target_model.state_dict())
        optimizer = optim.SGD(model.parameters(), lr=helper.params['lr'],
                              momentum=helper.params['momentum'],
                              weight_decay=helper.params['decay'])
        model.train()
        temp_local_epoch = start_epoch-1

        adversarial_index = -1
        localmodel_poison_epochs = helper.params['poison_epochs']
        if is_poison and state_key in helper.params['adversary_list']:
            for adver_idx in range(0, len(helper.params['adversary_list'])):
                if state_key == helper.params['adversary_list'][adver_idx]:
                    localmodel_poison_epochs = helper.params[str(adver_idx) + '_poison_epochs']
                    adversarial_index = adver_idx
                    main.logger.info(f'poison local model {state_key} will poison epochs: {localmodel_poison_epochs}')
                    break
            if len(helper.params['adversary_list']) == 1:
                adversarial_index = -1  # attack the global trigger

        trigger_names = []
        trigger_values = []
        if adversarial_index == -1:
            for j in range(0, helper.params['trigger_num']):
                for name in helper.params[str(j) + '_poison_trigger_names']:
                    trigger_names.append(name)
                for value in helper.params[str(j) + '_poison_trigger_values']:
                    trigger_values.append(value)
        else:
            trigger_names = helper.params[str(adversarial_index) + '_poison_trigger_names']
            trigger_values = helper.params[str(adversarial_index) + '_poison_trigger_values']

        for epoch in range(start_epoch, start_epoch + helper.params['aggr_epoch_interval']):
            # This is for calculating distances
            target_params_vars = dict()
            for name, param in target_model.named_parameters():
                # target_params_vars[name] = last_params_vars[name].clone().detach().requires_grad_(False)
                target_params_vars[name] = last_params_vars[name].detach().clone()

            if is_poison and state_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                main.logger.info('poison_now')
                _, acc_p, _, _ = test.Mytest_poison(helper, epoch, model, is_poison=True, 
                                                    visualize=False, agent_name_key=state_key)
                main.logger.info(acc_p)
                poison_lr = helper.params['poison_lr']

                if not helper.params['baseline']:
                    if acc_p > 20:
                        poison_lr /= 5
                    if acc_p > 60:
                        poison_lr /= 10

                internal_epoch_num = helper.params['internal_poison_epochs']
                step_lr = helper.params['poison_step_lr']

                poison_optim = optim.SGD(model.parameters(), lr=poison_lr, 
                                         momentum=helper.params['momentum'], weight_decay=helper.params['decay'])
                scheduler = optim.lr_scheduler.MultiStepLR(poison_optim, gamma=0.1,
                                                           milestones=[0.2*internal_epoch_num, 0.8*internal_epoch_num])
                # acc = acc_initial
                for internal_epoch in range(1, internal_epoch_num + 1):
                    temp_local_epoch += 1
                    poison_data = helper.statehelper_dic[state_key].get_poison_trainloader()
                    if step_lr:
                        scheduler.step()
                        main.logger.info(f'Current lr: {scheduler.get_lr()}')
                    data_iterator = poison_data
                    poison_data_count = 0
                    total_loss = 0.
                    correct = 0
                    data_size = 0
                    for batch_id, batch in enumerate(data_iterator):
                        for index in range(0, helper.params['poisoning_per_batch']):
                            if index >= len(batch[1]):
                                break
                            batch[1][index] = helper.params['poison_label_swap']
                            for j in range(0, len(trigger_names)):
                                name = trigger_names[j]
                                value = trigger_values[j]
                                batch[0][index][helper.feature_dict[name]] = value
                            poison_data_count += 1

                        data, targets = helper.statehelper_dic[state_key].get_batch(poison_data, batch, False)
                        poison_optim.zero_grad()
                        data_size += len(data)
                        output = model(data)
                        class_loss = F.cross_entropy(output, targets)
                        distance_loss = helper.model_dist_norm_var(model, target_params_vars)

                        loss = helper.params['alpha_loss'] * class_loss + (1-helper.params['alpha_loss']) * distance_loss
                        loss.backward()
                        # get gradients
                        if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()
                        poison_optim.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                    acc = 100.0 * (float(correct) / float(data_size))
                    total_l = total_loss / data_size

                    main.logger.info('_PoisonTrain {}, epoch {:3d}, local model {}, internal_epoch {:3d}, Avg loss: {:.4f},'
                                    'Accuracy: {}/{} ({:.4f}%), train_poison_data_count{}'.format(model.name, epoch, state_key, 
                                                          internal_epoch, total_l, correct, data_size, acc, poison_data_count))

                    csv_record.train_result.append([state_key, temp_local_epoch, epoch, internal_epoch, 
                                                    total_l.item(), acc, correct, data_size])
                    if helper.params['vis_train']:
                        model.train_vis(main.vis, temp_local_epoch, acc, loss=total_l, 
                                        eid=helper.params['environment_name'], is_poisoned=True, name=state_key)
                    num_samples_dict[state_key] = data_size

                # internal epoch finish
                main.logger.info(f'Global model norm: {helper.model_global_norm(target_model)}.')
                main.logger.info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                                 f'Distance: {helper.model_dist_norm(model, target_params_vars)}')

                # Adversary wants to scale his weights. Baseline model doesn't do this
                # Some NORM work happening here
                if not helper.params['baseline']:
                    clip_rate = helper.params['scale_weights_poison']
                    main.logger.info(f"Scaling by {clip_rate}")
                    for key, value in model.state_dict().items():
                        target_val = last_params_vars[key]
                        new_value = target_val + (value - target_val) * clip_rate
                        model.state_dict()[key].copy_(new_value)
                    distance = helper.model_dist_norm(model, target_params_vars)
                    main.logger.info(f'Scaled Norm after poisoning: '
                                     f'{helper.model_global_norm(model)}, distance: {distance}')

                distance = helper.model_dist_norm(model, target_params_vars)
                main.logger.info(f"Total norm for {curr_adversary_count} "
                                 f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")

            else:
                for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
                    temp_local_epoch += 1
                    train_data = helper.statehelper_dic[state_key].get_trainloader()
                    data_iterator = train_data
                    total_loss = 0.
                    correct = 0
                    data_size = 0
                    for batch_id, batch in enumerate(data_iterator):
                        optimizer.zero_grad()
                        data, targets = helper.statehelper_dic[state_key].get_batch(data_iterator, batch, eval=False)
                        data_size += len(data)
                        output = model(data)
                        loss = F.cross_entropy(output, targets)
                        loss.backward()

                        # get gradients
                        if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()

                        optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                    acc = 100.0 * (float(correct) / float(data_size))
                    total_l = total_loss / data_size

                    main.logger.info('_Train {}, epoch {:3d}, local model {}, internal_epoch {:3d}, Avg loss: {:.4f},'
                                     ' Accuracy: {}/{} ({:.4f}%)'.format(model.name, epoch, state_key, internal_epoch,
                                                                         total_l, correct, data_size, acc))
                    csv_record.train_result.append([state_key, temp_local_epoch, epoch, internal_epoch, 
                                                    total_l.item(), acc, correct, data_size])

                    if helper.params['vis_train']:
                        model.train_vis(main.vis, temp_local_epoch, acc, loss=total_l, 
                                        eid=helper.params['environment_name'], is_poisoned=False, name=state_key)
                    num_samples_dict[state_key] = data_size

            # test local model after internal epoch train finish
            """
            Attributes/Metrics over one epoch
            e_correct: no. of correct predictions in this epoch
            e_total: total samples in this epoch
            """
            e_loss, e_acc, e_correct, e_total = test.Mytest(helper, epoch, model, is_poison=False, 
                                                            visualize=True, agent_name_key=state_key)
            csv_record.test_result.append([state_key, epoch, e_loss, e_acc, e_correct, e_total])

            if is_poison:
                if state_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                    e_loss, e_acc, e_correct, e_total = test.Mytest_poison(helper, epoch, model, is_poison=True,
                                                                           visualize=True, agent_name_key=state_key)
                    csv_record.poisontest_result.append([state_key, epoch, e_loss, e_acc, e_correct, e_total])

                # test on local triggers
                if state_key in helper.params['adversary_list']:
                    if helper.params['vis_trigger_split_test']:
                        model.trigger_agent_test_vis(main.vis, epoch, acc=e_acc, loss=None,
                                                     eid=helper.params['environment_name'],
                                                     name=state_key + "_combine")

                    e_loss, e_acc, e_correct, e_total = test.Mytest_poison_agent_trigger(helper, model, 
                                                                                         agent_name_key=state_key)
                    csv_record.poisontriggertest_result.append([state_key, state_key + "_trigger", "", epoch, 
                                                                e_loss, e_acc, e_correct, e_total])
                    if helper.params['vis_trigger_split_test']:
                        model.trigger_agent_test_vis(main.vis, epoch, acc=e_acc, loss=None,
                                                     eid=helper.params['environment_name'],
                                                     name=state_key+"_trigger")
            # update the weight and bias
            local_model_update_dict = dict()
            for name, data in model.state_dict().items():
                local_model_update_dict[name] = torch.zeros(data.shape)
                local_model_update_dict[name] = (data - last_params_vars[name])
                last_params_vars[name] = copy.deepcopy(data)

            if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                epochs_local_upd_list.append(client_grad)
            else:
                epochs_local_upd_list.append(local_model_update_dict)

        epochs_submit_upd_dict[state_key] = epochs_local_upd_list

    return epochs_submit_upd_dict, num_samples_dict
