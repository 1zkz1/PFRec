import torch
import os
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from dataloader import MyDataloader
from torch.utils.data import DataLoader, Dataset

from model_save_load import ModelUtil
from my_metrics import IED, Precision_Recall, Ndcg
from mydataset import Amazon_MyModel_GAN8
from topNmodel import Dis_MyModel_GAN_8, Gen_MyModel_GAN_8
import torch.nn as nn
import torch.optim as optim
from tqdm.autonotebook import tqdm
from Config import get_config_data

def getExposure_dataset(data: pd.DataFrame, n_items: int, n_users: int, time_distance=5):

    record_items = np.zeros(shape=(n_users, time_distance), dtype=int)
    data_groupby_user = data.groupby('user')
    pad_num = 0
    for user, items in data_groupby_user:
        sorted_items = items.sort_values(by='timestamp', ascending=False)
        a = pd.DataFrame(sorted_items['item'][:time_distance]).to_numpy().T.ravel()
        need_pad_num = time_distance - len(a)
        while need_pad_num > 0:
            a = np.append(a, -1)
            pad_num += 1
            need_pad_num = time_distance - len(a)
            # print('1')
        record_items[user] = a
        # print()
    # b = np.where(record_items == -1)

    dataset_tensor = torch.tensor(record_items, dtype=torch.int)
    ied = IED(recom_TopK=dataset_tensor, n_items=n_items, n_users=n_users, topK=time_distance)
    ied_score, Exposure_dataset = ied.cal_IED_by_recomTopK(recomTopK=dataset_tensor, getExpo=True, n_items=n_items)

    # new_time_distance = time_distance + topK
    # Exposure_model = torch.ones_like(Exposure_dataset) * (1. / torch.tensor([3.]).log()) / n_users
    #
    # total_ied_score = ied.calIedFromExpo(Exposure_model + Exposure_dataset)

    return ied_score, Exposure_dataset

def getData(dataset_no=0, fold_no=0):
    dataset_name_list = [r'Amazon-beauty', r'Amazon-digital-music', r'Amazon-office-products',
                         r'Amazon-toys-and-games']
    fold_list = [1, 2, 3, 4, 5]
    assert 0 <= dataset_no <= 3
    assert 0 <= fold_no <= 4
    dataset_name = dataset_name_list[dataset_no]
    fold = fold_list[fold_no]

    data = MyDataloader(dataset_name=dataset_name, fold=fold)
    # # record[name[2]].append('{}_{}'.format(dataset_name, fold))
    #
    # train_pd = pd.read_csv(r'newData\Amazon-beauty\UN10_IN5\train_df_1_UN10_IN5.csv')
    # test_pd = pd.read_csv(r'newData\Amazon-beauty\UN10_IN5\test_df_1_UN10_IN5.csv')
    #
    # data.reloader(train_pd, test_pd)
    # print('data reload')
    print('DataSet:', dataset_name, ', fold:', fold)
    return data

def getPd_Tensor_array(data):
    array = data.toarray()
    tensor = torch.tensor(array)
    data_pd = pd.DataFrame(array)
    return array, tensor, data_pd

def getDataLoader(dataset: Dataset, batch_size, shuffle=False, drop_last=False):
    dataLoader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=drop_last
    )
    return dataLoader

def getTestDictFromDataFrame(test_PD: pd.DataFrame, test_skip=False, topK=5):
    test_groupby = test_PD.groupby('user')
    test_dict = {}
    for i, item in test_groupby:
        test_dict[i] = set([_ for _ in item['item']])
    return test_dict

def testAdd1(rating_array, predict, topK, ied_score=None, test_skip=False, PrT=True, getIed=False, Exposure_dataset=None, needAfterIED=False, test_PD=None, test_dict=None):
        if PrT:
            print('######### Test ########')

        assert test_PD is not None or test_dict is not None or test_dict is not None
        if needAfterIED:
            assert Exposure_dataset is not None and ied_score is not None

        n_users, n_items = predict.shape

        if test_dict is None:
            if test_PD is not None and isinstance(test_PD, pd.DataFrame):
                test_dict = getTestDictFromDataFrame(test_PD, test_skip=test_skip)

        if isinstance(topK, list):
            result_topK = {}
            for topK_i in topK:
                result_topK[topK_i] = testAdd1(rating_array, predict, topK=topK_i, PrT=False, getIed=getIed, Exposure_dataset=Exposure_dataset, needAfterIED=needAfterIED, test_dict=test_dict)
            return result_topK

        assert isinstance(topK, int)

        pre_record = {}
        predict_topK_list = []
        predict_dict = {}

        for i, _ in test_dict.items():
            temp = predict[i]
            a = torch.topk(temp, k=topK).indices
            assert len(a) == topK

            pre_record[i] = [rating_array[i][_] for _ in a]
            predict_dict[i] = set(a.numpy())
            predict_topK_list.append(a.numpy())

        # predict_topK_tensor = torch.Tensor(np.array(predict_topK_list))
        pred_topK_ied = IED(predict.squeeze(), n_items, n_users, topK)
        # ied = pred_topK_ied.cal_IED()
        ied = pred_topK_ied.cal_IED()
        if Exposure_dataset is not None and ied_score is not None:
            ied_after = pred_topK_ied.cal_IED(Add_expos=Exposure_dataset)
            ied_change = ied_after - ied_score

        precision, recall = Precision_Recall(predict_dict, test_dict, topK)

        ndcg_list = []
        for i, rel_list in pre_record.items():
            # print(i, rel_list)
            ndcg_list.append(Ndcg(rel_list))
        ndcg = np.sum(ndcg_list) / len(pre_record)
        # print(len(pre_record))
        var_ndcg = torch.var(torch.Tensor(np.array(ndcg_list)))


        if PrT:
            print('\ntop={}:'.format(topK))
            print('  precision@{}:{}'.format(topK, precision))
            print('  recall@{}:{}'.format(topK, recall))
            print('  ndcg@{}:{}'.format(topK, ndcg))
            print('  var_ndcg@{}:{}'.format(topK, var_ndcg))
            print('  IED@{}:{}'.format(topK, ied))
            if Exposure_dataset is not None:
                print('  IED_after@time_distance:{}'.format(ied_after))
                print('  IED_change@time_distance:{}'.format(ied_change))

        if getIed and not needAfterIED:
            return precision, recall, ndcg, var_ndcg, ied

        if getIed and needAfterIED:
            assert Exposure_dataset is not None
            return precision, recall, ndcg, var_ndcg.item(), ied.item(), ied_after.item(), ied_change.item()

        return precision, recall, ndcg, var_ndcg

def getBatchDict(idx: torch.Tensor, train_dict: dict):
    temp_dict = {}
    for i in idx:
        userid = i.item()
        temp_dict[userid] = train_dict[userid]
    return temp_dict

def getPredict(testDataLoader, net, n_items):
        # get predict
        # net = net.eval()
        # print('predict:', net.training)
        predict = torch.zeros(size=(1, n_items), dtype=torch.float)

        torch.cuda.empty_cache()
        with torch.no_grad():
            for i, (input, _) in enumerate(testDataLoader):
                # convert to cuda
                # torch.cuda.empty_cache()
                # input_tensor = input_tensor.to(device)
                # label_tensor = label_tensor.to(device)
                # pad = pad.to(device)
                # label_mask_tensor = label_mask_tensor.to(device)
                input = input.to(device)
                pred = net(input)
                pred = pred.cpu().detach()

                predict = torch.concat([predict, pred], dim=0)

        torch.cuda.empty_cache()
        predict = predict[1:, :]

        # net = net.train()

        return predict

def getValue(value: float, num=6):
    if num == 2:
        return_Value = r'{:.2f}'.format(value)
    elif num == 3:
        return_Value = r'{:.3f}'.format(value)
    elif num == 4:
        return_Value = r'{:.4f}'.format(value)
    elif num == 5:
        return_Value = r'{:.5f}'.format(value)
    elif num == 6:
        return_Value = r'{:.6f}'.format(value)
    elif num == 7:
        return_Value = r'{:.7f}'.format(value)
    else:
        return_Value = r'{:.8f}'.format(value)
    return float(return_Value)

def main():
    # get params from config
    train_batch_size = DataSet['train_batch_size']
    test_batch_size = DataSet['test_batch_size']
    epochs = DataSet['epochs']

    # Gen
    gen_hidden = DataSet['gen_hidden']
    gen_output_activation = DataSet['gen_output_activation']
    gen_lr = float(DataSet['gen_lr'])
    iter_num_gen = DataSet['iter_num_gen']
    gen_train = True
    alpha = DataSet['alpha']
    thread = DataSet['thread']
    # print('gen_lr:', gen_lr)

    # Dis
    dis_hidden = DataSet['dis_hidden']
    dis_output_activation = DataSet['dis_output_activation']
    # dis_output_activation = 'LeakyReLU'
    dis_lr = float(DataSet['dis_lr'])
    iter_num_dis = DataSet['iter_num_dis']
    dis_train = True

    if not use_step:
        iter_num_gen = 2
        iter_num_dis = 1

    # TensorBoard
    cmd = os.getcwd()
    path_name = DataSet['log_path']
    assert os.path.exists(path_name)
    path_log = r'{}\{}'.format(cmd, path_name)
    file_name = r'{}_test_no{}_alpha{}'.format(DataSetName, no + 1, alpha)
    if add_Expo and use_step:
        file_name = r'{}_test_no{}_alpha{}_addExpo_useStep'.format(DataSetName, no + 1, alpha)
    elif add_Expo and not use_step:
        file_name = r'{}_test_no{}_alpha{}_addExpo'.format(DataSetName, no + 1, alpha)
    elif not add_Expo and use_step:
        file_name = r'{}_test_no{}_alpha{}_useStep'.format(DataSetName, no + 1, alpha)
    elif not add_Expo and not use_step:
        file_name = r'{}_test_no{}_alpha{}'.format(DataSetName, no + 1, alpha)
    path = r'{}\{}'.format(path_log, file_name)
    print('path:', path)
    # return
    os.mkdir(path)
    writer = SummaryWriter(path)


    train_DataLoader = getDataLoader(train_dataset, train_batch_size, shuffle=True)
    test_DataLoader = getDataLoader(test_dataset, test_batch_size, shuffle=False)
    train_dict = getTestDictFromDataFrame(test_PD=train_pd)
    test_dict = getTestDictFromDataFrame(test_PD=test_pd)

    # network
    # Gen = Dis_MyModel_GAN_8(n_items, hidden_dim=gen_hidden, output_activation=gen_output_activation)
    Gen = Gen_MyModel_GAN_8(n_items, hidden_dim=gen_hidden)
    Gen = Gen.to(device)
    gen_optimizer = optim.Adam(Gen.parameters(), lr=gen_lr, eps=1e-6)
    gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=1, gamma=0.1)
    # gen_optimizer = optim.SGD(Gen.parameters(), lr=gen_lr)

    Dis = Dis_MyModel_GAN_8(n_items, hidden_dim=dis_hidden, output_activation=dis_output_activation)
    Dis = Dis.to(device)
    dis_optimizer = optim.Adam(Dis.parameters(), lr=dis_lr, eps=1e-6)
    # dis_optimizer = optim.SGD(Dis.parameters(), lr=dis_lr)
    dis_scheduler = optim.lr_scheduler.StepLR(dis_optimizer, step_size=1, gamma=0.1)

    netUtil = ModelUtil(Gen, gen_optimizer, epochs, 'MyModel_GAN_8')

    mse_loss = nn.MSELoss()
    sigmoid = nn.Sigmoid()
    not_save = False
    record_epoch = 0
    current_epoch = 0
    record_dis_loss = 0.
    status = True
    total_num = 0
    total_num1 = 0

    thread = DataSet['thread']
    bind_train_epochs = DataSet['bind_train_epochs']
    topK_test = DataSet['topK_test']
    record_interval = DataSet['record_interval']

    print('alpha:', alpha)
    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            pbar.set_description('alpha:{}'.format(alpha))
            epoch_dis_loss = 0.
            epoch_gen_loss = 0.
            epoch_gen_fair_loss = 0.
            epoch_dis_real_loss = 0.
            epoch_dis_fake_loss = 0.
            # final target: Dis(real) - Dis(fake) = 0
            for i, (input, idx) in enumerate(train_DataLoader):
                input = input.to(device)
                mask = input > 0.

                # dis
                for _ in range(iter_num_dis):
                    with torch.no_grad():
                        gen_optimizer.zero_grad()
                        gen_output = Gen(input)

                    dis_optimizer.zero_grad()
                    # output = Dis(torch.concat([input, gen_output.detach()], dim=1))
                    D_fake_output = Dis(gen_output.detach())
                    D_real_output = Dis(input)

                    dis_target_loss = torch.sum(D_fake_output * mask) / torch.sum(mask) - torch.sum(D_real_output * mask) / torch.sum(mask) + 1.
                    # dis_fair_loss = myIed(gen_output, topK=max_time_distance) - ied_score
                    dis_loss_total = dis_target_loss

                    dis_loss_total.backward()
                    if dis_train:
                        dis_optimizer.step()
                    epoch_dis_loss += dis_loss_total.item()

                # gen
                for _ in range(iter_num_gen):
                    gen_optimizer.zero_grad()
                    gen_output = Gen(input)
                    # gen_loss = mse_loss(pred * mask, mask.float())
                    # output = torch.concat([input, gen_output], dim=1)
                    D_fake_output = Dis(gen_output)
                    # D_real_output = Dis(input)

                    gen_loss = 0.5 - torch.sum(D_fake_output * mask) / torch.sum(mask)
                    if add_Expo:
                        gen_fair_loss = myIed(gen_output, topK=5, n_items=n_items, AddExposure=Exposure_dataset)
                    else:
                        gen_fair_loss = myIed(gen_output, topK=5, n_items=n_items)

                    gen_total_loss = gen_loss * alpha + gen_fair_loss * (1 - alpha)
                    # gen_total_loss = gen_loss + gen_fair_loss

                    # gen_loss = torch.sum(D_real_output * mask) / torch.sum(mask) - torch.sum(D_fake_output * mask) / torch.sum(mask)
                    # print('loss:', gen_loss.item())
                    # gen_loss.backward()
                    gen_total_loss.backward()
                    if gen_train:
                        gen_optimizer.step()
                    epoch_gen_loss += gen_loss.item()
                    epoch_gen_fair_loss += gen_fair_loss.item()

                # train_batch_dict = getBatchDict(idx, train_dict)
                # precision, recall, ndcg, var_ndcg = testAdd1(rating_mask_array, gen_output.cpu().detach(), 5, test_dict=train_batch_dict, PrT=False)
                # pbar.set_postfix({'pre@5': precision, 'rec@5': recall, 'ndcg@5': ndcg})

            # print('dis_loss:', epoch_dis_loss / iter_num_dis, ', gen_loss:', epoch_gen_loss / iter_num_gen)
            temp_dict = {}
            if (epoch + 1) > bind_train_epochs:
                if dis_train:
                    writer.add_scalar('gen_dis_train', scalar_value=1, global_step=epoch)
                if gen_train:
                    writer.add_scalar('gen_dis_train', scalar_value=0, global_step=epoch)
            if iter_num_dis > 0:
                epoch_aver_dis_loss = epoch_dis_loss / iter_num_dis
                writer.add_scalar('dis_loss/dis_target_loss', scalar_value=getValue(dis_target_loss), global_step=epoch)
                # writer.add_scalar('dis_loss/dis_fair_loss', scalar_value=dis_fair_loss, global_step=epoch)
                writer.add_scalar('dis_loss/total', scalar_value=getValue(epoch_aver_dis_loss), global_step=epoch)
                temp_dict['dis_loss'] = epoch_aver_dis_loss
                if epoch_aver_dis_loss < 0.1 and False:
                    iter_num_gen = 1
                    iter_num_dis = 0
            # writer.add_scalar('dis_loss/fake', scalar_value=epoch_dis_fake_loss / iter_num_dis, global_step=epoch)
            # writer.add_scalar('dis_loss/real', scalar_value=epoch_dis_real_loss / iter_num_dis, global_step=epoch)
            if iter_num_gen > 0:
                epoch_aver_gen_loss = epoch_gen_loss / iter_num_gen
                writer.add_scalar('gen/loss', scalar_value=getValue(epoch_aver_gen_loss), global_step=epoch)
                writer.add_scalar('gen/fair_loss', scalar_value=getValue(epoch_gen_fair_loss), global_step=epoch)
                writer.add_scalar('gen/gen_total_loss', scalar_value=getValue(gen_total_loss), global_step=epoch)
                temp_dict['gen_loss'] = epoch_aver_gen_loss
                temp_dict['gen_fair_loss'] = epoch_gen_fair_loss
                if epoch_aver_gen_loss < 0.1 and False:
                    iter_num_gen = 0
                    iter_num_dis = 1
            dis_curr_lr = dis_optimizer.param_groups[0]['lr']
            gen_curr_lr = gen_optimizer.param_groups[0]['lr']
            temp_dict['dis_curr_lr'] = dis_curr_lr
            temp_dict['gen_curr_lr'] = gen_curr_lr
            temp_dict['gen_train'] = gen_train
            temp_dict['dis_train'] = dis_train
            temp_dict['thread'] = thread

            pbar.set_postfix(temp_dict)

            if (epoch + 1) % record_interval == 0:
                predict = getPredict(test_DataLoader, Gen, n_items)
                # precision, recall, ndcg, var_ndcg, ied = testAdd1(rating_mask_array, predict, 5, test_dict=train_dict, PrT=False, getIed=True)
                precision, recall, ndcg, var_ndcg, ied, ied_after, ied_change = testAdd1(rating_mask_array, predict, topK=5, test_dict=train_dict, PrT=False, getIed=True, Exposure_dataset=Exposure_dataset, ied_score=ied_score, needAfterIED=True)
                writer.add_scalar('train@5/precision', scalar_value=getValue(precision), global_step=epoch)
                writer.add_scalar('train@5/recall', scalar_value=getValue(recall), global_step=epoch)
                writer.add_scalar('train@5/ndcg', scalar_value=getValue(ndcg), global_step=epoch)
                writer.add_scalar('train@5/ied', scalar_value=getValue(ied), global_step=epoch)
                writer.add_scalar('train@5/ied_after', scalar_value=getValue(ied_after), global_step=epoch)
                writer.add_scalar('train@5/ied_change', scalar_value=getValue(ied_change), global_step=epoch)
                # print('\ntrain  :: pre:', precision, 'rec:', recall, 'ndcg:', ndcg, 'var_ndcg:', var_ndcg.item())
                predict = predict - train_mask_tensor * 999

                for topk in topK_test:
                    # precision5, recall5, ndcg5, var_ndcg5, ied5 = testAdd1(rating_mask_array, predict, 5, test_dict=test_dict, PrT=False, getIed=True)
                    precision5, recall5, ndcg5, var_ndcg5, ied5, ied_after5, ied_change5 = testAdd1(rating_mask_array, predict, topK=topk, test_dict=test_dict, PrT=False, getIed=True, Exposure_dataset=Exposure_dataset, ied_score=ied_score, needAfterIED=True)
                    writer.add_scalar('test@{}/precision'.format(topk), scalar_value=getValue(precision5), global_step=epoch)
                    writer.add_scalar('test@{}/recall'.format(topk), scalar_value=getValue(recall5), global_step=epoch)
                    writer.add_scalar('test@{}/ndcg'.format(topk), scalar_value=getValue(ndcg5), global_step=epoch)
                    writer.add_scalar('test@{}/ied'.format(topk), scalar_value=getValue(ied5), global_step=epoch)
                    writer.add_scalar('test@{}/ied_after'.format(topk), scalar_value=getValue(ied_after5), global_step=epoch)
                    writer.add_scalar('test@{}/ied_change'.format(topk), scalar_value=getValue(ied_change5), global_step=epoch)

            if use_step and (epoch + 1) > current_epoch + 10 and (epoch + 1) > bind_train_epochs:
                # if epoch_aver_gen_loss < thread:
                if epoch_aver_gen_loss < thread:
                    # Dis(Gen(x)) > 0.5
                    dis_train = True
                    gen_train = False
                else:
                    # Dis(Gen(x)) < 0.5
                    dis_train = False
                    gen_train = True
                    total_num += 1

                if total_num > 5 and thread < 0.1:
                    thread = thread * 10
                current_epoch = epoch


            if (epoch + 1) % 10 == 0:
                print()

            pbar.update(1)

    predict = getPredict(test_DataLoader, Gen, n_items)
    predict = predict - train_mask_tensor * 999
    recom_tensor = torch.topk(predict, k=5).indices
    for topk in topK_test:
        precision, recall, ndcg, var_ndcg = testAdd1(rating_mask_array, predict, topk, test_dict=test_dict, PrT=True)

    return

def myIed(y_pred: torch.tensor, topK=5, n_items=None, AddExposure=None):
    # input (n_users, n_items)
    n_users, _ = y_pred.shape
    if n_items is None:
        n_items = _
    position = torch.ones(size=(n_users, n_items))
    indices = torch.topk(y_pred, k=topK, dim=1).indices
    temp = [_ for _ in range(n_items)]
    for i, rows_i in enumerate(indices):
        temp_list = list(set(temp) - set(rows_i.tolist()))
        position[i][temp_list] = 0.
    y_pred = y_pred * position.to(device)
    # mask = position > 0
    # bias = 1. / torch.log(1. + position)
    if AddExposure is None:
        expo = torch.mean(y_pred, dim=0)
    else:
        expo = torch.mean(y_pred, dim=0) + AddExposure.to(device)
    # print('expo shape:', expo.shape)

    return torch.sum(torch.abs(expo.unsqueeze(0) - expo.unsqueeze(1))) / (2. * n_items * torch.sum(expo))

def load_and_test(model_name, topK, path='models', needPredict=False, getNet=False, test_in_train=False):
    if not isinstance(model_name, str):
        model_metric = []
        for i in model_name:
            metric_dict = {}
            precision, recall, ndcg, var_ndcg, ied, ied_after, ied_change = load_and_test(i, path=path, topK=topK)
            metric_dict['precision'] = precision
            metric_dict['recall'] = recall
            metric_dict['ndcg'] = ndcg
            metric_dict['var_ndcg'] = var_ndcg
            metric_dict['ied'] = ied
            metric_dict['ied_after'] = ied_after
            metric_dict['ied_change'] = ied_change
            model_metric.append(metric_dict)
        return model_metric

    assert isinstance(model_name, str)

    if isinstance(topK, list) and isinstance(model_name, str):
        model_metric = {}
        for i in topK:
            metric_dict = {}
            precision, recall, ndcg, var_ndcg, ied, ied_after, ied_change = load_and_test(model_name, path=path, topK=i)
            metric_dict['precision'] = precision
            metric_dict['recall'] = recall
            metric_dict['ndcg'] = ndcg
            metric_dict['var_ndcg'] = var_ndcg
            metric_dict['ied'] = ied
            metric_dict['ied_after'] = ied_after
            metric_dict['ied_change'] = ied_change
            model_metric[i] = metric_dict
        return model_metric

    assert isinstance(topK, int)

    gen_hidden = 20000
    Gen = Gen_MyModel_GAN_8(n_items, hidden_dim=gen_hidden)
    gen_optimizer = optim.Adam(Gen.parameters(), lr=1e-5, eps=1e-6)
    Gen = Gen.to(device)

    netUtil = ModelUtil(Gen, gen_optimizer, 1, 'MyModel_GAN_8')

    net, optimizer = netUtil.load_model(path=path, name=model_name)
    test_DataLoader = getDataLoader(test_dataset, 10000, shuffle=False)
    predict = getPredict(test_DataLoader, Gen, n_items)

    predict_test = predict - 999 * train_mask_array
    test_dict = getTestDictFromDataFrame(test_PD=test_pd)
    train_dict = getTestDictFromDataFrame(test_PD=train_pd)
    if test_in_train:

        precision, recall, ndcg, var_ndcg, ied, ied_after, ied_change = testAdd1(rating_mask_array, predict, topK=topK,
                                                                                 test_dict=train_dict, PrT=True,
                                                                                 getIed=True,
                                                                                 Exposure_dataset=Exposure_dataset,
                                                                                 ied_score=ied_score, needAfterIED=True)
    else:
        precision, recall, ndcg, var_ndcg, ied, ied_after, ied_change = testAdd1(rating_mask_array, predict_test, topK=topK, test_dict=test_dict, PrT=True, getIed=True, Exposure_dataset=Exposure_dataset, ied_score=ied_score, needAfterIED=True)

    if getNet:
        return net
    if needPredict:
        return predict
    return precision, recall, ndcg, var_ndcg, ied, ied_after, ied_change

if __name__ == "__main__":
    device = torch.device('cuda:0')

    yaml_path = 'config.yaml'
    # yaml_path = 'config_xiaorong.yaml'
    config = get_config_data(yaml_path)

    mode = config['mode']
    Test = config['Test']
    print('#### mode:{} ####'.format(mode))

    if mode == 'Test':
        # print('Test')
        for DataSetName in Test:
            DataSet = Test[DataSetName]

            if DataSet['train']:
                fold_no_list = DataSet['fold_no']
                dataset_no = DataSet['dataset_no']
                # print(dataset_no)
                add_Expo = DataSet['add_Expo']
                use_step = DataSet['use_step']

                print('#### {}: Train add_expo:{} use_step:{} ####'.format(DataSetName, add_Expo, use_step))

                for no in fold_no_list:
                    print('###### train {}_no_{} ######'.format(DataSetName, no))

                    data = getData(dataset_no=dataset_no, fold_no=no)
                    train_pd, test_pd = data.get_dataset()
                    # rating_pd = pd.concat([train_pd, test_pd])

                    # train_pd, test_pd = pd.read_csv(r'Amazon-beauty/beauty_change_train.csv'), pd.read_csv(r'Amazon-beauty/beauty_change_test.csv')
                    rating_pd = pd.concat([train_pd, test_pd])
                    n_users = len(rating_pd['user'].unique())
                    n_items = len(rating_pd['item'].unique())

                    train_real_matrix = data.construct_real_matrix(train_pd, low=1.)
                    train_real_array, train_real_tensor, train_real_pd = getPd_Tensor_array(train_real_matrix)

                    train_mask_matrix = data.construct_one_valued_matrix(train_pd)
                    train_mask_array, train_mask_tensor, train_mask_pd = getPd_Tensor_array(train_mask_matrix)

                    test_real_matrix = data.construct_real_matrix(test_pd, low=1.)
                    test_real_array, test_real_tensor, test_real_pd = getPd_Tensor_array(test_real_matrix)

                    test_mask_matrix = data.construct_one_valued_matrix(test_pd)
                    test_mask_array, test_mask_tensor, test_mask_pd = getPd_Tensor_array(test_mask_matrix)

                    rating_raw_matrix = data.construct_real_matrix(rating_pd, low=1.)
                    rating_raw_array, rating_raw_tensor, rating_raw_pd = getPd_Tensor_array(rating_raw_matrix)

                    rating_mask_matrix = data.construct_one_valued_matrix(rating_pd)
                    rating_mask_array, rating_mask_tensor, rating_mask_pd = getPd_Tensor_array(rating_mask_matrix)

                    # dataset input (BS, n_users)

                    train_dataset = Amazon_MyModel_GAN8(rating=rating_mask_array)
                    test_dataset = Amazon_MyModel_GAN8(rating=rating_mask_array)

                    train_groupby_user = train_pd.groupby('user')
                    max_time_distance = 0
                    for user, items in train_groupby_user:
                        max_time_distance = len(items) if len(items) > max_time_distance else max_time_distance
                    print('max_time_distance:', max_time_distance)
                    time_distance = 10
                    ied_score, Exposure_dataset = getExposure_dataset(data=train_pd, n_items=n_items, n_users=n_users,
                                                                      time_distance=max_time_distance)
                    print('ied_score:', ied_score.item())

                    # path_prefix = r'log\MyModel_GAN_10\models\beauty'
                    # path = r'models_alpha0.2_epoch22000_pre_0.12661982825917253'
                    # topK_list = [5, 10, 15, 50, 100, 200]
                    #
                    # model_metric = load_and_test(model_name='MyModel_GAN_8', topK=topK_list, path=r'{}\{}'.format(path_prefix, path))
                    # # load_and_test(model_name='MyModel_GAN_8', topK=10, path=r'{}\{}'.format(path_prefix, path))
                    # # load_and_test(model_name='MyModel_GAN_8', topK=15, path=r'{}\{}'.format(path_prefix, path))
                    # print(model_metric)

                    main()
            else:
                print('#### {}: No Train ####'.format(DataSetName))
    else:
        print('XiaoRong')
        # add_Expo_list = [False, False, True, True]
        # use_step_list = [False, True, False, True]

        for DataSetName in Test:
            DataSet = Test[DataSetName]

            add_Expo_list = DataSet['add_Expo_list']
            use_step_list = DataSet['use_step_list']

            if DataSet['train']:
                fold_no_list = DataSet['fold_no']
                dataset_no = DataSet['dataset_no']
                # print(dataset_no)
                for add_Expo, use_step in zip(add_Expo_list, use_step_list):
                    print('#### {}: Train add_expo:{} use_step:{} ####'.format(DataSetName, add_Expo, use_step))

                    for no in fold_no_list:
                        print('###### train {}_no_{} ######'.format(DataSetName, no))

                        data = getData(dataset_no=dataset_no, fold_no=no)
                        train_pd, test_pd = data.get_dataset()
                        # rating_pd = pd.concat([train_pd, test_pd])

                        # train_pd, test_pd = pd.read_csv(r'Amazon-beauty/beauty_change_train.csv'), pd.read_csv(r'Amazon-beauty/beauty_change_test.csv')
                        rating_pd = pd.concat([train_pd, test_pd])
                        n_users = len(rating_pd['user'].unique())
                        n_items = len(rating_pd['item'].unique())

                        train_real_matrix = data.construct_real_matrix(train_pd, low=1.)
                        train_real_array, train_real_tensor, train_real_pd = getPd_Tensor_array(train_real_matrix)

                        train_mask_matrix = data.construct_one_valued_matrix(train_pd)
                        train_mask_array, train_mask_tensor, train_mask_pd = getPd_Tensor_array(train_mask_matrix)

                        test_real_matrix = data.construct_real_matrix(test_pd, low=1.)
                        test_real_array, test_real_tensor, test_real_pd = getPd_Tensor_array(test_real_matrix)

                        test_mask_matrix = data.construct_one_valued_matrix(test_pd)
                        test_mask_array, test_mask_tensor, test_mask_pd = getPd_Tensor_array(test_mask_matrix)

                        rating_raw_matrix = data.construct_real_matrix(rating_pd, low=1.)
                        rating_raw_array, rating_raw_tensor, rating_raw_pd = getPd_Tensor_array(rating_raw_matrix)

                        rating_mask_matrix = data.construct_one_valued_matrix(rating_pd)
                        rating_mask_array, rating_mask_tensor, rating_mask_pd = getPd_Tensor_array(rating_mask_matrix)

                        # dataset input (BS, n_users)

                        train_dataset = Amazon_MyModel_GAN8(rating=rating_mask_array)
                        test_dataset = Amazon_MyModel_GAN8(rating=rating_mask_array)

                        train_groupby_user = train_pd.groupby('user')
                        max_time_distance = 0
                        for user, items in train_groupby_user:
                            max_time_distance = len(items) if len(items) > max_time_distance else max_time_distance
                        print('max_time_distance:', max_time_distance)
                        time_distance = 10
                        ied_score, Exposure_dataset = getExposure_dataset(data=train_pd, n_items=n_items, n_users=n_users,
                                                                          time_distance=max_time_distance)
                        print('ied_score:', ied_score.item())


                        main()
            else:
                print('#### {}: No Train ####'.format(DataSetName))
