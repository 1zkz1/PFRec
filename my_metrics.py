import pandas as pd
import numpy as np
import torch


def Precision_Recall(R_topK_dict: dict, T_dict, topK: int, getlist=False):
    # R: predict length:topK  T: test set
    R_T_num = 0
    R_num = 0
    T_num = 0
    precision_k_list = []
    recall_k_list = []

    for i, R_i in R_topK_dict.items():
        # i:userid
        T_i = T_dict[i]
        R_T_num += len(R_i & T_i)
        R_num += len(R_i)
        T_num += len(T_i)
        precision_k_list.append(len(R_i & T_i) / topK)
        recall_k_list.append(len(R_i & T_i) / len(T_i))

    # precision_K = R_T_num / R_num
    precision_K = R_T_num / (len(R_topK_dict) * topK)
    # recall_K = R_T_num / T_num
    recall_K = sum(recall_k_list) / len(T_dict)
    # print('precision@{}:{}'.format(topK, precision_K))
    # print('recall@{}:{}'.format(topK, recall_K))

    del T_i
    if getlist:
        return precision_K, recall_K, precision_k_list, recall_k_list
    else:
        return precision_K, recall_K

def Ndcg(pred_rel):
    '''pred_rel: list(rel(i))'''
    dcg = 0
    dcg_other = 0
    for (index, rel) in enumerate(pred_rel):
        dcg += (rel * np.reciprocal(np.log2(index+2)))
        # dcg_other += ((2**rel -1) * np.reciprocal(np.log2(index+2)))
    # print("dcg " + str(dcg))
    # print('dcg_other:', dcg_other)
    if dcg == 0:
        return 0.
    idcg = 0
    idcg_other = 0
    for(index,rel) in enumerate(sorted(pred_rel, reverse=True)):
        idcg += (rel * np.reciprocal(np.log2(index+2)))
        # idcg_other += ((2**rel -1) * np.reciprocal(np.log2(index+2)))
    # print("idcg " + str(idcg))
    # print('idcg_other:', idcg_other)
    # return dcg_other / idcg_other

    return dcg/idcg


class IED():

    def __init__(self, recom_TopK: torch.Tensor, n_items, n_users, topK=5):
        super(IED, self).__init__()
        self.IED = 0.
        self.recom_TopK = recom_TopK
        self.topK = topK
        # matrix (m, topK)
        self.n_items = n_items
        self.n_users = n_users
        self.expos = torch.zeros(size=(1, n_items)).squeeze()

    def cal_exposure(self, itemid=None, recom_TopK=None):
        recom_TopK = self.recom_TopK if recom_TopK is None else recom_TopK

        total = torch.zeros(size=(self.n_users, self.n_items))
        for i, recom_TopK_i in enumerate(recom_TopK):
            recom_topK = torch.topk(recom_TopK_i, k=self.topK).indices
            for j, data in enumerate(recom_topK):
                if data < self.n_items and data >= 0:
                    total[i][data] += self.cal_b_u_v(j)


        # expos_mean = expos / total
        self.expos = torch.mean(total, dim=0)

        if itemid is None:
            return self.expos
        else:
            return self.expos[itemid]

    def get_ied(self, y_pred, topK=None):
        topK = topK if topK is not None else self.topK
        position = torch.argsort(torch.argsort(y_pred, descending=True, dim=1), dim=1).float() + 1.
        mask = position <= topK
        bias = 1. / torch.log(1. + position)
        expo = torch.mean(bias * mask, dim=0)
        # print('expo shape:', expo.shape)
        return torch.sum(torch.abs(expo.unsqueeze(0) - expo.unsqueeze(1))) / (2. * self.n_items * torch.sum(expo))

    def cal_b_u_v(self, index):
        # b_u_v = np.reciprocal(np.log2(index+2.))
        b_u_v = np.reciprocal(np.log(index + 2.))
        return b_u_v

    def cal_IED(self, Add_expos=None):
        self.cal_exposure()
        expos = self.expos if Add_expos is None else self.expos + Add_expos
        above = 0.
        below = 0.
        above = torch.sum(torch.abs(expos.unsqueeze(0) - expos.unsqueeze(1)))
        below = 2 * self.n_items * torch.sum(expos)
        IED = above / below
        self.IED = IED
        return IED

    def cal_IED_by_recomTopK(self, recomTopK: torch.Tensor, getExpo=False, n_items=None, AddExpo=None):
        n_users, topK = recomTopK.shape
        n_items = n_items if n_items is not None else self.n_items
        total = torch.zeros(size=(n_users, n_items))
        for i, recom_topk in enumerate(recomTopK):
            # i：userid  recom_topk:recom to i.
            for j, data in enumerate(recom_topk):
                data = int(data)
                if data < self.n_items and data >= 0:
                    total[i][data] += self.cal_b_u_v(j)
                # expos[data] += self.cal_b_u_v(j)
        expos = torch.mean(total, dim=0)
        assert expos.shape[0] == self.n_items

        # 计算IED
        above = 0.
        below = 0.
        expos = expos if AddExpo is None else expos + AddExpo
        above = torch.sum(torch.abs(expos.unsqueeze(0) - expos.unsqueeze(1)))
        below = 2 * self.n_items * torch.sum(expos)
        IED = above / below

        if getExpo:
            return IED, expos
        return IED

    def calIedFromExpo(self, Expo):
        expos = Expo
        above = torch.sum(torch.abs(expos.unsqueeze(0) - expos.unsqueeze(1)))
        below = 2 * self.n_items * torch.sum(expos)
        IED = above / below

        return IED




if __name__ == "__main__":
    a = torch.Tensor([[0.4, 0.3, 0.2, 0.42, 0.76, 0.32],
                      [0.3, 0.1, 0.3, 0.6, 0.3, 0.64]])
    a_ied = IED(a, 6, 2)
    # expo = a_ied.cal_exposure()
    expo = a_ied.cal_IED()
    print(expo)
    print(a_ied.get_ied(a))
