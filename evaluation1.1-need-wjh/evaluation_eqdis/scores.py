import numpy as np
import datetime
import os


# 只对预测结果取邻域。邻域内有一个闪电就算有闪电。边界的部分不填充0，直接砍掉。
class Cal_params_neighbor(object):
    def __init__(self, neighbor_size=0):
        # n1->TP  n2->FP  n3->FN  n4->TN
        self.n1sum = 0
        self.n2sum = 0
        self.n3sum = 0
        self.n4sum = 0
        self.neighbor_size = neighbor_size
        self.eps = 1e-10

    def _cal_neighbor(self, y_true, y_pred):
        m = y_true.shape[0]
        n = y_true.shape[1]
        # y_true_neighbor = y_true[self.neighbor_size:m-self.neighbor_size, self.neighbor_size:n-self.neighbor_size]
        # y_pred_neighbor = np.zeros_like(y_true_neighbor)
        y_pred_neighbor = y_pred[self.neighbor_size:m-self.neighbor_size, self.neighbor_size:n-self.neighbor_size]
        y_true_neighbor = np.zeros_like(y_pred_neighbor)
        for i in range(self.neighbor_size, m-self.neighbor_size):
            for j in range(self.neighbor_size, n-self.neighbor_size):
                tmp = np.sum(y_true[i - self.neighbor_size: i + self.neighbor_size + 1, j - self.neighbor_size: j + self.neighbor_size + 1])
                y_true_neighbor[i - self.neighbor_size, j - self.neighbor_size] = tmp
        y_true_neighbor[y_true_neighbor > 1] = 1
        return y_true_neighbor, y_pred_neighbor

    def _cal_fss(self, y_true, y_pred):
        m = y_true.shape[0]
        n = y_true.shape[1]
        fbs = 0
        nxy = 0
        sumo2 = 0
        summ2 = 0
        for i in range(self.neighbor_size, m-self.neighbor_size):
            for j in range(self.neighbor_size, n-self.neighbor_size):
                on = np.sum(y_true[i - self.neighbor_size: i + self.neighbor_size + 1, j - self.neighbor_size: j + self.neighbor_size + 1]) / (2 * self.neighbor_size + 1)
                mn = np.sum(y_pred[i - self.neighbor_size: i + self.neighbor_size + 1, j - self.neighbor_size: j + self.neighbor_size + 1]) / (2 * self.neighbor_size + 1)
                sumo2 += on ** 2
                summ2 += mn ** 2
                fbs += (on - mn) ** 2
                nxy += 1
        fbs = fbs / nxy
        fss = 1 - fbs / ((sumo2 + summ2) / nxy + self.eps)
        return fss

    def _POD_(self, n1, n3):
        if n1 == 0 and n3 == 0:
            return 9999
        return np.true_divide(n1, n1 + n3)

    def _FAR_(self, n1, n2):
        if n1 == 0 and n2 == 0:
            return 9999
        return np.true_divide(n2, n1 + n2)

    def _TS_(self, n1, n2, n3):
        if n1 == 0 and n2 == 0 and n3 == 0:
            return 9999
        return np.true_divide(n1, n1 + n2 + n3)

    def _ETS_(self, n1, n2, n3, n4):
        r = np.true_divide((n1 + n2) * (n1 + n3), n1 + n2 + n3 + n4 + self.eps)
        return np.true_divide(n1 - r, n1 + n2 + n3 - r + self.eps)

    def _FOM_(self, n1, n3):
        if n1 == 0 and n3 ==0 :
            return 9999
        return np.true_divide(n3, n1 + n3 + self.eps)

    def _BIAS_(self, n1, n2, n3):
        if n1 == 0 and n2 == 0 and n3 ==0:
            return 9999
        return np.true_divide(n1 + n2, n1 + n3 + self.eps)

    def _HSS_(self, n1, n2, n3, n4):

        return np.true_divide(2 * (n1 * n4 - n2 * n3), (n1 + n3) * (n3 + n4) + (n1 + n2) * (n2 + n4) + self.eps)

    def _PC_(self, n1, n2, n3, n4):
        return np.true_divide(n1 + n4, n1 + n2 + n3 + n4 + self.eps)

    def _all_eval(self, n1, n2, n3, n4):
        pod = self._POD_(n1, n3)
        far = self._FAR_(n1, n2)
        ts = self._TS_(n1, n2, n3)
        ets = self._ETS_(n1, n2, n3, n4)
        fom = self._FOM_(n1, n3)
        bias = self._BIAS_(n1, n2, n3)
        hss = self._HSS_(n1, n2, n3, n4)
        pc = self._PC_(n1, n2, n3, n4)
        return {'POD': pod, 'FAR': far, 'TS': ts, 'ETS': ets, 'FOM': fom, 'BIAS': bias, 'HSS': hss, 'PC': pc, 'N1': n1, 'N2': n2, 'N3': n3, 'N4': n4}

    def cal_batch_sum(self, y_true, y_pred):
        y_true, y_pred = self._cal_neighbor(y_true, y_pred)
        n1 = np.sum((y_pred > 0) & (y_true > 0))
        n2 = np.sum((y_pred > 0) & (y_true < 1))
        n3 = np.sum((y_pred < 1) & (y_true > 0))
        n4 = np.sum((y_pred < 1) & (y_true < 1))
        self.n1sum += n1
        self.n2sum += n2
        self.n3sum += n3
        self.n4sum += n4
        return self._all_eval(n1, n2, n3, n4)

    def cal_params_ones(self, y_true, y_pred):
        y_true, y_pred = self._cal_neighbor(y_true, y_pred)
        fss = self._cal_fss(y_true, y_pred)
        n1 = np.sum((y_pred > 0) & (y_true > 0))
        n2 = np.sum((y_pred > 0) & (y_true < 1))
        n3 = np.sum((y_pred < 1) & (y_true > 0))
        n4 = np.sum((y_pred < 1) & (y_true < 1))
        res = self._all_eval(n1, n2, n3, n4)
        res['FSS'] = fss
        return res

class Cal_FSS(object):
    def __init__(self, neighbor_size=0):
        self.neighbor_size = neighbor_size
        self.eps = 1e-10

    def _cal_fss(self, y_true, y_pred):
        m = y_true.shape[0]
        n = y_true.shape[1]
        fbs = 0
        nxy = 0
        sumo2 = 0
        summ2 = 0
        for i in range(self.neighbor_size, m-self.neighbor_size):
            for j in range(self.neighbor_size, n-self.neighbor_size):
                on = np.sum(y_true[i - self.neighbor_size: i + self.neighbor_size + 1, j - self.neighbor_size: j + self.neighbor_size + 1]) / (2 * self.neighbor_size + 1)
                mn = np.sum(y_pred[i - self.neighbor_size: i + self.neighbor_size + 1, j - self.neighbor_size: j + self.neighbor_size + 1]) / (2 * self.neighbor_size + 1)
                sumo2 += on ** 2
                summ2 += mn ** 2
                fbs += (on - mn) ** 2
                nxy += 1
        fbs = fbs / nxy
        fss = 1 - fbs / ((sumo2 + summ2) / nxy + self.eps)
        return fss

if __name__ == "__main__":
    pass