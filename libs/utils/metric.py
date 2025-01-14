import torch
import torch.nn.functional as F
import numpy as np

class Metric():
    """
    Metric类用来跟踪MAE与S-Measure指标
    
    要度量的性能指标如下:
    1. S-Measure: “Structure-measure: A New Way to Evaluate Foreground Maps” 论文中提出
    2. Mean Absolute Error(MAE)
    """
    def __init__(self, device):
        
        self.device = device
        
        self.mae_list = []
        self.smeasure_list = []
        
        self.best_smeasure = 0.0
    
    def get_mae(self, reduce="avg"):
        """
        根据reduce返回平均绝对误差的均值、加和或者MAE列表
        """
        assert reduce in ["average", "sum", "none"], f"Got unexpected reduce method: {reduce}"
        mae = torch.stack(self.mae_list, 0)

        if reduce == "average":
            return mae.mean()
        elif reduce == "sum":
            return mae.sum()
        else:
            return mae
    
    def get_smeasure(self, reduce="avg"):
        """
        根据reduce返回结构度量的均值、加和或者S-Measure列表
        """
        assert reduce in ["average", "sum", "none"], f"Got unexpected reduce method: {reduce}"
        sm = torch.stack(self.smeasure_list, 0)

        if reduce == "average":
            return sm.mean()
        elif reduce == "sum":
            return sm.sum()
        else:
            return sm

    
    def add(self, pred_label, gt_label):
        """
        输入：单张掩码以及其对应的预测掩码
        计算两者之间的平均绝对误差MAE与结构度量S-Measure, 并添加到对应的列表中
        以方便后续处理（如求均值，加和等）
        """
        self.mae_list.append(self.calculate_mae(pred_label,gt_label ))
        self.smeasure_list.append(self.calculate_smeasure(pred_label, gt_label))
        
    
    def calculate_mae(self, pred, gt):
                
        if pred.ndim == 3:
            assert pred.shape[0] == 1, "只能是单张图像"
            pred = pred.squeeze(0)
        if gt.ndim == 3:
            assert gt.shape[0] == 1, "只能是单张图像"
            gt = gt.squeeze(0)
        assert gt.shape == pred.shape, "预测图像与真值图像的尺寸必须一致"
        
        
        return torch.abs(pred - gt).mean()
        

    def calculate_smeasure(self, pred, gt, alpha=0.5):
        
        assert gt.shape[1:] == pred.shape[1:], "预测图像与真值图像的尺寸必须一致"
        
        if pred.ndim == 3:
            assert pred.shape[0] == 1, "只能是单张图像"
            pred = pred.squeeze(0)
        if gt.ndim == 3:
            assert gt.shape[0] == 1, "只能是单张图像"
            gt = gt.squeeze(0)
            
        y = gt.mean()
        if y == 0:
            x = pred.mean()
            Q = 1.0 - x
        elif y == 1:
            x = pred.mean()
            Q = x
        else:
            gt[gt>=0.5] = 1
            gt[gt<0.5] = 0
            Q = alpha * self._S_object(pred, gt) + (1-alpha) * self._S_region(pred, gt)
            if Q.item() < 0:
                Q = torch.FloatTensor([0.0])
        
        return Q
    
    def reset(self):
        """
        在开始新的validation或者test之前，需要重置MAE与S-Measure列表
        """
        self.mae_list = []
        self.smeasure_list = []

 
    def _S_object(self, pred, gt):
        fg = torch.where(gt==0, torch.zeros_like(pred), pred)
        bg = torch.where(gt==1, torch.zeros_like(pred), 1-pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1-gt)
        u = gt.mean()
        Q = u * o_fg + (1-u) * o_bg
        return Q

    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
        
        return score

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
        # print(Q)
        return Q
    
    def _centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            X = torch.eye(1).to(self.device) * round(cols / 2)
            Y = torch.eye(1).to(self.device) * round(rows / 2)
        else:
            total = gt.sum()
            i = torch.from_numpy(np.arange(0,cols)).to(self.device).float()
            j = torch.from_numpy(np.arange(0,rows)).to(self.device).float()
            X = torch.round((gt.sum(dim=0)*i).sum() / total)
            Y = torch.round((gt.sum(dim=1)*j).sum() / total)
        return X.long(), Y.long()
    
    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h*w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h*w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x)*(pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y)*(gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + 1e-20)
        
        aplha = 4 * x * y *sigma_xy
        beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q