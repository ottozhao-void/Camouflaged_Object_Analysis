import torch
import torch.nn.functional as F
import numpy as np

class SegmentationMetric:
    """
    Metric类用来跟踪MAE与S-Measure指标
    
    要度量的性能指标如下:
    1. S-Measure: "Structure-measure: A New Way to Evaluate Foreground Maps" 论文中提出
    2. Mean Absolute Error(MAE)
    """
    
    @staticmethod
    def get_mae(mae_list, reduce="average"):
        """
        根据reduce返回平均绝对误差的均值、加和或者MAE列表
        """
        assert reduce in ["average", "sum", "none"], f"Got unexpected reduce method: {reduce}"
        mae = torch.stack(mae_list, 0)

        if reduce == "average":
            return mae.mean()
        elif reduce == "sum":
            return mae.sum()
        else:
            return mae
    
    @staticmethod
    def get_smeasure(smeasure_list, reduce="average"):
        """
        根据reduce返回结构度量的均值、加和或者S-Measure列表
        """
        assert reduce in ["average", "sum", "none"], f"Got unexpected reduce method: {reduce}"
        sm = torch.stack(smeasure_list, 0)

        if reduce == "average":
            return sm.mean()
        elif reduce == "sum":
            return sm.sum()
        else:
            return sm
    
    @staticmethod
    def calculate_mae(pred, gt):
        if pred.ndim == 3:
            assert pred.shape[0] == 1, "只能是单张图像"
            pred = pred.squeeze(0)
        if gt.ndim == 3:
            assert gt.shape[0] == 1, "只能是单张图像"
            gt = gt.squeeze(0)
        assert gt.shape == pred.shape, "预测图像与真值图像的尺寸必须一致"
        
        return torch.abs(pred - gt).mean()
        
    @staticmethod
    def calculate_smeasure(pred, gt, alpha=0.5):
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
            Q = alpha * SegmentationMetric._S_object(pred, gt) + (1-alpha) * SegmentationMetric._S_region(pred, gt)
            if Q.item() < 0:
                Q = torch.FloatTensor([0.0])
        if isinstance(Q, float):
            raise ValueError("Q is a float")
        return Q

    @staticmethod
    def _S_object(pred, gt):
        fg = torch.where(gt==0, torch.zeros_like(pred), pred)
        bg = torch.where(gt==1, torch.zeros_like(pred), 1-pred)
        o_fg = SegmentationMetric._object(fg, gt)
        o_bg = SegmentationMetric._object(bg, 1-gt)
        u = gt.mean()
        Q = u * o_fg + (1-u) * o_bg
        return Q

    @staticmethod
    def _object(pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
        return score

    @staticmethod
    def _S_region(pred, gt):
        X, Y = SegmentationMetric._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = SegmentationMetric._divideGT(gt, X, Y)
        p1, p2, p3, p4 = SegmentationMetric._dividePrediction(pred, X, Y)
        Q1 = SegmentationMetric._ssim(p1, gt1)
        Q2 = SegmentationMetric._ssim(p2, gt2)
        Q3 = SegmentationMetric._ssim(p3, gt3)
        Q4 = SegmentationMetric._ssim(p4, gt4)
        Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
        return Q
    
    @staticmethod
    def _centroid(gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            X = torch.eye(1) * round(cols / 2)
            Y = torch.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            i = torch.from_numpy(np.arange(0,cols)).float()
            j = torch.from_numpy(np.arange(0,rows)).float()
            X = torch.round((gt.sum(dim=0)*i).sum() / total)
            Y = torch.round((gt.sum(dim=1)*j).sum() / total)
        return X.long(), Y.long()
    
    @staticmethod
    def _divideGT(gt, X, Y):
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

    @staticmethod
    def _dividePrediction(pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    @staticmethod
    def _ssim(pred, gt):
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
