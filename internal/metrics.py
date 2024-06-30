import math
from typing import Dict, List

import lpips
import torch
import torch.nn.functional as F

eps = 1e-8


class BaseMetricCalculator:
    """
    Base class for all metric calculators.
    """

    def __init__(
            self,
            divide_mask_ratio: bool = True,
            scale_invariant: bool = False,
    ):
        """
        Initialize the metric calculator.
        Args:
            divide_mask_ratio (bool): whether to divide the metric by the mask
                ratio.
            scale_invariant (bool): whether to scale the prediction to match
                the ground truth prior to computing the metric.
        """
        self.divide_mask_ratio = divide_mask_ratio
        self.scale_invariant = scale_invariant

    def __call__(
            self,
            pred: torch.Tensor,
            gt: torch.Tensor,
            mask: torch.Tensor = None,
            **kwargs
    ) -> torch.Tensor:
        """
        Compute some metric for the input two images `pred` and `gt` within
        the mask.
        Args:
            pred: (..., c) a batch of ray attributes,
            gt: (..., c) another batch of ray attributes,
            mask: (..., 1), the image mask within which error is computed.
                If not given, assumes all one.

        Returns:
            torch.Tensor: scalar, average metric. Put on 'cpu' device.
        """
        # pre-process
        if mask is None:
            mask = torch.ones(*pred.shape[:-1], 1, device=pred.device)
        else:
            assert len(mask.shape) == len(pred.shape) and mask.shape[-1] == 1
        pred = pred * mask
        gt = gt * mask

        if self.scale_invariant:
            # per-channel intensity adjustment
            dims = torch.arange(0, len(pred.shape) - 1).tolist()
            pred *= (gt.sum(dim=dims, keepdim=True)
                     / pred.sum(dim=dims, keepdim=True).clamp(min=eps))

        # compute
        metric = self.calc(pred, gt, **kwargs)

        # post-process
        if self.divide_mask_ratio:
            mask_ratio = (torch.sum(mask) / torch.tensor(mask.shape).prod())
            metric = metric / mask_ratio

        return metric.cpu()

    def calc(
            self,
            pred: torch.Tensor,
            gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute some metric for the input two images `pred` and `gt` (masked).
        Args:
            pred: (..., c) a batch of ray attributes,
            gt: (..., c) another batch of ray attributes.

        Returns:
            torch.Tensor: scalar, average metric, undivided by mask size.
        """
        raise NotImplementedError


class MeanLpDistCalculator(BaseMetricCalculator):
    """
    Compute the mean of differences to the power of p for two input images.
    """

    def __init__(self, p: float, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def calc(
            self,
            pred: torch.Tensor,
            gt: torch.Tensor,
    ) -> torch.Tensor:
        gt = gt.view(-1, gt.shape[-1])
        pred = pred.view(-1, pred.shape[-1])
        return (torch.abs(gt - pred) ** self.p).mean()


class SSIMCalculator(BaseMetricCalculator):
    """modified from https://github.com/jorge-pessoa/pytorch-msssim"""

    def __init__(self, w_size=11, **kwargs):
        super().__init__(**kwargs)
        self.w_size = w_size

    def gaussian(self, w_size, sigma):
        """
        Generate 1D Gaussian kernel.
        Args:
            w_size (int): the size of gaussian kernel.
            sigma (float): sigma of Gaussian distribution.

        Returns:
            torch.Tensor: [w_size] 1D Gaussian kernel.
        """
        gauss = torch.tensor(
            [math.exp(
                -(x - w_size // 2) ** 2
                / float(2 * sigma ** 2))
                for x in range(w_size)])
        return gauss / gauss.sum()

    def create_window(self, w_size: int, channel: int):
        """
        Create Gaussian window.
        Args:
            w_size (int): the size of gaussian kernel.
            channel (int): the number of channels of the window.

        Returns:
            torch.Tensor: [channel, 1, w_size, w_size] Gaussian window.
        """
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float(). \
            unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def calc(
            self,
            pred: torch.Tensor,
            gt: torch.Tensor,
            w_size: int = None,
    ):
        """Computes SSIM for the two input images `pred` and `gt`.

        Args:
            pred (torch.Tensor): [h, w, c] predicted image
            gt (torch.Tensor): [h, w, c] ground truth image
            w_size (int): the size of gaussian kernel, default to 11.

        Return:
            torch.Tensor: scalar, SSIM, larger the better.
        """
        if w_size is None:
            w_size = self.w_size

        # Value range can be different from 255. Other common ranges are 1
        # (sigmoid) and 2 (tanh).
        if torch.max(pred) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(pred) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val

        # change size
        pred = pred.permute(2, 0, 1).unsqueeze(0)
        gt = gt.permute(2, 0, 1).unsqueeze(0)

        padd = 0
        (_, channel, height, width) = pred.size()
        window = self.create_window(w_size, channel=channel).to(pred.device)

        mu1 = F.conv2d(pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(gt, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            pred * pred, window, padding=padd,
            groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(
            gt * gt, window, padding=padd,
            groups=channel) - mu2_sq
        sigma12 = F.conv2d(
            pred * gt, window, padding=padd,
            groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        # cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        # if size_average:
        #     ret = ssim_map.mean()
        # else:
        ret = ssim_map.mean(1).mean(1).mean(1)

        # if full:
        #     return ret, cs
        return ret[0]


class AngularErrorCalculator(BaseMetricCalculator):
    """
    Modified from matlab : colorangle.m, MATLAB V2019b
    angle = acos(RGB1' * RGB2 / (norm(RGB1) * norm(RGB2)));
    angle = 180 / pi * angle;
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calc(
            self,
            pred: torch.Tensor,
            gt: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the angular error between the two input `pred` and `gt`.
        Args:
            pred: (..., 3) the predicted normal
            gt: (..., 3) the ground truth normal

        Returns:
            torch.Tensor: scalar, angular error, smaller the better.
        """
        dot_prod = torch.sum(pred * gt, dim=-1)
        pred_norm = torch.sqrt(torch.sum(pred * pred, dim=-1))
        gt_norm = torch.sqrt(torch.sum(gt * gt, dim=-1))
        cosine = dot_prod / (pred_norm * gt_norm).clamp(min=eps)
        cosine[gt_norm < eps] = 1  # let ae = 0 when gt_norm = 0
        cosine = cosine.clamp(min=-1, max=1)
        ae = 180 / math.pi * torch.acos(cosine)
        # assert ae.shape == pred.shape[:-1]
        return ae.mean()


class LPIPSCalculator(BaseMetricCalculator):
    def __init__(self, net_device: torch.device, **kwargs):
        super().__init__(**kwargs)
        self.lpips_net = lpips.LPIPS(net="vgg").to(net_device)

    @torch.no_grad()
    def calc(
            self,
            pred: torch.Tensor,
            gt: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
        gt = gt.unsqueeze(0).permute(0, 3, 1, 2)
        metric = self.lpips_net(pred, gt)
        return metric.mean()



class MetricsGatherer(object):
    """
    Complete error metrics.
    """

    def __init__(
            self,
            measuring_list: List[str],
            net_device: torch.device,
            within_mask: bool = False,
            compute_extra_metrics: bool = False,
    ):
        """
        Build the error metric calculators.
        Args:
            measuring_list: the list of error metrics to measure.
            net_device: the device for some network (e.g. vgg for lpips) to
                compute the metrics. Note that, the metrics are still
                put on 'cpu' device.
            within_mask: whether to compute metrics only within the mask. May
                worsen the metric. Exceptions: angular error (always), SSIM
                (never), lpips (never).
            compute_extra_metrics: whether to compute LPIPS and SSIM in
                addition to PSNR.
        """
        self.supported_metrics = [
            'nr_rgb',
            'pbr_rgb',
            'pbr_rgb_diff',
            'pbr_rgb_spec',
            'normal',
            'albedo',
        ]
        for metric in measuring_list:
            assert metric in self.supported_metrics, (
                f'Unsupported metric: {metric}')
        self.measuring_list = measuring_list
        self.compute_extra_metrics = compute_extra_metrics

        self.device = net_device

        self.mse_fn = MeanLpDistCalculator(
            p=2, scale_invariant=False,
            divide_mask_ratio=within_mask)
        self.rmse_fn = lambda pred, gt, mask: (
            self.mse_fn(pred, gt, mask).sqrt())
        self.psnr_fn = lambda pred, gt, mask: (
                -10 * torch.log10(self.mse_fn(pred, gt, mask)))

        self.si_mse_fn = MeanLpDistCalculator(
            p=2, scale_invariant=True,
            divide_mask_ratio=within_mask)
        self.si_rmse_fn = lambda pred, gt, mask: (
            self.si_mse_fn(pred, gt, mask).sqrt())
        self.si_psnr_fn = lambda pred, gt, mask: (
                -10 * torch.log10(self.si_mse_fn(pred, gt, mask)))

        self.lpips_fn = LPIPSCalculator(
            scale_invariant=False,
            divide_mask_ratio=False,
            net_device=net_device
        )
        self.ssim_fn = SSIMCalculator(
            scale_invariant=False,
            divide_mask_ratio=False)
        self.angular_error_fn = AngularErrorCalculator(
            scale_invariant=False,
            divide_mask_ratio=True)

    def delete_from_measuring_list(self, delete_list: List[str]):
        """Delete the error metrics to measure.

        Args:
            delete_list: the list of deleted error metrics to measure.

        """
        for metric in delete_list:
            assert metric in self.supported_metrics, (
                f'Unsupported metric: {metric}')
        for metric in delete_list:
            self.measuring_list.remove(metric)

    def extend_measuring_list(self, new_list: List[str]):
        """Extend the list of error metrics to measure.

        Args:
            new_list: the list of added error metrics to measure.

        """
        for metric in new_list:
            assert metric in self.supported_metrics, (
                f'Unsupported metric: {metric}')
            if metric not in self.measuring_list:
                self.measuring_list.append(metric)

    def __call__(
            self,
            pred: Dict[str, torch.Tensor],
            gt: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute the error metrics for the input two dicts of ray attributes.

        Args:
            pred: predicted ray attributes as a dict.
            gt: Ground truth ray attributes as a dict.

        Return:
            Dict[str, torch.Tensor]: the error metrics as a dict. Note that all
                the values are put on 'cpu' device.
        """
        error_dict_all = {}
        mask: torch.Tensor = torch.ge(gt['alpha'], 0.5)
        if 'nr_rgb' in self.measuring_list:
            error_dict_all['rgb_PSNR'] = self.psnr_fn(
                pred['nr_rgb'], gt['rgb'], mask)

        if 'pbr_rgb' in self.measuring_list:
            error_dict_all['pbr_rgb_PSNR'] = self.psnr_fn(
                pred['pbr_rgb'], gt['rgb'], mask)
            if self.compute_extra_metrics:
                error_dict_all['pbr_rgb_LPIPS'] = self.lpips_fn(
                    pred['pbr_rgb'], gt['rgb'], mask)
                error_dict_all['pbr_rgb_SSIM'] = self.ssim_fn(
                    pred['pbr_rgb'], gt['rgb'], mask)

        if 'pbr_rgb_diff' in self.measuring_list and gt['rgb_diff'] is not None:
            error_dict_all['pbr_rgb_diff_PSNR'] = self.psnr_fn(
                pred['pbr_rgb_diff'], gt['rgb_diff'], mask)
            if self.compute_extra_metrics:
                error_dict_all['pbr_rgb_diff_LPIPS'] = self.lpips_fn(
                    pred['pbr_rgb_diff'], gt['rgb_diff'], mask)
                error_dict_all['pbr_rgb_diff_SSIM'] = self.ssim_fn(
                    pred['pbr_rgb_diff'], gt['rgb_diff'], mask)

        if 'pbr_rgb_spec' in self.measuring_list and gt['rgb_spec'] is not None:
            error_dict_all['pbr_rgb_spec_PSNR'] = self.psnr_fn(
                pred['pbr_rgb_spec'], gt['rgb_spec'], mask)
            if self.compute_extra_metrics:
                error_dict_all['pbr_rgb_spec_LPIPS'] = self.lpips_fn(
                    pred['pbr_rgb_spec'], gt['rgb_spec'], mask)
                error_dict_all['pbr_rgb_spec_SSIM'] = self.ssim_fn(
                    pred['pbr_rgb_spec'], gt['rgb_spec'], mask)

        if 'normal' in self.measuring_list and gt['normal'] is not None:
            error_dict_all['normal_MAngE'] = self.angular_error_fn(
                pred['normal'], gt['normal'], mask)

        if 'albedo' in self.measuring_list and gt['albedo'] is not None:
            # scale-invariant measurement for albedo
            error_dict_all['albedo_PSNR'] = self.si_psnr_fn(
                pred['base_color'], gt['albedo'], mask)
            if self.compute_extra_metrics:
                error_dict_all['albedo_LPIPS'] = self.lpips_fn(
                    pred['base_color'], gt['albedo'], mask)
                error_dict_all['albedo_SSIM'] = self.ssim_fn(
                    pred['base_color'], gt['albedo'], mask)

        return error_dict_all


def test_error_gatherer():
    h, w, c = 800, 800, 3
    vec_a = (torch.tensor((0.99, 0, 0), dtype=torch.float32)
             .view(-1, 3))
    vec_b = (torch.tensor((1, 0, 0), dtype=torch.float32)
             .view(-1, 3))

    pred_normal = torch.randn(h, w, 3, dtype=torch.float32)
    gt_normal = torch.randn(h, w, 3, dtype=torch.float32)
    mask = torch.ones(h, w, 1, dtype=torch.float32)

    mean_ang_err_fn = AngularErrorCalculator(
        divide_mask_ratio=False,
        scale_invariant=False)
    mean_ang_err = mean_ang_err_fn(pred_normal, gt_normal, mask)
    print('mean_ang_err:', mean_ang_err)

    rgb_pred = torch.rand(h * w, 3, dtype=torch.float32)
    rgb_gt = torch.rand(h * w, 3, dtype=torch.float32)
    mask = torch.ones(h * w, 1, dtype=torch.float32)

    metrics_gatherer = MetricsGatherer(
        measuring_list=['nr_rgb'],
        net_device=torch.device('cpu'),
        within_mask=False)
    error_dict = metrics_gatherer(
        pred={'nr_rgb': rgb_pred},
        gt={'rgb': rgb_gt, 'mask': mask})
    print('error_dict:', error_dict)


if __name__ == "__main__":
    test_error_gatherer()
