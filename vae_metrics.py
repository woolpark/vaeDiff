from typing import List

from torch import Tensor
from sewar.full_ref import rmse as rmse_
from sewar.full_ref import psnr as psnr_
from sewar.full_ref import uqi as uqi_
from sewar.full_ref import ssim as ssim_
from sewar.full_ref import ergas as ergas_
from sewar.full_ref import msssim as msssim_
from sewar.full_ref import scc as scc_
from sewar.full_ref import rase as rase_
from sewar.full_ref import sam as sam_
from sewar.full_ref import vifp as vifp_

lam = lambda x: x.data.squeeze(0).permute((1, 2, 0)).numpy()


def rmse(s: Tensor, t: Tensor):
    if s.is_cuda:
        s = s.cpu()
    if t.is_cuda:
        t = t.cpu()
    return rmse_(lam(s), lam(t))


def psnr(s: Tensor, t: Tensor):
    if s.is_cuda:
        s = s.cpu()
    if t.is_cuda:
        t = t.cpu()
    return psnr_(lam(s), lam(t), MAX=1)


def uqi(s: Tensor, t: Tensor):
    if s.is_cuda:
        s = s.cpu()
    if t.is_cuda:
        t = t.cpu()
    return uqi_(lam(s), lam(t))


def ssim(s: Tensor, t: Tensor):
    if s.is_cuda:
        s = s.cpu()
    if t.is_cuda:
        t = t.cpu()
    return ssim_(lam(s), lam(t), MAX=1)


def ergas(s: Tensor, t: Tensor):
    if s.is_cuda:
        s = s.cpu()
    if t.is_cuda:
        t = t.cpu()
    return ergas_(lam(s), lam(t))


def scc(s: Tensor, t: Tensor):
    if s.is_cuda:
        s = s.cpu()
    if t.is_cuda:
        t = t.cpu()
    return scc_(lam(s), lam(t))


def sam(s: Tensor, t: Tensor):
    if s.is_cuda:
        s = s.cpu()
    if t.is_cuda:
        t = t.cpu()
    return sam_(lam(s), lam(t))


def rase(s: Tensor, t: Tensor):
    if s.is_cuda:
        s = s.cpu()
    if t.is_cuda:
        t = t.cpu()
    return rase_(lam(s), lam(t))


def vifp(s: Tensor, t: Tensor):
    if s.is_cuda:
        s = s.cpu()
    if t.is_cuda:
        t = t.cpu()
    return vifp_(lam(s), lam(t))


def msssim(s: Tensor, t: Tensor):
    if s.is_cuda:
        s = s.cpu()
    if t.is_cuda:
        t = t.cpu()
    return msssim_(lam(s), lam(t), MAX=1)


def metric(s: Tensor, t: Tensor):
    """
    6 metrics: mse, psrn, qui, ssim, scc, vifp
    :param s:
    :param t:
    :return:
    """
    return {
        "RMSE": rmse(s, t),
        "PSRN": psnr(s, t),
        "SSIM": ssim(s, t),
        "UQI": uqi(s, t),
        "ERGAS": ergas(s, t),
        "VIFP": vifp(s, t),
        "RASE": rase(s, t),
        "SCC": scc(s, t),
        "SAM": sam(s, t),
        "MSSSIM": msssim(s, t),
    }


def metric_batch(source: List[Tensor], target: List[Tensor]):
    metrics = []
    for idx, t in enumerate(target):
        s = source[idx]
        metric = {
            "RMSE": rmse(s, t),
            "PSRN": psnr(s, t),
            "SSIM": ssim(s, t),
            "UQI": uqi(s, t),
            "ERGAS": ergas(s, t),
            "VIFP": vifp(s, t),
            "RASE": rase(s, t),
            "SCC": scc(s, t),
            "SAM": sam(s, t),
            "MSSSIM": msssim(s, t),
        }
        metrics.append(metric)
    return metrics
