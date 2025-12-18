import torch
import numpy as np
import joblib
from utils import *

def setup_source(FLAGS, device):
    if FLAGS.source == 'mle':
        mean_std = torch.load("data/cifar10_mean_std.pth")
        p_mean, p_sigma = mean_std['mean'].to(device), mean_std['std'].to(device)
        return {'sampler': None, 'mean': p_mean, 'std': p_sigma}
    else:
        p_mean, p_sigma = 0, 1
    if FLAGS.source == 'gaussian':
        return {'sampler': None, 'mean': p_mean, 'std': p_sigma}
    if FLAGS.source == 'gmm':
        pipe = joblib.load(f"data/cifar10_gmm{FLAGS.ncluster}_pca3072_full.pkl")
        stats = torch.load(f'data/cifar10_gmm{FLAGS.ncluster}_pca3072_full_stats.pt')
        # pca = pipe['pca']
        gmm = GMM_TORCH(gmm=pipe['gmm'], gmm_stats=stats, dtype=torch.float32, device=device)
        return {'sampler': gmm, 'mean': p_mean, 'std': p_sigma}
    if FLAGS.source == 'dct':
        rgb = False
        span = 3
        stats = torch.load(f'data/noise_dctmask{span*2}_{FLAGS.mask}_torch.pt', weights_only=True)
        root = 'data'
        dct = DCT_TORCH(span=span, dct_stats=stats, root=root, device=device, mask=FLAGS.mask, batch_size=FLAGS.batch_size, rgb=rgb)
        return {'sampler': dct, 'mean': p_mean, 'std': p_sigma}
    elif FLAGS.source == 'ffjord':
        ckpt_path = 'data/ffjord_ckpt.pth'
        ffjord = FFJORD(device=device, ckpt_path=ckpt_path)
        return {'sampler': ffjord, 'mean': p_mean, 'std': p_sigma}
    elif FLAGS.source == 'vonmises':
        if not FLAGS.vmf_all:
            raise ValueError("vonmises source requires --vmf_all=True flag. The non-vmf_all version is deprecated.")
        # Oracle vMF: use each data sample as mu
        vmf = VMF(params=None, kappa=FLAGS.kappa, normalize=FLAGS.normalize)
        return {'sampler': vmf, 'mean': p_mean, 'std': p_sigma}
    elif FLAGS.source == 'pruned':
        rjs = PrunedSample(ncandidates=1000, threshold=FLAGS.threshold, max_trials=10000)
        return {'sampler': rjs, 'mean': p_mean, 'std': p_sigma}
    elif FLAGS.source.startswith("sknn"):
        if FLAGS.flip:
            sknn = torch.load(f'./data/spherical_kmeans{FLAGS.ncluster}_flip.pt')
            if FLAGS.kappa != 0:
                sknn['kappa'] = torch.tensor(np.array([FLAGS.kappa] * FLAGS.ncluster), device=device)
            method = FLAGS.source.split("_")[1]
            sknn_sampler = SphericalKNN_Sampler(sknn=sknn, method=method, normalization=False, dtype=torch.float32, device=device)
            return {'sampler': sknn_sampler, 'mean': p_mean, 'std': p_sigma}
        else:
            sknn = torch.load(f'./data/spherical_knn{FLAGS.ncluster}.pt')
            method = FLAGS.source.split("_")[1]
            sknn_sampler = SphericalKNN_Sampler(sknn=sknn, method=method, normalization=True, dtype=torch.float32, device=device)
            return {'sampler': sknn_sampler, 'mean': p_mean, 'std': p_sigma}
    elif FLAGS.source.startswith("pca_pruned"):
        vectors = torch.load("data/pca_pruned_flip_under_1e-2.pt")
        pca_sampler = PCAPrunedSample(vectors, shape=[3, 32, 32], dtype=torch.float32, device=device, ncandidates=10, threshold=FLAGS.threshold, max_trials=100000)
        return {'sampler': pca_sampler, 'mean': p_mean, 'std': p_sigma}
    elif FLAGS.source == 'pca_acceptance':
        vectors = torch.load("data/pca_acceptance_flip_over_1e-1.pt")
        pca_sampler = PCAPrunedSample(vectors, shape=[3, 32, 32], dtype=torch.float32, device=device, ncandidates=1, threshold=FLAGS.threshold, max_trials=1000000)
        return {'sampler': pca_sampler, 'mean': p_mean, 'std': p_sigma}
    elif FLAGS.source.startswith("gmm_"):
        method = FLAGS.source.split("_")[1]
        gmm_ = torch.load(f'data/{method}{FLAGS.ncluster}_gmms.pt')
        gmm = GMM_TORCH(gmm=gmm_, dtype=torch.float32, device=device)
        # if FLAGS.guidance != 'None':
        #     kp = torch.tensor(torch.load(f'data/{FLAGS.source}{FLAGS.ncluster}_pairs.pt'), device=device, dtype=torch.float32)
        return {'sampler': gmm, 'mean': p_mean, 'std': p_sigma}
    elif FLAGS.source == 'mix_vmf_normchi':
        vmf = VMF(params=None, normalize=False)
        return {'sampler': vmf, 'mean': p_mean, 'std': p_sigma}
    else:
        # raise("Select correct source distribution")
        print("[WARNING] No source distribution is selected. Source distribution automatically becomes Gaussian Distribution.")
        return {'sampler': None, 'mean': p_mean, 'std': p_sigma}