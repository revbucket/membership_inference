""" Basic tools for computing pruning metrics:

Easiest pruning metrics:
- El2N : https://arxiv.org/abs/2107.07075
- Forgetting score:
- (Un)Supervised Prototypes:

"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
#from kmeans_pytorch import kmeans
from torch.cuda.amp import GradScaler, autocast
from fast_pytorch_kmeans import KMeans





# ===========================================================
# =           El2N + GraND                                  =
# ===========================================================
"""
Engineering notes:
- Build function to compute el2N for a minibatch
- Build function to compute GraND for a minibatch
n- Build function to stash El2N/GraND in mongo for examples
"""

@torch.no_grad()
def el2n_minibatch(model, minibatch, num_classes):
    """ Comptues El2n for a single model/batch
        Returns a vector of scores (of size |minibatch|)
    """
    with autocast():
        x, y = minibatch[0], minibatch[1]
        prob_vec = torch.softmax(model(x), dim=1)
        one_hot = F.one_hot(y, num_classes=num_classes).type(prob_vec.dtype)

        return torch.norm(one_hot - prob_vec, p=2, dim=1)


def grand_minibatch(model, minibatch):
    """ Computes GraND for a single model/batch
        No real fast way to do this, so have to loop over points
    """


    @torch.no_grad()
    def gradnorm(model=model):
        return torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters())).item()


    x, y = minibatch[0], minibatch[1]
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    output = torch.zeros(x.shape[0])
    with autocast():
        for i, (subx, suby) in enumerate(zip(x, y)):
            optimizer.zero_grad()
            suby = torch.LongTensor([suby]).to(subx.device)
            loss = F.cross_entropy(model(subx[None]), suby)
            loss.backward()
            output[i] = gradnorm(model)
        return output


def el2n_grand_loader(model, dataloader, scores=None, num_classes=100):
    """ Computes the el2n/grand score for a dataloader
    ARGS:
        model: standard pytorch model argument
        dataloader: should be an INDEX dataset [(x, y, idx), ...]
    RETURNS:
        collection of tuples like:
            [(idx, score1, score2,...)]
    """

    if scores is None:
        scores = ['el2n', 'grand']


    all_outputs = []

    for batch in dataloader:
        idxs = batch[-1]
        score_list = []
        if 'el2n' in scores:
            score_list.append(el2n_minibatch(model, batch, num_classes))
        if 'grand' in scores:
            score_list.append(grand_minibatch(model, batch))

        for idx, score_tup in zip(idxs, zip(*score_list)):
            all_outputs.append(idx + score_tup)

    return all_outputs







# =============================================
# =           SwaV Prototyping                =
# =============================================
@torch.no_grad()
def load_swav_model(device):
    swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    swav_model.fc = nn.Identity()
    swav_model = swav_model.eval().to(device)
    return swav_model

@torch.no_grad()
def make_selfsup_prototypes(swav_model, dataloader, k, device='cpu'):
    """ Uses SwaV to make k self-supervised prototypes
    Steps:
    1. Load SwaV model and compute reps for all datapoints
    2. Run kmeans on datapoints to collect clusters
    3. Return prototypes
    """

    reps = []
    for batch in dataloader:
        x = batch[0].to(device)
        reps.append(swav_model(x).cpu())

    all_reps = torch.cat(reps)
    kmeans = KMeans(n_clusters=k, mode='euclidean', verbose=1)
    _ = kmeans.fit_predict(all_reps)
    return kmeans.centroids



@torch.no_grad()
def selfsup_cluster_minibatch(swav_model, minibatch, cluster_centers, device):
    """ Computes a score as the minimum distance to a cluster center
        for each el in the minibatch
    ARGS:
        swav_model:
        minibatch: tuple where first el is data

    """

    reps = swav_model(x.to(device))
    k = cluster_centers.shape[0]
    N = reps.shape[0]
    # Need to compute pairwise distances (i, j)

    rep_exp = reps.unsqueeze(0).expand((k,) + reps.shape)
    cen_exp = cluster_centers.unsqueeze(1)
    return (rep_exp - cen_exp).pow(2).sum(dim=2).min(dim=1)[0].cpu().data


def selfsup_prototype_loader(dataloader, device, cluster_centers=None, k=None):
    swav_model = load_swav_model(device)
    if cluster_centers is None:
        cluster_centers = make_selfsup_prototypes(swav_model, dataloader, k, device)


    all_outputs = []
    for batch in dataloader:
        idxs = batch[-1]
        score = selfsup_cluster_minibatch(swav_model, batch, cluster_centers, device)

        for pair in zip(idxs, score):
            all_outputs.append(pair)
    return all_outputs

