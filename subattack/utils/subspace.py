import torch


def create_basis(pool, subspace_size):

    assert subspace_size <= pool.shape[0]

    # svd is memory eager
    # it has to be computed on cpu
    device = pool.device
    pool = pool.cpu()

    _, _, v = torch.svd(pool.view(pool.shape[0], -1))
    v = v.t()[:subspace_size]
    basis = v.view(v.shape[0], *pool.shape[1:])

    return basis.to(device)


def create_basis_inverted(pool, subspace_size):
    assert subspace_size <= pool.shape[0]

    # svd is memory eager
    # it has to be computed on cpu
    device = pool.device
    pool = pool.cpu()

    _, _, v = torch.svd(pool.view(pool.shape[0], -1))
    v = v.t()[-subspace_size:]
    basis = v.view(v.shape[0], *pool.shape[1:])

    return basis.to(device)


def create_basis_by_qr(pool):

    result, _ = torch.qr(pool.view(pool.shape[0], -1).t())
    return result.t().view(pool.shape)


def create_quasi_basis(differences, epsilon=1e-7):

    return (
        differences.view(differences.shape[0], -1)
        / differences.view(differences.shape[0], -1).norm(
            dim=1, keepdim=True).clamp(min=epsilon)
    ).view(differences.shape)
