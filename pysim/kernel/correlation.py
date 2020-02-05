def kernel_alignment(K_x: np.array, K_y: np.array, center: bool = False) -> float:
    """Gives a target kernel alignment score: how aligned the kernels are. Very
    useful for measures which depend on comparing two different kernels, e.g.
    Hilbert-Schmidt Independence Criterion (a.k.a. Maximum Mean Discrepency)
    
    Note: the centered target kernel alignment score is the same function
          with the center flag = True.
    
    Parameters
    ----------
    K_x : np.array, (n_samples, n_samples)
        The first kernel matrix, K(X,X')
    
    K_y : np.array, (n_samples, n_samples)
        The second kernel matrix, K(Y,Y')
        
    center : Bool, (default: False)
        The option to center the kernels (independently) before hand.
    
    Returns
    -------
    kta_score : float,
        (centered) target kernel alignment score.
    """

    # center kernels
    if center:
        K_x = KernelCenterer().fit_transform(K_x)
        K_y = KernelCenterer().fit_transform(K_y)

    # target kernel alignment
    return np.sum(K_x * K_y) / np.linalg.norm(K_x) / np.linalg.norm(K_y)
