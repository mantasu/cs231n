import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    
    norm_dot_product = (z_i/torch.linalg.norm(z_i)) @ (z_j/torch.linalg.norm(z_j))
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).
    
    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair. 
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
    
    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples
    
     # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
        ##############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Create lists of non-exponentiated unnormalized similarity scores
        sims_k = torch.tensor([sim(z_k, z) for z in out[np.arange(2*N)!=k]])
        sims_k_N = torch.tensor([sim(z_k_N, z) for z in out[np.arange(2*N)!=k+N]])

        # Compute l(k, k+N) and l(k+N, k), given the lists of similarity scores
        l1 = -((sim(z_k, z_k_N) / tau).exp() / (sims_k / tau).exp().sum()).log()
        l2 = -((sim(z_k_N, z_k) / tau).exp() / (sims_k_N / tau).exp().sum()).log()

        # Update the total loss
        total_loss += l1 + l2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
         ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
    
    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compute the normalized head projection outputs over each row of features
    norm_left = out_left / torch.linalg.norm(out_left, dim=1, keepdim=True)
    norm_right = out_right / torch.linalg.norm(out_right, dim=1, keepdim=True)

    # Compute the diagonal dot product directly by multiplying and summing
    pos_pairs = (norm_left * norm_right).sum(dim=1, keepdim=True)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    sim_matrix = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    ##############################################################################
    
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    norm_out = out / torch.linalg.norm(out, dim=1, keepdim=True)
    sim_matrix = norm_out @ norm_out.T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # TODO: Start of your code. Follow the hints.                                #
    ##############################################################################
    
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
    exponential = (sim_matrix / tau).exp().to(device)
    
    # This binary mask zeros out terms where k=i.
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
    # We apply the binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    denom = exponential.sum(dim=1)

    # Step 2: Compute similarity between positive pairs.
    # You can do this in two ways: 
    # Option 1: Extract the corresponding indices from sim_matrix. 
    # Option 2: Use sim_positive_pairs().
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    sim_pairs = sim_matrix[range(2*N), [*range(N, 2*N), *range(0, N)]]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Step 3: Compute the numerator value for all augmented samples.
    numerator = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    numerator = (sim_pairs / tau).exp().to(device)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    loss = -(numerator / denom).log().mean()
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return loss

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

pass

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))