# adapted from https://github.com/dbolya/tomesd
import torch
import numpy as np
from typing import Tuple, Callable


def do_nothing(x: torch.Tensor, mode:str=None):
    return x

def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)
    
def init_generator(device: torch.device, fallback: torch.Generator=None):
    """Forks the current default random generator given device."""
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback


def find_similarities(metric: torch.Tensor,
                      w: int, sx: int, r: int,
                      no_rand: bool = False,
                      generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions captions into src and dst subsets and return the indices to be merged.
    Dst captions are partitioned by choosing one randomy in each sx window.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: number of captions
     - sx: stride to sample dst
     - r: number of captions to remove (by merging)
     - no_rand: if true, disable randomness. samples first idx in each sx window as dst
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape
    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        wsx = w // sx

        # For each sx window, randomly assign one caption to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sx, size=(wsx, 1), device=generator.device, generator=generator).to(metric.device)
        
        # Scatter sampling indices of dst (-1) and src (0) sets
        idx_buffer_view = torch.zeros(wsx, sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=1, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.reshape(wsx * sx)

        # If caption-set is not divisible by sx, pad the rest to be src
        if (wsx * sx) < w:
            idx_buffer = torch.zeros(w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(-1, 1).argsort(dim=0)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = wsx
        a_idx = rand_idx[num_dst:, :] # src
        b_idx = rand_idx[:num_dst, :] # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the #captions in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged captions
        src_idx = edge_idx[..., :r, :]  # Merged captions
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

        # collect infices corresponding to the input caption-set, for each unm-src, mer-src and dst sets
        ori_unm_idx = gather(a_idx.unsqueeze(0), dim=-2, index=unm_idx).view(-1)
        ori_src_idx = gather(a_idx.unsqueeze(0), dim=-2, index=src_idx).view(-1)
        ori_dst_idx = gather(b_idx.unsqueeze(0), dim=-2, index=dst_idx).view(-1)
        b_idx = b_idx.view(-1)
    return ori_unm_idx, ori_src_idx, ori_dst_idx, b_idx


def group_captions(all_nar, offset, key_map, unm_idx, src_idx, srcdst_idx, dst_idx, debug=False):
    """
    Parse captions based on merging indices. Captions to be rephrased are grouped together.

    Args:
     - all_nar: set of all captions for each chunk
     - offset: chunk id
     - key_map: keys/timestamps of grouped captions in previous iteration (None for first iteration)
     - unm_idx: indices in src, that are not merged
     - src_idx: indices in src, that are merged
     - srcdst_idx: where each src idx to be merged maps to in dst
     - dst_idx: indices in dst
     - debug: if True, prints matched captions
    """
    unm_idx, src_idx, srcdst_idx, dst_idx = np.array(unm_idx), np.array(src_idx), np.array(srcdst_idx), np.array(dst_idx)
    unm_idx, dst_idx = np.sort(unm_idx), np.sort(dst_idx)
    all_nar = np.array(all_nar)

    # sample subsets of captions based on indices
    unm_nar = all_nar[unm_idx]
    src_nar = all_nar[src_idx]
    srcdst_nar = all_nar[srcdst_idx]
    dst_nar = all_nar[dst_idx]

    # compute frame ids by adding chunk offset
    if not key_map:
        unm_idx, src_idx, srcdst_idx, dst_idx = unm_idx+offset, src_idx+offset, srcdst_idx+offset, dst_idx+offset

    grouped = {}
    
    # unmered captions directly added as a group entry
    print_unm = '[unmerged] ...\n'
    for i, ni in zip(unm_idx, unm_nar):
        print_unm += f'({"{:2d}".format(i)}) {ni}\n'
        if key_map:
            grouped[key_map[i]] = [(ni,key_map[i])]
        else:
            grouped[i] = [(ni,i)]
    
    # destination captions directly added as a group entry
    print_dst = '[destination] ...\n'
    for i, ni in zip(dst_idx, dst_nar):
        print_dst += f'({"{:2d}".format(i)}) {ni}\n'
        if key_map:
            grouped[key_map[i]] = [(ni,key_map[i])]
        else:
            grouped[i] = [(ni,i)]
    
    # source captions to be merged, added to the corresponding destination entry in group
    print_srcdst = '[merged; src --> dst] ...\n'
    for i, ni, j, nj in zip(src_idx, src_nar, srcdst_idx, srcdst_nar):
        print_srcdst += f'({"{:2d}".format(i)}) --> ({"{:2d}".format(j)}) {ni} --> {nj}\n'
        if key_map:
            grouped[key_map[j]].append((ni,key_map[i]))
        else:
            grouped[j].append((ni,i))

    # for debugging
    if debug:
        print(print_unm + print_dst + print_srcdst)
    return grouped
