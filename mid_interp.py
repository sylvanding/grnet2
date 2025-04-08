from einops import rearrange, repeat
from pointops.functions import pointops

def get_knn_pts(k, pts, center_pts, return_idx=False):
    # input: (b, 3, n)
    # 1，GT，sample
    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, m, 3)
    center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
    # (b, m, k)
    knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
    # (b, 3, m, k)
    knn_pts = index_points(pts, knn_idx)

    if return_idx == False:
        return knn_pts
    else:
        return knn_pts, knn_idx

def midpoint_interpolate(sparse_pts, up_rate=4):
    # sparse_pts: (b, 3, 256)

    pts_num = sparse_pts.shape[-1]
    up_pts_num = int(pts_num * up_rate)
    k = int(2 * up_rate)
    # (b, 3, n, k)
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    # (b, 3, n, k)
    repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
    # (b, 3, n, k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    # (b, 3, (n k))
    mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
    # note that interpolated_pts already contain sparse_pts
    interpolated_pts = mid_pts
    # fps: (b, 3, up_pts_num)
    interpolated_pts = FPS(interpolated_pts, up_pts_num)

    if (normal):
        interpolated_pts = centroid + interpolated_pts * furthest_distance

    return interpolated_pts