import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

# from pointnet2_ops.pointnet2_utils import ( # Original import
#     furthest_point_sample,
#     grouping_operation,
#     ball_query,
#     QueryAndGroup,
#     GroupAll,
# )
# Import CUDA accelerated operations
from pointnet2_ops.pointnet2_utils import (
    furthest_point_sample,
    gather_operation,
    three_nn,
    three_interpolate,
    QueryAndGroup,
    GroupAll,
    grouping_operation,  # Keep grouping_operation if needed elsewhere or used internally by QueryAndGroup
    ball_query,  # Keep ball_query if needed elsewhere or used internally by QueryAndGroup
)
# Keep square_distance and index_points for now, in case they are used elsewhere
# Or if the CUDA interpolation (three_interpolate) needs them indirectly, although less likely.


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


# Keep original Python implementations for reference or potential fallback
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    # Use CUDA ball_query
    # Note: CUDA ball_query returns indices and counts
    group_idx, _ = ball_query(radius, nsample, xyz, new_xyz)
    # Handle potential padding index (often N) if necessary, similar to original logic
    # However, the CUDA grouping_operation handles indexing directly.
    # The original logic to replace N with group_first might not be needed
    # depending on how grouping_operation handles out-of-bounds/masked indices.
    # Let's rely on the CUDA ops for now.
    # group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # sqrdists = square_distance(new_xyz, xyz)
    # group_idx[sqrdists > radius**2] = N
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # mask = group_idx == N
    # group_idx[mask] = group_first[mask]
    return group_idx


# Keep original Python implementations for reference or potential fallback
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    # Use CUDA furthest_point_sample and gather_operation (equivalent to index_points)
    fps_idx = furthest_point_sample(xyz, npoint)  # [B, npoint]
    # Note: CUDA ops often expect B,C,N for features, B,N,C for xyz
    new_xyz = (
        gather_operation(xyz.transpose(1, 2).contiguous(), fps_idx)
        .transpose(1, 2)
        .contiguous()
    )  # [B, npoint, C]

    # Use CUDA ball_query and grouping_operation
    idx, _ = ball_query(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample]
    # Group XYZ coordinates
    grouped_xyz = grouping_operation(
        xyz.transpose(1, 2).contiguous(), idx
    )  # [B, C, npoint, nsample]
    grouped_xyz_norm = grouped_xyz - new_xyz.transpose(1, 2).contiguous().unsqueeze(
        -1
    )  # [B, C, npoint, nsample]

    if points is not None:
        # points: [B, N, D] -> [B, D, N] before grouping
        grouped_points = grouping_operation(
            points.transpose(1, 2).contiguous(), idx
        )  # [B, D, npoint, nsample]
        # Concatenate along the feature dimension (C+D)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], dim=1
        )  # [B, C+D, npoint, nsample]
    else:
        new_points = grouped_xyz_norm  # [B, C, npoint, nsample]

    if returnfps:
        # grouped_xyz needs to be B, npoint, nsample, C for compatibility if used
        return (
            new_xyz,
            new_points,
            grouped_xyz.permute(0, 2, 3, 1).contiguous(),
            fps_idx,
        )
    else:
        return new_xyz, new_points  # new_points is [B, C+D, npoint, nsample]


# Keep original Python implementations for reference or potential fallback
def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)  # [B, 1, C]
    grouped_xyz = xyz.unsqueeze(1)  # [B, 1, N, C]
    if points is not None:
        # points: [B, N, D]
        grouped_points = points.unsqueeze(1)  # [B, 1, N, D]
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # [B, 1, N, C+D]
    else:
        new_points = grouped_xyz  # [B, 1, N, C]
    # Output shape expected by conv layers later: [B, C+D, 1, N]
    return new_xyz, new_points.permute(0, 3, 1, 2).contiguous()


# Rewritten PointNetSetAbstraction using CUDA ops
class PointNetSetAbstraction(nn.Module):
    def __init__(
        self,
        npoint,
        radius,
        nsample,
        in_channel,
        mlp,
        group_all,
        use_xyz=True,
        include_abs_coordinate=False,
    ):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.group_all = group_all
        if self.group_all:
            self.grouper = GroupAll(use_xyz)
        else:
            # Use the CUDA QueryAndGroup module
            self.grouper = QueryAndGroup(
                radius,
                nsample,
                use_xyz=use_xyz,
                include_abs_coordinate=include_abs_coordinate,
            )

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        # Adjust input channel based on what QueryAndGroup/GroupAll provides
        if use_xyz:
            # QueryAndGroup concatenates relative xyz (3 dims)
            # optionally absolute xyz (+3 dims)
            # optionally center xyz (+3 dims) - we assume not used here unless specified
            last_channel = in_channel + 3
            if include_abs_coordinate:
                last_channel += 3
        else:
            last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N] where C=3
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_res: sample points feature data, [B, D', S]
        """
        # CUDA ops expect B, N, C for coordinates
        xyz_trans = xyz.permute(0, 2, 1).contiguous()  # [B, N, 3]
        B, N, C_xyz = xyz_trans.shape

        if points is not None:
            # points: [B, D, N]
            features = points  # Keep features as B, D, N for QueryAndGroup input
        else:
            features = None

        if self.group_all:
            # GroupAll expects xyz: [B, N, 3], features: [B, C, N]
            # It ignores new_xyz, we pass None or a dummy tensor
            # Output: [B, C+3, 1, N]
            new_points = self.grouper(xyz_trans, None, features)
            # Output xyz is just the center (0,0,0) - needs shape [B, 3, 1]
            new_xyz = (
                torch.zeros(B, 1, C_xyz, device=xyz.device).transpose(1, 2).contiguous()
            )  # [B, 3, 1]
        else:
            # Sample points using furthest_point_sample
            # fps_idx: [B, npoint]
            fps_idx = furthest_point_sample(xyz_trans, self.npoint)

            # Get sampled coordinates new_xyz: [B, 3, npoint]
            new_xyz = gather_operation(xyz, fps_idx)  # Input xyz is [B, 3, N]
            # Convert new_xyz to [B, npoint, 3] for QueryAndGroup
            new_xyz_trans = new_xyz.permute(0, 2, 1).contiguous()

            # Group features using QueryAndGroup
            # Input xyz: [B, N, 3], new_xyz: [B, npoint, 3], features: [B, D, N]
            # Output: [B, D+3(+3), npoint, nsample]
            new_points = self.grouper(xyz_trans, new_xyz_trans, features)

        # Apply MLPs
        # new_points shape: [B, Cin, npoint/1, nsample/N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Max pooling over samples
        # Input: [B, Cout, npoint/1, nsample/N]
        # Output: [B, Cout, npoint/1]
        new_points_res = torch.max(new_points, dim=3)[0]

        # new_xyz: [B, 3, S], new_points_res: [B, D', S]
        return new_xyz, new_points_res


class PointNetSetAbstractionMsg(nn.Module):
    # This class uses multiple radii and might be more complex to directly map
    # to QueryAndGroup which assumes a single radius/nsample per instance.
    # We can rewrite it using multiple QueryAndGroup instances.
    def __init__(
        self,
        npoint,
        radius_list,
        nsample_list,
        in_channel,
        mlp_list,
        use_xyz=True,
        include_abs_coordinate=False,
    ):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            mlp_spec = mlp_list[i]
            # Create a grouper for each scale
            grouper = QueryAndGroup(
                radius,
                nsample,
                use_xyz=use_xyz,
                include_abs_coordinate=include_abs_coordinate,
            )
            self.groupers.append(grouper)

            # Create MLP for each scale
            mlp_layers = nn.ModuleList()
            mlp_bns = nn.ModuleList()
            # Adjust input channel based on what QueryAndGroup provides
            if use_xyz:
                last_channel = in_channel + 3
                if include_abs_coordinate:
                    last_channel += 3
            else:
                last_channel = in_channel

            for out_channel in mlp_spec:
                mlp_layers.append(nn.Conv2d(last_channel, out_channel, 1))
                mlp_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            # Store MLP layers and BNs together for convenience
            self.mlps.append(nn.ModuleDict({"convs": mlp_layers, "bns": mlp_bns}))

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N] where C=3
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # CUDA ops expect B, N, C for coordinates
        xyz_trans = xyz.permute(0, 2, 1).contiguous()  # [B, N, 3]
        B, N, C_xyz = xyz_trans.shape

        if points is not None:
            # features: [B, D, N]
            features = points
        else:
            features = None

        # Sample points using furthest_point_sample
        # fps_idx: [B, npoint]
        fps_idx = furthest_point_sample(xyz_trans, self.npoint)

        # Get sampled coordinates new_xyz: [B, 3, npoint]
        new_xyz = gather_operation(xyz, fps_idx)  # Input xyz is [B, 3, N]
        # Convert new_xyz to [B, npoint, 3] for QueryAndGroup
        new_xyz_trans = new_xyz.permute(0, 2, 1).contiguous()

        new_points_list = []
        for i in range(len(self.groupers)):
            # Group features using QueryAndGroup for this scale
            # Input xyz: [B, N, 3], new_xyz: [B, npoint, 3], features: [B, D, N]
            # Output: [B, D+3(+3), npoint, nsample_i]
            grouped_points = self.groupers[i](xyz_trans, new_xyz_trans, features)

            # Apply MLP for this scale
            # grouped_points shape: [B, Cin_i, npoint, nsample_i]
            mlp_module = self.mlps[i]
            for j, conv in enumerate(mlp_module["convs"]):
                bn = mlp_module["bns"][j]
                grouped_points = F.relu(bn(conv(grouped_points)))

            # Max pooling over samples for this scale
            # Input: [B, Cout_i, npoint, nsample_i]
            # Output: [B, Cout_i, npoint]
            pooled_points = torch.max(grouped_points, dim=3)[0]
            new_points_list.append(pooled_points)

        # Concatenate features from all scales
        # Output: [B, sum(Cout_i), npoint]
        new_points_concat = torch.cat(new_points_list, dim=1)

        # new_xyz: [B, 3, S], new_points_concat: [B, D', S]
        return new_xyz, new_points_concat


# Rewritten PointNetFeaturePropagation using CUDA ops
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N] (points to be interpolated to)
            xyz2: sampled input points position data, [B, C, S] (points to interpolate from)
            points1: input points data, [B, D1, N] (features at xyz1, often from skip connection)
            points2: input points data, [B, D2, S] (features at xyz2)
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        # CUDA ops expect B, N, C for coordinates
        xyz1_trans = xyz1.permute(0, 2, 1).contiguous()  # [B, N, 3]
        xyz2_trans = xyz2.permute(0, 2, 1).contiguous()  # [B, S, 3]

        B, N, _ = xyz1_trans.shape
        _, S, D2 = points2.shape  # points2 is [B, D2, S]

        if S == 1:
            # Special case: If only one point to interpolate from, repeat its features
            interpolated_points = points2.repeat(1, 1, N)  # [B, D2, N]
        else:
            # Use CUDA three_nn and three_interpolate
            # Find 3 nearest neighbors of xyz1 in xyz2
            # dist: [B, N, 3], idx: [B, N, 3]
            dist, idx = three_nn(xyz1_trans, xyz2_trans)

            # Calculate interpolation weights (inverse distance)
            dist_recip = 1.0 / (dist + 1e-8)  # [B, N, 3]
            norm = torch.sum(dist_recip, dim=2, keepdim=True)  # [B, N, 1]
            weight = dist_recip / norm  # [B, N, 3]

            # Interpolate features from points2 using weights and indices
            # three_interpolate expects features: [B, D2, S], idx: [B, N, 3], weight: [B, N, 3]
            # Output: [B, D2, N]
            interpolated_points = three_interpolate(points2.contiguous(), idx, weight)

        if points1 is not None:
            # Concatenate interpolated features with skip connection features
            # points1: [B, D1, N], interpolated_points: [B, D2, N]
            # Result: [B, D1+D2, N]
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            # If no skip connection, just use interpolated features
            # Result: [B, D2, N]
            new_points = interpolated_points

        # Apply MLPs
        # Input: [B, Cin, N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Output: [B, D', N]
        return new_points
