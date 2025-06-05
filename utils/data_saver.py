import numpy as np
import os

import torch
from core.inference import denormalize_pc, descale_z

def save_pc_to_csv(cfg, pc, path, filename, denormalize_params=None):
    if not os.path.exists(path):
        os.makedirs(path)
    if isinstance(pc, torch.Tensor):
        pc = pc.cpu().numpy()
    assert pc.ndim == 2, "shape of pc must be (N, 3)"
    pc = pc[:, :3]
    if denormalize_params is not None:
        if cfg.DATASETS.SMLM.is_scale_z:
            pc = descale_z(pc, denormalize_params)
    pc = pc.astype(np.float16)
    filename = filename + '.csv'
    save_path = os.path.join(path, filename)
    np.savetxt(save_path, pc, delimiter=',', header='x [px],y [px],z [px]', comments='')
    print(f"Saved pc to {save_path}")
