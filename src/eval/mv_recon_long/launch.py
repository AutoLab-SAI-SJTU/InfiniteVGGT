import os
import sys
import argparse
import glob
import time
import copy
import torch
import numpy as np
import open3d as o3d
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.geometry import FrameDiskCache
from eval.mv_recon.utils import accuracy, completion


class Inference:
    def __init__(self, args, device="cuda"):
        self.args = args
        self.device = device
        
    def run(self):
        print(f"\n[Inference] Loading model from {self.args.checkpoint_path}...")
        model = StreamVGGT(total_budget=1200000)
        ckpt = torch.load(self.args.checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        model = model.to(self.device)
        model.eval()
        
        cache_dir = os.path.join(self.args.output_dir, "frames_cache")
        os.makedirs(cache_dir, exist_ok=True)
        frame_writer = FrameDiskCache(cache_dir)
        
        print(f"[Inference] Loading images from {self.args.input_dir}...")
        image_names = sorted(glob.glob(os.path.join(self.args.input_dir, "*")))
            
        if not image_names:
            raise FileNotFoundError(f"No images found in {self.args.input_dir}")
            
        images = load_and_preprocess_images(image_names).to(self.device)
        
        frames = [{"img": images[i].unsqueeze(0)} for i in range(images.shape[0])]
        
        print(f"[Inference] Processing {len(frames)} frames...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                model.inference(frames, frame_writer=frame_writer, cache_results=False)
        
        torch.cuda.empty_cache()
        return cache_dir

    def build_ply(self, cache_dir):
        ply_path = os.path.join(self.args.output_dir, "pred_cloud.ply")
        print(f"[Post-Process] Building PLY to {ply_path}...")
        
        stride = self.args.stride
        frame_paths = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))[::stride]
        
        raw_pts_list, colors_list, masks_list = [], [], []
        
        for path in frame_paths:
            payload = torch.load(path, map_location="cpu", weights_only=False) 
            view = payload["view"]
            raw_pts_list.append(payload["pred"]["pts3d_in_other_view"].float())
            colors_list.append(view["img"].float().permute(0, 2, 3, 1))
            
            mask = view.get("valid_mask")
            if not isinstance(mask, torch.Tensor):
                B, _, H, W = view["img"].shape
                mask = torch.ones((B, H, W), dtype=torch.bool)
            masks_list.append(mask.bool())

        # Helper to flatten
        def flatten_valid(data, mask):
            data_flat = data.reshape(data.shape[0], -1, data.shape[-1])
            mask_flat = mask.reshape(mask.shape[0], -1)
            pieces = [data_flat[b][mask_flat[b]] for b in range(data_flat.shape[0]) if mask_flat[b].any()]
            if not pieces: return torch.tensor([])
            return torch.cat(pieces, dim=0)

        raw_points = torch.cat([flatten_valid(p, m) for p, m in zip(raw_pts_list, masks_list)], dim=0)
        raw_colors = torch.cat([flatten_valid(c, m) for c, m in zip(colors_list, masks_list)], dim=0)
        
        if raw_points.shape[0] > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(raw_points.numpy().astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(np.clip(raw_colors.numpy(), 0.0, 1.0).astype(np.float64))
            o3d.io.write_point_cloud(ply_path, pcd)
            return ply_path
        else:
            raise RuntimeError("Generated point cloud is empty.")

class Evaluation:
    def __init__(self, gt_path):
        self.gt_path = gt_path
        self.voxel_size = 0.015
        self.seed = 42
    
    def preprocess(self, pcd):
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
        return pcd_clean

    def get_scale_and_init_transform(self, source, target):
        center_src = source.get_center()
        center_tgt = target.get_center()
        
        src_centered = np.asarray(source.points) - center_src
        tgt_centered = np.asarray(target.points) - center_tgt
        
        rms_src = np.sqrt(np.mean(np.sum(src_centered**2, axis=1)))
        rms_tgt = np.sqrt(np.mean(np.sum(tgt_centered**2, axis=1)))
        scale = 0.75 * (rms_tgt / rms_src) # set 1.0 in class/dormitory
        
        T_pre = np.eye(4); T_pre[:3, 3] = -center_src
        T_scale = np.diag([scale, scale, scale, 1.0])
        T_post = np.eye(4); T_post[:3, 3] = center_tgt
        
        return T_post @ T_scale @ T_pre

    def run(self, pred_path, output_metrics_path):
        print(f"\n[Eval] Comparing Pred: {pred_path} \n       vs GT: {self.gt_path}")
        
        np.random.seed(self.seed)
        o3d.utility.random.seed(self.seed)

        pred_pcd = o3d.io.read_point_cloud(pred_path)
        gt_pcd = o3d.io.read_point_cloud(self.gt_path)

        if len(pred_pcd.points) == 0:
            print("[Error] Prediction point cloud is empty.")
            return

        # Initial Alignment (Camera -> LiDAR frame)
        T_cam_wrt_lidar = np.array([
            [-0.012577, -0.999915, -0.003397, -0.03],
            [0.345639, -0.001159, -0.938367, -0.04],
            [0.938283, -0.012976, 0.345625, -0.03],
            [0.0, 0.0, 0.0, 1.0]
        ])
        pred_pcd.transform(np.linalg.inv(T_cam_wrt_lidar))

        pred_clean = self.preprocess(pred_pcd)
        gt_clean = self.preprocess(gt_pcd)

        n_pred, n_gt = len(pred_clean.points), len(gt_clean.points)
        if n_pred == 0 or n_gt == 0:
            print("[Error] Point cloud empty after preprocessing.")
            return

        if n_pred > n_gt:
            pred_eval = pred_clean.random_down_sample(n_gt / n_pred)
            gt_eval = gt_clean
        else:
            gt_eval = gt_clean.random_down_sample(n_pred / n_gt)
            pred_eval = pred_clean

        T_init = self.get_scale_and_init_transform(pred_eval, gt_eval)
        gt_eval.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*2, max_nn=30))
        
        try:
            reg = o3d.pipelines.registration.registration_icp(
                pred_eval, gt_eval, 0.05, T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
            )
            pred_eval.transform(reg.transformation)
        except Exception as e:
            print(f"[Warning] ICP Failed: {e}. Using Initial Transform.")
            pred_eval.transform(T_init)

        print("[Eval] Calculating Metrics...")
        gt_eval.estimate_normals()
        pred_eval.estimate_normals()
        
        gt_pts, pred_pts = gt_eval.points, pred_eval.points
        gt_norm, pred_norm = np.asarray(gt_eval.normals), np.asarray(pred_eval.normals)

        acc, acc_med, nc1, nc1_med = accuracy(gt_pts, pred_pts, gt_norm, pred_norm)
        comp, comp_med, nc2, nc2_med = completion(gt_pts, pred_pts, gt_norm, pred_norm)

        nc_mean = (nc1 + nc2) / 2
        nc_med = (nc1_med + nc2_med) / 2

        # Print to console
        print("-" * 30)
        print(f"Acc: {acc:.4f} | Comp: {comp:.4f}")
        print(f"NC1: {nc1:.4f} | NC2: {nc2:.4f}")
        print(f"NC_mean: {nc_mean:.4f} | NC_med: {nc_med:.4f}")
        print("-" * 30)

        # Save to file
        with open(output_metrics_path, "w") as f:
            f.write(f"Acc: {acc}\nComp: {comp}\nNC1: {nc1}\nNC2: {nc2}\n")
            f.write(f"Acc_med: {acc_med}\nComp_med: {comp_med}\n")
            f.write(f"NC_mean: {nc_mean}\nNC_med: {nc_med}\n")
        
        print(f"[Success] Metrics saved to {output_metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified Inference and Evaluation Pipeline")
    
    # Inference Args
    parser.add_argument("--weights", dest="checkpoint_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--stride", type=int, default=10, help="Stride for PLY generation")
    
    # Eval Args
    parser.add_argument("--gt_path", type=str, required=True, help="Path to Ground Truth .pcd/.ply")
    parser.add_argument("--skip_inference", action="store_true", help="Skip inference if PLY already exists")

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    ply_path = os.path.join(args.output_dir, "pred_cloud.ply")
    print(f"ply path: {ply_path}")
    
    if args.skip_inference and os.path.exists(ply_path):
        print(f"[Info] Skipping inference, using existing: {ply_path}")
    else:
        inferencer = Inference(args)
        cache_dir = inferencer.run()
        ply_path = inferencer.build_ply(cache_dir)
        
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    evaluator = Evaluation(gt_path=args.gt_path)
    evaluator.run(pred_path=ply_path, output_metrics_path=metrics_path)

if __name__ == "__main__":
    main()