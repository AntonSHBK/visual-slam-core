from __future__ import annotations
from typing import Any, List, Optional

import torch
import numpy as np

from visual_slam.config import Config
from visual_slam.map.keyframe import KeyFrame
from visual_slam.map.map_point import MapPoint
from visual_slam.optimization.base_optimizer import BaseOptimizer


class TorchOptimizer(BaseOptimizer):
    """Оптимизация Bundle Adjustment на основе PyTorch."""

    def __init__(
        self,
        config: Optional[Config] = None,
        logger: Optional[Any] = None,
        log_dir: Optional[str] = 'logs',
        device: str = "cuda"
    ) -> None:
        super().__init__(config, logger, log_dir=log_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def optimize_local(
        self,
        keyframes: List[KeyFrame],
        map_points: List[MapPoint],
    ) -> bool:
        """Локальная оптимизация."""
        pass

    def optimize_global(
        self,
        keyframes: List[KeyFrame],
        map_points: List[MapPoint],
    ) -> bool:
        """Глобальная оптимизация всей карты."""
        pass

    def optimize_initial(
        self,
        keyframes: List[KeyFrame],
        map_points: List[MapPoint],
    ) -> bool:
        
        if len(keyframes) < 2 or len(map_points) == 0:
            self.logger.warning("[TorchOptimizer] Недостаточно данных для optimize_initial")
            return False

        kf_ref: KeyFrame
        kf_cur: KeyFrame
        kf_ref, kf_cur = keyframes[:2]

        K = torch.tensor(kf_ref._camera.K, dtype=torch.float32, device=self.device)
        
        R_ref = torch.tensor(kf_ref.Rcw, dtype=torch.float32, device=self.device)
        t_ref = torch.tensor(kf_ref.tcw, dtype=torch.float32, device=self.device)

        R_cur = torch.tensor(kf_cur.Rcw, dtype=torch.float32, requires_grad=True, device=self.device)
        t_cur = torch.tensor(kf_cur.tcw, dtype=torch.float32, requires_grad=True, device=self.device)

        # Собираем 3D-точки и их наблюдения
        pts_3d = []
        obs_ref, obs_cur = [], []

        for mp in map_points:
            obs = mp.get_observations()
            if kf_ref.keyframe_id not in obs or kf_cur.keyframe_id not in obs:
                continue
            idx_ref = obs[kf_ref.keyframe_id]
            idx_cur = obs[kf_cur.keyframe_id]

            kp_ref = kf_ref.keypoints_left[idx_ref].pt
            kp_cur = kf_cur.keypoints_left[idx_cur].pt

            pts_3d.append(mp.position)
            obs_ref.append(kp_ref)
            obs_cur.append(kp_cur)

        X = torch.tensor(np.array(pts_3d, dtype=np.float32), requires_grad=True, device=self.device)
        pts_ref = torch.tensor(np.array(obs_ref, dtype=np.float32), device=self.device)
        pts_cur = torch.tensor(np.array(obs_cur, dtype=np.float32), device=self.device)
        
        lr = self.config.optimization.lr if hasattr(self.config.optimization, 'lr') else 1e-3
        n_iter = self.config.optimization.n_iter if hasattr(self.config.optimization, 'n_iter') else 150

        optimizer = torch.optim.Adam([R_cur, t_cur, X], lr=lr)
        loss_fn = torch.nn.SmoothL1Loss()

        for i in range(n_iter):
            optimizer.zero_grad()

            # --- Проекция для ref ---
            X_ref = (R_ref @ X.T + t_ref.reshape(3, 1)).T
            uv_ref = (K @ (X_ref / X_ref[:, 2:].clamp(min=1e-8)).T).T[:, :2]

            # --- Проекция для cur ---
            X_cur = (R_cur @ X.T + t_cur.reshape(3, 1)).T
            uv_cur = (K @ (X_cur / X_cur[:, 2:].clamp(min=1e-8)).T).T[:, :2]

            loss_ref = loss_fn(uv_ref, pts_ref)
            loss_cur = loss_fn(uv_cur, pts_cur)
            loss = loss_ref + loss_cur

            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                self.logger.info(f"[TorchOptimizer] Итерация {i}, loss={loss.item():.6f}")

        # Обновляем позу текущего кадра
        kf_cur.set_pose_Rt(
            R=R_cur.detach().cpu().numpy(), 
            t=t_cur.detach().cpu().numpy()
        )
        
        # Обновляем позиции 3D-точек
        for mp, pos in zip(map_points, X.detach().cpu().numpy()):
            mp.update_position(pos)

        self.logger.info("[TorchOptimizer] optimize_initial завершён успешно.")
        return True
