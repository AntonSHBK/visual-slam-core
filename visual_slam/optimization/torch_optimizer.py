from __future__ import annotations
from ast import Dict, Tuple
from typing import Any, List, Optional

import torch
from torch import Tensor
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
        
        # брать случайный subset точек (например 100 штук)
        # отбрасывать точки с большим reprojection error перед оптимизацией

        # ------------------------------------------------------
        # 1. Подготовка данных
        # ------------------------------------------------------

        self.logger.info(f'[TorchOptimizer] Количество keyframes - {len(keyframes)}, точек - {len(map_points)}.')
        
        kf_fixed = keyframes[0]
        kf_opt = keyframes[1:]

        K = torch.tensor(kf_fixed._camera.K, dtype=torch.float32, device=self.device)

        poses: Dict[int, Tuple[Tensor, Tensor, Tensor]] = {}  # kf_id -> (R0: Tensor, w: Tensor, t: Tensor)

        for kf in keyframes:
            R0 = torch.tensor(kf.R_w2c, dtype=torch.float32, device=self.device)
            t0 = torch.tensor(kf.t_w2c, dtype=torch.float32, device=self.device)

            if kf is kf_fixed:
                w = torch.zeros(3, dtype=torch.float32, device=self.device, requires_grad=False)
                t = t0.clone().detach().requires_grad_(False)
            else:
                w = torch.zeros(3, dtype=torch.float32, device=self.device, requires_grad=True)
                t = t0.clone().detach().requires_grad_(True)

            poses[kf.keyframe_id] = (R0, w, t)


        X_list = [mp.position for mp in map_points]
        X = torch.tensor(
            np.array(X_list, dtype=np.float32), device=self.device,
            requires_grad=True
        )

        # ------------------------------------------------------
        # 2. Наблюдения
        # ------------------------------------------------------

        kf_by_id = {kf.keyframe_id: kf for kf in keyframes}
        obs = []  # формат: (kf_id, mp_index, cam_id, uv)

        for j, mp in enumerate(map_points):

            mp_obs = mp.get_observations()  # { kf_id : { cam_id : kp_idx } }

            for kf_id, cams in mp_obs.items():
                if kf_id not in poses:
                    continue

                kf = kf_by_id.get(kf_id)
                if kf is None:
                    continue

                for cam_id, kp_idx in cams.items():
                    if cam_id >= len(kf.keypoints):
                        continue
                    if kp_idx >= len(kf.keypoints[cam_id]):
                        continue
                    kp = kf.keypoints[cam_id][kp_idx].pt
                    obs.append((kf_id, j, cam_id, kp))
                    
        self.logger.info(f"[TorchOptimizer] Количество сформированных наблюдений - {len(obs)}.")

        if len(obs) < 10:
            self.logger.warning("[TorchOptimizer] Мало наблюдений для локальной оптимизации.")
            return False

        # ------------------------------------------------------
        # 3. Оптимизация
        # ------------------------------------------------------
        lr = getattr(self.config.optimization, "lr", 5e-3)
        n_iter = getattr(self.config.optimization, "n_iter", 100)
        robust_delta = getattr(self.config.optimization, "huber_delta", 5.0)

        params = [X]
        for kf_id, (R0, w, t) in poses.items():
            if w.requires_grad:
                params.append(w)
            if t.requires_grad:
                params.append(t)

        opt = torch.optim.Adam(params, lr=lr)
        
        def so3_exp(w: torch.Tensor) -> torch.Tensor:
            """Экспонента над so(3) для вектора w."""
            theta = torch.norm(w) + 1e-8
            k = w / theta

            Kmat = torch.tensor([
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0]
            ], device=w.device)

            R = (torch.eye(3, device=w.device)
                + torch.sin(theta) * Kmat
                + (1 - torch.cos(theta)) * (Kmat @ Kmat))
            return R

        def huber(residual, delta=robust_delta):
            """Huber loss."""
            abs_r = torch.abs(residual)
            mask = abs_r < delta
            return torch.where(
                mask,
                0.5 * residual ** 2,
                delta * (abs_r - 0.5 * delta)
            )

        # ------------------------------------------------------
        # 4. Итерации
        # ------------------------------------------------------
        for it in range(n_iter):
            opt.zero_grad()
            total_loss = 0.0

            for kf_id, j, cam_id, uv in obs:
                R0: Tensor
                w: Tensor
                t: Tensor
                R0, w, t = poses[kf_id]

                if w.requires_grad:
                    R = so3_exp(w) @ R0
                else:
                    R = R0

                Xj = X[j]
                Xc = R @ Xj + t
                X_, Y_, Z_ = Xc[0], Xc[1], Xc[2] + 1e-8

                K = keyframes[0]._camera.K
                K = torch.tensor(K, dtype=torch.float32, device=self.device)

                u = K[0,0] * (X_/Z_) + K[0,2]
                v = K[1,1] * (Y_/Z_) + K[1,2]
                uv_pred = torch.stack([u, v])

                uv_tgt = torch.tensor(uv, dtype=torch.float32, device=self.device)
                res = uv_pred - uv_tgt

                total_loss += huber(res, robust_delta).sum()

            total_loss.backward()
            opt.step()

            if it % 10 == 0:
                self.logger.info(f"[Local BA] iter={it}, loss={total_loss.item():.3f}")

        # ------------------------------------------------------
        # 5. Обновление KeyFrame и MapPoint
        # ------------------------------------------------------
        for kf in keyframes:
            R0: Tensor
            w: Tensor
            t: Tensor
            R0, w, t = poses[kf.keyframe_id]

            if w.requires_grad:
                R_new = (so3_exp(w) @ R0).detach().cpu().numpy()
            else:
                R_new = R0.detach().cpu().numpy()

            t_new = t.detach().cpu().numpy().reshape(3, 1)
            kf.set_pose_Rt(R_new, t_new)

        for mp, new_pos in zip(map_points, X.detach().cpu().numpy()):
            mp.update_position(new_pos)

        self.logger.info("[Local BA] завершено успешно.")
        return True


    def optimize_initial(
        self,
        keyframes: List[KeyFrame],
        map_points: List[MapPoint],
    ) -> bool:

        if len(keyframes) < 2 or len(map_points) == 0:
            self.logger.warning("[TorchOptimizer] Недостаточно данных для optimize_initial")
            return False

        kf_ref, kf_cur = keyframes[:2]

        # Камера и позы ref
        K = torch.tensor(kf_ref._camera.K, dtype=torch.float32, device=self.device)

        R_ref = torch.tensor(kf_ref.R_w2c, dtype=torch.float32, device=self.device)
        t_ref = torch.tensor(kf_ref.t_w2c, dtype=torch.float32, device=self.device)

        # Параметры cur — оптимизируем
        R_cur = torch.tensor(kf_cur.R_w2c, dtype=torch.float32, requires_grad=True, device=self.device)
        t_cur = torch.tensor(kf_cur.t_w2c, dtype=torch.float32, requires_grad=True, device=self.device)

        # ------------------------------------------------------------
        # Сбор наблюдений
        # ------------------------------------------------------------
        pts_3d = []
        obs_ref = []
        obs_cur = []

        for mp in map_points:
            obs = mp.get_observations()     # { kf_id : { cam_id : kp_idx } }

            if kf_ref.keyframe_id not in obs:
                continue
            if kf_cur.keyframe_id not in obs:
                continue

            obs_ref_entry = obs[kf_ref.keyframe_id]
            obs_cur_entry = obs[kf_cur.keyframe_id]

            if 0 not in obs_ref_entry or 0 not in obs_cur_entry:
                continue

            kp_ref_idx = obs_ref_entry[0]
            kp_cur_idx = obs_cur_entry[0]

            if kp_ref_idx >= len(kf_ref.keypoints[0]):
                continue
            if kp_cur_idx >= len(kf_cur.keypoints[0]):
                continue

            kp_ref = kf_ref.keypoints[0][kp_ref_idx].pt
            kp_cur = kf_cur.keypoints[0][kp_cur_idx].pt

            pts_3d.append(mp.position)
            obs_ref.append(kp_ref)
            obs_cur.append(kp_cur)

        if len(pts_3d) < 10:
            self.logger.warning("[TorchOptimizer] Недостаточно наблюдений для optimize_initial.")
            return False

        # ------------------------------------------------------------
        # Подготовка тензоров
        # ------------------------------------------------------------
        X = torch.tensor(np.array(pts_3d, dtype=np.float32), requires_grad=True, device=self.device)
        pts_ref = torch.tensor(np.array(obs_ref, dtype=np.float32), device=self.device)
        pts_cur = torch.tensor(np.array(obs_cur, dtype=np.float32), device=self.device)

        # ------------------------------------------------------------
        # Оптимизация
        # ------------------------------------------------------------
        lr = getattr(self.config.optimization, "lr", 1e-3)
        n_iter = getattr(self.config.optimization, "n_iter", 150)

        optimizer = torch.optim.Adam([R_cur, t_cur, X], lr=lr)
        loss_fn = torch.nn.SmoothL1Loss()

        for i in range(n_iter):
            optimizer.zero_grad()

            # Проекция ref
            X_ref = (R_ref @ X.T + t_ref.reshape(3, 1)).T
            uv_ref = (K @ (X_ref / X_ref[:, 2:].clamp(min=1e-8)).T).T[:, :2]

            # Проекция cur
            X_cur = (R_cur @ X.T + t_cur.reshape(3, 1)).T
            uv_cur = (K @ (X_cur / X_cur[:, 2:].clamp(min=1e-8)).T).T[:, :2]

            loss_ref = loss_fn(uv_ref, pts_ref)
            loss_cur = loss_fn(uv_cur, pts_cur)
            loss = loss_ref + loss_cur

            loss.backward()
            optimizer.step()

        # ------------------------------------------------------------
        # Обновление оптимизированных параметров
        # ------------------------------------------------------------
        kf_cur.set_pose_Rt(
            R=R_cur.detach().cpu().numpy(),
            t=t_cur.detach().cpu().numpy()
        )

        for mp, pos in zip(map_points, X.detach().cpu().numpy()):
            mp.update_position(pos)

        self.logger.info("[TorchOptimizer] optimize_initial завершён успешно.")
        return True
    
    def optimize_global(self, keyframes, map_points):
        pass
