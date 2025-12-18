from typing import List, Tuple

import cv2
import numpy as np

from visual_slam.map.keyframe import KeyFrame
from visual_slam.map.map_point import MapPoint
from visual_slam.utils.matching import filter_matches
from visual_slam.utils.motion_estimation import (
    triangulate_points,
    filter_by_parallax,
    filter_points_by_depth
)
from visual_slam.utils.geometry import normalize

from .base import BaseKeyframeHandler


class MonoKeyframeHandler(BaseKeyframeHandler):
    
    def process_keyframe(self, kf: KeyFrame) -> Tuple[KeyFrame, List[MapPoint]]:
        neighbors = self._find_neighbors(kf)
        matches = self._find_matches(kf, neighbors)
        matches = self._process_existing_points(kf, neighbors, matches)
        new_mappoints = self._triangulate_new_points(kf, neighbors, matches)
        return kf, new_mappoints

    def _find_neighbors(self, kf: KeyFrame) -> List[KeyFrame]:
        max_neighbors = self.config.local_mapping.max_neighbors
        neighbors = self.map.get_keyframes()[-max_neighbors:]
        self.logger.info(
            f"[MonoHandler] KF {kf.keyframe_id}: найдено соседей — {len(neighbors)}."
        )
        return neighbors

    def _find_matches(
            self, 
            kf: KeyFrame, 
            neighbors: List[KeyFrame]
        ) -> dict[int, List[cv2.DMatch]]:

        des_new = kf.descriptors[0]
        kps_new = kf.keypoints[0]

        if des_new is None or len(des_new) == 0:
            self.logger.warning(f"[MonoHandler] KF {kf.keyframe_id}: нет дескрипторов.")
            return {}

        matches_dict = {}
        matcher = self.slam.feature_tracker

        for nb in neighbors:
            des_nb = nb.descriptors[0]
            kps_nb = nb.keypoints[0]

            if des_nb is None or len(des_nb) == 0:
                matches_dict[nb.keyframe_id] = []
                continue

            matches = matcher.match(des1=des_nb, des2=des_new)
            
            self.logger.info(
                f"[MonoHandler] KF {kf.keyframe_id} ↔ KF {nb.keyframe_id}: найдено матчей — {len(matches)}."
            )
            
            matches = filter_matches(
                matches=matches,
                kps1=kps_nb,
                kps2=kps_new,
                logger=self.logger,
                filtered_params=self.config.features.filtered_params
            )
            
            self.logger.info(
                f"[MonoHandler] KF {kf.keyframe_id} ↔ KF {nb.keyframe_id}: после фильтрации матчей — {len(matches)}."
            )
            
            matches_dict[nb.keyframe_id] = matches

        return matches_dict
    
    def _process_existing_points(
            self,
            kf: KeyFrame,
            neighbors: List[KeyFrame],
            matches: dict[int, List[cv2.DMatch]]
        ) -> dict[int, List[Tuple[int, int]]]:

        self.logger.info(f"[MonoHandler] KF {kf.keyframe_id}: поиск существующих MapPoint.")

        matches_for_triangulation = {}

        for nb in neighbors:
            nb_id = nb.keyframe_id
            pair_list = matches.get(nb_id, [])

            tri_list = []

            for m in pair_list:
                idx_nb = m.queryIdx
                idx_new = m.trainIdx

                mp_existing = nb.get_map_point(
                    cam_id=0,
                    kp_idx=idx_nb
                )
                
                if mp_existing is not None:
                    kf.add_map_point(
                        cam_id=0,
                        kp_idx=idx_new,
                        map_point=mp_existing
                    )
                    continue

                tri_list.append((idx_new, idx_nb))
            
            self.logger.info(
                f"[MonoHandler] KF {kf.keyframe_id} ↔ KF {nb_id}: для триангуляции — {len(tri_list)} точек."
            )

            matches_for_triangulation[nb_id] = tri_list

        return matches_for_triangulation

    def _triangulate_new_points(
            self,
            kf: KeyFrame,
            neighbors: List[KeyFrame],
            matches_for_triangulation: dict[int, List[Tuple[int, int]]]
        ) -> List[MapPoint]:

        new_map_points: List[MapPoint] = []

        for nb in neighbors:
            nb_id = nb.keyframe_id
            pair_list = matches_for_triangulation.get(nb_id, [])

            self.logger.info(
                f"[MonoHandler] KF {kf.keyframe_id} ↔ KF {nb_id}: триангуляция {len(pair_list)} точек"
            )

            # ---------------------------------------------------------
            # 1. Сбор пиксельных координат
            # ---------------------------------------------------------
            pts_new_px = []
            pts_nb_px = []

            for idx_new, idx_nb in pair_list:
                kp_new = kf.keypoints[0][idx_new].pt
                kp_nb  = nb.keypoints[0][idx_nb].pt

                pts_new_px.append(kp_new)
                pts_nb_px.append(kp_nb)

            pts_new_px = np.array(pts_new_px, dtype=np.float64)
            pts_nb_px  = np.array(pts_nb_px, dtype=np.float64)

            # ---------------------------------------------------------
            # 2. Нормализация точек
            # ---------------------------------------------------------
            Kinv = self.camera.Kinv

            pts_new_n = normalize(Kinv, pts_new_px)
            pts_nb_n  = normalize(Kinv, pts_nb_px)

            # ---------------------------------------------------------
            # 3. Триангуляция
            # ---------------------------------------------------------
            pts_3d, mask_triang = triangulate_points(
                T_w2c_ref=nb.T_w2c,
                T_w2c_cur=kf.T_w2c,
                pts_ref_n=pts_nb_n,
                pts_cur_n=pts_new_n
            )

            if pts_3d is None or len(pts_3d) == 0:
                continue

            pts_3d = pts_3d[mask_triang]
            pts_new_n = pts_new_n[mask_triang]
            pts_nb_n = pts_nb_n[mask_triang]
            pair_list = [pair_list[i] for i in range(len(mask_triang)) if mask_triang[i]]

            self.logger.info(
                f"[MonoHandler] KF {kf.keyframe_id}: после триангуляции — {len(pts_3d)}"
            )

            # ---------------------------------------------------------
            # 4. Фильтрация по глубине
            # ---------------------------------------------------------
            pts_3d, mask_depth = filter_points_by_depth(
                points_3d=pts_3d,
                T_w2c_ref=nb.T_w2c,
                T_w2c_cur=kf.T_w2c,
                min_depth=self.config.local_mapping.min_depth,
                max_depth=self.config.local_mapping.max_depth
            )

            pts_new_n = pts_new_n[mask_depth]
            pts_nb_n  = pts_nb_n[mask_depth]
            pair_list = [pair_list[i] for i in range(len(mask_depth)) if mask_depth[i]]

            self.logger.info(
                f"[MonoHandler] KF {kf.keyframe_id}: после фильтрации по глубине — {len(pts_3d)}"
            )

            # ---------------------------------------------------------
            # 5. Фильтрация по параллаксу
            # ---------------------------------------------------------
            mask_parallax, ang = filter_by_parallax(
                pts_ref_n=pts_nb_n,
                pts_cur_n=pts_new_n,
                pose_ref=nb.T_w2c,
                pose_cur=kf.T_w2c,
                min_parallax_deg=self.config.local_mapping.min_parallax_deg
            )

            pts_3d = pts_3d[mask_parallax]
            pair_list = [pair_list[i] for i in range(len(mask_parallax)) if mask_parallax[i]]

            self.logger.info(
                f"[MonoHandler] KF {kf.keyframe_id}: после фильтрации по параллаксу — {len(pts_3d)}"
            )

            # ---------------------------------------------------------
            # 6. Создание MapPoint и привязка наблюдений
            # ---------------------------------------------------------

            for (p3d, (idx_new, idx_nb)) in zip(pts_3d, pair_list):

                color = None
                img = kf.images[0]

                if img is not None:
                    kp_new = kf.keypoints[0][idx_new].pt
                    u_new, v_new = np.round(kp_new).astype(int)

                    if 0 <= v_new < img.shape[0] and 0 <= u_new < img.shape[1]:
                        px = img[v_new, u_new]

                        if px.ndim == 0 or len(px.shape) == 0:
                            color = np.array([px, px, px], dtype=np.float32)
                        else:
                            color = px.astype(np.float32)

                mp = MapPoint(position=p3d, color=color)

                mp.add_observation(kf.keyframe_id, 0, idx_new)
                mp.add_observation(nb.keyframe_id, 0, idx_nb)

                kf.add_map_point(0, idx_new, mp)
                nb.add_map_point(0, idx_nb, mp)

                new_map_points.append(mp)

                    
        return new_map_points