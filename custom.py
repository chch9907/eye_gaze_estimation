import datetime
import logging
import pathlib
from typing import List, Optional, Union

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from ptgaze.common import Face, FacePartsName, Visualizer
from ptgaze.gaze_estimator import GazeEstimator
from ptgaze.utils import (check_path_all, download_dlib_pretrained_model,
                    download_ethxgaze_model, download_mpiifacegaze_model,
                    download_mpiigaze_model, expanduser_all,
                    generate_dummy_camera_params)
from ptgaze.demo import Demo
from kalman_filter import GazeKalmanFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# head coordinate: pitch: up+ (red), yaw: left+ (green), roll: right+ (blue)

class GazeEstimation(Demo):
    def __init__(self, key_config: DictConfig):
        config = load_mode_config(key_config)
        super().__init__(config)
        
        self.faces = None
        self.non_detected_num = 0
        self.kf = GazeKalmanFilter(**config.kalman_filter)
        self._filtered_angles = None
    def _process_image(self, image: np.ndarray):  # overwrite function
        undistorted = cv2.undistort(
            image, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)

        self.visualizer.set_image(image.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)
        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)

        if self.config.demo.use_camera:
            self.visualizer.image = self.visualizer.image[:, ::-1]
        if self.writer:
            self.writer.write(self.visualizer.image)
        
        #========= custom modification ========
        # store face output
        if len(faces) == 0:
            self.faces = None
        else:
            head_pitches = []
            # if detects multiple faces, take the one with the highest pitch angle
            for face in faces:
                euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
                head_pitch, head_yaw, head_roll = face.change_coordinate_system(euler_angles)
                head_pitches.append(head_pitch)
            self.faces = faces[np.argmin(head_pitch)]  # only one face
        
        head_angles = self._get_head_poses()  # [pitch, yaw]
        # gaze_angles = self._get_gaze_angles()  # [pitch, yaw]
        self.kf.update(head_angles)
        _filtered_angles = self.kf.get_estimate()
        self._filtered_angles = _filtered_angles
        # print('raw angles', head_angles, '_filtered_angles', _filtered_angles)
        return _filtered_angles

    #====================== custom modification =================================    
    def _get_head_poses(self,) -> Optional[List[float]]:
        if self.faces is not None:
            euler_angles = self.faces.head_pose_rot.as_euler('XYZ', degrees=True)
            head_pitch, head_yaw, _ = self.faces.change_coordinate_system(euler_angles)
            return [head_pitch, head_yaw]
        else:
            return None
    
    def _get_gaze_angles(self,) -> Optional[List[float]]:
        if self.faces is not None:
            gaze_pitch, gaze_yaw = np.rad2deg(self.faces.vector_to_angle(self.faces.gaze_vector))
            return [gaze_pitch, gaze_yaw]
        else:
            return None
    
    def get_output_angles(self,) -> Optional[List[float]]:
        return self._filtered_angles
    

def load_mode_config(cfg: DictConfig) -> DictConfig:
    # print('cfg', cfg)
    package_root = pathlib.Path(__file__).parent.resolve()
    if cfg._mode == 'mpiigaze':
        path = package_root / 'ptgaze/data/configs/mpiigaze.yaml'
    elif cfg._mode == 'mpiifacegaze':
        path = package_root / 'ptgaze/data/configs/mpiifacegaze.yaml'
    elif cfg._mode == 'eth-xgaze':
        path = package_root / 'ptgaze/data/configs/eth-xgaze.yaml'
    else:
        raise ValueError
    mode_config = OmegaConf.load(path)
    mode_config.PACKAGE_ROOT = package_root.as_posix() + '/ptgaze'
    OmegaConf.resolve(mode_config)
    mode_config = OmegaConf.merge(mode_config, cfg)
    
    # print('mode_config', mode_config)
    # assert False
    expanduser_all(mode_config)
    
    if mode_config.face_detector.mode == 'dlib':
        download_dlib_pretrained_model()
    if cfg._mode:
        if mode_config.mode == 'MPIIGaze':
            download_mpiigaze_model()
        elif mode_config.mode == 'MPIIFaceGaze':
            download_mpiifacegaze_model()
        elif mode_config.mode == 'ETH-XGaze':
            download_ethxgaze_model()
            
    check_path_all(mode_config)
    return mode_config
    
    
    
@hydra.main(
    version_base=None,
    config_path='./custom_params',
    config_name='overall_config'
)
def main(cfg: DictConfig):
    demo = GazeEstimation(cfg)
    demo.run()
    output_angles = demo.get_output_angles()
    print('get pitch and yaw angles', output_angles)


if __name__ == "__main__":
    main()