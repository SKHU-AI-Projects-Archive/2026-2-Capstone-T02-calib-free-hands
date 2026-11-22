"""HandCalib benchmark Calibrator from 8f70b58, with a selectable runtime device."""

import cv2
import numpy as np
from rtmlib import Wholebody

FOCAL_ONLY = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
              | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
RIGHT_HAND = slice(112, 133)


class Calibrator:
    def __init__(self, target="hand", pattern_size=(7, 7), *, device="cuda"):
        self.target = target
        self.pattern_size = pattern_size
        if self.target == "hand":
            self.model = Wholebody(mode="performance", to_openpose=False, backend="onnxruntime", device=device)
        self.reset()

    def reset(self):
        self.observations = []

    def detect(self, frame):
        if self.target == "checkerboard":
            found, corners = cv2.findChessboardCornersSB(frame, self.pattern_size)
            return corners.reshape(-1, 2) if found else None
        keypoints, scores = self.model(frame)
        points, confidence = keypoints[0, RIGHT_HAND], scores[0, RIGHT_HAND]
        return points if confidence.min() > 3 else None

    def add(self, frame):
        self.size = frame.shape[1::-1]
        points = self.detect(frame)
        if points is not None:
            self.observations.append(points)

    def template(self, points):
        if self.target == "checkerboard":
            flat = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        else:
            shapes = points[..., 0] + 1j * points[..., 1]
            shapes = shapes - shapes.mean(axis=1, keepdims=True)
            scales = (shapes.conj() * shapes[0]).sum(axis=1) / (np.abs(shapes) ** 2).sum(axis=1)
            mean = (scales[:, None] * shapes).mean(axis=0)
            flat = np.c_[mean.real, mean.imag]
        return np.float32(flat - flat.mean(axis=0))

    def estimate(self):
        points = np.float32(self.observations)
        template = self.template(points)
        scores = [np.hypot(*cv2.findHomography(template, p)[0][2, :2]) for p in points]
        selected = np.flatnonzero(scores > np.median(scores))

        object_points = [np.float32(np.c_[template, np.zeros(len(template))])] * len(selected)
        image_points = list(points[selected])
        K = cv2.calibrateCamera(object_points, image_points, self.size, np.eye(3), None, flags=FOCAL_ONLY)[1]
        return K[0, 0]

    def run(self, source):
        cap = cv2.VideoCapture(source)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self.add(frame)
        cap.release()
        return self.estimate()
