
import sys
import os
import cv2
import numpy as np 
import json
import pickle
from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation
import hashlib
from tqdm import tqdm
import tensorflow as tf

class HandFacePersonSSDONNXOpenCV: 
    """The inference class for Alexa NU hand, face, and person object detection via SSD
    
    Given a test RGB image, this class provides the API to predict the object bounding boxes of hand, 
    face, and person classes. In particular, the detection results is organized as a list with each
    element of the format 
        [object_class, detection_confidence, bbox_left, bbox_top, bbox_right, bbox_bottom]
    """
    outputs = ['bboxes', 'classes']
    classes = ['hand', 'face', 'headtorso']
    scales = [10., 10., 5., 5.]
    colors =[(255, 0, 0), (0, 255, 127), (127, 0, 255)]
    num_max_dets = 100
    epsilon = 1e-7
    def __init__(self, 
                 model_file_path: str,
                 target_size: int=320,
                 thresh: float=0.2,
                 overlap_iou: float=0.5,
                ) :
        assert os.path.isfile(model_file_path), \
            f"ERROR: fail to locate the input model file {model_file_path}"
        assert model_file_path.endswith('.onnx'), \
            f"ERROR: only .onnx model is supported"

        anchor_file_path = os.path.join(os.path.dirname(model_file_path), 'boxprior.csv')
        assert os.path.isfile(anchor_file_path), \
            f"ERROR: fail to locate the box prior file {anchor_file_path}"
        
        with open(anchor_file_path) as IN :
            data_line = IN.readline()
            data_fields = data_line.strip().split(',')
            num_anchors = len(data_fields) // 4
            assert num_anchors * 4 <= len(data_fields)
            data_fields_valid = [float(x) for x in data_fields[:num_anchors*4]]
            
        self.anchors = np.array(data_fields_valid, dtype='float32').reshape([4,-1]).T
        self.detector = cv2.dnn.readNet(model_file_path)                       
        self.target_size = target_size
        self.thresh = thresh
        self.overlap_iou = overlap_iou
        return
    
    def _preprocess(self, full_image: np.ndarray) -> np.ndarray: 
        """preprocess full_image as the network's input
        """
        x = cv2.resize(full_image, (320, 320)).astype('float32')/127.5 - 1 
        return x[None]
    
    def __call__(self, 
                 full_image: np.ndarray, 
                 thresh: float=None,
                ) -> List:
        """hand, face and headtorso detection
        """
        thresh = thresh or self.thresh
        # 1. prepare input
        inps = self._preprocess(full_image)
        h, w = full_image.shape[:2]
        # 2. network inference
        self.detector.setInput(inps)
        bboxes, classes = self.detector.forward(self.outputs)
        raw_bboxes = np.squeeze(bboxes)
        raw_probas = np.squeeze(classes[...,1:])
        # 3. decoding outputs       
        rets = self._decoding(raw_bboxes,
                              raw_probas,
                              thresh,
                              img_height=h,
                              img_width=w,
                             )
        return rets
    
    def _decoding(self, 
                  raw_bboxes: np.ndarray, 
                  raw_probas: np.ndarray,
                  thresh: float,
                  img_height: int=None, 
                  img_width: int=None,
                 ) :
        """decode SSD bboxes with priors
        """
        def decode_one_bbox(bbox, prior) :
            ty, tx, th, tw = np.array(bbox)
            ycenter_a, xcenter_a, ha, wa = prior
            ty /= self.scales[0]
            tx /= self.scales[1]
            th /= self.scales[2]
            tw /= self.scales[3]
            w = np.exp(tw) * wa
            h = np.exp(th) * ha
            ycenter = ty * ha + ycenter_a
            xcenter = tx * wa + xcenter_a
            ymin = ycenter - h / 2.
            xmin = xcenter - w / 2.
            ymax = ycenter + h / 2.
            xmax = xcenter + w / 2.
            return [xmin, ymin, xmax, ymax]
        
        def non_maximum_suppression(bboxes) :
            # non maximum supporession for bbox selection
            if len(bboxes) == 0 :
                return []
            else :
                stacked_bboxes = np.row_stack(bboxes) # shape of [N, 4]
                left, top, right, bottom = stacked_bboxes.T # shape of [4, N]
                area = (right - left + self.epsilon) * (bottom - top + self.epsilon)
                indices = np.arange(len(left))
                picked = []
                while len(indices) > 0 :
                    # compute iou using the first bbox and the rest
                    first_idx = indices[0]
                    this_bbox = stacked_bboxes[first_idx:first_idx+1]
                    this_area = area[first_idx]
                    picked.append(first_idx)
                    # iou computation
                    if len(indices) > 1 :
                        # get the rest of bboxes
                        other_bboxes = stacked_bboxes[indices]
                        overlap_left = np.maximum(this_bbox[:,0], other_bboxes[:,0])
                        overlap_top = np.maximum(this_bbox[:,1], other_bboxes[:,1])
                        overlap_right = np.minimum(this_bbox[:,2], other_bboxes[:,2])
                        overlap_bottom = np.minimum(this_bbox[:,3], other_bboxes[:,3])
                        overlap_width = np.maximum(0, overlap_right - overlap_left)
                        overlap_height = np.maximum(0, overlap_bottom - overlap_top)
                        
                        # get the rest of bboxes areas
                        other_areas = area[indices]
                        # compute intersection and union areas
                        overlap_area = overlap_width * overlap_height
                        union_area = this_area + other_areas - overlap_area 
                        # compute iou
                        iou = overlap_area / union_area
                        # delete all locs with higher than allowed thresh
                        suppress_idxs = np.where(iou >= self.overlap_iou)[0]
                        indices = np.delete(indices, suppress_idxs.tolist())
                    else :
                        indices = np.delete(indices, [0])
                return picked
        
        max_probas = raw_probas.max(axis=-1)
        sorted_box_indices = np.argsort(max_probas)[::-1][:self.num_max_dets]
        candidates = []
        for i in sorted_box_indices :
            encoded_box = raw_bboxes[i]
            prior_box = self.anchors[i]
            decoded_box = decode_one_bbox(encoded_box, prior_box)
            cls_index = np.argmax(raw_probas[i])
            cls_proba = max_probas[i]
            if cls_proba < thresh :
                break
            candidates.append([self.classes[cls_index], cls_proba] + decoded_box)
        
        # apply non-maximum suppression
        bboxes = [this[2:] for this in candidates]
        picked_idxs = non_maximum_suppression(bboxes)
        nms_candidates = [candidates[i] for i in picked_idxs]
        # denormalize bboxes
        nms_candidates = [[c, 
                           p, 
                           x0 * img_width,
                           y0 * img_height,
                           x1 * img_width, 
                           y1 * img_height] \
                          for (c, p, x0, y0, x1, y1) \
                              in nms_candidates]
        return nms_candidates
    
    def annotate_results(self, img, rets) -> np.ndarray:
        """annotate object bboxes on the given input image
        """
        debug = np.array(img)
        thickness = max(3, int(min(min(img.shape[:2])//320 * 3, 7)))
        for this in rets :
            # parse a detection
            name, proba = this[:2]
            bbox = this[2:]
            left, top, right, bottom = bbox
            if np.mean(bbox) > 1 : # denormalized bboxes
                pass
            else :
                h, w = img.shape[:2]
                left *= w
                right *= w
                top *= h
                bottom *= h
            left, top, right, bottom = [int(v) for v in [left, top, right, bottom]]
            cls_index = self.classes.index(name)
            color = self.colors[cls_index]
            # plot on image
            cv2.rectangle(debug, 
                          (left, top), 
                          (right, bottom), 
                          color, 
                          thickness=thickness)
            cv2.rectangle(debug, (left-1, top-1), (right+1, top-int(25 * thickness/3)), color, thickness=-1)
            cv2.putText(debug, f'{name}: {proba:.2f}', (left, top-3), 
                        cv2.FONT_HERSHEY_TRIPLEX, 
                        thickness/3, 
                        thickness=thickness//3, 
                        color=(255-color[0],255-color[1],255-color[2]))
        return debug

####################################################################################################
# CenterNet (Hand, Face, Person) Model Inference
####################################################################################################
class HandFacePersonCenterNetONNXOpenCV:
    """The inference class for Alexa NU hand, face, and person object detection via centernet
    
    Given a test RGB image, this class provides the API to predict the object bounding boxes of hand, 
    face, and person classes. In particular, the detection results is organized as a list with each
    element of the format 
        [object_class, detection_confidence, bbox_left, bbox_top, bbox_right, bbox_bottom]
    
    """
    outputs = ['heatmaps', 'scales']
    classes = ['hand', 'face', 'person']
    colors =[(255, 0, 0), (0, 255, 127), (127, 0, 255)]
    epsilon = 1e-7
    def __init__(self, 
                 model_file_path: str,
                 target_size: int=320,
                 thresh: float=0.2,
                 use_normalized_bbox: bool=False,
                ) :
        """
        # INPUTS:
            model_file_path: file path the hand pose onnx model
            target_size: network's input image size
            thresh: minimum confidence value
            use_normalized_bbox: whether use normalized bbox in (0, 1) x (0, 1) 
                or abs bbox in (0, img_height-1) x (0, img_width-1)
        """
        assert os.path.isfile(model_file_path), \
            f"ERROR: fail to locate the input model file {model_file_path}"
        assert model_file_path.endswith('.onnx'), \
            f"ERROR: only .onnx model is supported"
        self.target_size = target_size
        self.thresh = thresh
        self.detector = cv2.dnn.readNet(model_file_path)
        self.use_normalized_bbox = use_normalized_bbox
        return
    
    def __call__(self, 
                 full_image: np.ndarray, 
                 thresh: float=None,
                ) -> List:
        """hand, face and person detection
        """
        thresh = thresh or self.thresh
        # 1. prepare input
        inps = self._preprocess(full_image)
        h, w = full_image.shape[:2]
        # 2. network inference
        self.detector.setInput(inps)
        heatmaps, scales = self.detector.forward(self.outputs)
        # 3. decoding outputs
        rets = self._decoding(heatmaps, 
                              scales, 
                              thresh,
                              img_height=h,
                              img_width=w,
                             )
        return rets

    def _local_maximum_detection(self, 
                                 heatmaps: np.ndarray, 
                                 thresh: float,
                                ) -> List:
        """detect local maximums of heatmaps
        """
        # apply the maximum filter
        heatmaps_max = cv2.dilate(heatmaps, np.ones([3,3]))
        # find local maximums
        local_max = (heatmaps - heatmaps_max + self.epsilon) >= 0
        # filter those below the required thresh
        mask = np.bitwise_and(
                    local_max, 
                    heatmaps > thresh
               )
        ys, xs, cs = np.where(mask > 0.5)
        yxc_list = list(zip(ys, xs, cs))
        return yxc_list
        
    def _decoding(self, 
                  heatmaps: np.ndarray, 
                  scales: np.ndarray, 
                  thresh: float,
                  img_height: int=None, 
                  img_width: int=None,
                 ) -> List:
        """decode object bounding boxes, probas, and classes from heatmaps and scales
        
        NOTE: we always return a normalized bounding box, i.e., in range of (0, 1)
        """
        # 1. normalize inputs
        if heatmaps.ndim == 4 :
            heatmaps = np.squeeze(heatmaps)
            scales = np.squeeze(scales)
            
        assert (heatmaps.ndim == 3) and (scales.ndim == 3), \
            f"ERROR: only single sample batch is supported"
        
        _, size = heatmaps.shape[:2]
        
        if self.use_normalized_bbox :
            x_coef = y_coef = 1./ size
        else :
            assert None not in [img_height, img_width], \
                "ERROR: original image height and width must be provided in decoding"
            x_coef = img_width / size
            y_coef = img_height / size
            
        yxc_list = self._local_maximum_detection(
                                    heatmaps, 
                                    thresh)

        outs = []
        for y, x, c in yxc_list :
            bh, bw = scales[y, x]
            left, right = x - bw/2, x + bw/2
            top, bottom = y - bh/2, y + bh/2
            proba = heatmaps[y, x, c]
            cls_name = self.classes[c]
            bbox = [left * x_coef, 
                    top * y_coef, 
                    right * x_coef, 
                    bottom * y_coef]
            outs.append([cls_name, proba] + bbox)
        return outs
    
    def _preprocess(self, full_image: np.ndarray) -> np.ndarray: 
        """preprocess full_image as the network's input
        """
        x = cv2.resize(full_image, (320, 320)).astype('float32') / 127.5 - 1
        return x[None]
    
    def annotate_results(self, img, rets) -> np.ndarray:
        """annotate object bboxes on the given input image
        """
        debug = np.array(img)
        thickness = max(3, int(min(min(img.shape[:2])//320 * 3, 7)))
        for this in rets :
            # parse a detection
            name, proba = this[:2]
            bbox = this[2:]
            left, top, right, bottom = bbox
            if np.mean(bbox) > 1 : # denormalized bboxes
                pass
            else :
                h, w = img.shape[:2]
                left *= w
                right *= w
                top *= h
                bottom *= h
            left, top, right, bottom = [int(v) for v in [left, top, right, bottom]]
            cls_index = self.classes.index(name)
            color = self.colors[cls_index]
            # plot on image
            cv2.rectangle(debug, 
                          (left, top), 
                          (right, bottom), 
                          color, 
                          thickness=thickness)
            cv2.rectangle(debug, (left-1, top-1), (right+1, top-int(25 * thickness/3)), color, thickness=-1)
            cv2.putText(debug, f'{name}: {proba:.2f}', (left, top-3), 
                        cv2.FONT_HERSHEY_TRIPLEX, 
                        thickness/3, 
                        thickness=thickness//3, 
                        color=(255-color[0],255-color[1],255-color[2]))
        return debug


####################################################################################################
# Face Mesh Model Inference
####################################################################################################
class NUFace3DOnnxOpenCV():
    def __init__(self,
                 onnx_path,
                 data_path=None):
        self.onnx_path = onnx_path
        if data_path is None :
            data_path = os.path.join(os.path.dirname(onnx_path), 
                                    'nuface3d_data.pkl')
        self.data_path = data_path 
        self.dnn_model = cv2.dnn.readNetFromONNX(onnx_path)
        self.load_mesh(data_path)
        self.input_size = [120, 120]

    def __call__(self, image: np.ndarray, bboxs: np.ndarray, return_dense=False) -> dict:
        '''
        Args:
            image: input RGB image
            bboxs: Nx4 bboxs
            return_dense: if true, also reconstruct the dense landmarks

        Returns:

        '''
        # a wrapper function for pytorch tensor in tensor out

        num_bbox = bboxs.shape[0]
        face_crops, transforms, inv_transforms = [], [], []
        for i in range(num_bbox):
            face_crop, transform, inv_transform = self.preprocess_img(image, bboxs[i, :], self.input_size)
            face_crops.append(face_crop)
            transforms.append(transform)
            inv_transforms.append(inv_transform)

        face_crops = np.array(face_crops)
        face_crops_bgr = face_crops[:, ::-1, :, :]

        self.dnn_model.setInput(face_crops_bgr)
        params_np = self.dnn_model.forward()
        pred_lmks = []
        pred_params = []
        for i in range(params_np.shape[0]):
            param_np = params_np[i, ...]
            param_np_denorm = self.denormalize_params(param_np)
            pred_params.append(param_np_denorm)
        pred_params = np.array(pred_params)

        preds = {'s': np.zeros([num_bbox, 1]), 'angle': np.zeros([num_bbox, 3]), 'T': np.zeros([num_bbox, 2]), 'w': np.zeros([num_bbox, 50]),
                 'lmk68': np.zeros([num_bbox, 68, 3])}
        if return_dense:
            preds['vertices'] = np.zeros([num_bbox, self.nver, 3])
        for i in range(num_bbox):
            inv_transform = inv_transforms[i]
            s, r, T, w = pred_params[i, 0], pred_params[i, 1:4], pred_params[i, 4:6], pred_params[i, 6:]
            angle = Rotation.from_rotvec(r).as_euler('xyz', degrees=False)
            s, angle, T = self.convert_coordinate_systems(s, angle, T, self.input_size[1])
            s, angle, T = self.transform_srt(s, angle, T, inv_transform)
            s, angle, T = self.convert_coordinate_systems(s, angle, T, image.shape[0])
            preds['s'][i, :], preds['angle'][i, :], preds['T'][i, :], preds['w'][i, :] = s, angle, T, w
            preds['lmk68'][i, ...] = self.reconstruct_lmks(s, angle, T, w).T
            preds['lmk68'][i, :, 1] = image.shape[0] + 1 - preds['lmk68'][i, :, 1]
            if return_dense:
                preds['vertices'][i, ...] = self.reconstruct_vertices(s, angle, T, w).T
                preds['vertices'][i, :, 1] = image.shape[0] + 1 - preds['vertices'][i, :, 1]
        return preds


    def transform_srt(self, s: float, angle: np.ndarray, T: np.ndarray, transform) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        transform scale, angle and translation together with 2d affine transform
        assuming scale, angle and translation is w.r.t image coordinates
        :param s: scale (1,)
        :param angle: euler angles in radians (3,)
        :param T: translation in 2d (2,) or in 3d (3,)
        :param transform: affine transform matrix
        :return: s, angle, T after transform
        """
        transform = transform.copy()
        R_trans = transform[:2, :2]
        t_trans = transform[:, 2]
        s_trans = np.linalg.norm(R_trans[:, 0])
        R_trans = R_trans / s_trans
        R3d = np.eye(3)
        R3d[:2, :2] = R_trans
        R_angle = Rotation.from_euler('xyz', angle, degrees=False).as_matrix()
        R_new = np.matmul(R3d, R_angle)

        angle = Rotation.from_matrix(R_new).as_euler('xyz', degrees=False)
        s *= s_trans
        T[:2] = s_trans * np.dot(R_trans, T[:2]) + t_trans

        return s, angle, T

    def convert_coordinate_systems(self, 
                                   s: float, 
                                   angle: np.ndarray, 
                                   T: np.ndarray, 
                                   img_h: float
                                  ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Can go back and forth between img and world coordinate systems.
        :param s: scale (1,)
        :param angle: euler angles in radians (3,)
        :param T: translation in 2d (2,)
        :param img_h: image height
        :return: s, angle, T in a different coordinate system
        """
        def srt2P(s: float, angle: np.ndarray, T: np.ndarray) -> np.ndarray:
            """
            :param s: scale (1,)
            :param angle: euler angles in radians (3,)
            :param T: translation in 2d (2,)
            :return: 4x4 similarity transform matrix
            """
            P = np.zeros((4, 4), dtype=np.float32)
            P[:T.shape[0], 3] = T
            R_mat = s * Rotation.from_euler('xyz', angle, degrees=False).as_matrix()
            P[:3, :3] = R_mat
            P[3, 3] = 1
            return P
        
        def P2srt(P: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
            """
            :param P: 4x4 similarity transform matrix
            :return: s, angle, T
            """
            T = P[:2, 3]
            R1 = P[0, :3]
            R2 = P[1, :3]
            R3 = P[2, :3]
            s = (np.linalg.norm(R1) + np.linalg.norm(R2) + np.linalg.norm(R3)) / 3.0
            R = P[:3, :3] / s
            angle = Rotation.from_matrix(R).as_euler('xyz', degrees=False)
            return s, angle, T
        P_old = srt2P(s, angle, T)
        P_new = np.zeros((4, 4))
        T3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        inv_T3 = np.linalg.inv(T3)
        P_new[:3, :3] = np.matmul(inv_T3, np.matmul(P_old[:3, :3].copy(), T3))
        P_new[:, 3] = P_old[:, 3]
        P_new[1, 3] = 1 + img_h - P_old[1, 3]
        P_new[2, 3] = - P_old[2, 3]
        s_new, angle_new, t_new = P2srt(P_new.copy())

        return s_new, angle_new, t_new

    def relax_bbox(self, bbox: np.ndarray, img_w: float, img_h: float, bbox_relax_factor: float = 0.1) -> np.ndarray:
        x1, y1, x2, y2 = bbox[:4]
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)
        x1 = max(0, cx - w * (1 + bbox_relax_factor) / 2)
        y1 = max(0, cy - h * (1 + bbox_relax_factor) / 2)
        x2 = min(img_w - 1, cx + w * (1 + bbox_relax_factor) / 2)
        y2 = min(img_h - 1, cy + h * (1 + bbox_relax_factor) / 2)
        bbox[:4] = np.array([x1, y1, x2, y2])
        return bbox

    def preprocess_img(self, img: np.ndarray, bbox: np.ndarray, input_size: List[int]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        '''
        :param img: height * width * 3 np.ndarray RGB image
        :param input_size: input size to resize image, for example [224, 224]
        :param bbox: np.array of [x1, y1, x2, y2, conf(optional)]
        :return: resized normalized image of a face crop and transform and inverse transform
        '''
        bbox = self.relax_bbox(bbox, img.shape[1], img.shape[0], bbox_relax_factor=0.1)
        transform, inv_transform = self.calc_affine_transforms(bbox, input_size)
        frame = cv2.warpAffine(img, transform, tuple(input_size))
        frame = np.transpose(frame, (2, 0, 1))
        frame = frame.astype(np.float32) / 255.0

        return frame, transform, inv_transform
    
    def calc_affine_transforms(self, 
                               bbox: np.array, 
                               input_size: list, 
                               rot: float = 0.0, 
                               sc: float = 1.0,
                               t: np.array = np.array([0, 0], dtype=np.float32)
                              ) -> Tuple[np.array, np.array]:
        """
        :param bbox: [x1, y1, x2, y2]
        :param input_size: [h, w], [224, 224] by default
        :param rot: rotation angle in degrees
        :param sc: scale in degrees, by default 1.0
        :param t: 2d translation
        :return: transform and inverse transform
        """
        def get_3rd_point(a, b):
            direct = a - b
            return b + np.array([-direct[1], direct[0]], dtype=np.float32)
        
        def get_dir(src_point, rot_rad):
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)

            src_result = [0, 0]
            src_result[0] = src_point[0] * cs - src_point[1] * sn
            src_result[1] = src_point[0] * sn + src_point[1] * cs

            return src_result
        def get_affine_transform(center,
                                 scale,
                                 rot,
                                 output_size,
                                 shift=np.array([0, 0], dtype=np.float32),
                                 inv=0):
            if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
                scale = np.array([scale, scale], dtype=np.float32)

            scale_tmp = scale
            src_w = scale_tmp[0]
            dst_w = output_size[0]
            dst_h = output_size[1]

            rot_rad = np.pi * rot / 180
            src_dir = get_dir([0, src_w * -0.5], rot_rad)
            dst_dir = np.array([0, dst_w * -0.5], np.float32)

            src = np.zeros((3, 2), dtype=np.float32)
            dst = np.zeros((3, 2), dtype=np.float32)
            src[0, :] = center + scale_tmp * shift
            src[1, :] = center + src_dir + scale_tmp * shift
            dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
            dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

            src[2:, :] = get_3rd_point(src[0, :], src[1, :])
            dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

            if inv:
                trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
            else:
                trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

            return trans

        ih, iw = input_size
        aspect_ratio = iw / ih
        center = (bbox[2:4] + bbox[:2]) / 2
        bbox_h, bbox_w = bbox[2:4] - bbox[:2]
        w, h = bbox_w, bbox_h
        if w > aspect_ratio * h:  # expands w, h to match aspect ratio
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w * 1.0, h * 1.0]) * sc
        translate = t
        translate[0] *= bbox_w  # bbox_w #img_w
        translate[1] *= bbox_h  # bbox_h # img_h
        center += translate

        T = get_affine_transform(center, scale, rot, input_size)
        inv_T = get_affine_transform(center, scale, rot, input_size, inv=1)

        return T, inv_T

    def load_mesh(self, data_path: str):
        self.model_data = pickle.load(open(data_path, 'rb'))
        self.nShapeParams = self.model_data['shapePC'].shape[1]
        self.nExpParams = self.model_data['expPC'].shape[1]
        self.nver = self.model_data['shapePC'].shape[0] // 3
        self.kpt_ind = self.model_data['kpt_ind']

        self.pca_vectors = np.zeros([self.nShapeParams + self.nExpParams, 3, self.nver])
        for i in range(self.nShapeParams):
            self.pca_vectors[i, ...] = self.model_data['shapePC'][:,
                                       i:(i + 1)].reshape(3, -1, order='F')
        for i in range(self.nExpParams):
            self.pca_vectors[self.nShapeParams + i, ...] = self.model_data['expPC'][:,i:(i + 1)
                                                           ].reshape(3, -1,order='F')

        self.mean_face = self.model_data['shapeMU'].reshape(3, -1, order='F')
        self.mean_params = self.model_data['mean_params']
        self.std_params = self.model_data['std_params']

    def normalize_params(self, params: np.ndarray, normalize_scale=False) -> np.ndarray:
        '''params : 56'''
        params[0] = params[0] * 1000
        params = (params - self.mean_params) / self.std_params
        return params

    def denormalize_params(self, params: np.ndarray) -> np.ndarray:
        '''params : 56'''
        params = (params * self.std_params) + self.mean_params
        params[0] = params[0] / 1000
        return params

    def parse_params(self, params: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        s = params[0]
        r = params[1:4]
        t = params[4:6]
        w = params[6:]
        R = cv2.Rodrigues(r)[0]
        return s, R, t, w

    def reconstruct_vertices(self, s, angle, T, w) -> np.ndarray:
        R = Rotation.from_euler('xyz', angle, degrees=False).as_matrix()
        shape3D = self.mean_face + np.sum(w[:, np.newaxis, np.newaxis] * self.pca_vectors, axis=0)
        vertices = s * np.dot(R, shape3D)  # (3, nver)
        vertices[:2, :] = vertices[:2, :] + T[:2, np.newaxis]
        # returning dense vertices on face crop with world coordinates
        return vertices

    def reconstruct_lmks(self, s, angle, T, w) -> np.ndarray:
        R = Rotation.from_euler('xyz', angle, degrees=False).as_matrix()
        shape3D = self.mean_face[..., self.kpt_ind] + np.sum(w[:, np.newaxis, np.newaxis] * self.pca_vectors[..., self.kpt_ind], axis=0)
        vertices = s * np.dot(R, shape3D)  # (3, nver)
        vertices[:2, :] = vertices[:2, :] + T[:2, np.newaxis]
        # returning dense vertices on face crop with world coordinates
        return vertices
    
    def annotate_results(self, 
                  img: np.ndarray, 
                  kpts_xy: np.ndarray, 
                  color = (0,255,0),
                  backend='cv2') -> np.ndarray:
        '''
        :param img: BGR image
        :param kpts_xy: Nx2 or Nx3 facial landmarks
        :param color: color for drawing, in (B,G,R) order
        :return: img with landmarks
        '''
        assert backend in ['pyplot', 'cv2'], \
            f"ERROR: the ploting backend={backend} is NOT supported"
        if backend == 'cv2' and img is None :
            raise IOError("ERROR: img input must be provided in cv2 backend")
        debug = np.array(img)
        for i, (x, y) in enumerate(kpts_xy[:,:2]) :
            if backend == 'pyplot' :
                pyplot.plot(x, y, color=color, marker='o', markersize=linewidth+2)
                if show_index : pyplot.annotate(str(i), (x, y), fontsize=15, color='w' if render else 'k')
            elif backend == 'cv2' :                
                cv2.circle(debug, (int(x), int(y)), radius=2 * min(3, max(1,img.shape[0]//320)), 
                           color=color, thickness=-1)
        return debug

    
####################################################################################################
# Face Semantic Parsing Model Inference
####################################################################################################
class FaceParsingTfLite(object) :
    """engine for face parsing pseudo annotation
    """
    facecls2idx={'BACKGROUND': 0,
                 'SKIN': 1,
                 'NOSE': 2,
                 'RIGHT_EYE': 3,
                 'LEFT_EYE': 4,
                 'RIGHT_BROW': 5,
                 'LEFT_BROW': 6,
                 'RIGHT_EAR': 7,
                 'LEFT_EAR': 8,
                 'MOUTH_INTERIOR': 9,
                 'TOP_LIP': 10,
                 'BOTTOM_LIP': 11,
                 'NECK': 12,
                 'HAIR': 13,
                 'BEARD': 14,
                 'CLOTHING': 15,
                 'GLASSES': 16,
                 'HEADWEAR': 17,
                 'FACEWEAR': 18,
                 'IGNORE': 19,
                }
    num_classes = 20
    def __init__(self, model_path) :
        # Load model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # Test the model on random input data.
        self.face_size = input_details[0]['shape'][1:3]
        self.model = interpreter
        self.inp_index = input_details[0]['index']
        self.out_index = output_details[0]['index']
        return
    
    def _convert_mask_to_polygons(self, img, y_best) :
        h, w = img.shape[:2]
        assert h == w, \
            "ERROR: input image shape must be square"
        factor = float( img.shape[0] / y_best.shape[0] )
        labels = np.argmax(y_best, axis=-1)
        polygon_lut = {}
        for name, idx in self.facecls2idx.items() :
            mask = (labels == idx).astype('uint8')
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            polygon_lut[name] = [ np.round( factor * cnt ).astype('int').tolist() for cnt in contours]
        return polygon_lut
    
    def _convert_polygons_to_mask(self, polygon_lut, height, width) :
        # convert a polygon dict to a stack of binary masks
        # NOTE:
        #   the number of masks is the number of semantic classes in `mappings`
        #   the semantic class of the i-th mask can be found in `._get_face_class_mapping` 
        masks = np.zeros([height, width, self.num_classes], dtype='float32')
        for (face_cls, polys) in (sorted(polygon_lut.items())) :
            if polys :
                contours = [np.array(x).reshape([-1, 1, 2]) for x in polys]
                mask = np.zeros([height, width])
                _ = cv2.fillPoly(mask, contours, color=1)
                idx = self.facecls2idx[face_cls]
                masks[...,idx] += mask
        return masks
    
    def _preprocess(self, rgb) :
        x = cv2.resize(rgb, self.face_size)
        return x
    
    def _predict(self, x) :
        self.model.set_tensor(self.inp_index, 
                              x[None].astype('float32'))
        self.model.invoke()
        out = self.model.get_tensor(self.out_index)[0]
        return out

    def __call__(self, aligned_face_rgb, return_mask=True ) :
        # preprocess
        x = self._preprocess(aligned_face_rgb)
        # pseudo annotation
        y = self._predict(x)
        polygons = self._convert_mask_to_polygons(aligned_face_rgb, y)
        mask = self._convert_polygons_to_mask(polygons, 
                                              aligned_face_rgb.shape[0],
                                              aligned_face_rgb.shape[1])
        if return_mask :
            return mask
        else :
            return polygons
        
        

class FaceAlignmentCropper :
    def __init__(self, 
                 ssd,
                 cnet, 
                 facemesh,
                 min_scale=1,
                 max_scale=8,
                 max_num_faces=1,
                 pad_face_perc=0.25,
                ) :
        self.ssd = ssd
        self.cnet = cnet
        self.facemesh = facemesh
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_num_faces = max_num_faces
        self.pad_face_perc = pad_face_perc
    
    def __call__(self, img, bbox=None, thresh=0.5, output_size=None) :
        # 1. ssd face detector
        if bbox is None :
            rets = self.ssd(img, thresh)
            bbox = []
            count = 0
            for d in rets :
                if d[0] == 'face' :
                    bbox.append([v for v in d[-4:]])
                    count += 1
                    if count >= self.max_num_faces: 
                        break
            # 2. try centernet detector if ssd fails
            if count == 0 :
                rets = self.cnet(img, thresh)
                for d in rets :
                    if d[0] == 'face' :
                        bbox.append([v for v in d[-4:]])
                        count += 1
                        if count >= self.max_num_faces: 
                            break
        else :
            bbox = [bbox]
        # 3. facial landmark detection
        crop_list = []
        for this in bbox :
            frets = self.facemesh(img, np.array([this]))
            kpts68 = frets['lmk68'][0][:,:2]
            src_pts = self._precompute_landmark_stats(img, kpts68)
            if output_size is None :
                mat, output_size = self._compute_transform_matrix(src_pts)
            else :
                mat = self._compute_transform_matrix_given_size(src_pts, output_size)
            if mat is not None :
                # this is a valid sample
                crop = cv2.warpPerspective(img, mat, (output_size, output_size))
                ann = self._prepare_annotation(kpts68, mat)
                crop_list.append([crop, ann, mat])
        return crop_list

    def _prepare_annotation(self, kpts68, mat) :
        kpts = np.squeeze(cv2.perspectiveTransform(kpts68[:,None,:], mat)).astype('float')
        lut = {
            'mat': [float(v) for v in mat.ravel().tolist()],
            'lmk68': kpts.ravel().tolist()
        }
        return lut
    
    def _precompute_landmark_stats(self, img, landmarks, em_scale=0.1) : 
        """compute the face alignment statistics
        NOTE: this alignment is NOT the classic 5 point based rigid alignment 
        which is widely used in dlib, LFW, etc., but the new transform that is
        used in StyleGAN. This is a re-implementation using OpenCV.
        """
        x_scale = self.pad_face_perc + 1.
        y_scale = 1
        lm = np.array(landmarks)
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        x *= x_scale
        y = np.flipud(x) * [-y_scale, y_scale]
        c = eye_avg + eye_to_mouth * em_scale
        src_pts = np.stack([c - x - y, 
                            c - x + y, 
                            c + x + y, 
                            c + x - y]).astype('float32')
        return src_pts
    
    def _compute_transform_matrix(self, src_pts) :
        inp_size = np.sum(src_pts[1] - src_pts[0])
        mat, output_size = None, None
        scale = int(inp_size // 128)
        if scale < self.min_scale :
            return None, None
        else :
            if scale > self.max_scale :
                scale = self.max_scale
                
            output_size = 128 * scale
            dst_pts = np.float32([
                            [0, 0],
                            [0, output_size-1],
                            [output_size-1, output_size-1],
                            [output_size-1, 0]
                        ])
            mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
            return mat, output_size
        
    def _compute_transform_matrix_given_size(self, src_pts, output_size) :
        dst_pts = np.float32([
                        [0, 0],
                        [0, output_size-1],
                        [output_size-1, output_size-1],
                        [output_size-1, 0]
                    ])
        mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return mat
    