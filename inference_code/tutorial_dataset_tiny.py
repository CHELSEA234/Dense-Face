import json
import cv2
import numpy as np
import csv
import os
import random
import torch
import pandas as pd

from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import OrderedDict
from math import cos, sin
# from helpers import *

class MyDatasetFace(Dataset):
    def __init__(
                self,
                dataset='celeba',
                data_dir="/home/ec2-user/CelebA/",
                keep_prompt=False,
                csv_name="celebaFacesDataset256_v2.csv",
                mode='train',
                interval=4,  # among how many samples, does the reference id show up.
                dict_file_name="CASIA_subject.txt",
                eval_mode=None
                ):
        if mode not in ['train', 'val', 'eval']:
            raise ValueError("Please check the mode.")
        self.eval_mode = eval_mode
        self.mode = mode
        self.interval = interval
        self.data_dir = os.path.join(data_dir, dataset)
        self.csv_name = os.path.join(self.data_dir, csv_name)
        self.img_list, self.con_list, self.cap_list, self.ref_list, self.feat_list = [], [], [], [], []
        self.img_bbox_list, self.ref_bbox_list = [], []
        self.key_point_list = []
        self.landmark_GT_list = []
        self.ref_bbox_dict = dict() # key, value pair is ref_id and bbox coordinates.
        self.sub_dict = self._create_dictionary(file_dir=self.data_dir,
                                                file_name=dict_file_name)

        csv_file = open(self.csv_name)
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        data_count = 0
        ref_p_count = []

        for row in tqdm(csv_reader):
            if line_count == 0:
                print("the title is: ")
                print(row)
                # import sys;sys.exit(0)
                pass
            else:
                if dataset == 'celeba':
                    file_p, mask_p, cap_p, ref_p = row[0], row[4], row[12], row[-1]
                    # file_p, mask_p, cap_p, ref_p = row[0], row[14], row[12], row[-1]
                    print(f"find the bounding box first.")
                    import sys;sys.exit(0)
                elif dataset == 'CASIA':
                    # file_p, mask_p, cap_p, ref_p = row[0], row[4], row[11], row[-1]
                    file_p, mask_p, cap_p, ref_p = row[0], row[4], row[-2], row[-4]
                    # file_p, mask_p, cap_p, ref_p = row[0], row[4], row[11], row[0]
                    bbox = row[9]
                    # ref_bbox = row[9]
                    ref_bbox = row[-3]
                    sub_id = row[10]
                    key_point = row[-1]
                    ld_file = row[3]

                sub_id = sub_id.zfill(7)
                if sub_id not in self.sub_dict:
                    continue

                ## caption
                cap_p = self._retrieve_caption_new(cap_p, target_word='person')
                if cap_p is None:
                    continue

                ## facenet feature
                feat_numpy = self._retrieve_facenet(file_p)
                if feat_numpy is None:
                    continue

                self.sub_dict[sub_id].append(data_count)
                data_count += 1

                self.feat_list.append(feat_numpy)
                self.img_list.append(os.path.join(self.data_dir, file_p))
                self.con_list.append(os.path.join(self.data_dir, mask_p))
                self.landmark_GT_list.append(ld_file)

                if keep_prompt:
                    self.cap_list.append(cap_p)
                else:
                    self.cap_list.append("A close-up photo of * .")
                self.ref_list.append(os.path.join(self.data_dir, ref_p))
                self.img_bbox_list.append(bbox)
                self.ref_bbox_list.append(ref_bbox)
                self.key_point_list.append(key_point)

                if ref_p not in ref_p_count:
                    ref_p_count.append(ref_p)

            line_count += 1
            if line_count == 1000:
               break

        csv_file.close()
        self.num_samples = len(self)
        self.sub_dict_lst = list(self.sub_dict.items())  # list of a tuple, (sub_id, lst[access_idx])
        print("===========================================================")
        print(f"The training samples {len(self.img_list)} has been loaded.")
        print(f"We have {len(ref_p_count)} different individuals.")
        print("===========================================================")

    def __len__(self):
        # return len(self.img_list)
        return len(self.sub_dict)

    def _create_dictionary(self, file_dir, file_name="CASIA_subject.txt"):
        '''
            create the dictionary, in which key-value are {subject_id}-{access_idx_list}
        '''
        od = OrderedDict()  # .
        file_name = os.path.join(file_dir, file_name)
        txt_file = open(file_name, 'r')
        lines = txt_file.readlines()
        for line in lines:
            sub_id = line.strip()
            od[sub_id] = []
        txt_file.close()
        return od

    def _resize(self, image):
        return cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)

    def draw_axis(self, yaw, pitch, roll, tdx=127.5, tdy=127.5, size=100):

        white_img = np.ones((256, 256, 3), dtype = np.uint8)
        white_img = 255* white_img

        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = 2*size * (sin(yaw)) + tdx
        y3 = 2*size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(white_img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
        cv2.line(white_img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
        cv2.line(white_img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

        return white_img

    def _retrieve_caption(self, json_f, model_name="blip2-flan-t5-xl"):
        '''
            return the caption from the json file.
        '''
        json_f = os.path.join(self.data_dir, json_f)
        json_f = open(json_f, "r")
        data = json.load(json_f)
        for key, value in data.items():
            return key
            # print(value.get("model")) # blip-flan-t5
            if value.get('model') == model_name:
                return key
        ## does not have blip-flan-t5 output.
        return key

    def _crop_img(self, image, bbox):
        '''
            cropping the image based on the input bouning box.
        '''
        bbox = bbox[1:-1].split(',')

        x1 = int(float(bbox[0].strip())*image.shape[1])
        y1 = int(float(bbox[1].strip())*image.shape[0])
        x2 = int(float(bbox[2].strip())*image.shape[1])
        y2 = int(float(bbox[3].strip())*image.shape[0])

        if x1 <= 0:
            x1 = 0
        if y1 <= 0:
            y1 = 0
        if x2 >= image.shape[1]:
            x2 = image.shape[1]
        if y2 >= image.shape[0]:
            y2 = image.shape[0]

        crop_img = image[y1:y2, x1:x2]
        if crop_img.shape[0] != 256 or crop_img.shape[1] != 256:
            crop_img = self._resize(crop_img)        
        return crop_img

    def _retrieve_facenet(self, file_p):
        """
            obtain the pre-cached facenet feature.
        """
        img_path = os.path.join(self.data_dir, file_p)
        file_p_suffix = file_p[-3:]
        feat_p = file_p.replace('images', 'feat_file_v2').replace(file_p_suffix, 'pth')
        feat_path = os.path.join(self.data_dir, feat_p)
        if os.path.isfile(feat_path):
            try:
                feat_numpy = torch.load(feat_path)
                if feat_numpy.shape == (1, 512):
                    return feat_numpy
                else:
                    return None
            except:
                return None
        else:
            return None

    def _retrieve_caption_new(self, cap_p, target_word='*'):
        '''modify the caption from the csv file.'''
        if 'man' in cap_p \
            or 'woman' in cap_p \
            or 'boy' in cap_p \
            or 'women' in cap_p \
            or 'men' in cap_p \
            or 'girl' in cap_p \
            or 'player' in cap_p \
            or 'officer' in cap_p \
            or 'soldier' in cap_p \
            or 'bruce lee' in cap_p:
            pass
        else:
            # continue
            return None

        ## man and men are included in woman and women.
        if 'woman' in cap_p:
            cap_p = cap_p.replace('woman', target_word)
        elif 'women' in cap_p:
            cap_p = cap_p.replace('women', target_word)
        elif 'man' in cap_p:
            cap_p = cap_p.replace('man', target_word)
        elif 'men' in cap_p:
            cap_p = cap_p.replace('men', target_word)

        cap_p = cap_p.replace('boy', target_word)
        cap_p = cap_p.replace('girl', target_word)
        cap_p = cap_p.replace('player', target_word)
        cap_p = cap_p.replace('officer', target_word)
        cap_p = cap_p.replace('solider', target_word)
        cap_p = cap_p.replace('bruce lee', target_word)

        return cap_p

    def _get_landmark(self, 
                    ld_file,
                    file_p,
                    image_ld_size=64,
                    viz=False,
                    interval=10):
        if viz: 
            image = cv2.imread(os.path.join(self.data_dir, file_p))
            image = cv2.resize(image, 
                                (image_ld_size, image_ld_size), 
                                interpolation = cv2.INTER_LINEAR)
        landmarks = pd.read_csv(os.path.join(self.data_dir, ld_file), header=None).to_numpy()
        landmarks = landmarks * image_ld_size
        # print("the landmarks number: ", landmarks.shape)

        landmark_map_lst = []
        for ld_idx, landmark in enumerate(landmarks):
            ## sampling
            if ld_idx % interval != 0:
                continue
            landmark_map = np.zeros((1, image_ld_size, image_ld_size), dtype='float32')
            x, y = landmark
            x, y = int(x), int(y)
            landmark_map = points_to_gaussian_heatmap((x, y), 64, 64, 1)
            landmark_scale = 1./landmark_map.max()
            landmark_map = landmark_map * landmark_scale
            landmark_map_lst.append(np.expand_dims(landmark_map, axis=0))
            if viz:
                if ld_idx < 100:
                    color_ = (0, 0, 255)
                elif 100 <= ld_idx and ld_idx < 200:
                    color_ = (0, 255, 255)
                elif 200 <= ld_idx and ld_idx < 300:
                    color_ = (255, 255, 255)
                else:
                    color_ = (255, 0, 255)
                cv2.circle(image, (x, y), 0, color_, -1)
                landmark_map = (landmark_map*255.).astype(np.uint8)
                cv2.imwrite(f"demo_{ld_idx}.png", landmark_map)
                cv2.imwrite(f"demo_{ld_idx}_overlaid.png", image)

        landmark_map_GT = np.concatenate(landmark_map_lst, axis=0)
        if viz:
            print("the num of used landmarks is: ", len(landmark_map_lst))
            import sys;sys.exit(0)
        else:
            return landmark_map_GT

    def _generate_pseudo_map(self, source_mask, viz=False,              
                            target_array = np.array([(227, 206, 166), 
                                                    (0, 127, 255), 
                                                    (111, 191, 253), 
                                                    (180, 120, 31), 
                                                    (153, 154, 251),
                                                    (138, 223, 178), 
                                                    (214, 178, 202), 
                                                    (28, 26, 227), 
                                                    (154, 61, 106), 
                                                    (44, 160, 51)])
                             ):
        matching_pixels = np.all(source_mask == target_array[:, None, None], axis=3).astype(int)
        matching_pixels = matching_pixels.reshape(10, 64, 4, 64, 4).mean(axis=(2, 4))
        matching_pixels = (matching_pixels >= 0.5).astype(np.uint8)
        if viz:
            cv2.imwrite("demo.png", source_mask)
            for i, binary_map in enumerate(matching_pixels):
                # Create a grayscale image from the binary map
                binary_image = (binary_map * 255).astype(np.uint8)

                # Write the image using OpenCV
                image_filename = f"binary_map_{i + 1}.png"
                cv2.imwrite(image_filename, binary_image)

                print(f"Saved: {image_filename}")
        return matching_pixels


    def __getitem__(self, batch_idx):

        while True:
            sub_idx_lst = self.sub_dict_lst[batch_idx][1]  # list of access idx
            if len(sub_idx_lst) == 0:
                batch_idx = batch_idx + 1
            else:
                break
        idx_src = sub_idx_lst[random.randint(0,len(sub_idx_lst)-1)]
        if self.eval_mode != 'face_swap':
            idx_tar = sub_idx_lst[random.randint(0,len(sub_idx_lst)-1)]
        else:
            while True:
                random_batch_idx = random.randint(0,len(self.sub_dict)-1)
                if random_batch_idx == batch_idx:
                    continue
                sub_idx_lst = self.sub_dict_lst[random_batch_idx][1]  # list of access idx
                if len(sub_idx_lst) == 0:
                    random_batch_idx = random_batch_idx + 1
                else:
                    break
            idx_tar = sub_idx_lst[random.randint(0,len(sub_idx_lst)-1)]
        loss_flag = 0

        # refer_filename = self.ref_list[idx_src]     # self.ref_list should never be used.
        refer_filename = self.img_list[idx_src]     # source image
        refer_feat = self.feat_list[idx_src]
        prompt = self.cap_list[idx_tar]
        pseduo_mask = self.con_list[idx_tar]        # pseduo mask
        target_filename = self.img_list[idx_tar]    # target image
        img_bbox = self.img_bbox_list[idx_tar]

        ## capture the dense landmark.
        target_ld_name = target_filename.replace('images', 'landmarks').replace('.jpg', '.npy')
        try:    # in case file exists but broken.
            target_ld = np.load(target_ld_name)
            assert target_ld.shape == (478, 64, 64)
        except:
            ld_file = self.landmark_GT_list[idx_tar]
            print(f"capture a failure on {ld_file}.")
            target_ld = self._get_landmark(ld_file, file_p=None, interval=1)

        ## generate pseudo mask.
        pseduo_mask_RGB = cv2.imread(pseduo_mask)
        pseduo_mask = self._generate_pseudo_map(pseduo_mask_RGB)

        ## now change the source to head pose.
        # source = cv2.imread(source_filename)
        pitch, yaw, roll = self.key_point_list[idx_tar][1:-1].split(',')
        pitch, yaw, roll = pitch.strip(), yaw.strip(), roll.strip()
        pitch, yaw, roll = float(pitch[1:-1]), float(yaw[1:-1]), float(roll[1:-1])
        source = self.draw_axis(yaw, pitch, roll)

        target = cv2.imread(target_filename)
        ## change the refer to the pre-cached arcface feature. 
        refer = refer_feat[0]
        refer_img = cv2.imread(refer_filename)

        # resize.
        if target.shape[0] != 256 or target.shape[1] != 256:
            target = self._resize(target)
        refer_img = self._resize(refer_img)

        if prompt == "":
            print(self.con_list[idx_tar])
            import sys;sys.exit(0)

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        refer_img = cv2.cvtColor(refer_img, cv2.COLOR_BGR2RGB)
        pseduo_mask_RGB = cv2.cvtColor(pseduo_mask_RGB, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        pseduo_mask_RGB = pseduo_mask_RGB.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        refer_img = (refer_img.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, refer=refer, 
                    refer_img=refer_img,
                    bbox_t=img_bbox, flag=loss_flag,
                    ld=target_ld,
                    p_mask=pseduo_mask,
                    target_path=target_filename,
                    control_path=pseduo_mask,
                    refer_path=refer_filename,
                    hint_mask=pseduo_mask_RGB)