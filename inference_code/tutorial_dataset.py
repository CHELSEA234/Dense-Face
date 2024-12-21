import json
import cv2
import numpy as np
import csv
import os
import random
import torch

from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import OrderedDict
from math import cos, sin

class MyDatasetFace(Dataset):
    def __init__(
                self,
                dataset='CelebA',
                data_dir="/home/ec2-user/CelebA/",
                keep_prompt=False,
                csv_name="celebaFacesDataset256_v2.csv",
                mode='train',
                interval=4,  # among how many samples, does the reference id show up.
                dict_file_name="CASIA_subject.txt"
                ):

        dataset_list = ['CelebA', "CASIA"]
        csv_name_lst = ["celebaFacesDataset_v2_headpose.csv", "casiaFacesDataset256_v3_headpose.csv"]
        dict_file_name_lst = ['CelebA_subject.txt', 'CASIA_subject.txt']
        # dataset_list = ['CelebA']
        # csv_name_lst = ["celebaFacesDataset_v2_headpose.csv"]
        # dict_file_name_lst = ['CelebA_subject.txt']

        self.img_list, self.con_list, self.cap_list, self.ref_list, self.feat_list = [], [], [], [], []
        self.img_bbox_list, self.ref_bbox_list = [], []
        self.key_point_list = []
        self.sub_dict = dict()
        data_count = 0

        for dataset_idx in range(2):
    
            dataset = dataset_list[dataset_idx]
            csv_name = csv_name_lst[dataset_idx]
            dict_file_name = dict_file_name_lst[dataset_idx]

            self.dataset = dataset
            self.data_dir = os.path.join(data_dir, dataset)
            self.csv_name = os.path.join(self.data_dir, csv_name)
            sub_dict = self._create_dictionary(file_dir=self.data_dir,
                                                file_name=dict_file_name)
            if len(self.sub_dict) == 0:
                self.sub_dict = sub_dict
            else:
                self.sub_dict = dict(list(self.sub_dict.items()) + list(sub_dict.items()))
            if self.dataset == "FF++":
                del self.sub_dict['087']
                del self.sub_dict['314']

            csv_file = open(self.csv_name)
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            ref_p_count = []

            for row in tqdm(csv_reader):
                if line_count == 0:
                    print("the title is: ")
                    print(row)
                    # import sys;sys.exit(0)
                    pass
                else:
                    if dataset == 'CelebA':
                        file_p, mask_p, cap_p, ref_p = row[0], row[4], row[-2], row[-1]
                        sub_id = row[10]
                        key_point = row[-1]
                        feat_p = file_p.replace('images', 'feat_file_v2').replace(file_p[-3:], 'pth')
                    elif dataset == 'CASIA':
                        file_p, mask_p, cap_p, ref_p = row[0], row[4], row[-2], row[-4]
                        sub_id = row[10]
                        key_point = row[-1]
                        sub_id = sub_id.zfill(7)
                        feat_p = file_p.replace('images', 'feat_file_v2').replace(file_p[-3:], 'pth')
                    elif dataset == "FF++":
                        file_p = row[0]
                        sub_id = file_p.split('/')[-2]
                        key_point = row[-1]
                        cap_p = "A close-up photo of man."
                        feat_p = file_p.replace(f'/{sub_id}/', f'/{sub_id}_feat_file_v2/').replace(file_p[-3:], 'pth')

                    if sub_id not in self.sub_dict:
                        continue

                    ## caption
                    cap_p = self._retrieve_caption_new(cap_p)
                    if cap_p is None:
                        continue

                    ## facenet feature
                    feat_path = os.path.join(self.data_dir, feat_p)
                    if not os.path.isfile(feat_path):
                        continue

                    feat_numpy = self._retrieve_facenet(file_p, sub_id)
                    if feat_numpy is None:
                        continue

                    self.sub_dict[sub_id].append(data_count)
                    data_count += 1

                    self.feat_list.append(feat_path)
                    self.img_list.append(os.path.join(self.data_dir, file_p))
                    # self.con_list.append(os.path.join(self.data_dir, mask_p))
                    if keep_prompt:
                        # debug:
                        cap_p = cap_p + dataset + sub_id
                        self.cap_list.append(cap_p)
                    else:
                        self.cap_list.append("A close-up photo of * .")
                    self.key_point_list.append(key_point)

                line_count += 1

            csv_file.close()
            print(f"the {dataset_idx} round over: ", len(self.img_list))

        for _ in list(self.sub_dict.keys()):
            if len(self.sub_dict[_]) == 0:
                del self.sub_dict[_]
        self.num_samples = len(self)
        self.sub_dict_lst = list(self.sub_dict.items())  # list of a tuple, (sub_id, lst[access_idx])
        print("===========================================================")
        print(f"The training samples {len(self.img_list)} has been loaded.")
        print(f"We have {len(self.sub_dict_lst)} different individuals.")
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
        return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)

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

        # cv2.imwrite("demo.png", image)
        crop_img = image[y1:y2, x1:x2]
        # print(image.shape, crop_img.shape, bbox)
        if crop_img.shape[0] != 256 or crop_img.shape[1] != 256:
            crop_img = self._resize(crop_img)        
        # cv2.imwrite("demo_crop.png", image)
        return crop_img

    def _retrieve_facenet(self, file_p, sub_id=None):
        """
            obtain the pre-cached facenet feature.
        """
        if self.dataset == "FF++":
            feat_path = file_p.replace(f'/{sub_id}/', f'/{sub_id}_feat_file_v2/').replace(file_p[-3:], 'pth')
        else:
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

    def _retrieve_caption_new(self, cap_p):
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
            cap_p = cap_p.replace('woman', "*")
        elif 'women' in cap_p:
            cap_p = cap_p.replace('women', "*")
        elif 'man' in cap_p:
            cap_p = cap_p.replace('man', "*")
        elif 'men' in cap_p:
            cap_p = cap_p.replace('men', "*")

        cap_p = cap_p.replace('boy', "*")
        cap_p = cap_p.replace('girl', "*")
        cap_p = cap_p.replace('player', "*")
        cap_p = cap_p.replace('officer', "*")
        cap_p = cap_p.replace('solider', "*")
        cap_p = cap_p.replace('bruce lee', "*")

        return cap_p

    def __getitem__(self, batch_idx):

        '''
            this dataset involves the regularization in the RGB domain.
            this regularization is the id-dependence.
            set interval as 10e6 to make the code for the diffusion process with RGB regularization.
        '''
        while True:   # sub_idx_lst may have zero images.
            try:
                sub_idx_lst = self.sub_dict_lst[batch_idx][1]  # list of access idx
                idx_src = sub_idx_lst[random.randint(0,len(sub_idx_lst)-1)]
                idx_tar = sub_idx_lst[random.randint(0,len(sub_idx_lst)-1)]
                # idx_src = random.choice(sub_idx_lst)
                # idx_tar = random.choice(sub_idx_lst)
                break
            except:
                print("the subject has no value is: ", self.sub_dict_lst[batch_idx])
                batch_idx += 1
                continue
        loss_flag = 0
        # print(idx_src, idx_tar)

        # refer_filename = self.ref_list[idx_src]     # self.ref_list should never be used.
        refer_filename = self.img_list[idx_src]     # source image
        refer_feat = self.feat_list[idx_src]
        prompt = self.cap_list[idx_tar]
        # source_filename = self.con_list[idx_tar]    # mask
        target_filename = self.img_list[idx_tar]    # target image
        # img_bbox = self.img_bbox_list[idx_tar]
        img_bbox = [0.22262095928192138, 0.176246280670166, 0.9826209592819214, 0.936246280670166]
        # pseduo_mask = self.con_list[idx_tar]        # pseduo mask
        pseduo_mask = refer_filename

        ## GX: now change the source to head pose.
        pitch, yaw, roll = self.key_point_list[idx_tar][1:-1].split(',')
        pitch, yaw, roll = pitch.strip(), yaw.strip(), roll.strip()
        pitch, yaw, roll = float(pitch[1:-1]), float(yaw[1:-1]), float(roll[1:-1])
        source = self.draw_axis(yaw, pitch, roll)

        target = cv2.imread(target_filename)
        ## change the refer to the pre-cached arcface feature. 
        refer = torch.load(refer_feat)[0]
        # print(refer.shape)
        if refer.shape != (512,):
            print(f"{refer_feat} has wrong shape.")
        refer_img = cv2.imread(refer_filename)

        # resize.
        if target.shape[0] != 256 or target.shape[1] != 256:
            target = self._resize(target)
        # refer = self._crop_img(refer, self.ref_bbox_list[idx]) 
        refer_img = self._resize(refer_img)

        if prompt == "":
            print(self.con_list[idx])
            import sys;sys.exit(0)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        refer_img = cv2.cvtColor(refer_img, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        refer_img = (refer_img.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, refer=refer, 
                    refer_img=refer_img,
                    bbox_t=img_bbox, flag=loss_flag,
                    target_path=target_filename,
                    control_path=pseduo_mask,
                    refer_path=refer_filename)