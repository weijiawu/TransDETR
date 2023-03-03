# ------------------------------------------------------------------------
# Copyright (c) 2021 Zhejiang University-model. All Rights Reserved.
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
MOT dataset which returns image_id for evaluation.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
import datasets.transforms as T
from models.structures import Instances
import os
from datasets.data_tools import get_vocabulary
import mmcv
import math
from PIL import Image, ImageDraw, ImageFont
from util.box_ops import box_cxcywh_to_xyxy


def cv2AddChineseText(image, text, position, textColor=(0, 0, 0), textSize=30):
    
    
    x1,y1 = position
    x2,y2 = len(text)* textSize/2 + x1, y1 + textSize
    
    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
    mask_1 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask_1, [points], 1)

    image,rgb = mask_image(image, mask_1,[255,255,255])
    
    
    if (isinstance(image, np.ndarray)):
        image = Image.fromarray(cv2.cvtColor(np.uint8(image.astype(np.float32)), cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    fontStyle = ImageFont.truetype(
        "./tools/simsun.ttc", textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)
    
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                    
    return image

def mask_image(image, mask_2d, rgb=None, valid = False):
    h, w = mask_2d.shape

    # mask_3d = np.ones((h, w), dtype="uint8") * 255
    mask_3d_color = np.zeros((h, w, 3), dtype="uint8")
    # mask_3d[mask_2d[:, :] == 1] = 0
    
        
    image.astype("uint8")
    mask = (mask_2d!=0).astype(bool)
    if rgb is None:
        rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        
    mask_3d_color[mask_2d[:, :] == 1] = rgb
    image[mask] = image[mask] * 0.5 + mask_3d_color[mask] * 0.5
    
    if valid:
        mask_3d_color[mask_2d[:, :] == 1] = [[0,0,0]]
        kernel = np.ones((5,5),np.uint8)
        mask_2d = cv2.dilate(mask_2d,kernel,iterations = 4)
        mask = (mask_2d!=0).astype(bool)
        image[mask] = image[mask] * 0 + mask_3d_color[mask] * 1
        return image,rgb
        
    return image,rgb


def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


class DetMOTDetection:
    def __init__(self, args, data_txt_path: str, seqs_folder, dataset2transform):
        self.args = args
        self.dataset2transform = dataset2transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.vis = args.vis
        self.video_dict = {}
        
        with open(data_txt_path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [osp.join(seqs_folder, x.strip()) for x in self.img_files]
            
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))
        self.img_files = self.img_files[:int(len(self.img_files))]
#         print(data_txt_path)
        if "BOVText" in data_txt_path:
            self.label_files = [(("/share/wuweijia/Data/VideoText/MOTR/BOVText/labels_with_ids/train" + x.split("/Frames")[1]).replace('.png', '.txt').replace('.jpg', '.txt'))
                            for x in self.img_files]
        if "VideoSynthText" in data_txt_path:
            self.label_files = [(x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt'))
                            for x in self.img_files]
        elif "SynthText" in data_txt_path:
            self.label_files = [(("/mmu-ocr/weijiawu/Data/VideoText/MOTR/SynthText/labels_with_ids/train" + x.split("/SynthText")[1]).replace('.png', '.txt').replace('.jpg', '.txt')) if len(x.split("/SynthText"))>1 else (x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt'))
                            for x in self.img_files]
        else:
            self.label_files = [(x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt'))
                            for x in self.img_files]
        
        # The number of images per sample: 1 + (num_frames - 1) * interval.
        # The number of valid samples: num_images - num_image_per_sample + 1.
        
        self.item_num = len(self.img_files) - (self.num_frames_per_batch - 1) * self.sample_interval

        self._register_videos()

        # video sampler.
        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))
        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            # Enable sampling length adjustment.
            assert len(self.lengths) > 0
            assert len(self.lengths) == len(self.sampler_steps) + 1
            for i in range(len(self.sampler_steps) - 1):
                assert self.sampler_steps[i] < self.sampler_steps[i + 1]
            self.item_num = len(self.img_files) - (self.lengths[-1] - 1) * self.sample_interval
            self.period_idx = 0
            self.num_frames_per_batch = self.lengths[0]
            self.current_epoch = 0
        
        # recognition  CHINESE LOWERCASE
        self.use_ctc = True
        self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE', use_ctc=self.use_ctc)
        self.max_word_num = 50
        self.max_word_len = 32
        
        
        self.vis = False
        
    def _register_videos(self):
        for label_name in self.label_files:
            video_name = '/'.join(label_name.split('/')[:-1])
            if video_name not in self.video_dict:
                print("register {}-th video: {} ".format(len(self.video_dict) + 1, video_name))
                self.video_dict[video_name] = len(self.video_dict)
                # assert len(self.video_dict) <= 300

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels']
        gt_instances.rotate = targets['rotate']
        gt_instances.texts_ignored = targets['texts_ignored']
        gt_instances.word = targets['word']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.area = targets['area']
        return gt_instances

    def _pre_single_frame(self, idx: int):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]

        img = Image.open(img_path).convert("RGB")
        targets = {}
        w, h = img._size
        assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)
        if osp.isfile(label_path):
            size = os.path.getsize(label_path)

            if size == 0:
                labels = np.array([])
            else:
#                 labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 7)
                labels0 = []
                words = []
                lines = mmcv.list_from_file(label_path)
                
                bboxes = []
                texts = []
                texts_ignored = []
                for i,line in enumerate(lines):
                    if i> self.max_word_num:
                        break
                    text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(' ')

                    # recognition words
                    if "#" in text[-1]:
#                         continue
                        texts_ignored.append(0)
                    else:
                        texts_ignored.append(1)
                    try:
                        labels0.append(list(map(float, text[:11])))
                    except:
                        print(text)

                    word = text[-1]
                    word = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])","",word.upper())
                    word = word.upper()
                    gt_word = np.full((self.max_word_len,), self.char2id['PAD'], dtype=np.int)
                    for j, char in enumerate(word):
                        if j > self.max_word_len - 1:
                            break
                        if char in self.char2id:
                            gt_word[j] = self.char2id[char]
                        else:
                            gt_word[j] = self.char2id['UNK']
                    if not self.use_ctc:
                        if len(word) > self.max_word_len - 1:
                            gt_word[-1] = self.char2id['EOS']
                        else:
                            gt_word[len(word)] = self.char2id['EOS']
                    words.append(gt_word)
                 
                if len(labels0)==0:
                    labels = np.array([])
                else:
                    # normalized cewh to pixel xyxy format
                    labels0 = np.array(labels0)
                    labels = np.array(labels0).copy()
                    labels[:, 2] = w * (labels0[:, 2] - labels0[:, 4] / 2)
                    labels[:, 3] = h * (labels0[:, 3] - labels0[:, 5] / 2)
                    labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4] / 2)
                    labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5] / 2)
        else:
            raise ValueError('invalid label path: {}'.format(label_path))
        
        
                
        video_name = '/'.join(label_path.split('/')[:-1])
        obj_idx_offset = self.video_dict[video_name] * 1000000  # 1000000 unique ids is enough for a video.
        if 'crowdhuman' in img_path:
            targets['dataset'] = 'CrowdHuman'
        elif 'MOT17' in img_path:
            targets['dataset'] = 'MOT17'
        elif 'ICDAR2015' in img_path:
            targets['dataset'] = 'ICDAR2015'
        elif 'ICDAR15' in img_path:
            targets['dataset'] = 'ICDAR15_Pre'
        elif 'COCOTextV2' in img_path:
            targets['dataset'] = 'COCOTextV2'
        elif 'VideoSynthText' in img_path:
            targets['dataset'] = 'VideoSynthText'
        elif 'FlowTextV2' in img_path:
            targets['dataset'] = 'FlowTextV2'
        elif 'FlowText' in img_path:
            targets['dataset'] = 'FlowText'
        elif 'SynthText' in img_path:
            targets['dataset'] = 'SynthText'
        elif 'VISD' in img_path:
            targets['dataset'] = 'VISD'
        elif 'YVT' in img_path:
            targets['dataset'] = 'YVT'
        elif 'UnrealText' in img_path:
            targets['dataset'] = 'UnrealText'
        elif 'BOVTextV2' in img_path:
            targets['dataset'] = 'BOVText'
        elif 'FlowImage' in img_path:
            targets['dataset'] = 'FlowImage'
        elif 'DSText' in img_path:
            targets['dataset'] = 'DSText'
        else:
            raise NotImplementedError()
        targets['boxes'] = []
        targets['rotate'] = []
        targets['area'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['upright_box'] = []
        targets['word'] = []
        targets['texts_ignored'] = []
        targets['obj_ids'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        
        for i,label in enumerate(labels):
            targets['rotate'].append(label[6])
            targets['boxes'].append(label[2:6].tolist())
            targets['area'].append(label[4] * label[5])
            targets['upright_box'].append(label[7:].tolist())
            targets['word'].append(words[i])
            targets['texts_ignored'].append(texts_ignored[i])
            targets['iscrowd'].append(0)
            targets['labels'].append(0)

            obj_id = label[1] + obj_idx_offset if label[1] >= 0 else label[1]
            targets['obj_ids'].append(obj_id)  # relative id

        targets['area'] = torch.as_tensor(targets['area'])
        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'], dtype=torch.int64)
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'])
        targets['word'] = torch.as_tensor(targets['word'], dtype=torch.long)
        targets['texts_ignored'] = torch.as_tensor(targets['texts_ignored'], dtype=torch.long)
        targets['rotate'] = torch.as_tensor(targets['rotate'], dtype=torch.float32)
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        targets['boxes'][:, 0::2].clamp_(min=0, max=w)
        targets['boxes'][:, 1::2].clamp_(min=0, max=h)
        
        targets['upright_box'] = torch.as_tensor(targets['upright_box'], dtype=torch.float32).reshape(-1, 4)
        targets['upright_box'][:, 0::2].clamp_(min=0, max=w)
        targets['upright_box'][:, 1::2].clamp_(min=0, max=h)
        return img, targets

    def _get_sample_range(self, start_idx):

        # take default sampling method for normal dataset.
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1, sample_interval
#         print(self.num_frames_per_batch)
        return default_range

    def pre_continuous_frames(self, start, end, interval=1):
        targets = []
        images = []
        for i in range(start, end, interval):
            img_i, targets_i = self._pre_single_frame(i)
            images.append(img_i)
            targets.append(targets_i)
        return images, targets

    def __getitem__(self, idx):
        sample_start, sample_end, sample_interval = self._get_sample_range(idx)
        
        images, targets = self.pre_continuous_frames(sample_start, sample_end, sample_interval)
        
        data = {}
        dataset_name = targets[0]['dataset']
        transform = self.dataset2transform[dataset_name]
        
        
        if transform is not None:
            images, targets = transform(images, targets)
        
        if self.vis:
            import random
            image_icdxx = random.randint(1,100)
            for idx1,(img,ann) in enumerate(zip(images,targets)):
                imge = img.permute(1,2,0)
#                 print(imge.shape)
               
                imge = (imge.cpu().numpy()*[0.229, 0.224, 0.225]+[0.485, 0.456, 0.406])*255
                h, w = imge.shape[:2]
                label = ann["labels"]
                area = ann["area"]
                iscrowd = ann["iscrowd"]
                rotates = ann["rotate"]
                obj_ids = ann["obj_ids"]
                words = ann["word"]
                texts_ignoreds = ann["texts_ignored"]
                boxes = ann["boxes"] * torch.tensor([w, h, w, h], dtype=torch.float32)
                image = imge.copy()
                boxes = box_cxcywh_to_xyxy(boxes)
                
                for obj_id ,box,rotate,word,texts_ignored in zip(obj_ids,boxes,rotates,words,texts_ignoreds):
                    x_min,y_min,x_max,y_max = box.cpu().numpy()
#                     print(x_min,y_min,x_max,y_max)
                    rotate_mat = get_rotate_mat(-rotate)
                    temp_x = np.array([[x_min, x_max, x_max, x_min]]) - (x_min+x_max)/2
                    temp_y = np.array([[y_min, y_min, y_max, y_max]]) - (y_min+y_max)/2
                    coordidates = np.concatenate((temp_x, temp_y), axis=0)
                    res = np.dot(rotate_mat, coordidates)
                    res[0,:] += (x_min+x_max)/2
                    res[1,:] += (y_min+y_max)/2
                    points = np.array([[res[0,0], res[1,0]], [res[0,1], res[1,1]], [res[0,2], res[1,2]], [res[0,3], res[1,3]]], np.int32)
                    if texts_ignored==0:
                        cv2.polylines(image, [points], True, (0,0,255), thickness=4)
                    else:
                        cv2.polylines(image, [points], True, (0,255,255), thickness=4)
                        
                    short_side = min(imge.shape[0],imge.shape[1])
                    text_size = int(short_side * 0.05)
#                     print(texts_ignored)
#                     image=cv2AddChineseText(image,str(texts_ignored.cpu().numpy()), (int(res[0,0]), int(res[1,0]) - text_size),(0,255,255), text_size)
#                     image=cv2AddChineseText(image,str(obj_id.cpu().numpy())+str(texts_ignored.cpu().numpy()), (int(res[0,0]), int(res[1,0]) - text_size),(0,255,255), text_size)

                cv2.imwrite("./exps/show/{}_{}.jpg".format(image_icdxx,idx1),image)
#                 cv2.imwrite("./exps/show/original_{}_{}.jpg".format(image_icdxx,idx1),imge)


        gt_instances = []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
        data.update({
            'imgs': images,
            'gt_instances': gt_instances,
        })
        if self.args.vis:
            data['ori_img'] = [target_i['ori_img'] for target_i in targets]
        return data

    def __len__(self):
        return self.item_num


class DetMOTDetectionValidation(DetMOTDetection):
    def __init__(self, args, seqs_folder, dataset2transform):
        args.data_txt_path = args.val_data_txt_path
        super().__init__(args, seqs_folder, dataset2transform)



def make_transforms_for_mot17(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
#             T.MotRandomHorizontalFlip(),
#             T.MotRandomResize(scales, max_size=1536),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([400, 500, 600]),
                    T.FixedMotRandomCrop(384, 600),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1536),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def make_transforms_for_BOVText(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
#             T.MotRandomHorizontalFlip(),
            T.MotRandomResize(scales, max_size=1536),
#             T.MotRandomSelect(
#                 T.MotRandomResize(scales, max_size=1536),
#                 T.MotCompose([
#                     T.MotRandomResize([400, 500, 600]),
#                     T.FixedMotRandomCrop(384, 600),
#                     T.MotRandomResize(scales, max_size=1536),
#                 ])
#             ),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1536),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def make_transforms_for_crowdhuman(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]
#     scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896]
    if image_set == 'train':
        return T.MotCompose([
#             T.MotRandomHorizontalFlip(),
            T.FixedMotRandomShift(bs=1),
#             T.MotRandomResize(scales, max_size=1536),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([400, 500, 600]),
                    T.FixedMotRandomCrop(384, 600),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            normalize,

        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_dataset2transform(args, image_set):
    mot17_train = make_transforms_for_mot17('train', args)
    mot17_test = make_transforms_for_mot17('val', args)
    BOVText_train = make_transforms_for_BOVText('train', args)

    crowdhuman_train = make_transforms_for_crowdhuman('train', args)
    dataset2transform_train = {'ICDAR2015': mot17_train,'DSText': mot17_train,  'VideoSynthText':mot17_train, 'FlowText':mot17_train, 'FlowTextV2':mot17_train,
                       'YVT': mot17_train,
                       'BOVText': BOVText_train, 'SynthText': crowdhuman_train, 'UnrealText': crowdhuman_train,
                       'COCOTextV2': crowdhuman_train, 'VISD': crowdhuman_train, 'FlowImage': crowdhuman_train,
                        "ICDAR15_Pre": crowdhuman_train}
    
    dataset2transform_val = {'ICDAR2015': mot17_test,'DSText': mot17_test, 'VideoSynthText':mot17_test, 'FlowText':mot17_test, 'YVT': mot17_test,
                     'FlowTextV2':mot17_test,
                     'BOVText': mot17_test, 'SynthText': mot17_test, 'UnrealText': mot17_test,
                     'COCOTextV2': mot17_test,'VISD': mot17_test,'FlowImage': mot17_test,"ICDAR15_Pre": crowdhuman_train}
    if image_set == 'train':
        return dataset2transform_train
    elif image_set == 'val':
        return dataset2transform_val
    else:
        raise NotImplementedError()


def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    dataset2transform = build_dataset2transform(args, image_set)
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform)
    if image_set == 'val':
        data_txt_path = args.data_txt_path_val
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script')
    args = parser.parse_args()
    
    data_txt_path = ""
    root = ""
    dataset2transform = ""
    dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform)

    
