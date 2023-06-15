import torch.nn as nn
import math
import torch
import torch.nn.functional as F
# from ..loss import acc
import numpy as np
# import Le
from datasets.data_tools import get_vocabulary

class PAN_PP_RecHead_CTC(nn.Module):

    def __init__(self,
                 character = 38,
                 input_dim=512,
                 hidden_dim=128,
                 feature_size=(8, 32)):
        super(PAN_PP_RecHead_CTC, self).__init__()
        
        # LOWERCASE  CHINESE
        if character == 4713:
            voc, char2id, id2char = get_vocabulary('CHINESE', use_ctc=True)
        else:
            voc, char2id, id2char = get_vocabulary('LOWERCASE', use_ctc=True)
        self.voc_size = len(voc)
        self.convs = SeqConvs(hidden_dim, feature_size)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, bidirectional=True)
        self.clf = nn.Linear(hidden_dim * 2, self.voc_size)
        
        # 用来做MaskROI
#         self.conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
#         self.bn = nn.BatchNorm2d(hidden_dim)
#         self.relu = nn.ReLU(inplace=True)
#         self.feature_size = feature_size

        # 解码使用
        self.char2id = char2id
        self.id2char = id2char
        self.blank = self.char2id['PAD']

        self.loss_weight = 0.2

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.half = False
        
    def _upsample(self, x, output_size):
        return F.upsample(x, size=output_size, mode='bilinear')


    def loss(self, preds, targets, rec_masks=None, reduce=True):
        """
        @param  :
                preds N 32, 4714
                targets = N 32
        -------
        @Returns  :
        -------
        """
        # 过滤掉rec mask的结果
        if torch.sum(rec_masks) == 0:
            return {'loss_rec': preds.new_full((1,), 0, dtype=torch.float32),
                    'acc_rec': preds.new_full((1,), 0, dtype=torch.float32)}
        preds = preds[rec_masks==1, :, :]
        targets = targets[rec_masks==1, :]
        preds = preds.permute(1, 0, 2) # 32 N 4714
        target_lengths = (targets != self.blank).long().sum(dim=-1)
        trimmed_targets = [t[:l] for t, l in zip(targets, target_lengths)] # 这里取出了制作label时所有的结果
        targets = torch.cat(trimmed_targets)
        x = F.log_softmax(preds, dim=-1)
        input_lengths = torch.full((x.size(1),), x.size(0), dtype=torch.long)
        loss_rec = F.ctc_loss(
                            x, targets, input_lengths, target_lengths,
                            blank=self.blank, zero_infinity=True
                        ) * self.loss_weight
        # acc_rec = acc(preds, target, reduce=True)
        # losses = {'loss_rec': loss_rec.view(-1), 'acc_rec': torch.tensor(0)}
        losses = {'loss_rec': loss_rec.view(-1), 'acc_rec': loss_rec.new_full((1,), 0, dtype=torch.float32)}
        return losses

    def forward(self, x, target=None):
        # x N 128 8 32
        if x.size(0) == 0:
            return x.new_zeros((x.size(2), 0, self.voc_size))
        x = self.convs(x).squeeze(dim=2) # N C W    2，128,32

        x = x.permute(2, 0, 1)  # WxNxC   32,2,128
        x, _ = self.rnn(x)     #    32,2,256
        preds = self.clf(x)    #    32,2,38

        out_rec = preds.permute(1, 0, 2) # N 32 38   N  W  C
        if self.training:
#             words, word_scores = self.decode(out_rec)
            return out_rec
        else:
            # words, word_scores, out_rec
#             words, word_scores = self.decode(out_rec)
            return out_rec

#     def decode(self, rec): 
#         rec_probs = F.softmax(rec, dim=2)

#         preds_max_prob, out_rec_decoded = rec_probs.max(dim=-1) # N 32
#         words = []
#         word_scores = []
#         num_words = out_rec_decoded.size(0)
#         for l in range(num_words):
#             s = ''
#             c_word_score = 0.0
#             num_chars = 0
#             word_preds_max_prob = preds_max_prob[l]
#             # word_score = word_preds_max_prob.cumprod(dim=0)[-1]
#             t = out_rec_decoded[l] # 32
#             for i in range(len(t)):
#                 if t[i].item() !=  self.blank and (not (i > 0 and t[i - 1].item() == t[i].item())):  # removing repeated characters and blank.
#                     s += self.id2char[t[i].item()]
#                     c_word_score += word_preds_max_prob[i]
#                     num_chars += 1
#             words.append(s)
#             word_scores.append(c_word_score/(num_chars+0.000001))
#         return words, word_scores

    def decode(self, rec): 
#         print(rec.shape)
        rec_probs = F.softmax(rec, dim=2)

        preds_max_prob, out_rec_decoded = rec_probs.max(dim=-1) # N 32
#         print(out_rec_decoded)
        words = []
        word_scores = []
        num_words = out_rec_decoded.size(0)
        for l in range(num_words):
            s = ''
            c_word_score = 0.0
            num_chars = 0
            word_preds_max_prob = preds_max_prob[l]
            # word_score = word_preds_max_prob.cumprod(dim=0)[-1]
            t = out_rec_decoded[l] # 32
            for i in range(len(t)):
                if t[i].item() !=  self.blank:
                    c_word_score += word_preds_max_prob[i]
                    num_chars += 1
                    if (not (i > 0 and t[i - 1].item() == t[i].item())):  # removing repeated characters and blank.
                        s += self.id2char[t[i].item()]
            words.append(s)
            word_scores.append(c_word_score/(num_chars+0.000001))
        return words, word_scores
    
class SeqConvs(nn.Module):
    def __init__(self, conv_dim, roi_size):
        super().__init__()

        height = roi_size[0] # 8 32
        downsample_level = math.log2(height) - 2
        assert math.isclose(downsample_level, int(downsample_level))
        downsample_level = int(downsample_level)

        conv_block = conv_with_kaiming_uniform(
            norm="BN", activation=True)
        convs = []
        for i in range(downsample_level):
            convs.append(conv_block(
                conv_dim, conv_dim, 3, stride=(2, 1)))
        convs.append(nn.Conv2d(conv_dim, conv_dim, kernel_size=(4, 1), bias=False))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)


def conv_with_kaiming_uniform(
        norm=None, activation=None,
        use_deformable=False, use_sep=False):
    
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1):

        conv_func = Conv2d
        groups = 1
        conv = conv_func(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            groups=groups,
            bias=(norm is None)
        )
            
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if norm is None:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if norm is not None and len(norm) > 0:
            norm_module = nn.BatchNorm2d(out_channels)
            module.append(norm_module)
        if activation is not None:
            module.append(nn.ReLU(inplace=False))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv
    return make_conv


class Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):

        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


if __name__ == '__main__':
    import sys
    sys.path.append('/home/wangjue_Cloud/wuweijia/Code/VideoSpotting/PAN_VTS/pan_pp')
    from dataset import dataset_tool
    voc, char2id, id2char = dataset_tool.get_vocabulary('LOWERCASE', use_ctc=True)
    rec = PAN_PP_RecHead_CTC(
                input_dim=512,
                hidden_dim=128,
                voc=voc,
                char2id= char2id,
                id2char = id2char,
                feature_size=(8, 32))
    target = torch.tensor([[1869, 4322, 1889,  119, 1018,  111, 3754, 4711, 4712, 4712, 4712, 4712,
         4712, 4712, 4712, 4712, 4712, 4712, 4712, 4712, 4712, 4712, 4712, 4712,
         4712, 4712, 4712, 4712, 4712, 4712, 4712, 4712], [1869, 4322, 1889,  119, 1018,  111, 3754, 4711, 4712, 4712, 4712, 4712,
         4712, 4712, 4712, 4712, 4712, 4712, 4712, 4712, 4712, 4712, 4712, 4712,
         4712, 4712, 4712, 4712, 4712, 4712, 4712, 4712]])
    # targets = [] # 
    rec.train()
    x = torch.empty((2, 128, 8, 32))
#     print(x.shape)
    y = rec.forward(x)
#     print(y.shape)

#     rec.loss(y, targets=target)
