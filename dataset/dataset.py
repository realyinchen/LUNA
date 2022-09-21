import copy
import csv
import functools
import glob
import math
import os
import random
from collections import namedtuple

import SimpleITK as sitk
import numpy as np
import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.coord_util import XyzTuple, xyz2irc
from utils.disk_util import getCache
from utils.log_util import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('CT')

CandidateInfoTuple = namedtuple('CandidateInfoTuple',
                                'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz')


@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all subsets yet.
    mhd_list = glob.glob('D:\\Data\\luna\\subset*\\*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    candidateInfo_list = []
    with open('D:\\Data\\luna\\annotations_with_malignancy.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            isMal_bool = {'False': False, 'True': True}[row[5]]

            candidateInfo_list.append(
                CandidateInfoTuple(
                    True,  # isNodule_bool
                    True,  # hasAnnotation_bool
                    isMal_bool,
                    annotationDiameter_mm,
                    series_uid,
                    annotationCenter_xyz)
            )

    with open('D:\\Data\\luna\\candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            if not isNodule_bool:
                candidateInfo_list.append(
                    CandidateInfoTuple(
                        False,  # isNodule_bool
                        False,  # hasAnnotation_bool
                        False,  # isMal_bool
                        0.0,
                        series_uid,
                        candidateCenter_xyz)
                )

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


@functools.lru_cache(1)
def getCandidateInfoDict(requireOnDisk_bool=True):
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid, []).append(candidateInfo_tup)

    return candidateInfo_dict


class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob('D:\\Data\\luna\\subset*\\{}.mhd'.format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        candidateInfo_list = getCandidateInfoDict()[self.series_uid]

        self.positiveInfo_list = [candidate_tup for candidate_tup in candidateInfo_list if candidate_tup.isNodule_bool]
        self.positive_mask = self.buildAnnotationMask(self.positiveInfo_list)
        self.positive_indexes = (self.positive_mask.sum(axis=(1, 2)).nonzero()[0].tolist())

    def buildAnnotationMask(self, positiveInfo_list, threshold_hu=-700):
        boundingBox_a = np.zeros_like(self.hu_a, dtype=np.bool)

        for candidateInfo_tup in positiveInfo_list:
            center_irc = xyz2irc(candidateInfo_tup.center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:
                while self.hu_a[ci + index_radius, cr, cc] > threshold_hu and \
                        self.hu_a[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and \
                        self.hu_a[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and \
                        self.hu_a[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            # assert index_radius > 0, repr([candidateInfo_tup.center_xyz, center_irc, self.hu_a[ci, cr, cc]])
            # assert row_radius > 0
            # assert col_radius > 0

            boundingBox_a[
            ci - index_radius: ci + index_radius + 1,
            cr - row_radius: cr + row_radius + 1,
            cc - col_radius: cc + col_radius + 1] = True

        mask_a = boundingBox_a & (self.hu_a > threshold_hu)

        return mask_a

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert 0 <= center_val < self.hu_a.shape[axis], repr(
                [self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]
        pos_chunk = self.positive_mask[tuple(slice_list)]

        return ct_chunk, pos_chunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, pos_chunk, center_irc


@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    ct = Ct(series_uid)
    return int(ct.hu_a.shape[0]), ct.positive_indexes


class Luna2dSegmentationDataset(Dataset):
    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None, contextSlices_count=3, fullCt_bool=False):
        self.contextSlices_count = contextSlices_count
        self.fullCt_bool = fullCt_bool

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(getCandidateInfoDict().keys())

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indexes = getCtSampleSize(series_uid)

            if self.fullCt_bool:
                self.sample_list += [(series_uid, slice_ndx) for slice_ndx in range(index_count)]
            else:
                self.sample_list += [(series_uid, slice_ndx) for slice_ndx in positive_indexes]

        self.candidateInfo_list = getCandidateInfoList()

        series_set = set(self.series_list)
        self.candidateInfo_list = [cit for cit in self.candidateInfo_list if cit.series_uid in series_set]

        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        log.info("{!r}: {} {} series, {} slices, {} nodules".format(
            self,
            len(self.series_list),
            {None: 'general', True: 'validation', False: 'training'}[isValSet_bool],
            len(self.sample_list),
            len(self.pos_list),
        ))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]
        return self.getitem_fullSlice(series_uid, slice_ndx)

    def getitem_fullSlice(self, series_uid, slice_ndx):
        ct = getCt(series_uid)
        ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512))

        start_ndx = slice_ndx - self.contextSlices_count
        end_ndx = slice_ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

        return ct_t, pos_t, ct.series_uid, slice_ndx


class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 2

    def __len__(self):
        return 300000

    def shuffleSamples(self):
        random.shuffle(self.candidateInfo_list)
        random.shuffle(self.pos_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.pos_list[ndx % len(self.pos_list)]
        return self.getitem_trainingCrop(candidateInfo_tup)

    def getitem_trainingCrop(self, candidateInfo_tup):
        ct_a, pos_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid, candidateInfo_tup.center_xyz, (7, 96, 96))
        pos_a = pos_a[3:4]

        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)
        ct_t = torch.from_numpy(ct_a[:, row_offset:row_offset+64, col_offset:col_offset+64]).to(torch.float32)
        pos_t = torch.from_numpy(pos_a[:, row_offset:row_offset+64, col_offset:col_offset+64]).to(torch.long)

        slice_ndx = center_irc.index

        return ct_t, pos_t, candidateInfo_tup.series_uid, slice_ndx


def getCtAugmentedCandidate(augmentation_dict, series_uid, center_xyz, width_irc, use_cache=True):
    if use_cache:
        ct_chunk, center_irc = getCtRawCandidate(series_uid, center_xyz, width_irc)
    else:
        ct = getCt(series_uid)
        ct_chunk, _, center_irc = ct.getRawCandidate(center_xyz, width_irc)

    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    transform_t = torch.eye(4)

    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i, i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i, 3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i, i] *= 1.0 + scale_float * random_float

    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        transform_t @= rotation_t

    affine_t = F.affine_grid(transform_t[:3].unsqueeze(0).to(torch.float32), ct_t.size(), align_corners=False)

    augmented_chunk = F.grid_sample(ct_t, affine_t, padding_mode='border', align_corners=False).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']
        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None, sortby_str='random', ratio_int=0,
                 augmentation_dict=None, candidateInfo_list=None):
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        if candidateInfo_list:
            self.candidateInfo_list = copy.copy(candidateInfo_list)
            self.use_cache = False
        else:
            self.candidateInfo_list = copy.copy(getCandidateInfoList())
            self.use_cache = True

        if series_uid:
            self.candidateInfo_list = [x for x in self.candidateInfo_list if x.series_uid == series_uid]

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.negative_list = [nt for nt in self.candidateInfo_list if not nt.isNodule_bool]
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        log.info("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
            len(self.negative_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.pos_list)

    def __len__(self):
        if self.ratio_int:
            return 200000
        else:
            return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            pos_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.negative_list)
                candidateInfo_tup = self.negative_list[neg_ndx]
            else:
                pos_ndx %= len(self.pos_list)
                candidateInfo_tup = self.pos_list[pos_ndx]
        else:
            candidateInfo_tup = self.candidateInfo_list[ndx]

        width_irc = (32, 48, 48)

        if self.augmentation_dict:
            candidate_t, center_irc = getCtAugmentedCandidate(
                self.augmentation_dict,
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
                self.use_cache)
        elif self.use_cache:
            candidate_a, center_irc = getCtRawCandidate(
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc)
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            ct = getCt(candidateInfo_tup.series_uid)
            candidate_a, _, center_irc = ct.getRawCandidate(candidateInfo_tup.center_xyz, width_irc)
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([not candidateInfo_tup.isNodule_bool, candidateInfo_tup.isNodule_bool], dtype=torch.long)

        return candidate_t, pos_t, candidateInfo_tup.series_uid, torch.tensor(center_irc)
