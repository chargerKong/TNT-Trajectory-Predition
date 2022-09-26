# About: script to processing argoverse forecasting dataset
# Author: Jianbang LIU @ RPAI, CUHK
# Date: 2021.07.16

import os
import argparse
from os.path import join as pjoin
import copy
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import sparse
import sys
sys.path.append("./")
import warnings

# import torchprint
from torch.utils.data import Dataset, DataLoader

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.utils.mpl_plotting_utils import visualize_centerline

from core.util.preprocessor.base import Preprocessor
from core.util.cubic_spline import Spline2D

warnings.filterwarnings("ignore")

RESCALE_LENGTH = 1.0    # the rescale length th turn the lane vector into equal distance pieces


class ArgoversePreprocessor(Preprocessor):
    def __init__(self,
                 root_dir,
                 split="train",
                 algo="tnt",
                 obs_horizon=20,
                 obs_range=100,
                 pred_horizon=30,
                 normalized=True,
                 save_dir=None):
        super(ArgoversePreprocessor, self).__init__(root_dir, algo, obs_horizon, obs_range, pred_horizon)

        self.LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
        self.COLOR_DICT = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}

        self.split = split
        self.normalized = normalized

        self.am = ArgoverseMap()
        self.loader = ArgoverseForecastingLoader(pjoin(self.root_dir, self.split+"_obs" if split == "test" else split))

        self.save_dir = save_dir

    def __getitem__(self, idx):
        f_path = self.loader.seq_list[idx]
        seq = self.loader.get(f_path)
        path, seq_f_name_ext = os.path.split(f_path)
        # 2.csv => ("2", ".csv")
        seq_f_name, ext = os.path.splitext(seq_f_name_ext)
        # 获得文件的具体数据，pd.DataFrame
        df = copy.deepcopy(seq.seq_df)
        return self.process_and_save(df, seq_id=seq_f_name, dir_=self.save_dir)

    def process(self, dataframe: pd.DataFrame,  seq_id, map_feat=True):
        data = self.read_argo_data(dataframe)
        data = self.get_obj_feats(data)

        data['graph'] = self.get_lane_graph(data)
        data['seq_id'] = seq_id
        # visualization for debug purpose
        # self.visualize_data(data)
        return pd.DataFrame(
            [[data[key] for key in data.keys()]],
            columns=[key for key in data.keys()]
        )

    def __len__(self):
        return len(self.loader)

    @staticmethod
    def read_argo_data(df: pd.DataFrame):
        """
        return: dict, key有 data['trajs'] 和 data['steps']
        data['trajs']: List[List[int]]. 第一个List是AGENT下的XY数据.第二个往后是其他trackingID的数据
        data['steps']: List[List[int]]. 第一个List是对应AGENT的数据 所在的时间戳序号。第二个往后是其他rackingID的数据的时间戳序号
                       表示该ID 追踪的时间，时长一共五秒，分为50个序号。
                       第一个AGENT跟踪时长为5秒，所以steps从0-49都有
        """
        city = df["CITY_NAME"].values[0]

        """TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME"""
        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
        mapping = dict()
        # 把所有的时间做序号，0，1，2，3，4，...
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), 1)
        # 给出原数据时间戳对应的时间序号
        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)
        # 按照追踪物体的类别分类，并且提取出物体类别
        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        # keys = [('00000000-0000-0000-0000-000000000000', 'AV'),('00000000-0000-0000-0000-000000019203', 'OTHERS')...]
        # 获取所有追踪物体的ID
        keys = list(objs.keys())
        # 获取相应追踪物体ID的类别
        obj_type = [x[1] for x in keys]
        # 找到类别的AGENT的序号
        agt_idx = obj_type.index('AGENT')
        # 找到类别为AGENT下的数据序号
        # [ 6,   18,  28,  40,  55,  66,  78,  90,  103, 115, 128, 140, 151,
        #   163, 175, 183, 195, 207, 219, 230, 238, 253, 263, 268, 286, 299,
        #   312, 328, 343, 359, 372, 387, 402, 410, 424, 438, 457, 471, 483,
        #   495, 509, 522, 535, 549, 566, 580, 596, 612, 628, 644]
        idcs = objs[keys[agt_idx]]
        # 提取出类别为AGENT下的数据
        agt_traj = trajs[idcs]
        # 提取出对应AGENT的数据，它所在的时间戳序号【0，1，2，3，4...】
        agt_step = steps[idcs]

        del keys[agt_idx]
        ctx_trajs, ctx_steps = [], []
        for key in keys:
            idcs = objs[key]
            # 找到其他某一类下的数据
            ctx_trajs.append(trajs[idcs])
            # 找到该类下的数据所对应的时间戳
            ctx_steps.append(steps[idcs])

        data = dict()
        data['city'] = city
        data['trajs'] = [agt_traj] + ctx_trajs
        data['steps'] = [agt_step] + ctx_steps
        return data

    def get_obj_feats(self, data):
        """
        return: data: 包括中心线，agent的轨迹
        """
        # get the origin and compute the oritentation of the target agent
        orig = data['trajs'][0][self.obs_horizon-1].copy().astype(np.float32)

        # comput the rotation matrix
        if self.normalized:
            pre, conf = self.am.get_lane_direction(data['trajs'][0][self.obs_horizon-1], data['city'])
            if conf <= 0.1:
                pre = (orig - data['trajs'][0][self.obs_horizon-4]) / 2.0
            theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2
            rot = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]], np.float32)
        else:
            # if not normalized, do not rotate.
            theta = None
            rot = np.asarray([
                [1.0, 0.0],
                [0.0, 1.0]], np.float32)

        # get the target candidates and candidate gt
        agt_traj_obs = data['trajs'][0][0: self.obs_horizon].copy().astype(np.float32)
        agt_traj_fut = data['trajs'][0][self.obs_horizon:self.obs_horizon+self.pred_horizon].copy().astype(np.float32)
<<<<<<< HEAD
        # 给出可能的中心线
        # 多条中心线，一条s to e 对应一个np.array
        ctr_line_candts = self.am.get_candidate_centerlines_for_traj(agt_traj_obs, data['city'])
=======
        ctr_line_candts = self.am.get_candidate_centerlines_for_traj(agt_traj_obs, data['city'], viz=False)
>>>>>>> 9180637f81304c3d9aaf28ea88e572d779618e36

        # rotate the center lines and find the reference center line
        agt_traj_fut = np.matmul(rot, (agt_traj_fut - orig.reshape(-1, 2)).T).T
        for i, _ in enumerate(ctr_line_candts):
            # 减去原点 做旋转
            ctr_line_candts[i] = np.matmul(rot, (ctr_line_candts[i] - orig.reshape(-1, 2)).T).T
<<<<<<< HEAD
        # 把多条中心线(多个list)进行合并（合并为一个list），并下采样
        tar_candts = self.lane_candidate_sampling(ctr_line_candts, viz=False)
        # ？？
=======

        tar_candts = self.lane_candidate_sampling(ctr_line_candts, [0, 0], viz=False)

>>>>>>> 9180637f81304c3d9aaf28ea88e572d779618e36
        if self.split == "test":
            tar_candts_gt, tar_offse_gt = np.zeros((tar_candts.shape[0], 1)), np.zeros((1, 2))
            splines, ref_idx = None, None
        else:
            splines, ref_idx = self.get_ref_centerline(ctr_line_candts, agt_traj_fut)
            tar_candts_gt, tar_offse_gt = self.get_candidate_gt(tar_candts, agt_traj_fut[-1])

        # self.plot_target_candidates(ctr_line_candts, agt_traj_obs, agt_traj_fut, tar_candts)
        # if not np.all(offse_gt < self.LANE_WIDTH[data['city']]):
        #     self.plot_target_candidates(ctr_line_candts, agt_traj_obs, agt_traj_fut, tar_candts)

        feats, ctrs, has_obss, gt_preds, has_preds = [], [], [], [], []
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        for traj, step in zip(data['trajs'], data['steps']):
            if self.obs_horizon-1 not in step:
                continue

            # normalize and rotate
            traj_nd = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T

            # collect the future prediction ground truth
            gt_pred = np.zeros((self.pred_horizon, 2), np.float32)
            has_pred = np.zeros(self.pred_horizon, np.bool)
            # 假如后3秒的数据有的话，把后3秒的数据，即后边30个index找到
            # step 为0 到 49之间的任意连续的序列，因为物体出现的时间不定
            # 找到step 在20-49之间的index, 设置为true
            # 比如是一段20 - 35之间为true的mask
            future_mask = np.logical_and(step >= self.obs_horizon, step < self.obs_horizon + self.pred_horizon)
            # post_step = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            # 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
            # 找到step 在20-49之间的index - 20
            post_step = step[future_mask] - self.obs_horizon
            # 截取出预测时间的轨迹XY，从原轨迹的第二十个开始取
            post_traj = traj_nd[future_mask]
            # 记录真实值的轨迹{点0，点1，...}
            gt_pred[post_step] = post_traj
            # array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
            # True,  True,  True,  True,  True,  True,  True ...
            # 表示预测部分有多少个时间是有数据点的
            # 比如在20之后有10个数据，has_pred 就有10个true
            # 可以方便后续使用gt_pred[has_pred]取出数据
            has_pred[post_step] = True

            # colect the observation
            obs_mask = step < self.obs_horizon
            step_obs = step[obs_mask]
            traj_obs = traj_nd[obs_mask]
            idcs = step_obs.argsort()
            step_obs = step_obs[idcs]
            traj_obs = traj_obs[idcs]

            for i in range(len(step_obs)):
                if step_obs[i] == self.obs_horizon - len(step_obs) + i:
                    break
            step_obs = step_obs[i:]
            traj_obs = traj_obs[i:]

            if len(step_obs) <= 1:
                continue
            # obs_horizon行3列
            feat = np.zeros((self.obs_horizon, 3), np.float32)
            # obs_horizon个false，即代表obs_horizon个时间戳下的观察数据
            has_obs = np.zeros(self.obs_horizon, np.bool)
            # feat 前两列为观察的轨迹
            feat[step_obs, :2] = traj_obs
            # feat 第三列为1.0
            feat[step_obs, 2] = 1.0
            has_obs[step_obs] = True
            # 判断最后一个点的x和y是不是超出了观测范围
            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue
            # 观察的轨迹XY
            feats.append(feat)                  # displacement vectors
            has_obss.append(has_obs)
            # 真实值的轨迹XY
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        # if len(feats) < 1:
        #     raise Exception()

        feats = np.asarray(feats, np.float32)
        has_obss = np.asarray(has_obss, np.bool)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool)

        # plot the splines
        # self.plot_reference_centerlines(ctr_line_candts, splines, feats[0], gt_preds[0], ref_idx)

        # # target candidate filtering
        # tar_candts = np.matmul(rot, (tar_candts - orig.reshape(-1, 2)).T).T
        # inlier = np.logical_and(np.fabs(tar_candts[:, 0]) <= self.obs_range, np.fabs(tar_candts[:, 1]) <= self.obs_range)
        # if not np.any(candts_gt[inlier]):
        #     raise Exception("The gt of target candidate exceeds the observation range!")

        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot

        data['feats'] = feats
        data['has_obss'] = has_obss

        data['has_preds'] = has_preds
        data['gt_preds'] = gt_preds
        data['tar_candts'] = tar_candts
        data['gt_candts'] = tar_candts_gt
        data['gt_tar_offset'] = tar_offse_gt

        data['ref_ctr_lines'] = splines         # the reference candidate centerlines Spline for prediction
        data['ref_cetr_idx'] = ref_idx          # the idx of the closest reference centerlines
        return data

    def get_lane_graph(self, data):
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        # 返回以orig为原点，曼哈顿距离半径为radius范围内的所有lane id
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius * 1.5)
        lane_ids = copy.deepcopy(lane_ids)

        lanes = dict()
        # 把对应lane_id的lane加入到lanes中
        # 每一个lane包括中心线，多边形等
        for lane_id in lane_ids:
            # 找该lane的车道中心线
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)

            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane

        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1

            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))

            x = np.zeros((num_segs, 2), np.float32)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))

        lane_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            lane_idcs.append(i * np.ones(len(ctr), np.int64))
            count += len(ctr)
        num_nodes = count
        lane_idcs = np.concatenate(lane_idcs, 0)

        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['lane_idcs'] = lane_idcs

        return graph

    def visualize_data(self, data):
        """
        visualize the extracted data, and exam the data
        """
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        # visualize the centerlines
        lines_ctrs = data['graph']['ctrs']
        lines_feats = data['graph']['feats']
        lane_idcs = data['graph']['lane_idcs']
        for i in np.unique(lane_idcs):
            line_ctr = lines_ctrs[lane_idcs == i]
            line_feat = lines_feats[lane_idcs == i]
            # confirm two centerline connected
            line_str = (2.0 * line_ctr - line_feat) / 2.0
            line_end = (2.0 * line_ctr[-1, :] + line_feat[-1, :]) / 2.0
            line = np.vstack([line_str, line_end.reshape(-1, 2)])
            visualize_centerline(line)

        # visualize the trajectory
        trajs = data['feats'][:, :, :2]
        has_obss = data['has_obss']
        preds = data['gt_preds']
        has_preds = data['has_preds']
        for i, [traj, has_obs, pred, has_pred] in enumerate(zip(trajs, has_obss, preds, has_preds)):
            self.plot_traj(traj[has_obs], pred[has_pred], i)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.show()
        # plt.show(block=False)
        # plt.pause(0.5)

    @staticmethod
    def get_ref_centerline(cline_list, pred_gt):
        if len(cline_list) == 1:
            return [Spline2D(x=cline_list[0][:, 0], y=cline_list[0][:, 1])], 0
        else:
            line_idx = 0
            ref_centerlines = [Spline2D(x=cline_list[i][:, 0], y=cline_list[i][:, 1]) for i in range(len(cline_list))]

            # search the closest point of the traj final position to each center line
            min_distances = []
            for line in ref_centerlines:
                xy = np.stack([line.x_fine, line.y_fine], axis=1)
                diff = xy - pred_gt[-1, :2]
                dis = np.hypot(diff[:, 0], diff[:, 1])
                min_distances.append(np.min(dis))
            line_idx = np.argmin(min_distances)
            return ref_centerlines, line_idx

    def plot_reference_centerlines(self, cline_list, splines, obs, pred, ref_line_idx):
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        for centerline_coords in cline_list:
            visualize_centerline(centerline_coords)

        for i, spline in enumerate(splines):
            xy = np.stack([spline.x_fine, spline.y_fine], axis=1)
            if i == ref_line_idx:
                plt.plot(xy[:, 0], xy[:, 1], "--", color="r", alpha=0.7, linewidth=1, zorder=10)
            else:
                plt.plot(xy[:, 0], xy[:, 1], "--", color="b", alpha=0.5, linewidth=1, zorder=10)

        self.plot_traj(obs, pred)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.show()
        # plt.show(block=False)
        # plt.pause(0.5)

    def plot_traj(self, obs, pred, traj_id=None):
        assert len(obs) != 0, "ERROR: The input trajectory is empty!"
        traj_na = "t{}".format(traj_id) if traj_id else "traj"
        obj_type = "AGENT" if traj_id == 0 else "OTHERS"

        plt.plot(obs[:, 0], obs[:, 1], color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)
        plt.plot(pred[:, 0], pred[:, 1], "d-", color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)

        plt.text(obs[0, 0], obs[0, 1], "{}_s".format(traj_na))

        if len(pred) == 0:
            plt.text(obs[-1, 0], obs[-1, 1], "{}_e".format(traj_na))
        else:
            plt.text(pred[-1, 0], pred[-1, 1], "{}_e".format(traj_na))


def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, default="../dataset")
    parser.add_argument("-d", "--dest", type=str, default="../dataset")
    parser.add_argument("-s", "--small", action='store_true', default=False)
    args = parser.parse_args()

    # args.root = "/home/jb/projects/Code/trajectory-prediction/TNT-Trajectory-Predition/dataset"
    raw_dir = os.path.join(args.root, "raw_data")
    interm_dir = os.path.join(args.dest, "interm_data" if not args.small else "interm_data_small")
    # print(args.root)
    # print(args.small)
    # for split in ["train", "val", "test"]:
    for split in ["val"]:
        # construct the preprocessor and dataloader
        argoverse_processor = ArgoversePreprocessor(root_dir=raw_dir, split=split, save_dir=interm_dir)
        loader = DataLoader(argoverse_processor,
                            batch_size=1 if sys.gettrace() else 16,     # 1 batch in debug mode
                            num_workers=0 if sys.gettrace() else 16,    # use only 0 worker in debug mode
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)

        for i, data in enumerate(tqdm(loader)):
            if args.small:
                if split == "train" and i >= 200:
                    break
                elif split == "val" and i >= 50:
                    break
                elif split == "test" and i >= 50:
                    break
