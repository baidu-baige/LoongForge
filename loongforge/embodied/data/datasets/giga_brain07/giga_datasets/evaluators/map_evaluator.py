import functools
import logging
from typing import Any, Iterable

import numpy as np
from terminaltables import AsciiTable

from ..structures import boxes3d_utils, boxes_utils


class MAPEvaluator:
    """Mean Average Precision evaluator for 2D/3D detections."""

    def __init__(
        self,
        dataset,
        label_names: list[str],
        iou_threshs: list[float],
        ignore_thresh: float = 0.5,
        ignore_mode: str = 'ioa',
        eval_mode: str = '2d',
        gt_names: tuple[str, str, str] | None = None,
        pred_names: tuple[str, str, str] | None = None,
        overlap_params: dict[str, Any] | None = None,
    ) -> None:
        assert eval_mode in ('2d', '3d')
        self.dataset = dataset
        self.label_names = label_names
        self.iou_threshs = iou_threshs
        self.ignore_thresh = ignore_thresh
        self.ignore_mode = ignore_mode
        self.eval_mode = eval_mode
        if eval_mode == '2d':
            self.gt_names = ('boxes', 'labels', 'ignores') if gt_names is None else gt_names
            self.pred_names = ('pred_boxes', 'pred_labels', 'pred_scores') if pred_names is None else pred_names
            self.overlap_func = boxes_utils.overlaps
        else:
            self.gt_names = ('boxes3d', 'labels3d', 'ignores3d') if gt_names is None else gt_names
            self.pred_names = ('pred_boxes3d', 'pred_labels3d', 'pred_scores3d') if pred_names is None else pred_names
            assert overlap_params is not None
            self.overlap_func = functools.partial(boxes3d_utils.overlaps, **overlap_params)

    def __call__(self, results: Iterable[dict[str, Any]]) -> dict[str, Any]:
        results = self.process_results(results)
        recall, precision, ap = self.eval_all_classes(results)
        log_str = self.print_summary(recall, ap)
        return dict(recall=recall, precision=precision, ap=ap, log_str=log_str)

    def process_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i in range(len(results)):
            result = results[i]
            pred_boxes = result[self.pred_names[0]]
            pred_labels = result[self.pred_names[1]]
            pred_scores = result[self.pred_names[2]]
            sorted_ind = np.argsort(-pred_scores)
            result[self.pred_names[0]] = pred_boxes[sorted_ind]
            result[self.pred_names[1]] = [pred_labels[_] for _ in sorted_ind]
            result[self.pred_names[2]] = pred_scores[sorted_ind]
        return results

    def eval_all_classes(self, results: list[dict[str, Any]]) -> tuple[list[dict[str, float]], list[dict[str, float]], list[dict[str, float]]]:
        all_rets = []
        for label_name in self.label_names:
            gt_boxes_list = []
            ignore_boxes_list = []
            pred_boxes_list = []
            pred_scores_list = []
            for i, result in enumerate(results):
                data_index = result.get('data_index', i)
                data_dict = self.dataset[data_index]
                # gt
                boxes = data_dict[self.gt_names[0]].astype(np.float32)
                labels = data_dict[self.gt_names[1]]
                keeps = [label_name == label for label in labels]
                boxes = boxes[keeps]
                if self.gt_names[2] in data_dict:
                    ignores = data_dict[self.gt_names[2]]
                    ignores = ignores[keeps]
                    gt_boxes = boxes[~ignores]
                    ignore_boxes = boxes[ignores]
                    gt_boxes_list.append(gt_boxes)
                    ignore_boxes_list.append(ignore_boxes)
                else:
                    gt_boxes_list.append(boxes)
                # pred
                pred_boxes = result[self.pred_names[0]].astype(np.float32)
                pred_labels = result[self.pred_names[1]]
                pred_scores = result[self.pred_names[2]].astype(np.float32)
                pred_keeps = [label_name == pred_label for pred_label in pred_labels]
                pred_boxes = pred_boxes[pred_keeps]
                pred_scores = pred_scores[pred_keeps]
                pred_boxes_list.append(pred_boxes)
                pred_scores_list.append(pred_scores)
            # Evaluate per-class across IoU thresholds
            rets = self.eval_single_class(gt_boxes_list, pred_boxes_list, pred_scores_list, ignore_boxes_list)
            all_rets.append(rets)
        recall = [{} for _ in self.iou_threshs]
        precision = [{} for _ in self.iou_threshs]
        ap = [{} for _ in self.iou_threshs]
        for i, label_name in enumerate(self.label_names):
            for iou_idx in range(len(self.iou_threshs)):
                recall_i, precision_i, ap_i = all_rets[i][iou_idx]
                recall[iou_idx][label_name] = recall_i
                precision[iou_idx][label_name] = precision_i
                ap[iou_idx][label_name] = ap_i
        return recall, precision, ap

    def eval_single_class(self, gt_boxes_list, pred_boxes_list, pred_scores_list, ignore_boxes_list) -> list[tuple[float, float, float]]:
        assert len(gt_boxes_list) == len(pred_boxes_list) == len(pred_scores_list)
        tps_list = []
        fps_list = []
        npos = 0
        num_images = len(gt_boxes_list)
        for i in range(num_images):
            gt_boxes = gt_boxes_list[i]
            pred_boxes = pred_boxes_list[i]
            ignore_boxes = ignore_boxes_list[i] if len(ignore_boxes_list) > 0 else []
            # Compute TPs/FPs for a single image at all overlap thresholds
            tps, fps = self.eval_single_class_single_image(gt_boxes, pred_boxes, ignore_boxes)
            tps_list.append(tps)
            fps_list.append(fps)
            npos += len(gt_boxes)
        tps = np.concatenate(tps_list, axis=1)
        fps = np.concatenate(fps_list, axis=1)
        pred_scores = np.concatenate(pred_scores_list, axis=0)
        sorted_ind = np.argsort(-pred_scores)
        tps = tps[:, sorted_ind]
        fps = fps[:, sorted_ind]
        ret = []
        for iou_idx in range(len(self.iou_threshs)):
            tp = np.cumsum(tps[iou_idx])
            fp = np.cumsum(fps[iou_idx])
            recall = tp / max(float(npos), 1)
            precision = tp / np.maximum(tp + fp, np.finfo(np.float32).eps)
            ap = average_precision(recall, precision)
            recall = recall[-1] if len(recall) > 0 else 0
            precision = precision[-1] if len(precision) > 0 else 0
            ret.append((recall, precision, ap))
        return ret

    def eval_single_class_single_image(self, gt_boxes, pred_boxes, ignore_boxes) -> tuple[np.ndarray, np.ndarray]:
        tps = np.zeros((len(self.iou_threshs), len(pred_boxes)), dtype=np.float32)
        fps = np.zeros((len(self.iou_threshs), len(pred_boxes)), dtype=np.float32)
        if len(pred_boxes) == 0:
            return tps, fps
        if len(gt_boxes) > 0:
            gt_overlaps = self.overlap_func(gt_boxes, pred_boxes)
        if len(ignore_boxes) > 0:
            ignore_overlaps = self.overlap_func(pred_boxes, ignore_boxes, mode=self.ignore_mode)
        match_gts = np.zeros((len(self.iou_threshs), len(gt_boxes)), dtype=np.bool_)
        for i in range(len(pred_boxes)):
            iou_max = -np.inf
            j_max = -1
            for j in range(len(gt_boxes)):
                iou = gt_overlaps[i, j]
                if iou > iou_max:
                    iou_max = iou
                    j_max = j
            match_ignore = False
            if len(ignore_boxes) > 0:
                for k in range(len(ignore_boxes)):
                    ignore_iou = ignore_overlaps[i, k]
                    if ignore_iou > self.ignore_thresh:
                        match_ignore = True
                        break
            for iou_idx, iou_thresh in enumerate(self.iou_threshs):
                # if match a gt, tp = 1, fp = 0
                # if match an ignored gt, tp = 0, fp = 0
                # if match no gt, tp = 0, fp = 1
                if iou_max > iou_thresh and not match_gts[iou_idx, j_max]:
                    tps[iou_idx, i] = 1.0
                    match_gts[iou_idx, j_max] = True
                elif not match_ignore:
                    fps[iou_idx, i] = 1.0
        return tps, fps

    def print_summary(self, recall, ap) -> str:
        f2str = lambda x: '{:.2f}'.format(x)  # noqa E731
        table_head = ['classes']
        table_foot = ['Overall']
        aps = []
        for j, iou_thresh in enumerate(self.iou_threshs):
            ap_name_j = 'AP_{:.2f}'.format(iou_thresh)
            ar_name_j = 'AR_{:.2f}'.format(iou_thresh)
            table_head.append(ap_name_j)
            table_head.append(ar_name_j)
            ap_j = float(np.mean(list(ap[j].values())))
            recall_j = float(np.mean(list(recall[j].values())))
            table_foot.append(f2str(ap_j * 100))
            table_foot.append(f2str(recall_j * 100))
            aps.append(ap_j)
        table_rows = []
        for i, label_name in enumerate(self.label_names):
            table_row = [label_name]
            for j in range(len(self.iou_threshs)):
                table_row.append(f2str(ap[j][label_name] * 100))
                table_row.append(f2str(recall[j][label_name] * 100))
            table_rows.append(table_row)
        table = AsciiTable([table_head] + table_rows + [table_foot])
        table.inner_footing_row_border = True
        log_str = '\n' + table.table
        logging.info(log_str)
        return log_str


def average_precision(recalls: np.ndarray, precisions: np.ndarray, mode: str = 'area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum((mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError('Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap
