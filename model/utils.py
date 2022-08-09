import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid import core, in_dygraph_mode
from paddle.fluid.dygraph import parallel_helper
from paddle.fluid.layer_helper import LayerHelper
from paddle.static import InputSpec


def identity(x):
    return x


def mish(x):
    return F.mish(x) if hasattr(F, mish) else x * F.tanh(F.softplus(x))


def silu(x):
    return F.silu(x)


def swish(x):
    return x * F.sigmoid(x)


TRT_ACT_SPEC = {'swish': swish, 'silu': swish}

ACT_SPEC = {'mish': mish, 'silu': silu}


def get_act_fn(act=None, trt=False):
    assert act is None or isinstance(act, (
        str, dict)), 'name of activation should be str, dict or None'
    if not act:
        return identity

    if isinstance(act, dict):
        name = act['name']
        act.pop('name')
        kwargs = act
    else:
        name = act
        kwargs = dict()

    if trt and name in TRT_ACT_SPEC:
        fn = TRT_ACT_SPEC[name]
    elif name in ACT_SPEC:
        fn = ACT_SPEC[name]
    else:
        fn = getattr(F, name)

    return lambda x: fn(x, **kwargs)


def get_static_shape(tensor):
    shape = paddle.shape(tensor)
    shape.stop_gradient = True
    return shape


def paddle_distributed_is_initialized():
    return core.is_compiled_with_dist() and parallel_helper._is_parallel_ctx_initialized()


@paddle.jit.not_to_static
def multiclass_nms(bboxes,
                   scores,
                   score_threshold,
                   nms_top_k,
                   keep_top_k,
                   nms_threshold=0.3,
                   normalized=True,
                   nms_eta=1.,
                   background_label=-1,
                   return_index=False,
                   return_rois_num=True,
                   rois_num=None,
                   name=None):
    """
    This operator is to do multi-class non maximum suppression (NMS) on
    boxes and scores.
    In the NMS step, this operator greedily selects a subset of detection bounding
    boxes that have high scores larger than score_threshold, if providing this
    threshold, then selects the largest nms_top_k confidences scores if nms_top_k
    is larger than -1. Then this operator pruns away boxes that have high IOU
    (intersection over union) overlap with already selected boxes by adaptive
    threshold NMS based on parameters of nms_threshold and nms_eta.
    Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
    per image if keep_top_k is larger than -1.
    Args:
        bboxes (Tensor): Two types of bboxes are supported:
                           1. (Tensor) A 3-D Tensor with shape
                           [N, M, 4 or 8 16 24 32] represents the
                           predicted locations of M bounding bboxes,
                           N is the batch size. Each bounding box has four
                           coordinate values and the layout is
                           [xmin, ymin, xmax, ymax], when box size equals to 4.
                           2. (LoDTensor) A 3-D Tensor with shape [M, C, 4]
                           M is the number of bounding boxes, C is the
                           class number
        scores (Tensor): Two types of scores are supported:
                           1. (Tensor) A 3-D Tensor with shape [N, C, M]
                           represents the predicted confidence predictions.
                           N is the batch size, C is the class number, M is
                           number of bounding boxes. For each category there
                           are total M scores which corresponding M bounding
                           boxes. Please note, M is equal to the 2nd dimension
                           of BBoxes.
                           2. (LoDTensor) A 2-D LoDTensor with shape [M, C].
                           M is the number of bbox, C is the class number.
                           In this case, input BBoxes should be the second
                           case with shape [M, C, 4].
        background_label (int): The index of background label, the background
                                label will be ignored. If set to -1, then all
                                categories will be considered. Default: 0
        score_threshold (float): Threshold to filter out bounding boxes with
                                 low confidence score. If not provided,
                                 consider all boxes.
        nms_top_k (int): Maximum number of detections to be kept according to
                         the confidences after the filtering detections based
                         on score_threshold.
        nms_threshold (float): The threshold to be used in NMS. Default: 0.3
        nms_eta (float): The threshold to be used in NMS. Default: 1.0
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
                          step. -1 means keeping all bboxes after NMS step.
        normalized (bool): Whether detections are normalized. Default: True
        return_index(bool): Whether return selected index. Default: False
        rois_num(Tensor): 1-D Tensor contains the number of RoIs in each image.
            The shape is [B] and data type is int32. B is the number of images.
            If it is not None then return a list of 1-D Tensor. Each element
            is the output RoIs' number of each image on the corresponding level
            and the shape is [B]. None by default.
        name(str): Name of the multiclass nms op. Default: None.
    Returns:
        A tuple with two Variables: (Out, Index) if return_index is True,
        otherwise, a tuple with one Variable(Out) is returned.
        Out: A 2-D LoDTensor with shape [No, 6] represents the detections.
        Each row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]
        or A 2-D LoDTensor with shape [No, 10] represents the detections.
        Each row has 10 values: [label, confidence, x1, y1, x2, y2, x3, y3,
        x4, y4]. No is the total number of detections.
        If all images have not detected results, all elements in LoD will be
        0, and output tensor is empty (None).
        Index: Only return when return_index is True. A 2-D LoDTensor with
        shape [No, 1] represents the selected index which type is Integer.
        The index is the absolute value cross batches. No is the same number
        as Out. If the index is used to gather other attribute such as age,
        one needs to reshape the input(N, M, 1) to (N * M, 1) as first, where
        N is the batch size and M is the number of boxes.
    Examples:
        .. code-block:: python

            import paddle
            from ppdet.modeling import ops
            boxes = paddle.static.data(name='bboxes', shape=[81, 4],
                                      dtype='float32', lod_level=1)
            scores = paddle.static.data(name='scores', shape=[81],
                                      dtype='float32', lod_level=1)
            out, index = ops.multiclass_nms(bboxes=boxes,
                                            scores=scores,
                                            background_label=0,
                                            score_threshold=0.5,
                                            nms_top_k=400,
                                            nms_threshold=0.3,
                                            keep_top_k=200,
                                            normalized=False,
                                            return_index=True)
    """
    helper = LayerHelper('multiclass_nms3', **locals())

    if in_dygraph_mode():
        attrs = ('background_label', background_label, 'score_threshold',
                 score_threshold, 'nms_top_k', nms_top_k, 'nms_threshold',
                 nms_threshold, 'keep_top_k', keep_top_k, 'nms_eta', nms_eta,
                 'normalized', normalized)
        output, index, nms_rois_num = core.ops.multiclass_nms3(bboxes, scores,
                                                               rois_num, *attrs)
        if not return_index:
            index = None
        return output, nms_rois_num, index

    else:
        output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
        index = helper.create_variable_for_type_inference(dtype='int32')

        inputs = {'BBoxes': bboxes, 'Scores': scores}
        outputs = {'Out': output, 'Index': index}

        if rois_num is not None:
            inputs['RoisNum'] = rois_num

        if return_rois_num:
            nms_rois_num = helper.create_variable_for_type_inference(
                dtype='int32')
            outputs['NmsRoisNum'] = nms_rois_num

        helper.append_op(
            type="multiclass_nms3",
            inputs=inputs,
            attrs={
                'background_label': background_label,
                'score_threshold': score_threshold,
                'nms_top_k': nms_top_k,
                'nms_threshold': nms_threshold,
                'keep_top_k': keep_top_k,
                'nms_eta': nms_eta,
                'normalized': normalized
            },
            outputs=outputs)
        output.stop_gradient = True
        index.stop_gradient = True
        if not return_index:
            index = None
        if not return_rois_num:
            nms_rois_num = None

        return output, nms_rois_num, index


def iou_similarity(x, y, box_normalized=True, name=None):
    """
    Computes intersection-over-union (IOU) between two box lists.
    Box list 'X' should be a LoDTensor and 'Y' is a common Tensor,
    boxes in 'Y' are shared by all instance of the batched inputs of X.
    Given two boxes A and B, the calculation of IOU is as follows:

    $$
    IOU(A, B) =
    \\frac{area(A\\cap B)}{area(A)+area(B)-area(A\\cap B)}
    $$

    Args:
        x (Tensor): Box list X is a 2-D Tensor with shape [N, 4] holds N
             boxes, each box is represented as [xmin, ymin, xmax, ymax],
             the shape of X is [N, 4]. [xmin, ymin] is the left top
             coordinate of the box if the input is image feature map, they
             are close to the origin of the coordinate system.
             [xmax, ymax] is the right bottom coordinate of the box.
             The data type is float32 or float64.
        y (Tensor): Box list Y holds M boxes, each box is represented as
             [xmin, ymin, xmax, ymax], the shape of X is [N, 4].
             [xmin, ymin] is the left top coordinate of the box if the
             input is image feature map, and [xmax, ymax] is the right
             bottom coordinate of the box. The data type is float32 or float64.
        box_normalized(bool): Whether treat the priorbox as a normalized box.
            Set true by default.
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        Tensor: The output of iou_similarity op, a tensor with shape [N, M]
              representing pairwise iou scores. The data type is same with x.

    Examples:
        .. code-block:: python

            import paddle
            from ppdet.modeling import ops
            paddle.enable_static()

            x = paddle.static.data(name='x', shape=[None, 4], dtype='float32')
            y = paddle.static.data(name='y', shape=[None, 4], dtype='float32')
            iou = ops.iou_similarity(x=x, y=y)
    """

    if in_dygraph_mode():
        out = core.ops.iou_similarity(x, y, 'box_normalized', box_normalized)
        return out
    else:
        helper = LayerHelper("iou_similarity", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type="iou_similarity",
            inputs={"X": x,
                    "Y": y},
            attrs={"box_normalized": box_normalized},
            outputs={"Out": out})
        return out


class JDEBBoxPostProcess(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['decode', 'nms']

    def __init__(self, num_classes=1, decode=None, nms=None, return_idx=True):
        super(JDEBBoxPostProcess, self).__init__()
        self.num_classes = num_classes
        self.decode = decode
        self.nms = nms
        self.return_idx = return_idx

        self.fake_bbox_pred = paddle.to_tensor(
            np.array(
                [[-1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype='float32'))
        self.fake_bbox_num = paddle.to_tensor(np.array([1], dtype='int32'))
        self.fake_nms_keep_idx = paddle.to_tensor(
            np.array(
                [[0]], dtype='int32'))

        self.fake_yolo_boxes_out = paddle.to_tensor(
            np.array(
                [[[0.0, 0.0, 0.0, 0.0]]], dtype='float32'))
        self.fake_yolo_scores_out = paddle.to_tensor(
            np.array(
                [[[0.0]]], dtype='float32'))
        self.fake_boxes_idx = paddle.to_tensor(np.array([[0]], dtype='int64'))

    def forward(self, head_out, anchors):
        """
        Decode the bbox and do NMS for JDE model.

        Args:
            head_out (list): Bbox_pred and cls_prob of bbox_head output.
            anchors (list): Anchors of JDE model.

        Returns:
            boxes_idx (Tensor): The index of kept bboxes after decode 'JDEBox'.
            bbox_pred (Tensor): The output is the prediction with shape [N, 6]
                including labels, scores and bboxes.
            bbox_num (Tensor): The number of prediction of each batch with shape [N].
            nms_keep_idx (Tensor): The index of kept bboxes after NMS.
        """
        boxes_idx, yolo_boxes_scores = self.decode(head_out, anchors)

        if len(boxes_idx) == 0:
            boxes_idx = self.fake_boxes_idx
            yolo_boxes_out = self.fake_yolo_boxes_out
            yolo_scores_out = self.fake_yolo_scores_out
        else:
            yolo_boxes = paddle.gather_nd(yolo_boxes_scores, boxes_idx)
            yolo_boxes_out = paddle.reshape(
                yolo_boxes[:, :4], shape=[1, len(boxes_idx), 4])
            yolo_scores_out = paddle.reshape(
                yolo_boxes[:, 4:5], shape=[1, 1, len(boxes_idx)])
            boxes_idx = boxes_idx[:, 1:]

        if self.return_idx:
            bbox_pred, bbox_num, nms_keep_idx = self.nms(
                yolo_boxes_out, yolo_scores_out, self.num_classes)
            if bbox_pred.shape[0] == 0:
                bbox_pred = self.fake_bbox_pred
                bbox_num = self.fake_bbox_num
                nms_keep_idx = self.fake_nms_keep_idx
            return boxes_idx, bbox_pred, bbox_num, nms_keep_idx
        else:
            bbox_pred, bbox_num, _ = self.nms(yolo_boxes_out, yolo_scores_out,
                                              self.num_classes)
            if bbox_pred.shape[0] == 0:
                bbox_pred = self.fake_bbox_pred
                bbox_num = self.fake_bbox_num
            return _, bbox_pred, bbox_num, _


def generate_anchors_for_grid_cell(feats,
                                   fpn_strides,
                                   grid_cell_size=5.0,
                                   grid_cell_offset=0.5):
    r"""
    Like ATSS, generate anchors based on grid size.
    Args:
        feats (List[Tensor]): shape[s, (b, c, h, w)]
        fpn_strides (tuple|list): shape[s], stride for each scale feature
        grid_cell_size (float): anchor size
        grid_cell_offset (float): The range is between 0 and 1.
    Returns:
        anchors (Tensor): shape[l, 4], "xmin, ymin, xmax, ymax" format.
        anchor_points (Tensor): shape[l, 2], "x, y" format.
        num_anchors_list (List[int]): shape[s], contains [s_1, s_2, ...].
        stride_tensor (Tensor): shape[l, 1], contains the stride for each scale.
    """
    assert len(feats) == len(fpn_strides)
    anchors = []
    anchor_points = []
    num_anchors_list = []
    stride_tensor = []
    for feat, stride in zip(feats, fpn_strides):
        _, _, h, w = feat.shape
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = (paddle.arange(end=w) + grid_cell_offset) * stride
        shift_y = (paddle.arange(end=h) + grid_cell_offset) * stride
        shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
        anchor = paddle.stack(
            [
                shift_x - cell_half_size, shift_y - cell_half_size,
                shift_x + cell_half_size, shift_y + cell_half_size
            ],
            axis=-1).astype(feat.dtype)
        anchor_point = paddle.stack(
            [shift_x, shift_y], axis=-1).astype(feat.dtype)

        anchors.append(anchor.reshape([-1, 4]))
        anchor_points.append(anchor_point.reshape([-1, 2]))
        num_anchors_list.append(len(anchors[-1]))
        stride_tensor.append(
            paddle.full(
                [num_anchors_list[-1], 1], stride, dtype=feat.dtype))
    anchors = paddle.concat(anchors)
    anchors.stop_gradient = True
    anchor_points = paddle.concat(anchor_points)
    anchor_points.stop_gradient = True
    stride_tensor = paddle.concat(stride_tensor)
    stride_tensor.stop_gradient = True
    return anchors, anchor_points, num_anchors_list, stride_tensor


def _prune_input_spec(input_spec, program, targets):
    device = paddle.get_device()
    paddle.enable_static()
    paddle.set_device(device)
    pruned_input_spec = [{}]
    program = program.clone()
    program = program._prune(targets=targets)
    global_block = program.global_block()
    for name, spec in input_spec[0].items():
        try:
            v = global_block.var(name)
            pruned_input_spec[0][name] = spec
        except Exception:
            pass
    paddle.disable_static(place=device)
    return pruned_input_spec


def get_infer_cfg_and_input_spec(model, image_shape):
    if len(image_shape) == 3:
        image_shape = [None] + image_shape
        im_shape = [None, 2]
        scale_factor = [None, 2]
    else:
        im_shape = [image_shape[0], 2]
        scale_factor = [image_shape[0], 2]

    for layer in model.sublayers():
        if hasattr(layer, 'convert_to_deploy'):
            layer.convert_to_deploy()

    model.fuse_norm = False
    input_spec = [{"image": InputSpec(shape=image_shape, name='image'),
                   "im_shape": InputSpec(shape=im_shape, name='im_shape'),
                   "scale_factor": InputSpec(shape=scale_factor, name='scale_factor')}]
    static_model = paddle.jit.to_static(model, input_spec=input_spec)
    pruned_input_spec = _prune_input_spec(input_spec, static_model.forward.main_program,
                                          static_model.forward.outputs)

    return static_model, pruned_input_spec
