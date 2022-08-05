import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle import to_tensor
import paddle.nn.functional as F
from paddle.fluid import core
from paddle.fluid.dygraph import parallel_helper



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
    return core.is_compiled_with_dist(
    ) and parallel_helper._is_parallel_ctx_initialized()


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
    attrs = ('background_label', background_label, 'score_threshold',
             score_threshold, 'nms_top_k', nms_top_k, 'nms_threshold',
             nms_threshold, 'keep_top_k', keep_top_k, 'nms_eta', nms_eta,
             'normalized', normalized)
    output, index, nms_rois_num = core.ops.multiclass_nms3(bboxes, scores,
                                                           rois_num, *attrs)
    if not return_index:
        index = None
    return output, nms_rois_num, index

