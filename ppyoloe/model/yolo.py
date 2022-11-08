from ppyoloe.model.cspresnet import CSPResNet
from ppyoloe.model.custom_pan import CustomCSPPAN
from ppyoloe.model.meta_arch import BaseArch
from ppyoloe.model.ppyoloe_head import PPYOLOEHead

__all__ = ["PPYOLOE_X", 'PPYOLOE_L', "PPYOLOE_M", "PPYOLOE_S"]


class PPYOLOE(BaseArch):
    def __init__(self,
                 num_classes,
                 depth_mult=0.33,
                 width_mult=0.50,
                 data_format='NCHW',
                 for_mot=False):
        """
        YOLOv3 network, see https://arxiv.org/abs/1804.02767

        Args:
            data_format (str): data format, NCHW or NHWC
            for_mot (bool): whether return other features for multi-object tracking
                models, default False in pure object detection models.
        """
        super(PPYOLOE, self).__init__(data_format=data_format)
        self.backbone = CSPResNet(width_mult=width_mult, depth_mult=depth_mult)
        self.neck = CustomCSPPAN(in_channels=self.backbone.out_channels[1:], width_mult=width_mult,
                                 depth_mult=depth_mult)
        self.yolo_head = PPYOLOEHead(in_channels=self.neck.out_channels, num_classes=num_classes)
        self.for_mot = for_mot

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        neck_feats = self.neck(body_feats, self.for_mot)

        if self.training:
            yolo_losses = self.yolo_head(neck_feats, self.inputs)
            return yolo_losses
        else:
            yolo_head_outs = self.yolo_head(neck_feats)
            bbox, bbox_num = self.yolo_head.post_process(
                yolo_head_outs, self.inputs['im_shape'], self.inputs['scale_factor'])
            output = {'bbox': bbox, 'bbox_num': bbox_num}
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()


def PPYOLOE_X(num_classes):
    model = PPYOLOE(num_classes=num_classes, depth_mult=1.33, width_mult=1.25)
    return model


def PPYOLOE_L(num_classes):
    model = PPYOLOE(num_classes=num_classes, depth_mult=1.0, width_mult=1.0)
    return model


def PPYOLOE_M(num_classes):
    model = PPYOLOE(num_classes=num_classes, depth_mult=0.67, width_mult=0.75)
    return model


def PPYOLOE_S(num_classes):
    model = PPYOLOE(num_classes=num_classes, depth_mult=0.33, width_mult=0.50)
    return model
