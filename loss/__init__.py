from loss.cross_dice_loss import cross_dice_loss, bce_dice_loss, cross_entropy_loss
from loss.oct_loss import (
    oct_cross_dice_loss,
    oct_topology_loss,
    oct_order_loss,
    oct_smooth_loss,
    oct_curvature_loss,
    oct_order_smooth_loss,
    oct_order_curvature_loss,
    oct_smooth_curvature_loss,
    oct_order_smooth_curvature_loss,
)
from loss.octa_ce_dice_loss import octa_ce_dice_loss
from loss.ASTP_loss import ASTPLoss


__all__ = {
    'cross_dice_loss': cross_dice_loss,
    'bce_dice_loss': bce_dice_loss,
    'cross_entropy_loss': cross_entropy_loss,
    'octa_ce_dice_loss': octa_ce_dice_loss,
    'ASTP_loss': ASTPLoss,
    'oct_cross_dice_loss': oct_cross_dice_loss,
    'oct_topology_loss': oct_topology_loss,
    'oct_order_loss': oct_order_loss,
    'oct_smooth_loss': oct_smooth_loss,
    'oct_curvature_loss': oct_curvature_loss,
    'oct_order_smooth_loss': oct_order_smooth_loss,
    'oct_order_curvature_loss': oct_order_curvature_loss,
    'oct_smooth_curvature_loss': oct_smooth_curvature_loss,
    'oct_order_smooth_curvature_loss': oct_order_smooth_curvature_loss,
}




def build_loss(hypes):
    name = hypes['loss']['core_method']

    return __all__[name]
