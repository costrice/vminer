"""
This file contains loss functions for training.
"""
from typing import Dict

import torch
import torch.nn.functional as F

from internal import scenes


class LossGatherer(object):
    """
    Calculate the loss terms for the model.
    All losses are put on the same device as the model.
    """
    def __init__(
            self,
            loss_weights: Dict[str, float],
            verbose: bool = True
    ):
        super().__init__()
        self.supported_loss_terms = [
            'alpha_l1',
            'nr_rgb_mse',
            'pbr_rgb_mse',
            'self_consistency_mse',
            # 'normal_diff_mse',
            'eikonal_mse',
            'normal_smoothness_mse',
            'material_smoothness_mse',
            'reg_light_intensity_l1',
            'reg_light_color_balance',
        ]
        self.loss_weights = {}
        self.verbose = verbose
        self.update_weights(loss_weights)

    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update the loss weights. If a loss weight is set to 0, it will be
        removed from the loss weights to save computation.
        Args:
            new_weights (Dict[str, float]): the new loss weights

        Returns:
            None
        """
        for key, value in new_weights.items():
            assert key in self.supported_loss_terms, \
                f'Unsupported loss term: {key}'
            if value < 1e-10:
                if key in self.loss_weights:
                    del self.loss_weights[key]
            else:
                self.loss_weights[key] = value
        if self.verbose:
            print('Updated loss weights:')
            print(self.loss_weights)

    def __call__(
            self,
            scene: scenes.SceneModel,
            pred: Dict[str, torch.Tensor],
            gt: Dict[str, torch.Tensor],
    ):
        loss_total = torch.tensor(0.0, device=scene.device)
        loss_dict = {}

        def add_loss_term(name: str, loss_term: torch.Tensor):
            """Add a loss term to the total loss.

            Args:
                name: the name of the loss term
                loss_term: the loss term

            Returns:
                None
            """
            nonlocal loss_total, loss_dict
            # cope with nan or inf
            if torch.isnan(loss_term).any() or torch.isinf(loss_term).any():
                print(f'Warning: Loss term {name} is nan or inf. Discarded.')
                loss_term = torch.tensor(0.0, device=scene.device)
            loss_term = loss_term.to(scene.device)
            loss_dict[name] = loss_term
            loss_total = loss_total + loss_term * self.loss_weights[name]

        # L1 loss for alpha
        if 'alpha_l1' in self.loss_weights:
            loss_term = F.l1_loss(pred['opacity'], gt['alpha'])
            add_loss_term('alpha_l1', loss_term)

        # MSE loss for RGB from radiance field
        if 'nr_rgb_mse' in self.loss_weights:
            # loss_term = F.huber_loss(pred['nr_rgb'], gt['rgb'])
            loss_term = F.mse_loss(pred['nr_rgb'], gt['rgb'])
            add_loss_term('nr_rgb_mse', loss_term)

        # MSE loss for RGB from PBR
        if 'pbr_rgb_mse' in self.loss_weights:
            loss_term = F.mse_loss(pred['pbr_rgb'],
                                   gt['rgb'])
            add_loss_term('pbr_rgb_mse', loss_term)

        # consistency loss between NR and PBR for each light source
        if 'self_consistency_mse' in self.loss_weights:
            loss_term = torch.tensor(0.0, device=scene.device)
            # only compute it when both near and far light sources are present,
            # and both nr and pbr are used
            if 'pbr_rgb_far' in pred and 'pbr_rgb_near' in pred \
                    and 'nr_rgb_far' in pred and 'nr_rgb_near' in pred:
                pbr_rgb_far = pred['pbr_rgb_far']
                pbr_rgb_near = pred['pbr_rgb_near']
                nr_rgb_far = pred['nr_rgb_far'].detach()
                nr_rgb_near = pred['nr_rgb_near'].detach()
                loss_term += F.mse_loss(pbr_rgb_far, nr_rgb_far)
                loss_term += F.mse_loss(pbr_rgb_near, nr_rgb_near)
            add_loss_term('self_consistency_mse', loss_term)

        # Eikonal loss for derived normal
        if 'eikonal_mse' in self.loss_weights:
            loss_term = (torch.norm(pred['sample_sdf_grad'], dim=-1, p=2)
                         - 1.0).pow(2).mean()
            # if 'sampled_sdf_grad_in_aabb' in pred:
            #     loss_term += (torch.norm(pred['sampled_sdf_grad_in_aabb'],
            #                              dim=-1, p=2) - 1.0).pow(2).mean()
            add_loss_term('eikonal_mse', loss_term)

        # MSE loss for normal smoothness
        if 'normal_smoothness_mse' in self.loss_weights:
            if pred['sample_normal'].shape[0] == 0:
                loss_term = torch.tensor(0.0, device=scene.device)
            else:
                mse = (pred['sample_normal'] - pred['sample_jittered_normal']
                       ).pow(2).mean(dim=-1)
                # aggregate use weight
                weights = pred['sample_weights']
                loss_term = (mse * weights).sum() / weights.sum()
                add_loss_term('normal_smoothness_mse', loss_term)

        # MSE loss for material smoothness
        if 'material_smoothness_mse' in self.loss_weights:
            loss_term = torch.tensor(0.0, device=scene.device)
            weights = pred['sample_weights']
            for k, v_jitter in pred.items():
                if k.startswith('sample_jittered_') and v_jitter.shape[0] > 0:
                    material_key = k[len('sample_jittered_'):]
                    v_orig = pred[f'sample_{material_key}']
                    mse = (v_jitter - v_orig).pow(2).mean(dim=-1)
                    loss_term_this = (mse * weights).sum() / weights.sum()
                    if material_key == 'base_color':
                        loss_term_this *= 0.1
                    loss_term += loss_term_this
            add_loss_term('material_smoothness_mse', loss_term)

        # L1 regularization for light intensity
        if 'reg_light_intensity_l1' in self.loss_weights:
            loss_term = scene.get_regularization_loss(
                reg_type='L1', component='light_intensity')
            add_loss_term('reg_light_intensity_l1', loss_term)

        if 'reg_light_color_balance' in self.loss_weights:
            loss_term = scene.get_regularization_loss(
                reg_type='L1', component='light_color_balance')
            add_loss_term('reg_light_color_balance', loss_term)

        loss_dict['total'] = loss_total
        return loss_dict
