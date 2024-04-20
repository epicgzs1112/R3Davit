# -*- coding: utf-8 -*-
#
# Developed by Liying Yang <lyyang69@gmail.com>
import os

import mcubes
import numpy as np
import open3d

from utils import logging
from datetime import datetime as dt
import torch
import torch.backends.cudnn
import torch.utils.data
from tqdm import tqdm

import utils.data_loaders
import utils.data_transforms
import utils.helpers
import core.pipeline_test as pipeline

from models.encoder.encoder import Encoder

from models.decoder.decoder import Decoder

from losses.losses import DiceLoss
from utils.average_meter import AverageMeter
def voxel_grid_to_mesh(vox_grid: np.array) -> open3d.geometry.TriangleMesh:
    """
        taken from: https://github.com/lmb-freiburg/what3d

        Converts a voxel grid represented as a numpy array into a mesh.
    """
    sp = vox_grid.shape
    if len(sp) != 3 or sp[0] != sp[1] or \
            sp[1] != sp[2] or sp[0] == 0:
        raise ValueError("Only non-empty cubic 3D grids are supported.")
    padded_grid = np.pad(vox_grid, ((1, 1), (1, 1), (1, 1)), 'constant')
    m_vert, m_tri = mcubes.marching_cubes(padded_grid, 0)
    m_vert = m_vert / (padded_grid.shape[0] - 1)
    out_mesh = open3d.geometry.TriangleMesh()
    out_mesh.vertices = open3d.utility.Vector3dVector(m_vert)
    out_mesh.triangles = open3d.utility.Vector3iVector(m_tri)
    return out_mesh


def calculate_fscore(list_pr: np.array, list_gt: np.array, th: float = 0.01) -> float:
    """
        based on: https://github.com/lmb-freiburg/what3d

        Calculates the F-score between two point clouds with the corresponding threshold value.
    """
    num_sampled_pts = 8192
    assert list_pr.shape == list_gt.shape
    b_size = list_gt.shape[0]

    list_gt, list_pr = list_gt.detach().cpu().numpy(), list_pr.detach().cpu().numpy()

    result = []

    for i in range(b_size):
        gt, pr = list_gt[i], list_pr[i]

        if (gt.sum() == 0 and pr.sum() != 0) or (gt.sum() != 0 and pr.sum() == 0):
            result.append(0)
            continue

        gt = voxel_grid_to_mesh(gt).sample_points_uniformly(num_sampled_pts)
        pr = voxel_grid_to_mesh(pr).sample_points_uniformly(num_sampled_pts)

        d1 = gt.compute_point_cloud_distance(pr)
        d2 = pr.compute_point_cloud_distance(gt)

        if len(d1) and len(d2):
            recall = float(sum(d < th for d in d2)) / float(len(d2))
            precision = float(sum(d < th for d in d1)) / float(len(d1))

            if recall + precision > 0:
                fscore = 2 * recall * precision / (recall + precision)
            else:
                fscore = 0
        else:
            fscore = 0
        result.append(fscore)

    return np.array(result)


def binarize_preds(predictions: torch.Tensor, threshold=0.5) -> torch.Tensor:
    """
        Apply threshold on the predictions
    :param predictions: Predicted voxel grid
    :param threshold: Threshold limit
    :return: Binarized voxel grid
    """
    return predictions.__ge__(threshold).int()



def test_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             test_file_num=None,
             test_writer=None,
             encoder=None,
             decoder=None,
          ):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # load data
    taxonomies, test_data_loader, test_file_num = pipeline.load_data(cfg, test_data_loader, test_file_num)

    # Set up networks
    if decoder is None or encoder is None :
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)


        encoder, decoder, epoch_idx = \
            pipeline.setup_network(cfg, encoder, decoder)

    # Set up loss functions
    loss_function = DiceLoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = []
    test_f1 = []
    taxonomies_list = []
    losses = AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()


    for_tqdm = tqdm(enumerate(test_data_loader), total=n_samples)
    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in for_tqdm:
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]
        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images).to(torch.cuda.current_device())
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume).to(torch.cuda.current_device())

            # Test the encoder, decoder and
            # encoder
            image_features = encoder(rendering_images)


            context = image_features

            # decoder
            generated_volume = decoder(context).squeeze(dim=1)
            generated_volume = generated_volume.clamp_max(1)
            threshold = 0.4
            pred_volume = generated_volume
            pred_volume = pred_volume > threshold
            if not os.path.exists('/root/autodl-tmp/R3DSWIN++/output/%s' % (taxonomy_id)):
                os.makedirs('/root/autodl-tmp/R3DSWIN++/output/%s' % (taxonomy_id))
            with open('/root/autodl-tmp/R3DSWIN++/output/%s/%s.binvox' % (taxonomy_id, sample_name),
                      'wb') as f:
                vox = utils.binvox_rw.Voxels(pred_volume.cpu().numpy(), (32,) * 3, (0,) * 3, 1, 'xzy')
                vox.write(f)
            # Loss
            # Loss
            loss = loss_function(generated_volume, ground_truth_volume)

            # Append loss and accuracy to average metrics
            loss = utils.helpers.reduce_value(loss)
            losses.update(loss.item())

            # IoU per sample
            sample_iou = []
            sample_f = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).unsqueeze(dim=0))
                pred_volume = binarize_preds(generated_volume, threshold=th)
                f1 = calculate_fscore(pred_volume, ground_truth_volume)
                f1 = torch.tensor(f1).to("cuda")
                sample_f.append((f1).unsqueeze(dim=0))
            test_iou.append(torch.cat(sample_iou).unsqueeze(dim=0))
            test_f1.append(torch.cat(sample_f).unsqueeze(dim=0))
            taxonomies_list.append(torch.tensor(list(taxonomies.keys()).index(taxonomy_id)).unsqueeze(dim=0))

            if torch.distributed.get_rank() == 0:
                # Print sample loss and IoU
                if (sample_idx + 1) % 50 == 0:
                    for_tqdm.update(50)
                    for_tqdm.set_description('Test[%d/%d] Taxonomy = %s Loss = %.4f' %
                                             (sample_idx + 1, n_samples, taxonomy_id, losses.avg))

                logging.debug('Test[%d/%d] Taxonomy = %s Sample = %s Loss = %.4f IoU = %s f2=%s' %
                              (sample_idx + 1, n_samples, taxonomy_id, sample_name,
                               loss.item(), ['%.4f' % si for si in sample_iou], ['%.4f' % si for si in sample_f]))

    test_iou = torch.cat(test_iou, dim=0)
    test_f1 = torch.cat(test_f1, dim=0)
    taxonomies_list = torch.cat(taxonomies_list).to(torch.cuda.current_device())

    test_iou = pipeline.combine_test_iou(test_iou, taxonomies_list, list(taxonomies.keys()), test_file_num)
    test_f1 = pipeline.combine_test_f1(test_f1, taxonomies_list, list(taxonomies.keys()), test_file_num)
    torch.cuda.synchronize(torch.device(torch.cuda.current_device()))

    if torch.distributed.get_rank() == 0:
        # Output testing results
        mean_iou = pipeline.output(cfg, test_iou, taxonomies)
        mean_f1 = pipeline.output2(cfg, test_f1, taxonomies)

        # Add testing results to TensorBoard
        max_iou = np.max(mean_iou)
        max_f1 = np.max(mean_f1)
        if test_writer is not None:
            test_writer.add_scalar('EpochLoss', losses.avg, epoch_idx)
            test_writer.add_scalar('IoU', max_iou, epoch_idx)
            test_writer.add_scalar('f1', max_f1, epoch_idx)

        print('The IoU score of %d-view-input is %.4f\n' % (cfg.CONST.N_VIEWS_RENDERING, max_iou))
        print('The f1 score of %d-view-input is %.4f\n' % (cfg.CONST.N_VIEWS_RENDERING, max_f1))

        return max_iou



