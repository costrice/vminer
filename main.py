import os
import random
import shutil
import sys

# enable openexr
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, Tuple

import configargparse
import numpy as np
import torch
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm

import options
from internal import (
    datasets,
    evaluate,
    export_mesh,
    global_vars,
    losses,
    metrics,
    renderers,
    scenes,
    utils,
)


def seed_everything(seed: int):
    """Set the seed for all random number generators."""
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    # os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr_with_name(
    optimizer: torch.optim.Optimizer,
    param_name: str,
):
    """Get the learning rate of a parameter group with the given name."""
    for param_group in optimizer.param_groups:
        if param_name == param_group["name"]:
            return param_group["lr"]
    raise ValueError(f"No param group with name {param_name} in optimizer.")


def get_command_line_and_config_file_args(
    parser: configargparse.ArgumentParser,
    args: configargparse.Namespace,
) -> Tuple[configargparse.Namespace, configargparse.Namespace]:
    """Separate the command line args and config file args from parsed args.

    Args:
        parser: the parser that parses the command line and config file
        args: the parsed args

    Returns:
        A tuple of two configargparse.Namespace objects, which are

        - the command line args
        - the config file args
    """
    args_source = parser.get_source_to_settings_dict()

    config_args = configargparse.Namespace()
    cli_args = configargparse.Namespace()
    for source, arg_dict in args_source.items():
        if source.startswith("config_file"):
            for key, value in arg_dict.items():
                setattr(config_args, key, getattr(args, key))
        if source == "command_line":
            arg_list = arg_dict[""][1]
            for cmd_item in arg_list:
                if cmd_item.startswith("--"):
                    cmd_item = cmd_item[2:]
                    setattr(cli_args, cmd_item, getattr(args, cmd_item))
            # for key, value in arg_dict.items():
            #     setattr(cli_args, key, getattr(args, key))

    return cli_args, config_args


def print_loss(loss_dict: Dict[str, torch.Tensor], weights: Dict[str, float]):
    """Not used."""
    it = global_vars.cur_iter
    print(f"Iteration {it}:")
    for key, value in loss_dict.items():
        if key == "total":
            print(f"{key}: {value.detach().item():.3f}")
        else:
            print(f"{key}: {value.detach().item():.3f} (weight = " f"{weights[key]})")


def enable_pbr(
    scene: scenes.SceneModel,
    loss_fn: losses.LossGatherer = None,
    metrics_fn: metrics.MetricsGatherer = None,
    optimizer: torch.optim.Optimizer = None,
):
    """Enable physically-based rendering (PBR) in the scene.

    Enable PBR by adding PBR-related loss terms and metrics, and adding
    material-related parts to the scene.field.

    Args:
        scene: the scene to enable PBR
        loss_fn: the loss gatherer to add PBR-related loss terms to
        metrics_fn: the metrics gatherer to add PBR-related metrics to
        optimizer: the optimizer to add PBR-related parameters to

    Returns:
        A tuple of objects, which are

        - scene (scenes.SceneModel): the scene with PBR-related parts added
        - loss_fn (losses.LossGatherer): the loss gatherer with PBR-related
            loss terms added
        - metrics_fn (metrics.MetricsGatherer): the metrics gatherer with
            PBR-related metrics added
        - optimizer (torch.optim.Optimizer): the optimizer with PBR-related
            parameters added
    """
    args = global_vars.args
    global_vars.pbr_enabled = True
    print(f"Enabling PBR...")

    if loss_fn is not None:
        loss_fn.update_weights(
            {
                "pbr_rgb_mse": args.pbr_rgb_weight,
                "reg_light_intensity_l1": args.light_intensity_l1_weight,
                "self_consistency_mse": args.self_consistency_weight,
                "material_smoothness_mse": args.material_smoothness_weight,
            }
        )
        if args.fix_shape_and_radiance_in_pbr:
            # remove old loss terms
            loss_fn.update_weights(
                {
                    "nr_rgb_mse": 0,
                    "alpha_l1": 0,
                    "eikonal_mse": 0,
                }
            )
        print(f"Current loss terms:" f"\n {loss_fn.loss_weights}")

    global_vars.need_jittered = args.material_smoothness_weight > 1e-6

    if metrics_fn is not None:
        pbr_metric_lists = [
            "pbr_rgb",
            "pbr_rgb_diff",
            "pbr_rgb_spec",
            "albedo",
        ]
        metrics_fn.extend_measuring_list(pbr_metric_lists)
        print(f"Current metrics:" f"\n {metrics_fn.measuring_list}")

    # add PBR-related parameters to optimizer
    scene.enable_pbr()
    param_groups_pbr = scene.get_optim_param_groups()
    if optimizer is not None:
        # build a new optimizer
        optimizer_new = torch.optim.AdamW(
            param_groups_pbr,
            betas=(0.9, 0.99),
            weight_decay=1e-4,
        )
        old_param_names = [
            param_group["name"] for param_group in optimizer.param_groups
        ]
        # reduce the learning rate of existing field parameters
        for param_group in optimizer_new.param_groups:
            if param_group["name"] in old_param_names:
                param_group["lr"] *= args.lr_geometry_final / args.lr_geometry_init
    else:
        optimizer_new = None

    return scene, loss_fn, metrics_fn, optimizer_new


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_iters: int,
    end_iters: int,
    lr_decay_target_ratio: float,
):
    """Build the learning rate scheduler."""
    # build lr scheduler: warmup + exponential decay
    lr_schedulers = []
    # milestones = []
    # add warmup
    lr_schedulers.append(
        torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1, total_iters=warmup_iters)
    )
    # milestones.append(warmup_iters)
    # add exponential decay
    lr_decay_factor = lr_decay_target_ratio ** (1 / (end_iters - warmup_iters))
    lr_schedulers.append(
        torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay_factor)
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, lr_schedulers, milestones=[warmup_iters]
    )

    return lr_scheduler


def reconstruct(
    scene: scenes.SceneModel,
    dataset_train: datasets.VMINerDataset,
    dataset_valid_view: datasets.VMINerDataset,
    dataset_test_relight: datasets.VMINerDataset,
    metrics_fn: metrics.MetricsGatherer,
    optimizer_state: Dict = None,
) -> scenes.SceneModel:
    """
    Reconstruct the scene from the input images.

    Args:
        scene: the loaded or initialized scene
        dataset_train: the training dataset
        dataset_valid_view: the validation dataset for novel view synthesis.
            This dataset should have the same lighting condition as the
            training dataset.
        dataset_test_relight: the test dataset for novel light synthesis.
            This dataset should have different lighting conditions from the training dataset, but within the same scene.
        metrics_fn: the metrics gatherer that measures the metrics given the ground truth and the predicted values
        optimizer_state: loaded optimizer state dict

    Returns:
        The reconstructed scene
    """
    args = global_vars.args
    visual_dir = global_vars.visual_dir
    ckpt_dir = global_vars.ckpt_dir
    writer = global_vars.writer

    # build optimizer
    # param_groups =
    optimizer = torch.optim.AdamW(  # Adam with weight decay
        scene.get_optim_param_groups(),
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )
    lr_scheduler = get_lr_scheduler(
        optimizer,
        warmup_iters=args.lr_warmup_iters,
        end_iters=args.enable_pbr_after,
        lr_decay_target_ratio=args.lr_geometry_final / args.lr_geometry_init,
    )

    # build loss fn
    loss_fn = losses.LossGatherer(
        {
            "nr_rgb_mse": 1.0,
            "alpha_l1": args.silhouette_weight,
            "eikonal_mse": args.eikonal_weight,
            "reg_light_color_balance": args.light_white_balance_weight,
            "normal_smoothness_mse": args.normal_smoothness_weight,
        }
    )
    global_vars.need_jittered = args.normal_smoothness_weight > 1e-6

    global_vars.pbr_enabled = False  # at the beginning, PBR is disabled
    filter_ray_at = args.filter_ray_at
    update_grad_at = torch.arange(0, args.n_iters, args.update_grad_every).tolist()
    pbar_refresh_at = torch.arange(0, args.n_iters, args.pbar_refresh_rate).tolist()
    eval_at = (
        torch.arange(args.eval_every, args.n_iters, args.eval_every).tolist()
        if args.eval_every < args.n_iters
        else []
    )
    save_ckpt_at = (
        torch.arange(
            args.save_ckpt_every,
            args.n_iters - args.save_ckpt_every // 10,
            args.save_ckpt_every,
        ).tolist()
        if args.save_ckpt_every < args.n_iters * 0.9
        else []
    )

    begin_iter = global_vars.cur_iter
    # if begin_iter is not 0, do some initialization
    if begin_iter > 0:
        # pop up the **_at lists
        while filter_ray_at and filter_ray_at[0] <= begin_iter:
            filter_ray_at.pop(0)
        while update_grad_at and update_grad_at[0] <= begin_iter:
            update_grad_at.pop(0)
        while pbar_refresh_at and pbar_refresh_at[0] <= begin_iter:
            pbar_refresh_at.pop(0)
        while eval_at and eval_at[0] <= begin_iter:
            eval_at.pop(0)
        while save_ckpt_at and save_ckpt_at[0] <= begin_iter:
            save_ckpt_at.pop(0)
        if begin_iter > args.enable_pbr_after:
            scene, loss_fn, metrics_fn, optimizer = enable_pbr(
                scene, loss_fn, metrics_fn, optimizer
            )
            # get lr scheduler
            lr_scheduler = get_lr_scheduler(
                optimizer,
                warmup_iters=args.lr_warmup_iters,
                end_iters=args.n_iters - args.enable_pbr_after,
                lr_decay_target_ratio=args.lr_material_final / args.lr_material_init,
            )

        print(
            f"\nBegin training at iteration {begin_iter}. Thus, "
            f"\nfilter_ray_at = {filter_ray_at}. "
        )

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # get training rays
    all_cond_rays = dataset_train.all_cond_rays
    # filter out invalid rays outside the bbox
    all_cond_rays = scene.filter_rays(
        all_cond_rays, use_ray_opacity=False, random_retention_ratio=0
    )

    sampler = datasets.TrainingRaySampler(
        all_cond_rays, args.batch_size_train, device=device, drop_last=True
    )

    # set AMP
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    iters = range(begin_iter, args.n_iters)
    train_pbar = tqdm(
        iters, file=sys.stdout, desc="Training", total=args.n_iters, initial=begin_iter
    )
    optimizer.zero_grad()
    for it in train_pbar:
        scene.train()
        global_vars.training = True
        global_vars.cur_iter = it

        # logging
        # writer.add_scalar(
        #     'LR/encoding',
        #     get_lr_with_name(optimizer, 'geo_encoding')
        #     if not global_vars.pbr_enabled
        #     else get_lr_with_name(optimizer, 'mat_encoding'),
        #     it)
        writer.add_scalar("inv_variance", scene.field.variance.inv_s.item(), it)

        # update occupancy grid
        if scene.occ_grid is not None and scene.occ_grid.training:
            scene.occ_grid.update_every_n_steps(
                step=it, occ_eval_fn=scene.field.closure_occ_eval_fn(), n=16
            )  # default n=16

        # sample batches
        inp_ray_attrs = next(sampler)
        n_rays = inp_ray_attrs["n_rays"]

        if args.white_bg:
            bg = torch.ones(n_rays, 1, device=device)
        else:
            bg = torch.zeros(n_rays, 1, device=device)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            pred_ray_attrs = renderers.render(
                scene,
                inp_ray_attrs,
                is_under_novel_light=False,
                do_pbr=global_vars.pbr_enabled,
                white_bg=bg,
            )

        # add white background
        inp_ray_attrs["rgb"] += (1 - inp_ray_attrs["alpha"]) * bg

        # compute losses
        loss = loss_fn(
            scene,
            pred_ray_attrs,
            inp_ray_attrs,
        )
        # cope with nan or inf loss
        if not (torch.isnan(loss["total"]) or torch.isinf(loss["total"])):
            # log loss to writer
            for key, value in loss.items():
                writer.add_scalar("Loss/" + key, value.detach().item(), it)

            # backprop
            if args.use_amp:
                scaler.scale(loss["total"]).backward()
            else:
                loss["total"].backward()

        if update_grad_at and it >= update_grad_at[0]:
            update_grad_at.pop(0)
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # logging
        if pbar_refresh_at and it >= pbar_refresh_at[0]:
            pbar_refresh_at.pop(0)
            metric_dict = {}
            if not (global_vars.pbr_enabled and args.fix_shape_and_radiance_in_pbr):
                rgb_psnr = -10 * torch.log10(loss["nr_rgb_mse"].detach()).item()
                metric_dict["nr_rgb_PSNR"] = rgb_psnr
            if global_vars.pbr_enabled:
                pbr_rgb_psnr = -10 * torch.log10(loss["pbr_rgb_mse"].detach()).item()
                metric_dict["pbr_rgb_PSNR"] = pbr_rgb_psnr
            for key, value in metric_dict.items():
                metric_dict[key] = f"{value:.2f}"
                writer.add_scalar(f"Metric/train/{key}", value, it)
            train_pbar.set_postfix(metric_dict)

        # visualize
        if eval_at and it >= eval_at[0]:
            eval_at.pop(0)
            if it >= args.eval_under_novel_light_after:
                if dataset_test_relight is not None:
                    # test under novel light
                    evaluate.evaluate_scene(
                        scene,
                        dataset_test=dataset_test_relight,
                        under_novel_light=True,
                        do_pbr=True,
                        metrics_fn=metrics_fn,
                        show_progress=False,
                        save_dir=visual_dir / "novel_light",
                    )
            else:
                if dataset_valid_view is not None:
                    #  test under only novel view without novel light
                    evaluate.evaluate_scene(
                        scene,
                        dataset_test=dataset_valid_view,
                        under_novel_light=False,
                        do_pbr=global_vars.pbr_enabled,
                        metrics_fn=metrics_fn,
                        show_progress=False,
                        save_dir=visual_dir / "novel_view",
                    )

        # save checkpoint
        if save_ckpt_at and it >= save_ckpt_at[0] or it == args.enable_pbr_after:
            if not it == args.enable_pbr_after:
                ckpt_path = ckpt_dir / f"ckpt_latest_{it:05d}.pth"
            else:
                ckpt_path = ckpt_dir / "ckpt_at_pbr_enabled.pth"
            if it >= save_ckpt_at[0]:
                save_ckpt_at.pop(0)
            # ckpt_path = ckpt_dir / f'ckpt_{it:05d}.pth'
            # delete last latest
            if not global_vars.pbr_enabled:
                for p in ckpt_dir.glob("ckpt_latest*"):
                    p.unlink()

            ckpt = {
                "iter": it,
                "optimizer": optimizer.state_dict(),
            }
            ckpt.update(scene.get_save_dict())
            torch.save(ckpt, ckpt_path)
            # print absolute path
            print("Latest checkpoint saved to ", ckpt_path.absolute())

        # update lr
        lr_scheduler.step()

        # if decrease_reg_loss_at and it >= decrease_reg_loss_at[0]:
        #     decrease_reg_loss_at.pop(0)
        #     loss_fn = update_reg_loss(loss_fn)

        if not global_vars.pbr_enabled and it >= args.enable_pbr_after:
            scene, loss_fn, metrics_fn, optimizer = enable_pbr(
                scene, loss_fn, metrics_fn, optimizer
            )
            # get lr scheduler
            lr_scheduler = get_lr_scheduler(
                optimizer,
                warmup_iters=args.lr_warmup_iters,
                end_iters=args.n_iters - args.enable_pbr_after,
                lr_decay_target_ratio=args.lr_material_final / args.lr_material_init,
            )

        # filter out training rays
        if filter_ray_at and it >= filter_ray_at[0]:
            filter_ray_at.pop(0)
            all_cond_rays = scene.filter_rays(all_cond_rays, use_ray_opacity=True)
            scene.train()
            sampler = datasets.TrainingRaySampler(
                all_cond_rays, args.batch_size_train, device=device, drop_last=True
            )
            print("Training rays filtered.")

    print("Training finished.")
    train_pbar.close()
    global_vars.training = False
    global_vars.cur_iter = args.n_iters

    # save final checkpoint
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"ckpt_final.pth"
    ckpt = {"iter": args.n_iters}
    ckpt.update(scene.get_save_dict())
    torch.save(ckpt, ckpt_path)
    print("Final checkpoint saved to ", ckpt_path.absolute())

    print("Training finished.")

    return scene


def build_scene_from_file(
    ckpt_path: str,
    cli_args: configargparse.Namespace,
    config_file_args: configargparse.Namespace,
):
    """Load the scene from the dictionary.

    The old arguments saved in the checkpoint will be overwritten by the
    new arguments parsed from the command line and the config file. The
    command line arguments have the highest priority.

    Args:
        ckpt_path: the path to the checkpoint file. The checkpoint file
            should be a dictionary containing the following keys:

            - model_args (Dict): the arguments of the model
            - state_dict (Dict): the state dict of the model
            - iter (int): the iteration number of the model

        cli_args: the arguments parsed from the command line
        config_file_args: the arguments parsed from the config file

    Returns:
        A tuple containing:

        - model (models.TensorBase): the loaded model
        - iter (int): the iteration number of the model
    """
    ckpt_path = Path(ckpt_path)
    device = global_vars.device

    loaded_dict = torch.load(ckpt_path, map_location="cpu")
    # load model arguments
    model_args = loaded_dict["model_args"]
    model_args.update({"device": device})
    # overwrite args
    overwritten_args = model_args["args"]
    for key, value in vars(config_file_args).items():
        setattr(overwritten_args, key, value)
    for key, value in vars(cli_args).items():
        setattr(overwritten_args, key, value)
    global_vars.args = overwritten_args
    # build model from arguments
    print("\nModel arguments saved in the checkpoint:")
    pprint(model_args)
    scene = scenes.SceneModel(
        field_type=model_args["field_type"],
        args=overwritten_args,
        device=model_args["device"],
        n_far_lights=model_args["n_far_lights"],
        n_near_lights=model_args["n_near_lights"],
        near_light_pos_types=model_args["near_light_pos_types"],
        aabb=model_args["aabb"],
    )

    # set states
    global_vars.cur_iter = loaded_dict["iter"]
    if global_vars.cur_iter >= args.enable_pbr_after:
        scene = enable_pbr(scene)[0]

    # load state dict
    scene.load_state_dict(loaded_dict["state_dict"])

    # get optimizer state
    if "optimizer" in loaded_dict:
        optimizer_state = loaded_dict["optimizer"]
    else:
        optimizer_state = None

    return scene, loaded_dict["iter"], optimizer_state


def main(
    args: configargparse.Namespace,
    cli_args: configargparse.Namespace,
    cfg_file_args: configargparse.Namespace,
):
    """Main function for training, exporting mesh, or evaluating."""
    # set global variables
    global_vars.args = args
    global_vars.training = False
    global_vars.device = device

    # initialize logging
    exp_name = args.exp_name
    exp_name_timestamped = exp_name + f"-{utils.current_time()}"
    if args.add_timestamp:
        exp_fullname = exp_name_timestamped
    else:
        exp_fullname = exp_name

    writer = SummaryWriter(
        log_dir="runs/" + exp_name_timestamped,
    )
    global_vars.writer = writer
    writer.add_scalar("seed", args.seed, 0)

    # # TODO: add multi-gpu support
    # if args.log_to_wandb:
    #     wandb.init(
    #         project='NearLight',
    #         name=exp_fullname,
    #         config=vars(args),
    #     )

    log_dir = Path(args.out_dir) / exp_fullname
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Log directory: {log_dir}")

    if args.reconstruct:
        ckpt_dir = log_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint directory: {ckpt_dir}")
    else:
        ckpt_dir = None

    if args.test_novel_view or args.test_novel_light:
        visual_dir = log_dir / "visuals"
        visual_dir.mkdir(parents=True, exist_ok=True)
        print(f"Visual directory: {visual_dir}")
    else:
        visual_dir = None

    # if export mesh, create mesh directory
    if args.export_mesh:
        mesh_dir = log_dir / "mesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        print(f"Mesh directory: {mesh_dir}")
    else:
        mesh_dir = None

    global_vars.log_dir = log_dir
    global_vars.ckpt_dir = ckpt_dir
    global_vars.visual_dir = visual_dir
    global_vars.mesh_dir = mesh_dir

    # copy the config file to the log directory
    shutil.copy(args.config, log_dir)

    # get dataset
    data_root = Path(args.data_root)
    dataset_train = None
    if args.reconstruct:
        dataset_train = datasets.VMINerDataset(
            data_root / args.train_data_dir,
            split="train",
            device=device,
            remove_ignored=args.remove_ignored_pixels,
        )

    dataset_valid_view = None
    if args.test_novel_view:
        dataset_valid_view = datasets.VMINerDataset(
            data_root / args.novel_view_test_data_dir,
            split="test",
            read_extra=True,
            device=device,
        )

    dataset_test_relight = None
    if args.test_novel_light:
        dataset_test_relight = datasets.VMINerDataset(
            data_root / args.novel_light_test_data_dir,
            split="test",
            read_extra=True,
            read_gt_far_lights=True,
            device=device,
        )

    if args.ckpt is not None:
        # load from checkpoint
        scene, begin_iter, optimizer_state = build_scene_from_file(
            args.ckpt, cli_args, cfg_file_args
        )
    else:
        # initialize from scratch
        scene = scenes.SceneModel(
            field_type=args.field_type,
            args=args,
            device=device,
            n_far_lights=dataset_train.n_far_lights,
            n_near_lights=dataset_train.n_near_lights,
            near_light_pos_types=dataset_train.near_lights_meta["pos_type"],
        )
        begin_iter = 0
        optimizer_state = None

    if args.test_novel_view or args.test_novel_light:
        # build metrics fn
        metrics_fn = metrics.MetricsGatherer(
            measuring_list=[
                "nr_rgb",
                "normal",
            ],
            net_device=scene.device,
            within_mask=False,
        )
        global_vars.cur_iter = begin_iter
        if begin_iter >= args.enable_pbr_after:
            metrics_fn = enable_pbr(scene, metrics_fn=metrics_fn)[2]
    else:
        metrics_fn = None

    if args.test_before_train:
        if args.test_novel_view:
            evaluate.evaluate_scene(
                scene,
                dataset_test=dataset_valid_view,
                under_novel_light=False,
                do_pbr=global_vars.pbr_enabled,
                metrics_fn=metrics_fn,
                show_progress=True,
                save_dir=global_vars.visual_dir / "novel_view",
            )
        if args.test_novel_light:
            evaluate.evaluate_scene(
                scene,
                dataset_test=dataset_test_relight,
                under_novel_light=True,
                do_pbr=True,
                metrics_fn=metrics_fn,
                show_progress=True,
                save_dir=global_vars.visual_dir / "novel_light",
            )

    if args.reconstruct:
        # train the model
        scene = reconstruct(
            scene,
            dataset_train,
            dataset_valid_view,
            dataset_test_relight,
            metrics_fn,
            optimizer_state,
        )

    if args.export_mesh:
        export_mesh.export_mesh(scene)

    # evaluate after training
    if args.test_after_train:
        if args.test_novel_view:
            evaluate.evaluate_scene(
                scene,
                dataset_test=dataset_valid_view,
                under_novel_light=False,
                do_pbr=global_vars.pbr_enabled,
                metrics_fn=metrics_fn,
                show_progress=True,
                save_dir=visual_dir / "novel_view",
            )
        if args.test_novel_light:
            evaluate.evaluate_scene(
                scene,
                dataset_test=dataset_test_relight,
                under_novel_light=True,
                do_pbr=True,
                metrics_fn=metrics_fn,
                show_progress=True,
                save_dir=visual_dir / "novel_light",
            )

    # wandb.finish()


if __name__ == "__main__":
    # disable shared GPU memory
    torch.cuda.set_per_process_memory_fraction(0.9)
    # # detect anomaly
    # torch.autograd.set_detect_anomaly(True, check_nan=True)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # parse arguments
    parser = options.construct_parser()
    args = parser.parse_args()
    print("----------Used Options----------")
    print(parser.format_values())

    # set -1 in args to n_iters + 1
    for key in options.iter_number_options:
        if getattr(args, key) == -1:
            setattr(args, key, args.n_iters + 1)

    # set up iter numbers
    prolong_multi = args.prolong_multi
    if prolong_multi > 1:
        for key in options.iter_number_options:
            if isinstance(getattr(args, key), list):
                setattr(args, key, [it * prolong_multi for it in getattr(args, key)])
            else:
                setattr(args, key, getattr(args, key) * prolong_multi)
        print(f"Prolonging training by {prolong_multi}x. Affected: ")
        print(options.iter_number_options)

    # check args
    assert (
        args.enable_pbr_after
        <= args.enable_sec_vis_for_pbr_after
        <= args.enable_indirect_after
    ), (
        "enable_sec_vis_for_pbr_after must be >= enable_pbr_after, "
        "and enable_indirect_after must be >= enable_sec_vis_for_pbr_after."
    )
    assert (
        args.eval_under_novel_light_after >= args.enable_pbr_after
    ), "eval_under_novel_light_after must be >= enable_pbr_after."
    # if args.use_compile:
    #     raise NotImplementedError('torch.compile can not work yet.')
    if args.test_novel_view:
        assert args.novel_view_test_data_dir is not None, (
            "novel_view_test_data_dir must be specified when testing "
            "under novel view."
        )
    if args.test_novel_light:
        assert args.novel_light_test_data_dir is not None, (
            "novel_light_test_data_dir must be specified when testing "
            "under novel light."
        )
    if args.reconstruct:
        assert (
            args.train_data_dir is not None
        ), "train_data_dir must be specified when reconstruction is needed."
        if args.eval_under_novel_light_after <= args.n_iters:
            if not args.test_novel_light:
                print(
                    "Warning: test_novel_light is set to False, "
                    "but eval_under_novel_light_after <= n_iters. "
                    "Thus, will not eval after this iteration."
                )
        if args.eval_under_novel_light_after > 0:
            if not args.test_novel_view:
                print(
                    "Warning: test_novel_view is set to False, "
                    "but eval_under_novel_light_after > 0. "
                    "Thus, will not eval before this iteration."
                )
    if args.test_before_train:
        assert (
            args.reconstruct
        ), "test_before_train can only be used when reconstruct is True."

    # set random seed
    if args.seed is None:
        args.seed = int(time.time() * 1000) % 1000
    seed_everything(args.seed)
    print(f"Random seed: {args.seed}.")

    # TODO: add support for multi-gpu training
    N_GPUS = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running with {N_GPUS} GPU(s)...")

    # get command line and config file args
    cli_args, cfg_file_args = get_command_line_and_config_file_args(parser, args)

    main(args, cli_args, cfg_file_args)
