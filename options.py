import configargparse

iter_number_options = []


def construct_parser():
    parser = configargparse.ArgumentParser()
    
    # path options
    parser.add_argument(
        '--config', is_config_file=True,
        help='config file path'
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        help='experiment name'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='random seed'
    )
    parser.add_argument(
        '--data_root',
        type=str, default=r'./data',
        help='path to the root of the dataset'
    )
    parser.add_argument(
        '--train_data_dir',
        type=str,
        help='path to training data relative to data_root'
    )
    parser.add_argument(
        '--novel_view_test_data_dir',
        type=str,
        help='path to test data for novel view synthesis relative to data_root'
    )
    parser.add_argument(
        '--novel_light_test_data_dir',
        type=str,
        help='path to test data for rendering with novel light relative to '
             'data_root'
    )
    parser.add_argument(
        '--out_dir',
        type=str, default='./log',
        help='where to store ckpts and logs'
    )
    parser.add_argument(
        '--add_timestamp', action='store_true',
        help='add timestamp to save dir'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        help='specify the checkpoint to load'
    )
    
    # main options
    parser.add_argument(
        '--reconstruct',
        action='store_true', default=False,
        help='whether to reconstruct the scene (i.e. train)'
    )
    parser.add_argument(
        '--test_novel_view',
        action='store_true', default=False,
        help='whether to test under novel view after training. '
             'need to specify novel_view_test_data_dir'
    )
    parser.add_argument(
        '--test_novel_light',
        action='store_true', default=False,
        help='whether to test under both novel view and light after training. '
             'need to specify novel_light_test_data_dir'
    )
    parser.add_argument(
        '--test_before_train',
        action='store_true', default=False,
        help='whether to evaluate before training'
    )
    parser.add_argument(
        '--test_after_train',
        action='store_true', default=False,
        help='whether to evaluate after training'
    )
    parser.add_argument(
        '--export_mesh',
        action='store_true', default=False,
        help='whether to export mesh at last'
    )
    
    # training schedule
    parser.add_argument(
        '--remove_ignored_pixels',
        action='store_true', default=False,
        help='Remove pixels that are marked as ignored (such as transient objects) '
             'from the training data.'
    )
    parser.add_argument(
        '--prolong_multi',
        type=int, default=1,
        help='prolong the training schedule by this factor'
    )
    parser.add_argument(
        '--n_iters',
        type=int, default=20000,
        help='number of iterations to train'
    )
    iter_number_options.append('n_iters')
    parser.add_argument(
        '--lr_warmup_iters',
        type=int, default=200,
        help='number of iterations for lr warmup'
    )
    iter_number_options.append('lr_warmup_iters')
    parser.add_argument(
        '--white_bg',
        action='store_true', default=False,
        help='render with white background in training (if random bg is not '
             'activated yet) and testing. '
    )
    parser.add_argument(
        '--filter_ray_at',
        type=int, nargs='*',
        help='filter out training rays using AABB and alpha mask at which '
             'iterations'
    )
    iter_number_options.append('filter_ray_at')
    parser.add_argument(
        '--enable_pbr_after',
        type=int, default=-1,
        help='Enable PBR after this iteration. This should typically be set '
             'after updating the alpha mask volume, otherwise it will be very '
             'slow. -1 means never enable PBR.'
    )
    iter_number_options.append('enable_pbr_after')
    parser.add_argument(
        '--enable_sec_vis_for_pbr_after',
        type=int, default=-1,
        help='Enable secondary visibility for PBR after this iteration. '
             'This should be bigger than enable_pbr_after. -1 means never.'
    )
    iter_number_options.append('enable_sec_vis_for_pbr_after')
    # parser.add_argument(
    #     '--enable_sec_vis_for_neur_near_after',
    #     type=int, default=-1,
    #     help='Enable secondary visibility for near light neural rendering '
    #          'after this iteration. -1 means never.')
    # iter_number_options.append('enable_sec_vis_for_neur_near_after')
    parser.add_argument(
        '--enable_indirect_after',
        type=int, default=-1,
        help='Enable indirect lighting for PBR after this iteration. '
             'This should be bigger than enable_sec_vis_for_pbr_after. '
             '-1 means never enable indirect lighting.'
    )
    iter_number_options.append('enable_indirect_after')
    parser.add_argument(
        '--fix_shape_and_radiance_in_pbr',
        action='store_true', default=False,
        help='whether to fix shape and radiance after enabling PBR'
    )
    parser.add_argument(
        '--detach_secondary',
        action='store_true', default=False,
        help='whether to detach secondary visibility and indirect lighting '
             'from the computation graph, usually for speeding up training.'
    )
    parser.add_argument(
        '--eval_under_novel_light_after',
        type=int, default=-1,
        help='Evaluate under novel light after this iteration in training. '
             '-1 means never.'
    )
    iter_number_options.append('eval_under_novel_light_after')
    parser.add_argument(
        '--anneal_end_iter',
        type=int, default=0,
        help='Annealing for some factor that is optimized during training. '
             '0 means no annealing.'
    )
    iter_number_options.append('anneal_end_iter')
    
    # logging options
    parser.add_argument(
        '--pbar_refresh_rate',
        type=int, default=10,
        help='refresh rate of progress bar (showing metrics)'
    )
    parser.add_argument(
        '--save_ckpt_every',
        type=int, default=-1,
        help='save checkpoint every N iterations. If <=0, only save the '
             'checkpoint at the end of training'
    )
    iter_number_options.append('save_ckpt_every')
    parser.add_argument(
        '--n_visual_train',
        type=int, default=5,
        help='number of images to be visualized during training'
    )
    parser.add_argument(
        '--n_visual_test',
        type=int, default=5,
        help='number of images to be visualized after training'
    )
    parser.add_argument(
        '--eval_every',
        type=int, default=-1,
        help='frequency of visualizing the image during training. If <= 0, '
             'only visualize at the end of training'
    )
    iter_number_options.append('eval_every')
    
    # field options
    parser.add_argument(
        '--field_type',
        type=str, default='InstantNSR', choices=['InstantNSR'],
        help='type of neural field to use'
    )
    parser.add_argument(
        '--n_geo_feature_dims',
        type=int, default=13,
        help='number of geometric features produced by the sdf decoder'
    )
    parser.add_argument(
        '--n_light_code_dims',
        type=int, default=16,
        help='dimension of light code for each light condition'
    )
    parser.add_argument(
        '--hash_grid_n_levels',
        type=int, default=16,
        help='number of levels in the hash grid'
    )
    parser.add_argument(
        '--hash_grid_coarsest_reso',
        type=int, default=16,
        help='coarsest resolution of the hash grid'
    )
    parser.add_argument(
        '--hash_grid_finest_reso',
        type=int, default=2048,
        help='finest resolution of the hash grid'
    )
    parser.add_argument(
        '--brdf_type',
        type=str, default='Disney', choices=['GGX', 'Disney'],
        help='type of BRDF to use'
    )
    parser.add_argument(
        '--sphere_init_radius',
        type=float, default=0.5,
        help='radius of the initial sphere output by SDF decoder'
    )
    
    # light options
    parser.add_argument(
        '--far_light_type',
        type=str, default='sg', choices=['sg', 'pixel'],
        help='type of far light used in training'
    )
    parser.add_argument(
        '--n_sg_lobes',
        type=int, default=128,
        help='number of SG lobes for each far light condition'
    )
    parser.add_argument(
        '--near_light_type',
        type=str, default='sh_point', choices=['sh_point'],
        help='type of near light'
    )
    parser.add_argument(
        '--near_light_sh_order',
        type=int, default=0,
        help='SH order for near light isotropic radiance'
    )
    parser.add_argument(
        '--envmap_hw',
        type=int, nargs=2, default=[32, 64],
        help='height and width of the environment map used or '
             'baked for output'
    )
    
    # loss weights
    parser.add_argument(
        '--silhouette_weight',
        type=float, default=0.0,
        help='weight for silhouette loss'
    )
    parser.add_argument(
        '--pbr_rgb_weight',
        type=float, default=1.0,
        help='weight for physical based rendered RGB loss'
    )
    parser.add_argument(
        '--eikonal_weight',
        type=float, default=1e-4,
        help='weight for SDF eikonal loss'
    )
    parser.add_argument(
        '--self_consistency_weight',
        type=float, default=0.1,
        help='weight for consistency between PB and neural rendered RGB for '
             'near and far lights (separately computed then summed up)'
    )
    parser.add_argument(
        '--material_smoothness_weight',
        type=float, default=0,
        help='weight for material\'s smoothness loss'
    )
    parser.add_argument(
        '--normal_smoothness_weight',
        type=float, default=0,
        help='weight for normal\'s smoothness loss'
    )
    parser.add_argument(
        '--light_intensity_l1_weight',
        type=float, default=0.0,
        help='weight for light intensity l1 norm loss'
    )
    parser.add_argument(
        '--light_white_balance_weight',
        type=float, default=0.0,
        help='weight for light white balance loss'
    )
    parser.add_argument(
        '--jitter_range',
        type=float, default=0.01,
        help='range for jittering position when computing material smoothness '
             'loss'
    )
    
    # batch size options
    parser.add_argument(
        '--batch_size_train',
        type=int, default=4096,
        help='ray chunk size for training'
    )
    parser.add_argument(
        '--update_grad_every',
        type=int, default=1,
        help='update gradient every N iterations (to sum up gradients from '
             'multiple light conditions). '
    )
    parser.add_argument(
        '--chunk_size_nr_train',
        type=int, default=65536,
        help='chunk size for tensor field rendering during training'
    )
    parser.add_argument(
        '--chunk_size_nr_test',
        type=int, default=1048576,
        help='chunk size for tensor field rendering during testing'
    )
    parser.add_argument(
        '--chunk_size_pbr_train',
        type=int, default=4096,
        help='chunk size for PBR during training'
    )
    parser.add_argument(
        '--chunk_size_pbr_test',
        type=int, default=65536,
        help='chunk size for PBR during testing'
    )
    
    # learning rate options
    parser.add_argument(
        '--lr_light',
        type=float, default=2e-2,
        help='learning rate for scene light parameters.'
    )
    parser.add_argument('--lr_geometry_init', type=float)
    parser.add_argument('--lr_geometry_final', type=float)
    parser.add_argument('--lr_variance_init', type=float)
    parser.add_argument('--lr_material_init', type=float)
    parser.add_argument('--lr_material_final', type=float)
    
    # rendering options
    parser.add_argument(
        '--use_occ_grid',
        action='store_true', default=False,
        help='whether to use NeRFAcc occuapncy grid to accelerate '
             'ray marching'
    )
    parser.add_argument(
        '--ray_march_weight_thres',
        type=float, default=2.5e-3,
        help='threshold for masking points in ray marching'
    )
    parser.add_argument(
        '--alpha_mask_thres',
        type=float, default=2.5e-3,
        help='threshold for creating alpha mask volume'
    )
    parser.add_argument(
        '--scene_aabb',
        type=float, nargs=6,
        default=[-1.5, -1.5, -1.5, 1.5, 1.5, 1.5],
        help='AABB of the scene, used for sampling points inside the scene '
             'and will be automatically determined and updated during '
             'training. The format is [xmin, ymin, zmin, xmax, ymax, zmax]. '
    )
    parser.add_argument(
        '--aabb_margin_ratio',
        type=float, default=0.05,
        help='leave additional margin when shrinking AABB. '
    )
    parser.add_argument(
        '--primary_near_far',
        type=float, nargs=2,
        default=[0.5, 20.0],
        help='near and far bound for primary rays. It is used only for '
             'clipping the ray marching distance because sampling is performed '
             'using AABB.'
    )
    parser.add_argument(
        '--n_samples_per_ray',
        type=int, default=1024,
        help='the maximum number of samples per ray.'
    )
    parser.add_argument(
        '--incident_sampling_method',
        type=str, default='importance',
        choices=['importance', 'stratified'],
        help='method for sampling incident directions. Can be importance '
             'sampling or stratified sampling.'
    )
    parser.add_argument(
        '--n_sec_sample_dirs_train',
        type=int, default=16,
        help='number of secondary ray directions to sample for each primary '
             'intersection during training'
    )
    parser.add_argument(
        '--n_sec_sample_dirs_test',
        type=int, default=128,
        help='number of secondary ray directions to sample for each primary '
             'intersection during testing'
    )
    parser.add_argument(
        '--second_near_far',
        type=float, nargs=2,
        default=[0.05, 1.5],
        help='near and far bound for secondary rays'
    )
    parser.add_argument(
        '--max_second_ray_segments',
        type=int, default=96,
        help='the upper limit of sampled points for each secondary rays. The '
             'rule is the same as n_samples_per_ray'
    )
    parser.add_argument(
        '--secondary_as_surface',
        action='store_true', default=False,
        help='if True, query in tensor field a single point (whose depth is '
             'still decided by volume rendering) for its radiance, rather than '
             'volume rendering a ray\'s radiance.'
    )
    parser.add_argument(
        '--use_prefiltered_envmap_for_diffuse',
        action='store_true', default=False,
        help='if True, use prefiltered environment map for diffuse lighting '
             'if available. This is only useful when the environment map is '
             'baked for output.'
    )
    
    # export mesh options
    parser.add_argument(
        '--mesh_grid_reso',
        type=int, default=512,
        help='the grid resolution for isosurface extraction when exporting '
             'mesh. Actual resolution of each dimension depends on the AABB '
             'of the scene.'
    )
    parser.add_argument(
        '--simplify_mesh',
        action='store_true', default=False,
        help='whether to simplify the mesh when exporting mesh'
    )
    parser.add_argument(
        '--simplify_mesh_target_num_faces',
        type=int, default=131072,
        help='the target number of faces after simplification when exporting '
             'mesh'
    )
    parser.add_argument(
        '--mesh_bake_texture',
        action='store_true', default=False,
        help='whether to bake texture maps when exporting mesh'
    )
    parser.add_argument(
        '--mesh_texture_reso',
        type=int, default=4096,
        help='the resolution of the baked texture map when exporting mesh'
    )
    
    # misc options
    parser.add_argument(
        '--use_amp',
        action='store_true', default=False,
        help='whether to use automatic mixed precision'
    )
    parser.add_argument(
        '--log_to_wandb',
        action='store_true', default=False,
        help='Whether to log to wandb'
    )
    
    return parser


def parse_cmd_argument(cmd=None):
    parser = construct_parser()
    
    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()
