import pickle
import argparse
import numpy as np
from tqdm import tqdm
import polyscope as ps
from os import path as osp
from PIL import Image, ImageEnhance
from datasets import build_dataset
from utils.options import parse
from utils.geometry_util import torch2np
from utils.texture_util import *
from utils.visualization_util import *

ps.set_allow_headless_backends(True) 
ps.init()
ps.set_ground_plane_mode("shadow_only")
ps.set_view_projection_mode("orthographic")
ps.set_SSAA_factor(4)
ps.set_window_size(3840, 2160)

def visualize_pipeline(root_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = parse(args.opt, root_path, is_train=False)
    ours_results_root = opt["path"]["results_root"]
    ours_p2p_list = pickle.load(open(osp.join(ours_results_root, "p2p.pkl"), "rb"))
    if not osp.exists(osp.join(ours_results_root, "polyscope_visualization")):
        os.makedirs(osp.join(ours_results_root, "polyscope_visualization"))
    dataset_opt = opt["datasets"].popitem()[1]
    dataset_opt.update({"return_dist": False, "return_elas_evecs": False, "cache": False})
    test_set = build_dataset(dataset_opt)
    orient_calib_R = orientation_calibration_by_dataset(test_set)

    # color and texture
    limbs_colors = np.array([(120, 60, 180), (240, 60, 100), (60, 200, 200), (60, 80, 180), (200, 120, 60), (160, 220, 80)])/255.0 #l/r hands, r/l feet, head, tail: Purple, Pink, Cyan, Blue, Orange, Green
    texture_grid_bw_8 = np.array(Image.open("figures/texture_grid_bw_8.png").convert("RGB")) / 255.0

    for i in tqdm(range(len(test_set))):
        data = test_set[i]
        data_x, data_y = data["first"], data["second"]
        verts_x, verts_y = torch2np(data_x["verts"]) @ orient_calib_R, torch2np(data_y["verts"]) @ orient_calib_R
        faces_x, faces_y = torch2np(data_x["faces"]), torch2np(data_y["faces"])
        evecs_x, evecs_y = torch2np(data_x["evecs"]), torch2np(data_y["evecs"])
        evecs_trans_x, evecs_trans_y = torch2np(data_x["evecs_trans"]), torch2np(data_y["evecs_trans"])
        ps_ours_x = ps.register_surface_mesh("ours_first", verts_x , faces_x, material="wax", smooth_shade=True)
        ps_ours_y = ps.register_surface_mesh("ours_second", verts_y + [1.2,0,0], faces_y, material="wax", smooth_shade=True)

        # predicted correspondence
        p2p = ours_p2p_list[i]

        # texture_grid_bw_8
        Cxy = evecs_trans_y @ evecs_x[p2p]
        Pyx = evecs_y @ Cxy @ evecs_trans_x
        uv_x = generate_tex_coords(verts_x)
        uv_y = Pyx @ uv_x
        ps_ours_x.add_parameterization_quantity("uv", uv_x)
        ps_ours_y.add_parameterization_quantity("uv", uv_y)
        ps_ours_x.add_color_quantity("texture", texture_grid_bw_8, defined_on='texture', param_name="uv", enabled=True)
        ps_ours_y.add_color_quantity("texture", texture_grid_bw_8, defined_on='texture', param_name="uv", enabled=True)
        texture_image = Image.fromarray(ps.screenshot_to_buffer())

        # color limbs
        limbs_indices = limbs_indices_by_dataset(i, data_x, test_set)
        limbs_colors = limbs_colors[:len(limbs_indices)]
        harmonic_colors_x = harmonic_interpolation(verts_x, faces_x, limbs_indices, limbs_colors)
        ps_ours_x.add_color_quantity("colors", harmonic_colors_x, defined_on='vertices', enabled=True)
        ps_ours_y.add_color_quantity("colors", harmonic_colors_x[p2p], defined_on='vertices', enabled=True)
        color_image = Image.fromarray(ps.screenshot_to_buffer())

        # blend color + texture_grid_bw_8
        blended_image = Image.blend(texture_image, color_image, 0.3)
        enhanced_image = ImageEnhance.Color(blended_image).enhance(8)
        final_image = enhanced_image.crop(enhanced_image.getbbox())
        final_image.save(osp.join(ours_results_root, "polyscope_visualization", f"{i}_{data_x['name']}_{data_y['name']}.png"))
        
        # normal transfer
        # normal_x = igl.per_vertex_normals(verts_x, faces_x) * [-1,-1,1]
        # normal_color_x = normal_x + 0.35
        # ps_ours_x.add_color_quantity("normal", normal_color_x, defined_on='vertices', enabled=True)
        # ps_ours_y.add_color_quantity("normal", normal_color_x[p2p], defined_on='vertices', enabled=True)
        # normal_image = Image.fromarray(ps.screenshot_to_buffer())
        # normal_image = normal_image.crop(normal_image.getbbox())
        # normal_image.save(osp.join(ours_results_root, "polyscope_visualization", f"{i}_{data_x['name']}_{data_y['name']}.png"))

        # vertex transfer
        # ps_ours_x = ps.register_surface_mesh("ours_first", verts_x , faces_x, material="wax", smooth_shade=False, color=[1,1,1], edge_width=0.2)
        # ps_ours_y = ps.register_surface_mesh("ours_second", verts_x[p2p] + [1.2,0,0], faces_y, material="wax", smooth_shade=False, color=[1,1,1], edge_width=0.2)
        # vertex_image = Image.fromarray(ps.screenshot_to_buffer())
        # vertex_image = vertex_image.crop(vertex_image.getbbox())
        # vertex_image.save(osp.join(ours_results_root, "polyscope_visualization", f"{i}_{data_x['name']}_{data_y['name']}.png"))

    print(f"Visualization saved to {osp.join(ours_results_root, 'polyscope_visualization')}")

if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    visualize_pipeline(root_path)
