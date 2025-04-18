import argparse
import os
from os.path import join

import cv2
import torch
from matplotlib import pyplot as plt
import numpy as np
from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from .drawing import plot_images, plot_lines, plot_color_line_matches_opencv, plot_keypoints, plot_matches
from .models.two_view_pipeline import TwoViewPipeline
from PIL import Image

def interpolate_linear(p0, p1, frac):
    interped = np.zeros_like(p0)
    N, _, _ = p0.shape
    for i in range(N):
        path0 = np.sum(np.abs(p1[i, 0] - p0[i, 0])) + np.sum(np.abs(p1[i, 1] - p0[i, 1]))
        path1 = np.sum(np.abs(p1[i, 0] - p0[i, 1])) + np.sum(np.abs(p1[i, 1] - p0[i, 0]))
        if path0 < path1:
            interped[i] = p0[i] + (p1[i] - p0[i]) * frac
        else:
            interped[i, 0] = p0[i, 0] + (p1[i, 1] - p0[i, 0]) * frac
            interped[i, 1] = p0[i, 1] + (p1[i, 0] - p0[i, 1]) * frac

    return interped


def create_gif_from_frames(frames, gif_path, duration=100):
    pil_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    
    pil_images[0].save(
        gif_path, 
        save_all=True, 
        append_images=pil_images[1:], 
        duration=duration, 
        loop=0
    )

def main():
    # Parse input parameters
    parser = argparse.ArgumentParser(
        prog='GlueStick Demo',
        description='Demo app to show the point and line matches obtained by GlueStick')
    parser.add_argument('-img1', default='/hdd/zty/code/2024work/code_src/svd_keyframe_interpolation-main/examples/example_004/frame1.png')
    parser.add_argument('-img2', default='/hdd/zty/code/2024work/code_src/svd_keyframe_interpolation-main/examples/example_004/frame2.png')
    parser.add_argument('--max_pts', type=int, default=1000)
    parser.add_argument('--max_lines', type=int, default=300)
    parser.add_argument('--skip-imshow', default=True, action='store_true')
    args = parser.parse_args()

    # Evaluation config
    conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            'name': 'wireframe',
            'sp_params': {
                'force_num_keypoints': False,
                'max_num_keypoints': args.max_pts,
            },
            'wireframe_params': {
                'merge_points': True,
                'merge_line_endpoints': True,
            },
            'max_n_lines': args.max_lines,
        },
        'matcher': {
            'name': 'gluestick',
            'weights': str(GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
            'trainable': False,
        },
        'ground_truth': {
            'from_pose_depth': False,
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipeline_model = TwoViewPipeline(conf).to(device).eval()

    gray0 = cv2.imread(args.img1, 0)
    gray1 = cv2.imread(args.img2, 0)

    torch_gray0, torch_gray1 = numpy_image_to_torch(gray0), numpy_image_to_torch(gray1)
    torch_gray0, torch_gray1 = torch_gray0.to(device)[None], torch_gray1.to(device)[None]
    x = {'image0': torch_gray0, 'image1': torch_gray1}
    pred = pipeline_model(x)

    pred = batch_to_np(pred)
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]

    line_seg0, line_seg1 = pred["lines0"], pred["lines1"]
    line_matches = pred["line_matches0"]

    valid_matches = m0 != -1
    match_indices = m0[valid_matches]
    matched_kps0 = kp0[valid_matches]
    matched_kps1 = kp1[match_indices]

    valid_matches = line_matches != -1
    match_indices = line_matches[valid_matches]
    matched_lines0 = line_seg0[valid_matches]
    matched_lines1 = line_seg1[match_indices]

    

    # Plot the matches
    img0, img1 = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
    # plot_images([img0, img1], ['Image 1 - detected lines', 'Image 2 - detected lines'], dpi=200, pad=2.0)
    # plot_lines([line_seg0, line_seg1], ps=4, lw=2)
    # plt.gcf().canvas.manager.set_window_title('Detected Lines')
    # plt.savefig('detected_lines.png')

    # plot_images([img0, img1], ['Image 1 - detected points', 'Image 2 - detected points'], dpi=200, pad=2.0)
    # plot_keypoints([kp0, kp1], colors='c')
    # plt.gcf().canvas.manager.set_window_title('Detected Points')
    # plt.savefig('detected_points.png')

    # plot_images([img0, img1], ['Image 1 - line matches', 'Image 2 - line matches'], dpi=200, pad=2.0)
    # plot_color_line_matches_opencv(img0, [matched_lines0, matched_lines1], lw=2)
    # plt.gcf().canvas.manager.set_window_title('Line Matches')
    # plt.savefig('line_matches.png')

    # plot_images([img0, img1], ['Image 1 - point matches', 'Image 2 - point matches'], dpi=200, pad=2.0)
    # plot_matches(matched_kps0, matched_kps1, 'green', lw=1, ps=0)
    # plt.gcf().canvas.manager.set_window_title('Point Matches')
    # plt.savefig('pointmatches.png')

    num_frames = 23
    all_interp_matches = [matched_lines0]
    for i in range(num_frames):
        frac = i / (num_frames - 1) 

        interped = interpolate_linear(matched_lines0, matched_lines1, frac)
        all_interp_matches.append(interped)
    all_interp_matches.append(matched_lines1)
    images = plot_color_line_matches_opencv(img0, all_interp_matches, save_path='./results/02',lw=2)
    create_gif_from_frames(images, 'linematches2.gif', duration=100)

    if not args.skip_imshow:
        plt.show()


if __name__ == '__main__':
    main()
