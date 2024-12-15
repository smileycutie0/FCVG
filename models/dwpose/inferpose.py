import torch
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image
from preprocess import get_image_pose
import cv2
import math
from PIL import Image
from util import draw_pose
import numpy as np
import imageio

ASPECT_RATIO = 9 / 16

def load_image(image_path, resolution=576, sample_stride=2):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): input video pose path
        image_path (str): reference image path
        resolution (int, optional):  Defaults to 576.
        sample_stride (int, optional): Defaults to 2.
    """
    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels) # (c, h, w)
    h, w = image_pixels.shape[-2:]
    ############################ compute target h/w according to original aspect ratio ###############################
    if h>w:
        w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    else:
        w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution
    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()

    return image_pixels

def interpolate_linear(p0, p1, frac):
    return p0 + (p1 - p0) * frac

def interp_poses(pose_md1, pose_md2, num_frames=23):
    bodies1 = pose_md1['bodies']
    candidate1 = bodies1['candidate']
    subset1 = bodies1['subset']
    score = bodies1['score']
    hands_score = pose_md1['hands_score']
    faces_score = pose_md1['faces_score']

    bodies2 = pose_md2['bodies']
    candidate2 = bodies2['candidate']
    subset2 = bodies2['subset']

    interped_pose = []

    for i in range(num_frames):
        frac = i / (num_frames - 1) 

        interp_candidate = [interpolate_linear(c1, c2, frac) for c1, c2 in zip(candidate1, candidate2)]  # assume one person
        interp_faces = [interpolate_linear(f1, f2, frac) for f1, f2 in zip(pose_md1['faces'], pose_md2['faces'])]
        interp_hands = [interpolate_linear(h1, h2, frac) for h1, h2 in zip(pose_md1['hands'], pose_md2['hands'])]

        frame_data = {
            'bodies': {
                'candidate': interp_candidate,
                'subset': subset1,
                'score': score
            },
            'faces': interp_faces,
            'faces_score': faces_score,
            'hands': interp_hands,
            'hands_score':hands_score
        }
        # print(frame_data)
        interped_pose.append(frame_data)

    return interped_pose

img1 = '/hdd/zty/code/2024work/code_src/svd_keyframe_interpolation-main/examples/example_004/frame1.png'
img2 = '/hdd/zty/code/2024work/code_src/svd_keyframe_interpolation-main/examples/example_004/frame2.png'
img1 = load_image(img1)
img2 = load_image(img2)

pose_image, pose = get_image_pose(img1)
pose_image2, pose2 = get_image_pose(img2)
interped = interp_poses(pose, pose2, num_frames=23)
gif_list = [pose_image.transpose((1, 2, 0))]

print(pose)
print(interped[0])
for idx, frame in enumerate(interped):
    height, width, _ = img1.shape

    interp_image = draw_pose(frame, height, width).transpose((1, 2, 0))

    gif_list.append(interp_image)
    Image.fromarray(interp_image).save('./results/08/out{:02d}.jpg'.format(idx+1))


gif_list.append(pose_image2.transpose((1, 2, 0)))
imageio.mimsave('./results/08/out.gif', gif_list, duration=1/10)

Image.fromarray(pose_image.transpose((1, 2, 0))).save('./results/08/out00.jpg')
Image.fromarray(pose_image2.transpose((1, 2, 0))).save('./results/08/out24.jpg')


