import os
import torch
import numpy as np
from PIL import Image
from pipeline.pipeline_FCVG import StableVideoDiffusionPipelineControlNeXtReverse
from models.controlnext_vid_svd import ControlNeXtSDVModel
from models.unet_spatio_temporal_condition_controlnext import UNetSpatioTemporalConditionControlNeXtModel
from transformers import CLIPVisionModelWithProjection
import re 
from diffusers import AutoencoderKLTemporalDecoder
import time
from decord import VideoReader
import argparse
from safetensors.torch import load_file
# from utils.pre_process import preprocess
from models.gluestick.models.two_view_pipeline import TwoViewPipeline
from models.gluestick import GLUESTICK_ROOT, batch_to_np, numpy_image_to_torch
from models.gluestick.drawing import plot_color_line_matches_opencv
import cv2
from models.dwpose.preprocess import get_image_pose
from models.dwpose.util import draw_pose
from utils.util import *

class Traverse_Logger:
    def __init__(self, result_dir, filename='inference_log.txt'):
        self.log_file_path = os.path.join(result_dir, filename)
        open_type = 'a' if os.path.exists(self.log_file_path) else 'w'
        self.log_file = open(self.log_file_path, open_type)

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')
        

def load_tensor(tensor_path):
    if os.path.splitext(tensor_path)[1] == '.bin':
        return torch.load(tensor_path)
    elif os.path.splitext(tensor_path)[1] == ".safetensors":
        return load_file(tensor_path)
    else:
        print("without supported tensors")
        os._exit()
        
def create_gif_from_frames(frames, gif_path, duration=100):
    pil_images = [Image.fromarray(frame) for frame in frames]
    
    pil_images[0].save(
        gif_path, 
        save_all=True, 
        append_images=pil_images[1:], 
        duration=duration, 
        loop=0
    )
    
def infer_gluestick_interp(model, image1, image2, interp_frames=23, save_matching_path=None, save_gif=False, lw=2):

    gray0 = np.array(image1.convert('L'))
    gray1 = np.array(image2.convert('L'))

    torch_gray0, torch_gray1 = numpy_image_to_torch(gray0), numpy_image_to_torch(gray1)
    torch_gray0, torch_gray1 = torch_gray0.to(device)[None], torch_gray1.to(device)[None]
    x = {'image0': torch_gray0, 'image1': torch_gray1}
    pred = model(x)

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

    img0, img1 = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
    if matched_lines0 is None:
        img = np.zeros_like(img0)
        images = []
        for i in range(interp_frames+2):
            images.append(img)
        return images
    all_interp_matches = [matched_lines0]
    for i in range(interp_frames):
        frac = i / (interp_frames - 1) 

        interped = interpolate_matches_linear(matched_lines0, matched_lines1, frac)
        all_interp_matches.append(interped)
    all_interp_matches.append(matched_lines1)
    images = plot_color_line_matches_opencv(img0, all_interp_matches, save_path=save_matching_path, lw=lw)

    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    return images


def infer_poses_interp(image1, image2, num_frames=23, save_pose_path=None):

    image1 = np.array(image1)
    image2 = np.array(image2)
    height, width, _ = image1.shape
    pose_image, pose1 = get_image_pose(image1)
    pose_image2, pose2 = get_image_pose(image2)

    bodies1 = pose1['bodies']
    candidate1 = bodies1['candidate']
    subset1 = bodies1['subset']
    bodies2 = pose2['bodies']
    candidate2 = bodies2['candidate']
    subset2 = bodies2['subset']
    new_match = match_bodies(candidate1, candidate2, subset1, subset2)
    interped_pose = [pose_image.transpose((1, 2, 0))]
    numbers = np.arange(min(len(candidate1), len(candidate2)))
    subset = np.array(np.split(numbers, min(len(subset1), len(subset2))))
    candidate2_idx = np.concatenate(subset[new_match])
    candidate2 = candidate2[candidate2_idx]
    

    for i in range(num_frames):
        frac = i / (num_frames - 1) 
        
        interp_candidate = [interpolate_linear(c1, c2, frac) for c1, c2 in zip(candidate1, candidate2)] 
        interp_faces = [interpolate_linear(f1, f2, frac) for f1, f2 in zip(pose1['faces'], pose2['faces'][new_match])]
        hands_match = new_match + [x + len(new_match) for x in new_match]
        interp_hands = [interpolate_linear(h1, h2, frac) for h1, h2 in zip(pose1['hands'], pose2['hands'][hands_match])]


        interp_hands_score = [interpolate_linear(h1, h2, frac) for h1, h2 in zip(pose1['hands_score'], pose2['hands_score'][hands_match])]
        interp_faces_score = [interpolate_linear(h1, h2, frac) for h1, h2 in zip(pose1['faces_score'], pose2['faces_score'][new_match])]
        interp_score = [interpolate_linear(h1, h2, frac) for h1, h2 in zip(bodies1['score'], bodies2['score'][new_match])]


        frame_data = {
            'bodies': {
                'candidate': interp_candidate,
                'subset': subset,
                'score': interp_score
            },
            'faces': interp_faces,
            'faces_score': interp_faces_score,
            'hands': interp_hands,
            'hands_score':interp_hands_score
        }
    

        interp_image = draw_pose(frame_data, height, width).transpose((1, 2, 0))
        interped_pose.append(interp_image)
    interped_pose.append(pose_image2.transpose((1, 2, 0)))
    if save_pose_path:
        for i in range(len(interped_pose)):
            Image.fromarray(interped_pose[i]).save(os.path.join(save_pose_path, 'pose{:02d}.png'.format(i)))

    return interped_pose

def crop_and_resize(image, size=(1024, 576)):
    target_width, target_height = size
    original_width, original_height = image.size

    target_ratio = target_width / target_height
    original_ratio = original_width / original_height

    if original_ratio > target_ratio:
        new_width = int(original_height * target_ratio)
        left = (original_width - new_width) // 2
        right = left + new_width
        top = 0
        bottom = original_height
    else:
        new_height = int(original_width / target_ratio)
        top = (original_height - new_height) // 2
        bottom = top + new_height
        left = 0
        right = original_width

    cropped_image = image.crop((left, top, right, bottom))
    resized_image = cropped_image.resize(size)

    return resized_image

def save_images(images, path='./results'):
    save_folder = path
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # save_images = os.path.join(save_folder, 'images')
    # if not os.path.exists(save_images):
    #     os.mkdir(save_images)
    for i in range(len(images)):
        save_images_path = os.path.join(save_folder, '{:02d}.png'.format(i))
        images[i].save(save_images_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )

    parser.add_argument("--pretrained_model_name_or_path", type=str, default='stabilityai/stable-video-diffusion-img2vid-xt-1-1')
    parser.add_argument("--height", type=int, default=576, required=False)
    parser.add_argument("--width", type=int, default=1024, required=False)
    parser.add_argument("--guidance_scale", type=float, default=3., required=False)
    parser.add_argument("--num_inference_steps",type=int,default=25,required=False)
    parser.add_argument("--controlnext_path",type=str,default='checkpoints/controlnext.safetensors',)
    parser.add_argument("--unet_path",type=str,default='checkpoints/unet.safetensors',)
    parser.add_argument("--max_frame_num",type=int,default=25,required=False)
    parser.add_argument("--dataset_path",type=str,default='./datasets/')
    parser.add_argument("--output_dir",type=str,default='./infer_results')

    parser.add_argument("--batch_frames",type=int,default=25,required=False)
    parser.add_argument("--input_gaps",type=int,default=None)
    
    parser.add_argument("--overlap",type=int,default=6,required=False)

    parser.add_argument("--sample_stride",type=int,default=2,required=False)
    parser.add_argument('--max_pts', type=int, default=1000)
    parser.add_argument('--max_lines', type=int, default=300)
    args = parser.parse_args()
    return args

# Main script
if __name__ == "__main__":
    args = parse_args()
    # Gluestick evaluation config
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

    unet = UNetSpatioTemporalConditionControlNeXtModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
    )
    controlnext = ControlNeXtSDVModel()
    controlnext.load_state_dict(load_tensor(args.controlnext_path))
    unet.load_state_dict(load_tensor(args.unet_path), strict=False)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae")
    
    pipeline = StableVideoDiffusionPipelineControlNeXtReverse.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnext=controlnext, 
        unet=unet,
        vae=vae,
        image_encoder=image_encoder)
    # pipeline.to(dtype=torch.float16)
    pipeline.enable_model_cpu_offload()

    # logger = Traverse_Logger(args.output_dir)
    # now_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    # logger.write_log('Inference - {}'.format(now_time))
    
    data_name = sorted(os.listdir(args.dataset_path))
    data_path = [os.path.join(args.dataset_path, d) for d in data_name]
    # input_images_path = [os.path.join(args.image_dir, img) for img in sorted(os.listdir(args.image_dir))]
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for idx, pth in enumerate(data_path):
        video_paths = [os.path.join(pth, v) for v in sorted(os.listdir(pth))]
        current_data_name = data_name[idx]
        save_data_folder = os.path.join(args.output_dir, current_data_name)
        if not os.path.exists(save_data_folder):
            os.mkdir(save_data_folder)
        
        for video_path in video_paths:
            basename = os.path.basename(video_path)
            video_name = current_data_name + '  ' + basename
            
            save_folder = os.path.join(save_data_folder, basename)
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
        
            # input_images_folder = os.path.join(video_path, 'images')
            images_path = [os.path.join(video_path, img) for img in sorted(os.listdir(video_path))]
            
            assert len(images_path) == args.max_frame_num
            input_image0 = images_path[0]
            if not args.input_gaps:
                input_image1 = images_path[-1]
            else:
                input_image1 = images_path[args.input_gaps+1]
            
            image1 = Image.open(input_image0).convert('RGB')
            image1 = crop_and_resize(image1, (args.width, args.height))
            image2 = Image.open(input_image1).convert('RGB')
            image2 = crop_and_resize(image2, (args.width, args.height))
            
            interped_matching = infer_gluestick_interp(pipeline_model, image1, image2, interp_frames=args.max_frame_num-2, save_matching_path=None, lw=1)
            interped_pose = infer_poses_interp(image1, image2, num_frames=args.max_frame_num-2, save_pose_path=None)
            fusion_control = []
            # save_fusion_path = os.path.join(save_folder, 'fusion')
            # if not os.path.exists(save_fusion_path):
            #     os.mkdir(save_fusion_path)
            for i in range(len(interped_pose)):
                fusion = interped_matching[i].copy()
                fusion[interped_pose[i] != 0] = interped_pose[i][interped_pose[i] != 0]
                # Image.fromarray(fusion).save(os.path.join(save_fusion_path, 'fusion{:02d}.png'.format(i)))
                fusion_control.append(Image.fromarray(fusion))
                
            # create_gif_from_frames(save_fusion_path, os.path.join(save_folder, 'fusion.gif'), duration=100)
                
            validation_control_images = [img.convert('RGB') for img in fusion_control]
            validation_control_images = [crop_and_resize(img, (args.width, args.height)) for img in validation_control_images]
            
            final_result = []
            frames = args.batch_frames
            num_frames = args.max_frame_num
            
            for i in range(num_frames):
                validation_control_images[i] = Image.fromarray(np.array(validation_control_images[i]))
            
            assert len(validation_control_images) == args.max_frame_num
            
            video_frames = pipeline(
                image1,
                image2, 
                validation_control_images, 
                decode_chunk_size=2,
                num_frames=num_frames,
                motion_bucket_id=127.0, 
                fps=7,
                controlnext_cond_scale=1.0, 
                width=args.width, 
                height=args.height, 
                min_guidance_scale=args.guidance_scale, 
                max_guidance_scale=args.guidance_scale, 
                frames_per_batch=frames, 
                num_inference_steps=args.num_inference_steps, 
                overlap=args.overlap).frames
            
            flattened_batch_output = [img for sublist in video_frames for img in sublist]

            save_images(flattened_batch_output, path=save_folder) 
            gif_path = os.path.join(save_folder, '{}.gif'.format(video_path[-6:]))
            flattened_batch_output[0].save(gif_path, save_all=True, append_images=flattened_batch_output[1:], loop=0, duration=100)
            
            

