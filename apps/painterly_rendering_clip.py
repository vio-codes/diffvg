"""
Scream: python painterly_rendering.py imgs/scream.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0 --use_lpips_loss
Baboon: python painterly_rendering.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 250
Baboon Lpips: python painterly_rendering.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 500 --use_lpips_loss
Kitty: python painterly_rendering.py imgs/kitty.jpg --num_paths 1024 --use_blob
"""
import pydiffvg
import torch
import skimage
import ttools.modules
import skimage.io
import skimage.color
import random
import argparse
import math
import torch.nn as nn
from torch.autograd import Variable
import clip_utils
pydiffvg.set_print_timing(True)

gamma = 1


def main(args):
    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    text_features = clip_utils.embed_text(args.target)
    #target = torch.nn.functional.interpolate(target, size = [256, 256], mode = 'area')
    canvas_width, canvas_height = 224, 224
    num_paths = args.num_paths

    shapes = []
    shape_groups = []

    for i in range(num_paths):
        num_segments = random.randint(3, 5)
        num_control_points = torch.zeros(
            num_segments, dtype=torch.int32) + 2
        points = []
        p0 = (random.random(), random.random())
        points.append(p0)
        for j in range(num_segments):
            radius = 0.05
            p1 = (p0[0] + radius * (random.random() - 0.5),
                  p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5),
                  p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5),
                  p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            if j < num_segments - 1:
                points.append(p3)
                p0 = p3
        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(num_control_points=num_control_points,
                             points=points,
                             stroke_width=torch.tensor(1.0),
                             is_closed=True)
        shapes.append(path)
        x_max = torch.max(path.points[:, 0])
        x_min = torch.min(path.points[:, 0])
        y_max = torch.max(path.points[:, 1])
        y_min = torch.min(path.points[:, 1])
        gradient = pydiffvg.LinearGradient(begin=torch.tensor([x_min, y_min]),
                                           end=torch.tensor([x_max, y_max]),
                                           offsets=torch.tensor(
                                               [0.0, 0.5, 1.0]),
                                           stop_colors=torch.tensor([[random.random(),
                                                                      random.random(),
                                                                      random.random(),
                                                                      random.random()],
                                                                     [random.random(),
                                                                      random.random(),
                                                                      random.random(),
                                                                      random.random()],
                                                                     [random.random(),
                                                                      random.random(),
                                                                      random.random(),
                                                                      random.random()]]))
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                         fill_color=gradient)
        shape_groups.append(path_group)

    points_vars = []
    color_vars = []
    begin_vars = []
    end_vars = []
    offsets_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)

    for group in shape_groups:
        group.fill_color.end.requires_grad = True
        end_vars.extend([group.fill_color.end])
        group.fill_color.begin.requires_grad = True
        begin_vars.extend([group.fill_color.begin])
        group.fill_color.offsets.requires_grad = True
        offsets_vars.extend([group.fill_color.offsets])
        group.fill_color.stop_colors.requires_grad = True
        color_vars.extend([group.fill_color.stop_colors])

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)
    begin_optim = torch.optim.Adam(begin_vars, lr=1.0)
    end_optim = torch.optim.Adam(end_vars, lr=1.0)
    offsets_optim = torch.optim.Adam(offsets_vars, lr=.001)
    # Adam iterations.
    for t in range(args.num_iter):
        print('iteration:', t)
        points_optim.zero_grad()
        color_optim.zero_grad()
        begin_optim.zero_grad()
        end_optim.zero_grad()
        offsets_optim.zero_grad()

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width,  # width
                     canvas_height,  # height
                     2,   # num_samples_x
                     2,   # num_samples_y
                     t,   # seed
                     None,
                     *scene_args)

        # Save the intermediate render.
       
        pydiffvg.imwrite(
            img.cpu(), 'results/painterly_clip/iter_{}.png'.format(t), gamma=gamma)

        pydiffvg.save_ln_gradient_svg('results/painterly_clip/iter_{}.svg'.format(t),
                                      canvas_width, canvas_height, shapes, shape_groups)

        image_features = clip_utils.embed_image(img)
        
        #IoU losl
        inputs = torch.flatten(image_features)
        targets = torch.flatten(text_features)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        smooth = 1
        IoU = (intersection + smooth)/(union + smooth)
        #dice                  
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
                
        #loss = -torch.cosine_similarity(text_features, image_features, dim=-1).mean()
        # torch.cosine_similarity(image_features, text_features, dim=-1).mean()

        #loss = 1 - dice
        loss = (image_features - text_features).pow(2).mean()
        print('render loss:', loss.item())
        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        points_optim.step()
        color_optim.step()
        begin_optim.step()
        end_optim.step()
        offsets_optim.step()

        for group in shape_groups:
            group.fill_color.stop_colors.data.clamp_(0.0, 1.0)
            group.fill_color.offsets[0].data.clamp_(0.0, 0.33)
            group.fill_color.offsets[1].data.clamp_(0.33, 0.66)
            group.fill_color.offsets[2].data.clamp_(0.66, 1.0)
            group.fill_color.begin[0].data.clamp_(0.0, canvas_width)
            group.fill_color.begin[1].data.clamp_(0.0, canvas_height)
            group.fill_color.end[0].data.clamp_(0.0, canvas_width)
            group.fill_color.end[1].data.clamp_(0.0, canvas_height)
        for path in shapes:
            path.points.requires_grad = True
            path.points[:, 0].data.clamp_(0.0, canvas_width)
            path.points[:, 1].data.clamp_(0.0, canvas_height)

        # Render the final result
    img = render(canvas_width,  # width
                 canvas_height,  # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 t,   # seed
                 None,
                 *scene_args)
    # Forward pass: render the image.
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
        img.shape[0], img.shape[1], 3, device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), "/content/final.png", gamma=gamma)
    pydiffvg.save_ln_gradient_svg('/content/final.svg',
                                  canvas_width, canvas_height, shapes, shape_groups)
    if args.debug:
        from subprocess import call
        call(["ffmpeg", "-framerate", "24", "-i",
              "results/painterly_clip/iter_%d.png", "-c:v", "libx264", "-preset", "veryslow",
              "-crf", "20", "-vf", "format=yuv420p", "-movflags", "+faststart",
              "/content/final.mp4"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="target text")
    parser.add_argument("--num_paths", type=int, default=512)
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--debug", dest='debug', action='store_true')
    args = parser.parse_args()
    main(args)
