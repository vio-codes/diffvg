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
import skimage.io
import random
import argparse
import math

pydiffvg.set_print_timing(True)

gamma = 1

def convert_svg2png(svg_file, png_file, resolution = 72):
    import wand.image
    from wand.api import library
    import wand.color
    with open(svg_file, "r") as svg_file:
        with wand.image.Image() as image:
            with wand.color.Color('transparent') as background_color:
                library.MagickSetBackgroundColor(image.wand, 
                                                 background_color.resource) 
            svg_blob = svg_file.read().encode('utf-8')
            image.read(blob=svg_blob, resolution = resolution)
            png_image = image.make_blob("png32")
    with open(png_file, "wb") as out:
        out.write(png_image)

def convert_svg2png2(svg_file, png_file):
    import wand.image
    with Image(filename=svg_file) as original:
        with original.convert('png') as converted:
            converted.save(filename=png_file) 
            converted.format = 'svg'  
            converted.save(filename=png_file+".svg")     

def main(args):
    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    #target = torch.from_numpy(skimage.io.imread('imgs/lena.png')).to(torch.float32) / 255.0
    target = torch.from_numpy(skimage.io.imread(
        args.target)).to(torch.float32) / 255.0
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2)  # NHWC -> NCHW
    #target = torch.nn.functional.interpolate(target, size = [256, 256], mode = 'area')
    canvas_width, canvas_height = target.shape[3], target.shape[2]
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
        gradient = pydiffvg.LinearGradient(begin=torch.tensor([random.random()*canvas_width, random.random()*canvas_height]),
                                           end=torch.tensor(
                                               [random.random()*canvas_width, random.random()*canvas_height]),
                                           offsets=torch.tensor([0.0, 0.5, 1.0]),
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

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)


    points_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)

    for group in shape_groups:
            group.fill_color.end.requires_grad = True
            group.fill_color.begin.requires_grad = True
            group.fill_color.offsets.requires_grad = True
            group.fill_color.stop_colors.requires_grad = True
            color_vars.extend(
                [group.fill_color.begin, group.fill_color.end, group.fill_color.offsets, group.fill_color.stop_colors])
 

    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)

    color_optim = torch.optim.Adam(color_vars, lr=0.1)
    # Adam iterations.
    for t in range(args.num_iter):
        print('iteration:', t)
        points_optim.zero_grad()
        color_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups)
        pydiffvg.save_svg('results/painterly_svg/iter_{}.svg'.format(t),
                              canvas_width, canvas_height, shapes, shape_groups)
        convert_svg2png2('results/painterly_svg/iter_{}.svg'.format(t),'results/painterly_svg/iter_{}.png'.format(t))                      
        #TODO  dice loss
        img = torch.from_numpy(skimage.io.imread('results/painterly_svg/iter_{}.png'.format(t))).to(torch.float32) / 255.0
        img= target.pow(gamma)
        img = target.to(pydiffvg.get_device())
        img = target.unsqueeze(0)
        img = target.permute(0, 3, 1, 2)

        loss = (img - target).pow(2).mean()
        print('render loss:', loss.item())

        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        #TODO break color_optim into more optimizers
        points_optim.step()
        color_optim.step()

        for group in shape_groups:
            #TODO Clam all the data
            group.fill_color.stop_colors.data.clamp_(0.0, 1.0)

        # Render the final result
    pydiffvg.save_svg('/content/final.svg',
                      canvas_width, canvas_height, shapes, shape_groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="target image path")
    parser.add_argument("--num_paths", type=int, default=512)
    parser.add_argument("--num_iter", type=int, default=500)
    args = parser.parse_args()
    main(args)
