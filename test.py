#!/usr/bin/env python3
import os
from options.test_options import TestOptions
from data.dummy_dataset import DummyDataset
from models.sine_cycle_gan_model import SineCycleGanModel
from util.visualizer import save_images
from util import html

if __name__ == '__main__':
    opt = TestOptions().parse()

    # instantiate dataset
    dataset = DummyDataset(opt)
    dataset_size = len(dataset)
    print(f'==> Running inference on {dataset_size} samples')

    # create model and load checkpoint
    model = SineCycleGanModel(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    # prepare web directory
    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    print('==> Saving results to', web_dir)
    webpage = html.HTML(web_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}')

    # inference
    for i, data in enumerate(dataset):
        if opt.num_test and i >= opt.num_test:
            break
        model.set_input(data) 
        model.test()                     # run forward pass under torch.no_grad()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()  # here it’s just the dummy index

        if i % 5 == 0:
            print(f'Processing ({i:04d})-th sample: {img_path}')

        save_images(
            webpage,
            visuals,
            img_path,
            aspect_ratio=1.0,
            width=opt.display_winsize,
            use_wandb=opt.use_wandb
        )

    webpage.save()
    print('Finished inference!')

