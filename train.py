import time
from options.train_options import TrainOptions 
from data.dummy_dataset import DummyDataset
from torch.utils.data import DataLoader 
from models.sine_cycle_gan_model import SineCycleGanModel
from util.visualizer import Visualizer 

if __name__ == '__main__':
    opt = TrainOptions().parse() 

    # create dummy dataset and dataloader 
    dataset = DummyDataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.isTrain)
    dataset_size = len(dataset)
    print(f'==> Number of training samples: {dataset_size}')

    # initialize model and setup
    model = SineCycleGanModel(opt)
    model.setup(opt)

    visualizer = Visualizer(opt)
    total_iters = 0
    
    # training loop 
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataloader):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            # ===== display results =========
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # save the latest model 
            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)


        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
