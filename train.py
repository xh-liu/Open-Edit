import sys
from collections import OrderedDict
import data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.OpenEdit_trainer import OpenEditTrainer
from options.train_options import TrainOptions

def main_worker(gpu, world_size, opt):
    print('Use GPU: {} for training'.format(gpu))
    world_size = opt.world_size
    rank = gpu
    opt.gpu = gpu
    dist.init_process_group(backend='nccl', init_method=opt.dist_url, world_size=world_size, rank=rank)
    torch.cuda.set_device(gpu)

    # load the dataset
    dataloader = data.create_dataloader(opt, world_size, rank)
    
    # create trainer for our model
    trainer = OpenEditTrainer(opt)
    
    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader), world_size, rank)
    
    # create tool for visualization
    visualizer = Visualizer(opt, rank)
    
    for epoch in iter_counter.training_epochs():
        if opt.mpdist:
            dataloader.sampler.set_epoch(epoch)
        iter_counter.record_epoch_start(epoch)
        for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()
    
            # Training
            # train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i)

            # train discriminator
            if not opt.no_disc and i % opt.G_steps_per_D == 0:
                trainer.run_discriminator_one_step(data_i)

            iter_counter.record_iteration_end()

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter, 
                                                iter_counter.model_time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            
        visuals = OrderedDict([('synthesized_image', trainer.get_latest_generated()),
                                   ('real_image', data_i['image'])])
        visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if rank == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if (epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs) and (rank == 0):
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save(epoch)
    
    print('Training was successfully finished.')
    
if __name__ == '__main__':
    global TrainOptions
    TrainOptions = TrainOptions()
    opt = TrainOptions.parse(save=True)
    opt.world_size = opt.num_gpu
    opt.mpdist = True

    mp.set_start_method('spawn', force=True)
    mp.spawn(main_worker, nprocs=opt.world_size, args=(opt.world_size, opt))
