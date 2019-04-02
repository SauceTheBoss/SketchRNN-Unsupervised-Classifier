# import argparse
from SketchAAE import SketchAAE
from SketchDataset import ComboDataset
from HParams import HParams
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils import make_image, make_image2, make_image3, SetSeed
import gc
from visualize import visualize_movement, vis_inner

# parser = argparse.ArgumentParser(description='SketchAAE')

# parser.add_argument('--softmax_train', type=bool, default=True, metavar='True',
#                     help='use softmax in reverse training for input categories (default: True)')

# args = parser.parse_args()

def training_detached(hp: HParams):
    model = SketchAAE(hp)

    writer = SummaryWriter() if hp.tensorboard else None
    inner_lf = 1
    epoch = 0
    while epoch < 10000 and not hp.fast_debug:
        inner_loss, inner_lf, inner_lc, inner_ls = model.train_inner(train=True)
        print("epoch",epoch,"inner_lf",inner_lf,"inner_lc",inner_lc,"inner_ls",inner_ls,"inner_loss",inner_loss)
        model.lr_decay(inner=True)
        if epoch%1000==0:
            vis_inner(hp, model, writer, epoch)
        # if epoch%5==0:
        #     writer.add_scalar("inner/inner_loss", inner_loss, epoch)
        #     writer.add_scalar("inner/inner_hist", inner_hist, epoch)
        if (inner_lc > 30) and inner_lf < 0.1:
            print("resetting")
            model = SketchAAE(hp)
        epoch += 1

    ds = ComboDataset(hp, model.device)
    ds.CacheToSharedMemory()
    model.Nmax = ds.Nmax


    epoch = 0
    loop_active = True
    old_buckets = None
    while loop_active:
        dl = DataLoader(ds, hp.batch_size, shuffle=True)
        for	_, batch_sample in enumerate(dl):
            epoch += 1

            batch, lengths, _,  = batch_sample
            batch = batch.transpose(0,1)

            r_loss, r_ls, r_lp, r_map = model.train_reconstruction(batch, lengths)
            model.lr_decay(outer=True)

            print("epoch",epoch,"r_loss",r_loss,"r_ls",r_ls,"r_lp",r_lp,"r_map",r_map)

            if hp.tensorboard and epoch%10==0:
                writer.add_scalar("l/r_loss", r_loss, epoch)
                writer.add_scalar("recon/r_ls", r_ls, epoch)
                writer.add_scalar("recon/r_lp", r_lp, epoch)
                writer.add_scalar("recon/r_lp", r_map, epoch)
            
            if hp.tensorboard and epoch%250==0 or hp.fast_debug:
                sketch, length = batch.transpose(0,1)[0].detach(), lengths[0].detach()
                writer.add_figure(tag='Original', figure=make_image(sketch.cpu().numpy()), global_step=epoch)
                writer.add_figure(tag='Generated', figure=make_image3(model.conditional_generation(sketch, length, ds.Nmax)), global_step=epoch)

            if hp.tensorboard and ((epoch%1000==0 and epoch < 10000) or epoch%2000==0 or hp.fast_debug):
                old_buckets = visualize_movement(hp, ds, writer, model, epoch, old_buckets)
                for cat in range(model.hp.cat_dims):
                    tag_name = "CatGen/"+str(cat)+"cat"
                    seqs = writer.add_figure(tag=tag_name, figure=make_image3(model.generation_for_category(cat)), global_step=epoch)

            if epoch >= hp.max_epochs:
                loop_active = False
                break

        gc.collect()
        
        


if __name__=="__main__":
    hp_main = HParams()
    training_detached(hp_main)

#from torch.autograd.profiler import profile
# with profile(use_cuda=True) as prof:
#             val = prof.key_averages()
#             print(val)
#             print(prof.table("cpu_time_total"))