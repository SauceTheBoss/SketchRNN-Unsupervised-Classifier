import math
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from HParams import HParams
from SketchAAE import SketchAAE
from SketchDataset import ComboDataset

from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import gc

def vis_inner(hp: HParams, model: SketchAAE, writer: SummaryWriter, epoch):
    def _np(item):
        return item.detach().cpu().numpy()

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    print("visualizing inner")
    with torch.no_grad():
        nBuckets = 50
        nLoops = 200
        new_buckets = np.zeros((nBuckets,nBuckets))
        edges = np.linspace(0, 1, num=nBuckets+1)
        model.eval()
        for _ in range(nLoops):
            input_encoder = (torch.randn(hp.viz_batch_size, model.innerEncoder.input_size, device=model.device))
            new_cats, new_styles = model.innerEncoder(input_encoder)

            np_new_styles = _np(new_styles)

            new_x = np_new_styles[:,0].flatten()
            new_y = np_new_styles[:,1].flatten()
            H, _, _ = np.histogram2d(x=new_x, y=new_y, bins=nBuckets, range=((0,1),(0,1)))
            new_buckets[:,:] += H[:,:]

        fig = plt.figure()
        ax = fig.add_subplot(111, title='test',
                aspect='equal', xlim=edges[[0, -1]], ylim=edges[[0, -1]])
        im = NonUniformImage(ax, interpolation='nearest')
        xcenters = (edges[:-1] + edges[1:]) / 2
        ycenters = (edges[:-1] + edges[1:]) / 2
        new_buckets.T
        im.set_data(xcenters, ycenters, new_buckets)
        ax.images.append(im)

        tag_name = "test/viz"
        writer.add_figure(tag=tag_name, figure=fig, global_step=epoch)

        fig = plt.figure()
        ax = fig.add_subplot(111, title='normalized',
                aspect='equal', xlim=edges[[0, -1]], ylim=edges[[0, -1]])
        im = NonUniformImage(ax, interpolation='nearest')
        xcenters = (edges[:-1] + edges[1:]) / 2
        ycenters = (edges[:-1] + edges[1:]) / 2
        avg = (nLoops*hp.viz_batch_size)/nBuckets**2
        new_buckets += 1
        new_buckets = (np.sqrt(new_buckets)/np.sqrt(new_buckets.max())) * avg
        im.set_data(xcenters, ycenters, new_buckets)
        ax.images.append(im)

        tag_name = "test/norm"
        writer.add_figure(tag=tag_name, figure=fig, global_step=epoch)


def visualize_movement(hp: HParams, ds: ComboDataset, writer: SummaryWriter, model: SketchAAE, epoch, old_buckets):
    def _np(item):
        return item.detach().cpu().numpy()

    with torch.no_grad():
        if writer is None:
            return

        if old_buckets is None:
            old_buckets = np.zeros((hp.cat_dims, hp.viz_buckets, hp.viz_buckets))
        new_buckets = np.zeros((hp.cat_dims, hp.viz_buckets, hp.viz_buckets))
        chg_buckets = np.zeros((hp.cat_dims, hp.viz_buckets, hp.viz_buckets))

        category_buckets = np.zeros((ds.NumCategoryTruths(), hp.cat_dims, hp.viz_buckets, hp.viz_buckets))
        
        dl = DataLoader(ds, hp.viz_batch_size)
        model.eval()
        print("visualize_movement")
        
            
        for	iter, batch_sample in enumerate(dl):
            batch, lengths, batch_indexes = batch_sample
            batch_len = len(batch_indexes)

            batch = batch.transpose(0,1)

            new_cats, new_styles = model.encoder(batch, batch_len)
            new_cats = torch.argmax(new_cats, dim=1)

            np_new_styles = _np(new_styles)
            np_new_cats = _np(new_cats)

            for i in range(hp.cat_dims):
                in_category = np.nonzero((np_new_cats==i))
                new_x = np_new_styles[in_category,0].flatten()
                new_y = np_new_styles[in_category,1].flatten()
                H, _, _ = np.histogram2d(x=new_x, y=new_y, bins=hp.viz_buckets, range=((0,1),(0,1)))
                new_buckets[i,:,:] += H[:,:]

            truths = ds.GetCategoryTruths(batch_indexes)
            for t_inx in range(ds.NumCategoryTruths()):
                for c_inx in range(hp.cat_dims):
                    t1 = (np_new_cats==c_inx)
                    t2 = (truths==t_inx)
                    in_category = np.nonzero((np_new_cats==c_inx) & (truths==t_inx))
                    new_x = np_new_styles[in_category,0].flatten()
                    new_y = np_new_styles[in_category,1].flatten()
                    H, _, _ = np.histogram2d(x=new_x, y=new_y, bins=hp.viz_buckets, range=((0,1),(0,1)))
                    category_buckets[t_inx,c_inx,:,:] += H[:,:]

            gc.collect()

        del dl
        gc.collect()

        for i in range(hp.cat_dims):
            chg_buckets[i,:,:] = new_buckets[i,:,:] - old_buckets[i,:,:]

        edges = np.linspace(0, 1, num=hp.viz_buckets+1)
        xcenters = (edges[:-1] + edges[1:]) / 2
        ycenters = (edges[:-1] + edges[1:]) / 2
        avg = ds.length/hp.viz_buckets**2

        for i in range(hp.cat_dims):
            tag_name = "newbuckets/"+str(i)+"cat"
            h_val = new_buckets[i]
            h_val.T

            fig = plt.figure()
            ax = fig.add_subplot(111, title='new buckets',
                    aspect='equal', xlim=edges[[0, -1]], ylim=edges[[0, -1]])
            im = NonUniformImage(ax, interpolation='nearest')
            im.set_data(xcenters, ycenters, h_val)
            ax.images.append(im)
            writer.add_figure(tag=tag_name, figure=fig, global_step=epoch)

            fig = plt.figure()
            ax = fig.add_subplot(111, title='normalized',
                    aspect='equal', xlim=edges[[0, -1]], ylim=edges[[0, -1]])
            im = NonUniformImage(ax, interpolation='nearest')
            h_val = (np.sqrt(h_val + 1)/np.sqrt(h_val.max() + 1)) * avg
            im.set_data(xcenters, ycenters, h_val)
            ax.images.append(im)
            writer.add_figure(tag=tag_name + "_norm", figure=fig, global_step=epoch)

        for i in range(hp.cat_dims):
            fig = plt.figure()
            ax = fig.add_subplot(111, title='changes in buckets',
                    aspect='equal', xlim=edges[[0, -1]], ylim=edges[[0, -1]])
            im = NonUniformImage(ax, interpolation='nearest')
            h_val = chg_buckets[i]
            h_val.T
            im.set_data(xcenters, ycenters, h_val)
            ax.images.append(im)

            tag_name = "bucketchanges/"+str(i)+"cat"
            writer.add_figure(tag=tag_name, figure=fig, global_step=epoch)

        for t_inx in range(ds.NumCategoryTruths()):
            for c_inx in range(hp.cat_dims):
                tag_name = "truth_buckets_"+ds.cat_labels[t_inx]+"/"+str(c_inx)
                h_val = category_buckets[t_inx, c_inx]
                h_val.T

                fig = plt.figure()
                ax = fig.add_subplot(111,title='cat'+str(c_inx),
                    aspect='equal', xlim=edges[[0, -1]], ylim=edges[[0, -1]])
                im = NonUniformImage(ax, interpolation='nearest')
                im.set_data(xcenters, ycenters, h_val)
                ax.images.append(im)
                writer.add_figure(tag=tag_name, figure=fig, global_step=epoch)

                fig = plt.figure()
                ax = fig.add_subplot(111,title='norm'+str(c_inx),
                        aspect='equal', xlim=edges[[0, -1]], ylim=edges[[0, -1]])
                im = NonUniformImage(ax, interpolation='nearest')
                h_val = (np.sqrt(h_val + 1)/np.sqrt(h_val.max() + 1)) * avg
                im.set_data(xcenters, ycenters, h_val)
                ax.images.append(im)
                writer.add_figure(tag=tag_name + "_norm", figure=fig, global_step=epoch)

        return new_buckets
        