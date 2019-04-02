import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from HParams import HParams
from models_AAE import DecoderAAE, EncoderAAE, InnerDecoder, InnerEncoder

TINY = 1e-15

class SketchAAE():
    def __init__(self, hp: HParams):
        self.hp = hp
        self.device = torch.device("cuda" if torch.cuda.is_available() and not hp.fast_debug else "cpu") #and os.name != 'nt'  #and not hp.fast_debug
        self.innerEncoder = InnerEncoder(self.hp).to(self.device)
        self.innerDecoder = InnerDecoder(self.hp).to(self.device)
        self.encoder = EncoderAAE(self.innerEncoder, self.hp).to(self.device)
        self.decoder = DecoderAAE(self.innerDecoder, self.hp).to(self.device)
        self.create_optims()

    def create_optims(self):
        self.train(inner=False, outer=True, skip_zero_grad=True)
        self.encoder_params =filter(lambda p: p.requires_grad, self.encoder.parameters())
        self.decoder_params =filter(lambda p: p.requires_grad, self.decoder.parameters())
        self.fw_encoder_optimizer = optim.Adam(self.encoder_params, self.hp.lr)
        self.fw_decoder_optimizer = optim.Adam(self.decoder_params, self.hp.lr)
        self.inner_encoder_optimizer = optim.Adam(self.innerEncoder.parameters(), self.hp.inner_lr)
        self.inner_decoder_optimizer = optim.Adam(self.innerDecoder.parameters(), self.hp.inner_lr)

    def zero_grad_models(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.innerEncoder.zero_grad()
        self.innerDecoder.zero_grad()

    def zero_grad_optims(self):
        self.fw_encoder_optimizer.zero_grad()
        self.fw_decoder_optimizer.zero_grad()
        self.inner_encoder_optimizer.zero_grad()
        self.inner_decoder_optimizer.zero_grad()

    def reset_inner(self):
        self.innerEncoder.reset_grads()
        self.innerDecoder.reset_grads()

    def lr_decay(self, outer=False, inner=False):
        def opt_decay(optimizer, hp: HParams):
            """Decay learning rate by a factor of lr_decay"""
            for param_group in optimizer.param_groups:
                if param_group['lr']>hp.min_lr:
                    param_group['lr'] *= hp.lr_decay

        if outer:
            opt_decay(self.fw_encoder_optimizer, self.hp)
            opt_decay(self.fw_decoder_optimizer, self.hp)

        if inner:
            opt_decay(self.inner_encoder_optimizer, self.hp)
            opt_decay(self.inner_decoder_optimizer, self.hp)

    
    def train(self, outer=True, inner=True, skip_zero_grad=False):
        self.encoder.train(outer)
        self.decoder.train(outer)
        self.innerEncoder.train(inner)
        self.innerDecoder.train(inner)

        for param in self.encoder.parameters():
            param.requires_grad = outer
        for param in self.decoder.parameters():
            param.requires_grad = outer
        for param in self.innerEncoder.parameters():
            param.requires_grad = inner
        for param in self.innerDecoder.parameters():
            param.requires_grad = inner

        if not skip_zero_grad:
            self.zero_grad_models()
            self.zero_grad_optims()

    def eval(self):
        self.train(outer=False, inner=False)

    def optim_step(self, inner=True, outer=True):
        if inner:
            nn.utils.clip_grad_norm_(self.innerEncoder.parameters(), self.hp.grad_clip)
            nn.utils.clip_grad_norm_(self.innerDecoder.parameters(), self.hp.grad_clip)

            self.inner_encoder_optimizer.step()
            self.inner_decoder_optimizer.step()
        
        if outer:
            nn.utils.clip_grad_norm_(self.encoder_params, self.hp.grad_clip)
            nn.utils.clip_grad_norm_(self.decoder_params, self.hp.grad_clip)

            self.fw_encoder_optimizer.step()
            self.fw_decoder_optimizer.step()

        self.zero_grad_models()
        self.zero_grad_optims()        



    def train_inner(self, train=True):
        def sample_categorical(batch_size, n_classes):
            cat = np.random.randint(0, n_classes, batch_size)
            onehot = np.eye(n_classes)[cat].astype('float32')
            return torch.from_numpy(onehot)

        def CELoss(x_pred,x_target,use_mean=True):
            assert x_target.size() == x_pred.size(), "size fail ! "+str(x_target.size()) + " " + str(x_pred.size())
            logged_x_pred = torch.log(TINY+x_pred)
            if use_mean:
                cost_value = torch.mean(-torch.sum(x_target * logged_x_pred, dim=1))
            else:
                cost_value = -torch.sum(x_target * logged_x_pred)
            return cost_value

        self.train(inner=train, outer=False)
        
        
        b_size = self.hp.batch_size
        input_onehot = sample_categorical(b_size, n_classes=self.hp.cat_dims).to(self.device)
        input_style = torch.rand(b_size, self.hp.style_dims, device=self.device) #
        #not sure if we should do a softmax on input_onehot...
        cat_style = torch.cat([input_onehot, input_style],1).requires_grad_()
        z = self.innerDecoder(cat_style)
        out_onehot, out_style = self.innerEncoder(z)
        LC = CELoss(out_onehot, input_onehot)
        LS = F.mse_loss(out_style, input_style)
        L3 = MapLoss(out_style, varPct=0.01)
        L1 = LC+LS+L3
        if train:  
            L1.backward()

        self.optim_step(inner=train, outer=False)

        input_encoder = F.softsign(torch.randn(b_size, self.innerEncoder.input_size, device=self.device)).requires_grad_()
        out_cat, out_style2 = self.innerEncoder(input_encoder)
        cat_style2 = torch.cat([out_cat, out_style2],1)
        output_decoder = self.innerDecoder(cat_style2)

        LF = F.mse_loss(output_decoder, input_encoder)
        L2 = LF
        if train:
            L2.backward()

        self.optim_step(inner=train, outer=False)
        
        loss = L1+L2

        return val(loss), val(LF), val(LC), val(LS)





    def train_reconstruction(self, batch, lengths):
        self.train(inner=False, outer=True)
        LS, LP, map_loss = self._forward(batch, lengths)
        loss = LS+LP+map_loss
        loss.backward()
        self.optim_step(inner=False, outer=True)

        return val(loss), val(LS), val(LP), val(map_loss)

    def _forward(self, batch, lengths):
        batch_len = batch.size()[1]
        cat, style = self.encoder(batch, batch_len)
        sos = torch.stack([torch.tensor([0,0,1,0,0], device=self.device, dtype=torch.float)]*batch_len).unsqueeze(0)
        inputs = torch.cat([sos, batch],0)

        z = torch.cat([cat, style],1)

        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, self.q, _, _ = self.decoder(inputs, z, self.Nmax, hidden_cell=None)
        mask,dx,dy,p = self.make_target(batch, lengths)
        LS, LP = self.reconstruction_loss(mask,dx,dy,p)
        map_loss = MapLoss(style, cntPct=0.00000001, varPct=0.000000001) #moving center, leave var at 0.000000001
        return LS, LP, map_loss

    def make_target(self, batch, lengths):
        eos = torch.stack([torch.tensor([0,0,0,0,1], device=self.device, dtype=torch.float)]*batch.size()[1]).unsqueeze(0)
        batch = torch.cat([batch, eos], 0)
        mask = torch.zeros(self.Nmax+1, batch.size()[1], device=self.device)
        for indice,length in enumerate(lengths):
            mask[:length,indice] = 1
        dx = torch.stack([batch.data[:,:,0]]*self.hp.M,2)
        dy = torch.stack([batch.data[:,:,1]]*self.hp.M,2)
        p1 = batch.data[:,:,2]
        p2 = batch.data[:,:,3]
        p3 = batch.data[:,:,4]
        p = torch.stack([p1,p2,p3],2)
        return mask,dx,dy,p

    def bivariate_normal_pdf(self, dx, dy):
        z_x = ((dx-self.mu_x)/self.sigma_x)**2
        z_y = ((dy-self.mu_y)/self.sigma_y)**2
        z_xy = (dx-self.mu_x)*(dy-self.mu_y)/(self.sigma_x*self.sigma_y)
        z = z_x + z_y -2*self.rho_xy*z_xy
        exp = torch.exp(-z/(2*(1-self.rho_xy**2)))
        norm = 2*np.pi*self.sigma_x*self.sigma_y*torch.sqrt(1-self.rho_xy**2)
        return exp/norm

    def reconstruction_loss(self, mask, dx, dy, p):
        len = mask.size()[1]
        pdf = self.bivariate_normal_pdf(dx, dy)
        ls_all = mask*torch.log(TINY+torch.sum(self.pi * pdf, 2))
        #not checking min because it performs worse
        LS = -torch.sum(ls_all)/float(self.Nmax*len)

        lp_all = p*torch.log(TINY+self.q)
        #lp_zeros = torch.zeros_like(lp_all).detach()
        #lp_all = torch.min(lp_all, lp_zeros)
        LP = -torch.sum(lp_all)/float(self.Nmax*len)
        return LS, LP

    def generation_for_category(self, category, x_count=10, y_count=10, x_offset=8, y_offset=8):
        with torch.no_grad():
            self.eval()

            cat = torch.zeros((self.hp.cat_dims), device=self.device)
            cat[category] = 1
            cat = F.softmax(cat, dim=0)
            out = []
            style = torch.zeros((self.hp.style_dims), device=self.device)
            xedges = np.linspace(0, 1, num=x_count)
            yedges = np.linspace(0, 1, num=y_count)
            for y in range(y_count):
                for x in range(x_count):
                    style[0] = xedges[x]
                    style[1] = yedges[y]
                    z = torch.cat([cat, style]).unsqueeze(0)
                    s = torch.tensor([0,0,1,0,0], device=self.device, dtype=torch.float).view(1,1,-1)
                    seq_x = []
                    seq_y = []
                    seq_z = []
                    hidden_cell = None
                    for i in range(self.hp.max_seq_length):
                        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                            self.rho_xy, self.q, hidden, cell = \
                                self.decoder(s, z, 0, hidden_cell=hidden_cell)
                        hidden_cell = (hidden, cell)
                        # sample from parameters:
                        s, dx, dy, pen_down, eos = self.sample_next_state()
                        #------
                        if i == self.hp.max_seq_length - 1:
                            eos = True
                        seq_x.append(dx)
                        seq_y.append(dy)
                        seq_z.append(pen_down or eos)
                        if eos:
                            break

                    # visualize result:
                    x_sample = np.cumsum(seq_x, 0)
                    y_sample = np.cumsum(seq_y, 0)
                    
                    if x_sample.min() < 0:
                        x_sample[:] += -x_sample.min()

                    if y_sample.min() < 0:
                        y_sample[:] += -y_sample.min()
                    
                    x_sample = np.abs(x_sample)
                    y_sample = np.abs(y_sample)
                    width = x_sample.max()
                    height = y_sample.max()

                    v_max = max(width, height)
                    if v_max > (x_offset-1) or v_max > (y_offset-1):
                        x_scale = ((x_offset-1) / v_max)
                        y_scale = ((y_offset-1) / v_max)
                        x_sample[:] *= x_scale
                        y_sample[:] *= y_scale

                    #switching x/y here is intentional
                    x_sample[:] += (y * y_offset) 
                    y_sample[:] -= (x * x_offset)

                    z_sample = np.array(seq_z)
                    sequence = np.stack([x_sample,y_sample,z_sample]).T
                    out.append(sequence)

            return np.concatenate(out)

    def conditional_generation(self, batch, lengths, Nmax):
       
        # should remove dropouts:
        self.eval()

        with torch.no_grad():
            # encode:
            batch = torch.unsqueeze(batch, 1)
            cat, style = self.encoder(batch, 1)
            #cat = F.softmax(cat, dim=0)
            z = torch.cat([cat, style],1)
            s = torch.tensor([0,0,1,0,0], device=self.device, dtype=torch.float).view(1,1,-1)
            seq_x = []
            seq_y = []
            seq_z = []
            hidden_cell = None
            for i in range(Nmax):
                # decode:
                self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                    self.rho_xy, self.q, hidden, cell = \
                        self.decoder(s, z, Nmax, hidden_cell=hidden_cell)
                hidden_cell = (hidden, cell)
                # sample from parameters:
                s, dx, dy, pen_down, eos = self.sample_next_state()
                #------
                seq_x.append(dx)
                seq_y.append(dy)
                seq_z.append(pen_down)
                if eos:
                    print("count: ", i)
                    break

            # visualize result:
            x_sample = np.cumsum(seq_x, 0)
            y_sample = np.cumsum(seq_y, 0)
            z_sample = np.array(seq_z)
            sequence = np.stack([x_sample,y_sample,z_sample]).T
            return sequence

    def sample_next_state(self):

        # get mixture indice:
        pi = self.adjust_temp_tensor(self.pi.data[0,0,:]).cpu().numpy()
        pi_idx = np.random.choice(self.hp.M, p=pi)
        # get pen state:
        q = self.adjust_temp_tensor(self.q.data[0,0,:]).cpu().numpy()
        q_idx = np.random.choice(3, p=q)
        # get mixture params:
        mu_x = self.mu_x.data[0,0,pi_idx].cpu().numpy()
        mu_y = self.mu_y.data[0,0,pi_idx].cpu().numpy()
        sigma_x = self.sigma_x.data[0,0,pi_idx].cpu().numpy()
        sigma_y = self.sigma_y.data[0,0,pi_idx].cpu().numpy()
        rho_xy = self.rho_xy.data[0,0,pi_idx].cpu().numpy()
        x,y = self.sample_bivariate_normal(mu_x,mu_y,sigma_x,sigma_y,rho_xy,greedy=False)
        next_state = torch.zeros(5, device=self.device)
        next_state[0] = x
        next_state[1] = y
        next_state[q_idx+2] = 1
        return next_state.view(1,1,-1),x,y,q_idx==1,q_idx==2


    def adjust_temp_tensor(self,pi_pdf):
        pi_pdf = torch.log(pi_pdf)/self.hp.temperature
        pi_pdf -= pi_pdf.max()
        pi_pdf = torch.exp(pi_pdf)
        pi_pdf /= pi_pdf.sum()
        return pi_pdf

    def sample_bivariate_normal(self,mu_x,mu_y,sigma_x,sigma_y,rho_xy, greedy=False):
        # inputs must be floats
        if greedy:
            return mu_x,mu_y
        mean = [mu_x, mu_y]
        sigma_x *= np.sqrt(self.hp.temperature)
        sigma_y *= np.sqrt(self.hp.temperature)
        cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],\
            [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]





    def Export(self, epoch):
        state = {
			'encoder': self.encoder.state_dict(),
			'decoder': self.decoder.state_dict(),
            'innerEncoder': self.innerEncoder.state_dict(),
			'innerDecoder': self.innerDecoder.state_dict(),
			'encoder_optimizer': self.fw_encoder_optimizer.state_dict(),
			'decoder_optimizer': self.fw_decoder_optimizer.state_dict(),
            'inner_encoder_optimizer': self.inner_encoder_optimizer.state_dict(),
			'inner_decoder_optimizer': self.inner_decoder_optimizer.state_dict(),
            'hp': self.hp,
            'epoch': epoch
		}
        return state

    def Import(self, state):
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
        self.fw_encoder_optimizer.load_state_dict(state["encoder_optimizer"])
        self.fw_decoder_optimizer.load_state_dict(state["decoder_optimizer"])
        self.innerEncoder.load_state_dict(state["innerEncoder"])
        self.innerDecoder.load_state_dict(state["innerDecoder"])
        self.inner_encoder_optimizer.load_state_dict(state["inner_encoder_optimizer"])
        self.inner_decoder_optimizer.load_state_dict(state["inner_decoder_optimizer"])
        self.hp = state["hp"]
        return state["epoch"]

def val(loss):
    return loss.detach().cpu().item()

def MapLoss(map, cntPct = 1, varPct = 1):
    L_CENTERED = torch.abs(torch.mean(map) - 0.5) * cntPct
    var = torch.var(map)
    L_HIST = (((var+1)/var)-1) * varPct 
    return L_CENTERED + L_HIST