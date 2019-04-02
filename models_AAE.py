import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from HParams import HParams

TINY = 1e-15

class IdentityBlock(nn.Module):
    def __init__(self, input_size, output_size, blocks):
        super(IdentityBlock, self).__init__()

        layers = []
        first_size = last_size = input_size
        for count, (layers_count, size, dropout) in enumerate(blocks):
            if count == 0:
                last_size = first_size = size
            for i in range(layers_count):
                layers.append(Block(last_size, size, dropout))
            last_size = size


        self.gen_seq = nn.Sequential(
            nn.Linear(input_size, first_size),
            nn.Sequential(*layers),
            nn.Linear(last_size, output_size),
        )

        self.downsample = nn.Linear(input_size, output_size)

    def forward(self, input):
        rtn = self.gen_seq(input) + self.downsample(input)
        return rtn

class Block(nn.Module):
    def __init__(self, first_size, layer_size, dropout):
        super(Block, self).__init__()

        self.gen_seq = nn.Sequential(
            nn.Linear(first_size, layer_size),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(layer_size),
            nn.Linear(layer_size, layer_size),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(layer_size),
            nn.Linear(layer_size, layer_size),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(layer_size)
        )

    def forward(self, input):
        return self.gen_seq(input)

class InnerEncoder(nn.Module):
    def __init__(self, hp: HParams):
        super(InnerEncoder, self).__init__()

        self.input_size = 2*hp.enc_hidden_size
        self.hidden_size = hp.enc_hidden_size

        self.gen_seq = IdentityBlock(self.input_size, self.hidden_size, hp.encoder_base_blocks)

        self.lin_style = IdentityBlock(self.hidden_size, hp.style_dims, hp.encoder_s_blocks) # nn.Linear(self.hidden_size, hp.style_dims)
        self.lin_cat = IdentityBlock(self.hidden_size, hp.cat_dims, hp.encoder_c_blocks) #nn.Linear(self.hidden_size, hp.cat_dims)

        self.noise = GaussianNoise()

    def forward(self, input):
        z = self.gen_seq(2*torch.tanh(input))

        style = torch.sigmoid(self.noise(self.lin_style(z)))
        cat = F.softmax(self.lin_cat(z))

        return cat, style

    def reset_grads(self):
        init_layers(self)

class InnerDecoder(nn.Module):
    def __init__(self, hp: HParams):
        super(InnerDecoder, self).__init__()

        self.input_size = hp.style_dims + hp.cat_dims
        self.output_size = 2*hp.enc_hidden_size
        
        self.gen_seq = IdentityBlock(self.input_size, self.output_size, hp.decoder_base_blocks)

    def forward(self, cat_style):
        return self.gen_seq(cat_style)

    def reset_grads(self):
        init_layers(self)


# Encoder
class EncoderAAE(nn.Module):
    def __init__(self, encoder: InnerEncoder, hp: HParams):
        super(EncoderAAE, self).__init__()
        # bidirectional lstm:
        self.encoder = encoder
        input_size = int(self.encoder.input_size/2)

        self.lstm = nn.LSTM(5, input_size, bidirectional=True)

        self.gen_seq = IdentityBlock(self.encoder.input_size, self.encoder.input_size, hp.encoder_base_blocks)
        
        init_layers(self)

        self.train()
        self.hp = hp

    def forward(self, inputs, batch_size, hidden_cell=None):
        if hidden_cell is None:
            # then must init with zeros
            hidden = torch.zeros(2, batch_size, self.hp.enc_hidden_size, device=inputs.device)
            cell = torch.zeros(2, batch_size, self.hp.enc_hidden_size, device=inputs.device)
            hidden_cell = (hidden, cell)
        _, (hidden,cell) = self.lstm(inputs, hidden_cell)
        # hidden is (2, batch_size, hidden_size), we want (batch_size, 2*hidden_size):
        hidden_forward, hidden_backward = torch.split(hidden,1,0)
        hidden_concat = torch.cat([hidden_forward.squeeze(0), hidden_backward.squeeze(0)],1)
        hidden_concat = F.dropout(hidden_concat, training=self.training, p=self.hp.dropout)  #adding relu here does not train well... 

        cat, style = self.encoder(self.gen_seq(hidden_concat))

        return cat, style




# Decoder
class DecoderAAE(nn.Module):
    def __init__(self, decoder: InnerDecoder, hp: HParams):
        super(DecoderAAE, self).__init__()
        self.lin_hc = nn.Linear(hp.style_dims + hp.cat_dims, 2*hp.dec_hidden_size)
        
        self.decoder = decoder
        self.gen_seq = IdentityBlock(self.decoder.output_size, self.decoder.output_size, hp.decoder_base_blocks)

        self.lstm = nn.LSTM(self.decoder.output_size+5, hp.dec_hidden_size)
        # create proba distribution parameters from hiddens:
        self.fc_params = nn.Linear(hp.dec_hidden_size,6*hp.M+3)

        init_layers(self)

        self.train()
        self.hp = hp

    def forward(self, inputs, cat_style, Nmax, hidden_cell=None):
        if hidden_cell is None:
            # then we must init from cat_style
            hidden,cell = torch.split(torch.tanh(self.lin_hc(cat_style)),self.hp.dec_hidden_size,1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        
        z = self.gen_seq(self.decoder(cat_style))

        if self.training:
            z_stack = torch.stack([z]*(Nmax+1))
            z_gen = torch.cat([inputs, z_stack],2)
        else:
            z = z.unsqueeze(0)
            z_gen = torch.cat([inputs, z], 2)

        outputs,(hidden,cell) = self.lstm(z_gen, hidden_cell)

        # in training we feed the lstm with the whole input in one shot
        # and use all outputs contained in 'outputs', while in generate
        # mode we just feed with the last generated sample:
        if Nmax > 0: #self.training
            fc_input = outputs.view(-1, self.hp.dec_hidden_size)
        else:
            fc_input = hidden.view(-1, self.hp.dec_hidden_size)

        fc_input = F.dropout(fc_input, training=self.training, p=self.hp.dropout) #adding relu here does not train well... 
        y = self.fc_params(fc_input)
        # separate pen and mixture params:
        params = torch.split(y,6,1)
        params_mixture = torch.stack(params[:-1]) # trajectory
        params_pen = params[-1] # pen up/down
        # identify mixture params:
        pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy = torch.split(params_mixture,1,2)
        # preprocess params::
        if self.training:
            len_out = Nmax+1
        else:
            len_out = 1
        
        pi = pi.transpose(0,1).squeeze().view(1,-1,self.hp.M)
        pi = F.softmax(pi,dim=2).view(len_out,-1,self.hp.M)
        mu_x = mu_x.transpose(0,1).squeeze().contiguous().view(len_out,-1,self.hp.M)
        mu_y = mu_y.transpose(0,1).squeeze().contiguous().view(len_out,-1,self.hp.M)
        sigma_x = torch.exp(sigma_x.transpose(0,1).squeeze()).view(len_out,-1,self.hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0,1).squeeze()).view(len_out,-1,self.hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0,1).squeeze()).view(len_out,-1,self.hp.M)
        q = F.softmax(params_pen,dim=1).view(len_out,-1,3)

        return pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy,q,hidden,cell

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            x = x + torch.randn_like(x) * scale
        return x 

def init_layers(module: nn.Module):
    for m in module.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.zero_() 