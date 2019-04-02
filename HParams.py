class HParams():
    def __init__(self):
        self.sketches_folder = 'sketches'
        self.models_folder = 'models'
        self.max_epochs = 50000

        self.M = 20
        self.batch_size = 100  
        self.lr = 0.001
        self.inner_lr = 0.001
        self.lr_decay = 0.9998
        self.min_lr = 0.00001
        self.dropout = 0.1
        self.grad_clip = 1.
        self.temperature = 0.4
        self.max_seq_length = 200

        self.style_dims = 2
        self.cat_dims = 2

        self.enc_hidden_size = 256
        self.dec_hidden_size = 256
        self.decoder_base_blocks = [(4,256, 0.2)]
        self.encoder_base_blocks = [(4,256, 0.2)]
        self.encoder_c_blocks = [(2,128, 0.2), (1,32, 0.2)]
        self.encoder_s_blocks = [(2,128, 0.2), (1,32, 0.2)]
        

        self.viz_buckets = 200
        self.viz_batch_size = 500


        self.fast_debug = False
        self.tensorboard = True
