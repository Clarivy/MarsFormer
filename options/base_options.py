import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='label2city', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--gpu_id', type=str, default='0')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='pix2pixHD', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')

        # input/output sizes       
        self.parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
        self.parser.add_argument('--vertice_dim', type=int, default=14062 * 3, help='number of vertices to this size')
        self.parser.add_argument('--max_len', type=int, default=400, help='number of maximum frame num')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./data/GNPFA/') 

        # for dataset dividing
        self.parser.add_argument("--train_subjects", type=str, default=
            "014_blendshape "
            "044blendshape "
            "045blendshape "
            "046blendshape "
            "047blendshape "
            "064blendshape "
        )
        self.parser.add_argument("--valid_subjects", type=str, default=
            "Ada_OURuv_aligned "
        )

        # for displays
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test
        self.train_subjects = sorted([i for i in self.opt.train_subjects.split(" ")])
        self.valid_subjects = sorted([i for i in self.opt.valid_subjects.split(" ")])

        self.opt.gpu_id = int(self.opt.gpu_id)
        torch.cuda.set_device(self.opt.gpu_id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
