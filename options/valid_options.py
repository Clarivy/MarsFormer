from .base_options import BaseOptions

class ValidOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./data/results/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='valid', help='train, val, debug, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--no_obj', action='store_true', help='whether to build obj frames')

        self.parser.add_argument("--condition_subject", type=str, default=
            "014_blendshape",
            # "044blendshape "
            # "045blendshape "
            # "046blendshape "
            # "047blendshape "
            # "064blendshape "
            help='subject to condition on, must be in train_subjects'
        )
        self.isTrain = False
