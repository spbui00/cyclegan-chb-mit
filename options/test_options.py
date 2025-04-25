from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """Defines options for testing (inference)."""
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # overwrite some training defaults
        parser.set_defaults(preprocess='none', batch_size=1, serial_batches=True, no_flip=True)
        # test‐specific options
        parser.add_argument('--n_blocks', type=int, default=6, help='number of ResNet blocks in the 1D generator (must match training)')
        parser.add_argument('--results_dir', type=str, default='./results', help='where to save the test HTML')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--num_test', type=int, default=0, help='how many samples to run; 0 means all')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time. This is used to turn off dropout and batchnorm')
        self.isTrain = False
        return parser

