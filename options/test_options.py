from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('model', nargs=1, help='trained model')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--save_scores', action='store_true', help='save the output scores')
        parser.add_argument('--save_path', default='scores.txt', help='output path')
        parser.add_argument('--threshold', type=float, default=0.75, help='threshold for logits to labels')
        return parser