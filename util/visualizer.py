import numpy as np
import os
import ntpath
import time
from . import util
from psbody.mesh import Mesh

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.name = opt.name
        from tensorboardX import SummaryWriter
        self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.frame_visualizer = FrameVisulizer(opt.template_path)

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        for tag, value in errors.items():
            self.writer.add_scalar(tag, value, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.9f) ' % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.9f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
    
    def plot_valid_errors(self, errors, epoch):
        for tag, value in errors.items():
            self.writer.add_scalar("valid/" + tag, value, epoch)
    
    def print_valid_errors(self, errors, epoch):
        message = f'Validation at epoch {epoch}: '
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.9f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
    
class FrameVisulizer():
    
    def __init__(
        self,
        template_path: str,
    ) -> None:
        self.template = Mesh(filename = template_path)
    
    def _visualize_array(self, frames: np.ndarray, output_dir: str):
        frames = frames.squeeze()
        for index, frame in enumerate(frames):
            frame_mesh = Mesh(v = frame.reshape(-1, 3), f = self.template.f)
            frame_mesh.write_obj(os.path.join(f"{output_dir}", f"{index:06}.obj"))
    
    def _visualize_file(self, frames_path: str, output_dir: str):
        frames = np.load(frames_path).squeeze()
        return self._visualize_array(frames, output_dir)

    def visualize(self, frames, output_dir):
        if isinstance(frames, np.ndarray):
            return self._visualize_array(frames, output_dir)
        if isinstance(frames, str):
            return self._visualize_file(frames, output_dir)
        raise TypeError("frames must be either a numpy array or a path to a numpy array")
    
    def __call__(self, frames, output_dir):
        return self.visualize(frames, output_dir)