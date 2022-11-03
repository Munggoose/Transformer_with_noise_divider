import torch
import os

class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.device = self._device_set()
        self.model = self._build_model().to(self.device)
    

    def _build_model(self):
        raise NotImplementedError
    
    
    def _device_set(self):
        if self.args.use_gpu:
            os.environ['CUDA_VISIBLE_DEIVCES'] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

        def _get_data(self,*args, **kwargs):
            pass
        
        def vali(self, *args, **kwargs):
            pass

        def train(self, *args, **kwargs):
            pass

        def test(self, *args, **kwargs):
            pass


