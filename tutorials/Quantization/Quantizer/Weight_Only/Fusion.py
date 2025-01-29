import torch
from quant.ModelAnalyzer import ModelAnalyzer
import torch.ao.quantization.quantize_fx as quantize_fx

class Fuse(ModelAnalyzer):

    def __init__(self, model, calibiration_data=None):

        self.model = model
        self.fused_model = None
        self.calibiration_data = calibiration_data
        self.fuse_model()

    def fuse_model(self):

        ma = ModelAnalyzer(self.model, self.calibiration_data)
        mapped_layers = ma.mapped_layers
        self.fused_model = self.fuse_layers(fusable_layers=mapped_layers['fusable_layers'])

    def fuse_layers(self, fusable_layers):
        fusing_layers= []
        for outer_idx in range(len(fusable_layers)):
            if(len(fusable_layers[outer_idx])>1):
                pipo = []
                for idx in range(len(fusable_layers[outer_idx])):
                    pipo.append(fusable_layers[outer_idx][idx][6:])
                fusing_layers.append(pipo)

        try: #Try with Custom Fusion
            model_fused = torch.quantization.fuse_modules(self.model, fusing_layers, inplace=True)
        except:
            model_fused = quantize_fx.fuse_fx(self.model)

        return model_fused
    
