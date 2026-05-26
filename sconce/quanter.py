import copy
import torch
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

# Warning suppression is handled centrally in utils.py (imported at package init).


class quantization:
    def __init__(self):
        pass

    def qat(self):
        print(
            "\n \n========================== Quantization-Aware Training(QAT) ===================================" )

        model_to_quantize = copy.deepcopy(self.model)
        model_to_quantize.train()

        batch = next(iter(self.dataloader["test"]))
        example_input = batch[0] if isinstance(batch, (list, tuple)) else batch
        example_input = example_input[:1].to(self.device or "cpu")
        example_inputs = (example_input,)

        backend = self.qat_config or "x86"
        qconfig_mapping = get_default_qat_qconfig_mapping(backend)
        model_prepared = prepare_qat_fx(
            model_to_quantize,
            qconfig_mapping,
            example_inputs,
        )

        self.model = model_prepared
        self.train()

        model_fp32_trained = copy.deepcopy(self.model).to("cpu").eval()
        model_int8 = convert_fx(model_fp32_trained)

        return model_int8, model_fp32_trained
