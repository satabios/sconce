import os

import snntorch
from snntorch import functional as SF
from snntorch import utils

import torch
from torch import nn

import time
from prettytable import PrettyTable
from tqdm import tqdm

class PerformanceEval:
    def __init__(self, dataloader, snn, snn_num_steps) -> None:
        self.dataloader = dataloader
        self.snn = snn
        self.snn_num_steps = snn_num_steps

    def load_torchscript_model(self, model_filepath, device):
        model = torch.jit.load(model_filepath, map_location=device)
        return model

    def measure_inference_latency(
			self, model, device, input_data, num_samples=100, num_warmups=10
	):
        model.to(device)
        model.eval()
        
        x = input_data.to(device)
        
        with torch.no_grad():
            for _ in range(num_warmups):
                _ = model(x)
        torch.cuda.synchronize()
        
        with torch.no_grad():
            start_time = time.time()
            for _ in range(num_samples):
                _ = model(x)
                torch.cuda.synchronize()
            end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_ave = elapsed_time / num_samples
        
        return elapsed_time_ave

    @torch.no_grad()
    def measure_latency(self, model, dummy_input, n_warmup=20, n_test=100):
        """
		Measures the average latency of a given PyTorch model by running it on a dummy input multiple times.

		Args:
		  model (nn.Module): The PyTorch model to measure the latency of.
		  dummy_input (torch.Tensor): A dummy input to the model.
		  n_warmup (int, optional): The number of warmup iterations to run before measuring the latency. Defaults to 20.
		  n_test (int, optional): The number of iterations to run to measure the latency. Defaults to 100.

		Returns:
		  float: The average latency of the model in milliseconds.
		"""
        model = model.to("cpu")
        
        model.eval()
        
        dummy_input = dummy_input.to("cpu")
        
        if self.snn:
            if isinstance(model, nn.Sequential):
                for layer_id in range(len(model)):
                    layer = model[layer_id]
                    if isinstance((layer), snntorch._neurons.leaky.Leaky):
                        layer.mem = layer.mem.to("cpu")
            else:
                for module in model.modules():
                    if isinstance((module), snntorch._neurons.leaky.Leaky):
                        module.mem = module.mem.to("cpu")
        
        # warmup
        for _ in range(n_warmup):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        # real test
        t1 = time.time()
        for _ in range(n_test):
            _ = model(dummy_input)
            torch.cuda.synchronize()
        t2 = time.time()
        return (t2 - t1) / n_test  # average latency in ms

    @torch.no_grad()
    def evaluate(self, model, device=None, Tqdm=True, verbose=False):
        """
        Evaluates the model on the test dataset and returns the accuracy.

        Args:
            verbose (bool): If True, prints the test accuracy.

        Returns:
            float: The test accuracy as a percentage.
        """
        if device != None:
            final_device = device
        else:
            final_device = self.device
        
        model.to(final_device)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            local_acc = []
            if Tqdm:
                loader = tqdm(self.dataloader["test"], desc="test", leave=False)
            else:
                loader = self.dataloader["test"]
            for i, data in enumerate(loader):
                images, labels = data
                images, labels = images.to(final_device), labels.to(final_device)
                # if ( "venum" in self.prune_mode ):
                #     out = self.model(images)
                #     total = len(images)
                #     return
                if self.snn:
                    outputs = self.forward_pass_snn(images, mem_out_rec=None)
                    correct += SF.accuracy_rate(outputs, labels) * outputs.size(1)
                    total += outputs.size(1)
                
                else:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0) - 1
                    correct += (predicted == labels).sum().item()
            
            acc = 100 * correct / total
            if verbose:
                print("Test Accuracy: {} %".format(acc))
            return acc
          
    def get_model_size_weights(self, mdl):
        """
        Calculates the size of the model's weights in megabytes.

        Args:
            mdl (torch.nn.Module): The model whose weights size needs to be calculated.

        Returns:
            float: The size of the model's weights in megabytes.
        """
        torch.save(mdl.state_dict(), "tmp.pt")
        mdl_size = round(os.path.getsize("tmp.pt") / 1e6, 3)
        os.remove("tmp.pt")
        return mdl_size

    def get_num_parameters(self, model: nn.Module, count_nonzero_only=False) -> int:
        """
        Calculates the total number of parameters in a given PyTorch model.

        :param model (nn.Module): The PyTorch model.
        :param count_nonzero_only (bool, optional): If True, only counts the number of non-zero parameters.
                                                    If False, counts all parameters. Defaults to False.

        """
        
        num_counted_elements = 0
        for param in model.parameters():
            if count_nonzero_only:
                num_counted_elements += param.count_nonzero()
            else:
                num_counted_elements += param.numel()
        return num_counted_elements
    
    def compare_models(self, model_list, model_tags=None):
        """
        Compares the performance of two PyTorch models: an original dense model and a pruned and fine-tuned model.
        Prints a table of metrics including latency, MACs, and model size for both models and their reduction ratios.

        Args:
        - original_dense_model: a PyTorch model object representing the original dense model
        - pruned_fine_tuned_model: a PyTorch model object representing the pruned and fine-tuned model

        Returns: None
        """
        
        table_data = {
            "latency": ["Latency (ms/sample)"],
            "accuracy": ["Accuracy (%)"],
            "params": ["Params (M)"],
            "size": ["Size (MiB)"],
            "mac": ["MAC (M)"],
            # "energy": ["Energy (Joules)"],
        }
        table = PrettyTable()
        table.field_names = ["", "Original Model", "Pruned Model", "Quantized Model"]
        
        accuracies, latency, params, model_size, macs = [], [], [], [], []
        skip = 3
        
        input_shape = list(next(iter(self.dataloader["test"]))[0].size())
        input_shape[0] = 1
        dummy_input = torch.randn(input_shape).to("cpu")
        file_name_list = ["original_model", "pruned_model", "quantized_model"]
        if not os.path.exists("weights/"):
            os.makedirs("weights/")
        for model, model_file_name in zip(model_list, file_name_list):
            # print(str(model))
            # # Parse through snn model and send to cpu
            if self.snn:
                if isinstance(model, nn.Sequential):
                    for layer_id in range(len(model)):
                        layer = model[layer_id]
                        if isinstance((layer), snntorch._neurons.leaky.Leaky):
                            layer.mem = layer.mem.to("cpu")
                else:
                    for module in model.modules():
                        if isinstance((layer), snntorch._neurons.leaky.Leaky):
                            layer.mem = layer.mem.to("cpu")
            
            table_data["accuracy"].append(
                round(self.evaluate(model=model, device="cpu"), 3)
            )
            table_data["latency"].append(
                round(
                    self.measure_latency(model=model.to("cpu"), dummy_input=dummy_input)
                    * 1000,
                    1,
                )
            )
            table_data["size"].append(self.get_model_size_weights(model))
            
            try:
                model_params = self.get_num_parameters(model, count_nonzero_only=True)
                if torch.is_tensor(model_params):
                    model_params = model_params.item()
                model_params = round(model_params / 1e6, 2)
                if model_params == 0:
                    table_data["params"].append("*")
                else:
                    table_data["params"].append(model_params)
            except RuntimeError as e:
                table_data["params"].append("*")
            if skip == 1:
                table_data["mac"].append("*")
                pass
            else:
                try:
                    mac = self.get_model_macs(model, dummy_input)
                    table_data["mac"].append(round(mac / 1e6))
                except AttributeError as e:
                    table_data["mac"].append("-")
            
            ########################################
            
            folder_file_name = "weights/" + model_file_name + "/"
            if not os.path.exists(folder_file_name):
                os.makedirs(folder_file_name)
            folder_file_name += model_file_name
            torch.save(model, folder_file_name + ".pt")
            torch.save(model.state_dict(), folder_file_name + "_weights.pt")
            traced_model = torch.jit.trace(model, dummy_input)
            torch.jit.save(torch.jit.script(traced_model), folder_file_name + ".pt")
            
            ########################################
            # Save model with pt,.pth and jit
            skip -= 1
        
        for key, value in table_data.items():
            table.add_row(value)
        print(
            "\n \n============================== Comparison Table =============================="
        )
        print(table)