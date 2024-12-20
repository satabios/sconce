import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchprofile import profile_macs
import os
import onnxruntime as ort
import time
import snntorch
from prettytable import PrettyTable
import numpy as np

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("default")
warnings.filterwarnings("ignore", category=DeprecationWarning)

class performance:


    ########## Model Profiling ##########

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

            #JIT Export
            traced_model = torch.jit.trace(model, dummy_input)
            torch.jit.save(torch.jit.script(traced_model), folder_file_name + ".pt")

            #ONNX Export
            torch.onnx.export(
                model,                         
                dummy_input,                   # Dummy input tensor
                folder_file_name+".onnx",      # ONNX file output path
                export_params=True,            # Store the trained parameter weights
                opset_version=18,              # ONNX opset version
                do_constant_folding=True,      # Optimize the graph by folding constants
                input_names=["input"],         # Input tensor name
                output_names=["output"],       # Output tensor name
                # dynamic_axes={                 # Dynamic axes for variable input sizes
                #     "input": {0: "batch_size"},
                #     "output": {0: "batch_size"}
                # }
            )

            ########################################
            # Save model with pt,.pth and jit
            skip -= 1

        for key, value in table_data.items():
            table.add_row(value)
        print(
            "\n \n============================== Comparison Table ==================================================="
        )
        print(table)

        print(
            "\n \n===================== Profiling Original and Optimized Model ======================================="
        )

        prof_table = PrettyTable()

        # Define the column headers
        prof_table.field_names = ["Metrics", "Original Model", "Optimized Model", "Speed-Up/Compression Achieved"]

        prof_table.add_row(["Latency", str(table_data['latency'][1]), str(table_data['latency'][-1]),
                            f"{table_data['latency'][1] / table_data['latency'][-1]:.2f} x"])
        prof_table.add_row(["Parameters", str(table_data['params'][1]), str(table_data['params'][2]),
                            f"{((table_data['params'][1] - table_data['params'][2]) / table_data['params'][1]) * 100:.2f} %"])
        prof_table.add_row(["Memory", str(table_data['size'][1]), str(table_data['size'][-1]),
                            f"{((table_data['size'][1] - table_data['size'][-1]) / table_data['size'][1]) * 100:.2f} %"])
        prof_table.add_row(["MACs(≈ 2FLOPs)", str(table_data['mac'][1]), str(table_data['mac'][2]),
                            f"{((table_data['mac'][1] - table_data['mac'][2]) / table_data['mac'][1]) * 100:.2f} %"])
        prof_table.add_row(["Accuracy", str(table_data['accuracy'][1]), str(table_data['accuracy'][-1]),
                            f"{((table_data['accuracy'][-1] - table_data['accuracy'][1]) / table_data['accuracy'][1]) * 100:.2f} %"])

        print(prof_table)

        #ONNX Model Profiling
        print(
            "\n \n=================================== ONNX Runtimes ================================================="
        )


        models_to_attend = {
            "original_model": "Original Model",
            "quantized_model": "Optimized Model"
        }
        devices = ["CPU", "CUDA"]

        dummy_input = dummy_input.cpu().detach().numpy()

        table = PrettyTable()
        table.field_names = ["ONNX Model", "CPU Latency (ms)", "GPU Latency (ms)"]

        for model_name, display_name in models_to_attend.items():
            latencies = {"CPU": None, "CUDA": None}
            model_filepath = f"weights/{model_name}/{model_name}.onnx"
            
            for device in devices:
                try:
                    session = ort.InferenceSession(model_filepath, providers=[f"{device}ExecutionProvider"])
                    input_name = session.get_inputs()[0].name
                    output_name = session.get_outputs()[0].name
                    
                    # Warm-up runs
                    for _ in range(5):
                        session.run([output_name], {input_name: dummy_input})
                    
                    num_iterations = 100
                    total_time = 0.0
                    for _ in range(num_iterations):
                        start_time = time.time()
                        session.run([output_name], {input_name: dummy_input})
                        total_time += (time.time() - start_time)
                    
                    # Calculate average latency in milliseconds
                    average_latency = (total_time / num_iterations) * 1000
                    latencies[device] = average_latency
                except Exception as e:
                    print(f"Error running model '{model_name}' on {device}: {e}")

            table.add_row([
                display_name,
                f"{latencies['CPU']:.2f}" if latencies["CPU"] else "N/A",
                f"{latencies['CUDA']:.2f}" if latencies["CUDA"] else "N/A",
            ])

        print(table)
        
        


    def get_model_macs(self, model, inputs) -> int:
        """
        Calculates the number of multiply-accumulate operations (MACs) required to run the given model with the given inputs.

        Args:
          model: The model to profile.
          inputs: The inputs to the model.

        Returns:
          The number of MACs required to run the model with the given inputs.
        """
        return profile_macs(model, inputs)

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

    def save_torchscript_model(self, model, model_dir, model_filename):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_filepath = os.path.join(model_dir, model_filename)

        torch.jit.save(torch.jit.script(model), model_filepath)

    def load_torchscript_model(self, model_filepath, device):
        model = torch.jit.load(model_filepath, map_location=device)

        return model

    def get_sparsity(self, tensor: torch.Tensor) -> float:
        """
        calculate the sparsity of the given tensor
            sparsity = #zeros / #elements = 1 - #nonzeros / #elements
        """
        return 1 - float(tensor.count_nonzero()) / tensor.numel()

    def get_model_sparsity(self, model: nn.Module) -> float:
        """

        Calculate the sparsity of the given PyTorch model.

        Sparsity is defined as the ratio of the number of zero-valued weights to the total number of weights in the model.
        This function iterates over all parameters in the model and counts the number of non-zero values and the total
        number of values.

        Args:
          model (nn.Module): The PyTorch model to calculate sparsity for.

        Returns:
          float: The sparsity of the model, defined as 1 - (# non-zero weights / # total weights).

        calculate the sparsity of the given model
            sparsity = #zeros / #elements = 1 - #nonzeros / #elements

        """
        num_nonzeros, num_elements = 0, 0
        for param in model.parameters():
            num_nonzeros += param.count_nonzero()
            num_elements += param.numel()
        return 1 - float(num_nonzeros) / num_elements

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

    def print_model_size(self, mdl):
        torch.save(mdl.state_dict(), "tmp.pt")
        print("%.2f MB" % (os.path.getsize("tmp.pt") / 1e6))
        os.remove("tmp.pt")

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

    def get_model_size(
            self, model: nn.Module, data_width=32, count_nonzero_only=False
    ) -> int:
        """
        calculate the model size in bits
        :param data_width: #bits per element
        :param count_nonzero_only: only count nonzero weights
        """
        return self.get_num_parameters(model, count_nonzero_only) * data_width

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

    def plot_weight_distribution(self, bins=256, count_nonzero_only=False):
        """
        Plots the weight distribution of the model's named parameters.

        Args:
          bins (int): Number of bins to use in the histogram. Default is 256.
          count_nonzero_only (bool): If True, only non-zero weights will be plotted. Default is False.

        Returns:
          None
        """
        fig, axes = plt.subplots(3, 3, figsize=(10, 6))
        axes = axes.ravel()
        plot_index = 0
        for name, param in self.model.named_parameters():
            if param.dim() > 1:
                ax = axes[plot_index]
                if count_nonzero_only:
                    param_cpu = param.detach().view(-1).cpu()
                    param_cpu = param_cpu[param_cpu != 0].view(-1)
                    ax.hist(param_cpu, bins=bins, density=True, color="blue", alpha=0.5)
                else:
                    ax.hist(
                        param.detach().view(-1).cpu(),
                        bins=bins,
                        density=True,
                        color="blue",
                        alpha=0.5,
                    )
                ax.set_xlabel(name)
                ax.set_ylabel("density")
                plot_index += 1
        fig.suptitle("Histogram of Weights")
        fig.tight_layout()
        fig.subplots_adjust(top=0.925)
        plt.show()
