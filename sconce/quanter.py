import torch.ao.quantization.quantize_fx as quantize_fx
import copy
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("default")
warnings.filterwarnings("ignore", category=DeprecationWarning)


class quantization:

    def qat(self):

        print(

            "\n \n========================== Quantization-Aware Training(QAT) ===================================" )
        ########### 2.1 ##################3
        # import torch
        # from torch.ao.quantization import (
        #     get_default_qconfig_mapping,
        #     get_default_qat_qconfig_mapping,
        #     QConfigMapping,
        # )


        # model_to_quantize = copy.deepcopy(self.model)
        # qconfig_mapping = get_default_qat_qconfig_mapping(self.qat_config)
        # model_to_quantize.train()
        # # prepare
        # example_inputs = next(iter(self.dataloader['test']))[0][:1, :]
        # model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_mapping, example_inputs)

        # self.model = model_prepared
        # self.train()
        # self.model.eval()
        # model_quantized = quantize_fx.convert_fx(self.model.to('cpu'))

        # model_fp32_trained = copy.deepcopy(model_quantized)
        # model_int8 = quantize_fx.fuse_fx(model_fp32_trained)
        import torch
        from torch.ao.quantization import (
            get_default_qat_qconfig_mapping,
            QConfigMapping,
        )
        from torch.ao.quantization.observer import default_observer, default_per_channel_weight_observer
        from torch.ao.quantization.qconfig import QConfig
        from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx, fuse_fx
        import copy

        from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
        from torch.ao.quantization.qconfig import QConfig

        def get_int8_qconfig_mapping():
            return QConfig(
                activation=MinMaxObserver.with_args(dtype=torch.qint8),  # INT8 activations
                weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8)  # INT8 weights
            )


        # Updated script
        model_to_quantize = copy.deepcopy(self.model)

        # Define a QConfigMapping explicitly for INT8
        qconfig_mapping = QConfigMapping().set_global(get_int8_qconfig_mapping())

        # Enable training mode for QAT
        model_to_quantize.train()

        # Prepare the model for QAT
        example_inputs = next(iter(self.dataloader['test']))[0][:1, :]  # Example input for FX preparation
        model_prepared = prepare_qat_fx(model_to_quantize, qconfig_mapping, example_inputs)

        # Train the prepared model
        self.model = model_prepared
        self.train()

        # Convert the trained model to evaluation mode and quantize
        self.model.eval()
        model_quantized = convert_fx(self.model.to('cpu'))

        # Fuse layers and finalize INT8 quantized model
        model_fp32_trained = copy.deepcopy(model_quantized)
        model_int8 = fuse_fx(model_fp32_trained)

# The resulting model_int8 contains INT8 quantized parameters and activations.

        #
        # def get_all_layers(model, parent_name=""):
        #     layers = []
        #     for name, module in model.named_children():
        #         full_name = f"{parent_name}.{name}" if parent_name else name
        #         layers.append((full_name, module))
        #         if isinstance(module, nn.Module):
        #             layers.extend(get_all_layers(module, parent_name=full_name))
        #     return layers
        #
        # fusing_layers = [
        #     torch.nn.modules.conv.Conv2d,
        #     torch.nn.modules.batchnorm.BatchNorm2d,
        #     torch.nn.modules.activation.ReLU,
        #     torch.nn.modules.linear.Linear,
        #     torch.nn.modules.batchnorm.BatchNorm1d,
        # ]
        #
        # def detect_sequences(lst):
        #     detected_sequences = []
        #
        #     i = 0
        #     while i < len(lst):
        #         if i + 2 < len(lst) and [type(l) for l in lst[i : i + 3]] == [
        #             fusing_layers[0],
        #             fusing_layers[1],
        #             fusing_layers[2],
        #         ]:
        #             detected_sequences.append(
        #                 np.take(name_list, [i for i in range(i, i + 3)]).tolist()
        #             )
        #             i += 3
        #         elif i + 1 < len(lst) and [type(l) for l in lst[i : i + 2]] == [
        #             fusing_layers[0],
        #             fusing_layers[1],
        #         ]:
        #             detected_sequences.append(
        #                 np.take(name_list, [i for i in range(i, i + 2)]).tolist()
        #             )
        #             i += 2
        #         # if i + 1 < len(lst) and [ type(l) for l in lst[i:i+2]] == [fusing_layers[0], fusing_layers[2]]:
        #         #     detected_sequences.append(np.take(name_list,[i for i in range(i,i+2)]).tolist())
        #         #     i += 2
        #         # elif i + 1 < len(lst) and [ type(l) for l in lst[i:i+2]] == [fusing_layers[1], fusing_layers[2]]:
        #         #     detected_sequences.append(np.take(name_list,[i for i in range(i,i+2)]).tolist())
        #         #     i += 2
        #         elif i + 1 < len(lst) and [type(l) for l in lst[i : i + 2]] == [
        #             fusing_layers[3],
        #             fusing_layers[2],
        #         ]:
        #             detected_sequences.append(
        #                 np.take(name_list, [i for i in range(i, i + 2)]).tolist()
        #             )
        #             i += 2
        #         elif i + 1 < len(lst) and [type(l) for l in lst[i : i + 2]] == [
        #             fusing_layers[3],
        #             fusing_layers[4],
        #         ]:
        #             detected_sequences.append(
        #                 np.take(name_list, [i for i in range(i, i + 2)]).tolist()
        #             )
        #             i += 2
        #         else:
        #             i += 1
        #
        #     return detected_sequences
        #
        # original_model = copy.deepcopy(self.model)
        #
        # model_fp32 = copy.deepcopy(self.model)
        # model_fp32 = nn.Sequential(
        #     torch.quantization.QuantStub(), model_fp32, torch.quantization.DeQuantStub()
        # )
        #
        # model_fp32.eval()
        #
        # all_layers = get_all_layers(model_fp32)
        # name_list = []
        # layer_list = []
        # for name, module in all_layers:
        #     name_list.append(name)
        #     layer_list.append(module)
        #
        # fusion_layers = detect_sequences(layer_list)
        #
        # model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig(
        #     self.qat_config
        # )
        #
        # # fuse the activations to preceding layers, where applicable
        # # this needs to be done manually depending on the model architecture
        # model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, fusion_layers)
        #
        # # Prepare the model for QAT. This inserts observers and fake_quants in
        # # the model needs to be set to train for QAT logic to work
        # # the model that will observe weight and activation tensors during calibration.
        # model_fp32_prepared = torch.ao.quantization.prepare_qat(
        #     model_fp32_fused.train()
        # )
        # self.model = model_fp32_prepared
        # self.train()
        #
        # # Convert the observed model to a quantized model. This does several things:
        # # quantizes the weights, computes and stores the scale and bias value to be
        # # used with each activation tensor, fuses modules where appropriate,
        # # and replaces key operators with quantized implementations.
        # model_fp32_trained = copy.deepcopy(self.model)
        # model_fp32_trained.to("cpu")
        # model_fp32_trained.eval()
        #
        # model_int8 = torch.ao.quantization.convert(model_fp32_trained, inplace=True)
        # model_int8.eval()
        # # # torch.save(model_int8, 'quantized_model.pt')
        # # # torch.save(
        # # #     model_int8.state_dict(),
        # # #     self.experiment_name + "_quantized" + ".pth",
        # # # # )
        # # input_shape = list(next(iter(self.dataloader["test"]))[0].size())
        # # input_shape[0] = 1
        # # current_device = "cpu"
        # # dummy_input = torch.randn(input_shape).to(current_device)
        # # #
        # # # self.params.append([self.evaluate(model=model_int8),
        # # #                     self.measure_latency(model=model_int8, dummy_input=dummy_input),
        # # #                     self.get_num_parameters(model=model_int8),
        # # #                     self.get_model_size(model=model_int8, count_nonzero_only=True)])
        # #
        # # save_file_name = self.experiment_name + "_int8.pt"
        # # self.save_torchscript_model(model=model_int8, model_dir="./", model_filename=save_file_name)
        # #
        # # quantized_jit_model = self.load_torchscript_model(model_filepath=self.experiment_name + "_int8.pt", device='cpu')
        # #
        # # _, fp32_eval_accuracy = self.evaluate_model(model=original_model, test_loader=self.dataloader['test'], device='cpu', criterion=None)
        # # _, int8_eval_accuracy = self.evaluate_model(model=model_int8, test_loader=self.dataloader['test'], device='cpu',
        # #                                        criterion=None)
        # # _, int8_jit_eval_accuracy = self.evaluate_model(model=quantized_jit_model, test_loader=self.dataloader['test'],
        # #                                             device='cpu', criterion=None)
        # # print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
        # # print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))
        # # print("INT8 JIT evaluation accuracy: {:.3f}".format(int8_eval_accuracy))
        # #
        # #
        # # fp32_cpu_inference_latency = self.measure_inference_latency(model=original_model, device='cpu',
        # #                                                        input_data=dummy_input, num_samples=100)
        # # int8_cpu_inference_latency = self.measure_inference_latency(model=model_int8, device='cpu',
        # #                                                        input_data=dummy_input, num_samples=100)
        # # int8_jit_cpu_inference_latency = self.measure_inference_latency(model=quantized_jit_model, device='cpu',
        # #                                                            input_data=dummy_input, num_samples=100)
        # # fp32_gpu_inference_latency = self.measure_inference_latency(model=original_model, device='cuda',
        # #                                                        input_data=dummy_input, num_samples=100)
        # #
        # # print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
        # # print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
        # # print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
        # # print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))

        return model_int8, model_fp32_trained
