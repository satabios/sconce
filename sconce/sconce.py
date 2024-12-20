import copy
import random
import time
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from snntorch import utils
from collections import namedtuple
from snntorch import functional as SF

from pruner import prune
from quanter import  quantization
from perf import performance

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("default")
warnings.filterwarnings("ignore", category=DeprecationWarning)

random.seed(321)
np.random.seed(432)
torch.manual_seed(223)

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
	torch.cuda.synchronize()



class sconce(quantization, performance, prune):
	def __init__(self):
		"""
		A class for training and evaluating neural networks with various optimization techniques.

		Attributes:
		- criterion: loss function used for training the model
		- batch_size: size of the batch used for training
		- validate: whether to validate the model during training
		- save: whether to save the model after training
		- goal: the goal of the model (e.g. classification)
		- experiment_name: name of the experiment
		- epochs: number of epochs for training
		- learning_rate: learning rate for the optimizer
		- dense_model_valid_acc: validation accuracy of the dense model
		- fine_tune_epochs: number of epochs for fine-tuning
		- fine_tune: whether to fine-tune the model
		- prune_model: whether to prune the model
		- prune_mode: mode of pruning (e.g. global, local)
		- quantization: whether to quantize the model
		- num_finetune_epochs: number of epochs for fine-tuning after pruning
		- best_sparse_model_checkpoint: checkpoint for the best sparse model
		- degradation_value: degradation value for pruning
		- degradation_value_local: local degradation value for pruning
		- model: the neural network model
		- criterion: loss function used for training the model
		- optimizer: optimizer used for training the model
		- scheduler: learning rate scheduler
		- dataloader: data loader for training and validation data
		- callbacks: callbacks for training the model
		- sparsity_dict: dictionary of sparsity values for each layer
		- masks: masks for pruning
		- Codebook: named tuple for codebook
		- codebook: codebook for quantization
		- channel_pruning_ratio: ratio of channels to prune
		- snn: whether to use spiking neural network
		- accuracy_function: function for calculating accuracy
		- bitwidth: bitwidth for quantization
		- device: device used for training the model
		"""
		self.criterion = nn.CrossEntropyLoss()
		self.batch_size = 64
		self.validate = True
		self.save = False
		self.goal = "classficiation"
		self.experiment_name = None
		self.epochs = None
		self.learning_rate = 1e-4
		self.dense_model_valid_acc = 0
		self.params = []
		self.qat_config = "x86"
		
		self.fine_tune_epochs = 10
		self.fine_tune = False
		self.prune_model = True
		self.prune_mode = ""
		self.quantization = True
		self.num_finetune_epochs = 5
		self.best_sparse_model_checkpoint = {}
		self.degradation_value = 1.2
		self.degradation_value_local = 1.2
		self.model = None
		self.criterion = None
		self.optimizer = None
		self.scheduler = None
		self.dataloader = None
		self.callbacks = None
		self.sparsity_dict = None
		self.masks = {}
		self.comparison = True
		self.Codebook = namedtuple("Codebook", ["centroids", "labels"])
		self.codebook = None
		self.channel_pruning_ratio = None
		self.snn = False
		self.snn_num_steps = 50
		self.accuracy_function = None
		
		self.layer_of_interest = []
		self.venum_sorted_list = []
		self.conv_layer = []
		self.linear_layer = []
		self.handles = []
		self.temp_sparsity_list = []
		self.prune_indexes = []
		self.record_prune_indexes = False
		self.layer_idx = 0
		
		self.bitwidth = 4
		
		self.device = None
	
	def forward_pass_snn(self, data, mem_out_rec=None):
		"""
		Perform a forward pass through the spiking neural network (SNN).

		Args:
			data: Input data for the SNN.
			mem_out_rec: Optional tensor to record the membrane potential at each time step.

		Returns:
			If mem_out_rec is not None, returns a tuple containing the spike outputs and membrane potentials
			as tensors. Otherwise, returns only the spike outputs as a tensor.
		"""
		spk_rec = []
		mem_rec = []
		utils.reset(self.model)  # resets hidden states for all LIF neurons in net
		
		for step in range(self.snn_num_steps):  # data.size(0) = number of time steps
			spk_out, mem_out = self.model(data)
			spk_rec.append(spk_out)
			mem_rec.append(mem_out)
			if mem_out_rec is not None:
				mem_rec.append(mem_out)
		if mem_out_rec is not None:
			return torch.stack(spk_rec), torch.stack(mem_rec)
		else:
			return torch.stack(spk_rec)
	
	def train(self, model=None) -> None:
		"""
		Trains the model for a specified number of epochs using the specified dataloader and optimizer.
		If fine-tuning is enabled, the number of epochs is set to `num_finetune_epochs`.
		The function also saves the model state after each epoch if the validation accuracy improves.
		If `model` is provided, applies supervised Knowledge Distillation (KD) using the provided teacher model.
		"""
		torch.cuda.empty_cache()
		self.model.to(self.device)
		if model is not None:
			model.to(self.device)
			model.eval()  # Ensure the teacher model stays in evaluation mode during training

		val_acc = 0
		running_loss = 0.0

		epochs = self.epochs if not self.fine_tune else self.num_finetune_epochs
		for epoch in range(epochs):
			self.model.train()
			validation_acc = 0

			for i, data in enumerate(
					tqdm(self.dataloader["train"], desc="train", leave=False)
			):
				# Move the data from CPU to GPU
				if self.goal != "autoencoder":
					inputs, targets = data
					inputs, targets = inputs.to(self.device), targets.to(self.device)
				elif self.goal == "autoencoder":
					inputs, targets = data.to(self.device), data.to(self.device)

				# Reset the gradients (from the last iteration)
				self.optimizer.zero_grad()

				# Forward pass
				if self.snn:
					outputs = self.forward_pass_snn(inputs)
					SF.accuracy_rate(outputs, targets) / 100
				else:
					outputs = self.model(inputs)

				# Compute the regular loss
				loss = self.criterion(outputs, targets)

				# Add Knowledge Distillation loss if `model` (teacher) is provided
				if model is not None:
					with torch.no_grad():
						teacher_outputs = model(inputs)
					temperature = 3.0 
					kd_loss = (
						torch.nn.functional.kl_div(
							torch.nn.functional.log_softmax(outputs / temperature, dim=1),
							torch.nn.functional.softmax(teacher_outputs / temperature, dim=1),
							reduction="batchmean",
						)
						* (temperature ** 2)
					)
					# Combine losses
					alpha = 0.5  # Weighting factor between regular loss and KD loss
					loss = alpha * loss + (1 - alpha) * kd_loss

				# Backward propagation
				loss.backward()

				# Update optimizer and LR scheduler
				self.optimizer.step()
				if self.scheduler is not None:
					self.scheduler.step()

				if self.callbacks is not None:
					for callback in self.callbacks:
						callback()

				running_loss += loss.item()

			running_loss = 0.0

			# Evaluate validation accuracy
			validation_acc = self.evaluate()
			if validation_acc > val_acc:
				print(
					f"Epoch:{epoch + 1} Train Loss: {running_loss / 2000:.5f} Validation Accuracy: {validation_acc:.5f}"
				)
				torch.save(
					copy.deepcopy(self.model.state_dict()),
					self.experiment_name + ".pth",
				)

	
	@torch.no_grad()
	def evaluate(self, model=None, device=None, Tqdm=True, verbose=False):
		"""
		Evaluates the model on the test dataset and returns the accuracy.

		Args:
		  verbose (bool): If True, prints the test accuracy.

		Returns:
		  float: The test accuracy as a percentage.
		"""
		if model != None:
			self.model = model
		if device != None:
			final_device = device
		else:
			final_device = self.device
		
		self.model.to(final_device)
		self.model.eval()
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
					outputs = self.model(images)
					_, predicted = torch.max(outputs.data, 1)
					total += labels.size(0) - 1
					correct += (predicted == labels).sum().item()
			
			acc = 100 * correct / total
			if verbose:
				print("Test Accuracy: {} %".format(acc))
			return acc
	

	
	def compress(self, verbose=True) -> None:
		"""
		Compresses the neural network model using either Granular-Magnitude Pruning (GMP) or Channel-Wise Pruning (CWP).
		If GMP is used, the sensitivity of each layer is first scanned and then the Fine-Grained Pruning is applied.
		If CWP is used, the Channel-Wise Pruning is applied directly.
		After pruning, the model is fine-tuned using Stochastic Gradient Descent (SGD) optimizer with Cosine Annealing
		Learning Rate Scheduler.
		The original dense model and the pruned fine-tuned model are saved in separate files.
		Finally, the validation accuracy and the size of the pruned model are printed.

		Args:
		  verbose (bool): If True, prints the validation accuracy and the size of the pruned model. Default is True.

		Returns:
		  None
		"""

		#Pruning
		sensitivity_start_time, sensitivity_start_end = 0, 0
		original_experiment_name = self.experiment_name
		if self.snn:
			original_dense_model = self.model
		
		else:
			original_dense_model = copy.deepcopy(self.model)
		
		input_shape = list(next(iter(self.dataloader["test"]))[0].size())
		input_shape[0] = 1
		
		current_device = next(original_dense_model.parameters()).device
		dummy_input = torch.randn(input_shape).to(current_device)
		
		dense_model_size = self.get_model_size(
			model=self.model, count_nonzero_only=True
		)
		print(f"\nOriginal Dense Model Size Model={dense_model_size / MiB:.2f} MiB")
		dense_validation_acc = self.evaluate(verbose=False)
		print("Original Model Validation Accuracy:", dense_validation_acc, "%")
		self.dense_model_valid_acc = dense_validation_acc
		
		if self.prune_mode == "GMP":
			print("Granular-Magnitude Pruning")
			sensitivity_start_time = time.time()
			self.sensitivity_scan(
				dense_model_accuracy=dense_validation_acc, verbose=False
			)
			sensitivity_start_end = time.time()
			print(
				"Sensitivity Scan Time(mins):",
				(sensitivity_start_end - sensitivity_start_time) / 60,
			)
			
			# Sparsity
			# for each Layer: {'backbone.conv0.weight': 0.45000000000000007, 'backbone.conv1.weight': 0.7500000000000002,
			#                  'backbone.conv2.weight': 0.7000000000000002, 'backbone.conv3.weight': 0.6500000000000001,
			#                  'backbone.conv4.weight': 0.6000000000000002, 'backbone.conv5.weight': 0.7000000000000002,
			#                  'backbone.conv6.weight': 0.7000000000000002, 'backbone.conv7.weight': 0.8500000000000002,
			#                  'classifier.weight': 0.9500000000000003}
			
			# self.sparsity_dict = {'0.weight': 0.6500000000000001, '3.weight': 0.5000000000000001, '7.weight': 0.7000000000000002}
			# self.sparsity_dict = {'backbone.conv0.weight': 0.20000000000000004, 'backbone.conv1.weight': 0.45000000000000007, 'backbone.conv2.weight': 0.25000000000000006, 'backbone.conv3.weight': 0.25000000000000006, 'backbone.conv4.weight': 0.25000000000000006, 'backbone.conv5.weight': 0.25000000000000006, 'backbone.conv6.weight': 0.3500000000000001, 'backbone.conv7.weight': 0.3500000000000001, 'classifier.weight': 0.7000000000000002}
			
			self.GMP_Pruning()  # FineGrained Pruning
			self.callbacks = [lambda: self.GMP_apply()]
			print(f"Sparsity for each Layer: {self.sparsity_dict}")
			self.fine_tune = True
		
		elif self.prune_mode == "CWP":
			print("\n Channel-Wise Pruning")
			sensitivity_start_time = time.time()
			self.sensitivity_scan(
				dense_model_accuracy=dense_validation_acc, verbose=False
			)
			sensitivity_start_end = time.time()
			print(
				"Sensitivity Scan Time(mins):",
				(sensitivity_start_end - sensitivity_start_time) / 60, "\n"
			)
			
			# self.sparsity_dict = {'backbone.conv0.weight': 0.15000000000000002, 'backbone.conv1.weight': 0.15, 'backbone.conv2.weight': 0.15, 'backbone.conv3.weight': 0.15000000000000002, 'backbone.conv4.weight': 0.20000000000000004, 'backbone.conv5.weight': 0.20000000000000004, 'backbone.conv6.weight': 0.45000000000000007}
			print("Sparsity for each Layer: ")
			for k, v in self.sparsity_dict.items():
				print(f"Layer Name: {k}: Sparsity:{v*100:.2f}%")

			self.CWP_Pruning()  # Channelwise Pruning
			self.fine_tune = True
		

		print(
			"\nPruning Time Consumed (mins):", (time.time() - sensitivity_start_end) / 60
		)
		print(
			"Total Pruning Time Consumed (mins):",
			(time.time() - sensitivity_start_time) / 60,
		)
		
		pruned_model = copy.deepcopy(self.model)
		
		current_device = next(pruned_model.parameters()).device
		dummy_input = torch.randn(input_shape).to(current_device)
		
		pruned_model_size = self.get_model_size(
			model=pruned_model, count_nonzero_only=True
		)
		
		print(
			f"\nPruned Model has size={pruned_model_size / MiB:.2f} MiB(non-zeros) = {pruned_model_size / dense_model_size * 100:.2f}% of Original model size"
		)
		pruned_model_acc = self.evaluate()
		print(
			f"\nPruned Model has Accuracy={pruned_model_acc :.2f} % = {pruned_model_acc - dense_validation_acc :.2f}% of Original model Accuracy"
		)
		
		if self.fine_tune:
			print("\n \n==================== Fine-Tuning ========================================")
			self.optimizer = torch.optim.SGD(
				self.model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4
			)
			self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
				self.optimizer, self.num_finetune_epochs
			)
			
			self.train(model=copy.deepcopy(original_dense_model))
			save_file_name = self.experiment_name + "_pruned_fine_tuned" + ".pt"
			self.save_torchscript_model(
				model=self.model, model_dir="./", model_filename=save_file_name
			)

			
			pruned_model = copy.deepcopy(self.model)
		
		fine_tuned_pruned_model_size = self.get_model_size(
			model=pruned_model, count_nonzero_only=True
		)
		fine_tuned_validation_acc = self.evaluate(verbose=False)
		
		if verbose:
			print(
				f"Fine-Tuned Sparse model has size={fine_tuned_pruned_model_size / MiB:.2f} MiB = {fine_tuned_pruned_model_size / dense_model_size * 100:.2f}% of Original model size"
			)
			print(
				"Fine-Tuned Pruned Model Validation Accuracy:",
				fine_tuned_validation_acc,
			)

		#Quantization
		quantized_model, model_fp32_trained = self.qat()
		
		model_list = [original_dense_model, pruned_model, quantized_model]
		
		self.compare_models(model_list=model_list)
	
	def evaluate_model(self, model, test_loader, device, criterion=None):
		model.eval()
		model.to(device)
		
		running_loss = 0
		running_corrects = 0
		
		for inputs, labels in test_loader:
			inputs = inputs.to(device)
			labels = labels.to(device)
			
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			
			if criterion is not None:
				loss = criterion(outputs, labels).item()
			else:
				loss = 0
			
			# statistics
			running_loss += loss * inputs.size(0)
			running_corrects += torch.sum(preds == labels.data)
		
		eval_loss = running_loss / len(test_loader.dataset)
		eval_accuracy = running_corrects / len(test_loader.dataset)
		
		return eval_loss, eval_accuracy
	

	
