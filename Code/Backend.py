import os
from deepxde.model import Model, TrainState
from deepxde.nn.pytorch.nn import NN
from deepxde.nn import activations, initializers
from deepxde import config, optimizers, utils, display
from deepxde.utils.external import save_loss_history, save_best_state, plot_loss_history, plot_best_state
from deepxde import gradients as grad
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F

from Code.utils import linear_normalize, replace_group_name
from Plot import SimulationPlot




class Model2(Model):
    def __init__(self, data, data_val, net, task_name, meta_learning, alpha=None, pde_num=4, output_dir=None):
        super().__init__(data, net)
        self.data_val = data_val
        self.lambda_pde = 1
        self.lambda_icbc = 1
        self.alpha = alpha
        self.lambda_values = []
        self.task_name = task_name
        self.meta_learning = meta_learning
        self.pde_num = pde_num
        self.output_dir = output_dir
        self.outputs_losses_val = None
        self.train_state = TrainState2()
        self.losshistory = LossHistory2(meta_learning)
        self.no_weight = False
        self.frozen = False

    def change_data(self, new_data):
        self.data = new_data

    def meta_learning_loss_weights(self, pde_init=1, data_init=1):
        # Define learnable scalar parameters for PDE and data initial weights
        self.pde_loss_weight = torch.nn.Parameter(torch.tensor(pde_init, dtype=torch.float32).cuda())  # Initial weight for PDE loss
        self.data_loss_weight = torch.nn.Parameter(torch.tensor(data_init, dtype=torch.float32).cuda())  # Initial weight for data loss

        # Repeat the scalar parameters to create weight tensors for PDE and data
        pde_weights = self.pde_loss_weight.repeat(self.pde_num)  # Repeat for the number of PDE loss components
        data_weights = self.data_loss_weight.repeat(len(self.data.bcs))  # Repeat for the number of data loss components

        # Concatenate the weight tensors into a single tensor
        loss_weights = torch.cat([pde_weights, data_weights], dim=0)

        # Convert to a learnable parameter and apply softmax normalization
        # loss_weights = torch.nn.Parameter(F.softmax(loss_weights, dim=0))
        loss_weights = torch.nn.Parameter(linear_normalize(loss_weights))

        return loss_weights

    @utils.timing
    def compile(
        self,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=None,
        decay=None,
        loss_weights=None,
        external_trainable_variables=None,
        dynamic_loss_weight=False,
        train_step_num=1,
        frozen=False,
    ):
        self.dynamic_loss_weight = dynamic_loss_weight
        self.train_step_num = train_step_num
        self.frozen = frozen
        # self.lr = lr
        
        if self.meta_learning and self.train_step_num == 2:
            loss_weights = self.meta_learning_loss_weights(loss_weights[0], loss_weights[-1])


        super().compile(
            optimizer,
            lr,
            loss,
            metrics,
            decay,
            loss_weights,
            external_trainable_variables,
        )

        

    def _compile_pytorch(self, lr, loss_fn, decay):
        """pytorch"""

        def outputs(training, inputs):
            self.net.train(mode=training)
            with torch.no_grad():
                if isinstance(inputs, tuple):
                    inputs = tuple(
                        map(lambda x: torch.as_tensor(x).requires_grad_(), inputs)
                    )
                else:
                    inputs = torch.as_tensor(inputs)
                    inputs.requires_grad_()
            # Clear cached Jacobians and Hessians.
            grad.clear()
            return self.net(inputs)

        def outputs_losses(training, inputs, targets, auxiliary_vars, losses_fn):
            self.net.auxiliary_vars = None
            if auxiliary_vars is not None:
                self.net.auxiliary_vars = torch.as_tensor(auxiliary_vars)
            self.net.train(mode=training)
            if isinstance(inputs, tuple):
                inputs = tuple(
                    map(lambda x: torch.as_tensor(x).requires_grad_(), inputs)
                )
            else:
                inputs = torch.as_tensor(inputs)
                inputs.requires_grad_()
            outputs_ = self.net(inputs)
            # Data losses
            if targets is not None:
                targets = torch.as_tensor(targets)
            # if forward-mode AD is used, then a forward call needs to be passed
            aux = [self.net] if config.autodiff == "forward" else None
            losses = losses_fn(targets, outputs_, loss_fn, inputs, self, aux=aux)
            if not isinstance(losses, list):
                losses = [losses]
            losses = torch.stack(losses)


            # Weighted losses
            if self.loss_weights is not None and not self.no_weight:
                losses *= torch.as_tensor(self.loss_weights)

            # Clear cached Jacobians and Hessians.
            grad.clear()

            return outputs_, losses



        def outputs_losses_train(inputs, targets, auxiliary_vars):
            return outputs_losses(
                True, inputs, targets, auxiliary_vars, self.data.losses_train
            )
        
        def outputs_losses_val(inputs, targets, auxiliary_vars):
            return outputs_losses(
                True, inputs, targets, auxiliary_vars, self.data_val.losses_train
            )

        def outputs_losses_test(inputs, targets, auxiliary_vars):
            return outputs_losses(
                False, inputs, targets, auxiliary_vars, self.data.losses_test
            )

        # Another way is using per-parameter options
        # https://pytorch.org/docs/stable/optim.html#per-parameter-options,
        # but not all optimizers (such as L-BFGS) support this.
        
        trainable_variables = (
                [param for param in self.net.parameters() if param.requires_grad]
                + self.external_trainable_variables
            )

        if self.train_step_num == 2 and self.frozen:
            for param in self.net.parameters():
                param.requires_grad = False
            for name, param in self.net.named_parameters():
                if "2" in name or "3" in name:
                    param.requires_grad = True
            trainable_variables = (
                [param for param in self.net.parameters() if param.requires_grad]
                + self.external_trainable_variables
            )

        if self.net.regularizer is None:
            self.opt, self.lr_scheduler = optimizers.get(
                trainable_variables, self.opt_name, learning_rate=lr, decay=decay
            )
        else:
            if self.net.regularizer[0] == "l2":
                self.opt, self.lr_scheduler = optimizers.get(
                    trainable_variables,
                    self.opt_name,
                    learning_rate=lr,
                    decay=decay,
                    weight_decay=self.net.regularizer[1],
                )
            else:
                raise NotImplementedError(
                    f"{self.net.regularizer[0]} regularization to be implemented for "
                    "backend pytorch."
                )

        if self.train_step_num == 2 and self.meta_learning:
            self.opt_meta = optim.Adam([self.loss_weights], lr=0.001)


        def train_step(inputs, targets, auxiliary_vars):
            
            def update_loss_weights():
                if (self.train_step_num == 2) & (self.iter % 100 == 0):
                    losses = outputs_losses_train(inputs, targets, auxiliary_vars)[1]

                    data_num = len(self.data.bcs)
                    pde_num = len(losses) - data_num

                    data_loss = torch.sum(losses[-data_num:])
                    pde_loss = torch.sum(losses[:pde_num])

                    data_gradient = torch.autograd.grad(outputs=data_loss, inputs=self.net.parameters(), create_graph=True)
                    gradient_norms = torch.stack([torch.norm(tensor, p='fro') / (tensor.numel() ** 0.5) for tensor in data_gradient])
                    gradient_max = torch.quantile(gradient_norms, 0.95).item()

                    pde_gradient = torch.autograd.grad(outputs=pde_loss, inputs=self.net.parameters(), create_graph=True)
                    gradient_norms = torch.stack([torch.norm(tensor, p='fro') / (tensor.numel() ** 0.5) for tensor in pde_gradient])
                    gradient_pde = torch.mean(gradient_norms).item()

                    # icbc_gradient = torch.autograd.grad(outputs=icbc_loss, inputs=self.net.parameters(), create_graph=True)
                    # gradient_norms = torch.stack([torch.norm(tensor) for tensor in icbc_gradient])
                    # gradient_icbc = torch.mean(gradient_norms).item()

                    hat_lambda_pde = gradient_max / (gradient_pde + 1e-8)  # Avoid division by zero
                    self.lambda_pde = (1 - self.alpha) * self.lambda_pde + self.alpha * hat_lambda_pde

                    # hat_lambda_icbc = gradient_max / (gradient_icbc + 1e-8)  # Avoid division by zero
                    # self.lambda_icbc = (1 - self.alpha) * self.lambda_icbc + self.alpha * hat_lambda_icbc

                    self.lambda_values.append(self.lambda_pde)
                    print(f"Lambda PDE: {self.lambda_pde:.3f}")

                    SimulationPlot.simlulation_plot_lambdas(output_image_path=f'Output/Plot/Simulation/{self.task_name}_simualtion_lambdas.png', lambdas=self.lambda_values)

                    self.loss_weights = [self.lambda_pde] * pde_num + [1] * data_num


            def closure():
                losses = outputs_losses_train(inputs, targets, auxiliary_vars)[1]
                total_loss = torch.sum(losses)
                self.opt.zero_grad()
                total_loss.backward()
                return total_loss

            if self.dynamic_loss_weight:
                update_loss_weights()

            self.opt.step(closure)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        def train_step_meta(inputs, targets, auxiliary_vars):
            
            def closure():
                # self.no_weight = True
                losses = -outputs_losses_val(inputs, targets, auxiliary_vars)[1] #
                self.no_weight = False

                total_loss = torch.sum(losses)
                self.opt_meta.zero_grad()
                total_loss.backward()

                return total_loss

            self.opt_meta.step(closure)

            with torch.no_grad():
                self.loss_weights.data = linear_normalize(self.loss_weights)
                
        # Callables
        self.outputs = outputs
        self.outputs_losses_train = outputs_losses_train
        self.outputs_losses_val = outputs_losses_val
        self.outputs_losses_test = outputs_losses_test
        self.train_step = train_step
        self.train_step_meta = train_step_meta

    

    @utils.timing
    def train(
        self,
        iterations=None,
        batch_size=None,
        display_every=1000,
        disregard_previous_best=False,
        callbacks=None,
        model_restore_path=None,
        model_save_path=None,
        epochs=None,
    ):
        self.model_save_path = model_save_path

        return super().train(iterations, batch_size, display_every, disregard_previous_best, callbacks, model_restore_path, model_save_path, epochs)

    def _train_sgd(self, iterations, display_every):

        def save_lambda_values_to_csv():
            df = pd.DataFrame(self.lambda_values, columns=['lambda_pde'])
            df.to_csv(os.path.join(os.path.dirname(self.model_save_path), 'lambdas.csv'), index=False)

        for i in range(iterations):
            self.iter = i

            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size)
            )

            self.train_step(
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars,
            )

            self.train_state.epoch += 1
            self.train_state.step += 1

            if self.meta_learning and self.train_state.step % 100 == 0 and self.train_step_num == 2:
                self.train_state.set_data_val(
                    *self.data_val.train_next_batch(self.batch_size)
                )
                self.train_step_meta(
                    self.train_state.X_val,
                    self.train_state.y_val,
                    self.train_state.val_aux_vars,
                )

            if self.train_state.step % display_every == 0 or i + 1 == iterations:
                self._test()

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

        if self.dynamic_loss_weight:
            save_lambda_values_to_csv()

    def _test(self):
        # TODO Now only print the training loss in rank 0. The correct way is to print the average training loss of all ranks.

        self.no_weight = True
        (
            self.train_state.y_pred_train,
            self.train_state.loss_train,
        ) = self._outputs_losses(
            True,
            self.train_state.X_train,
            self.train_state.y_train,
            self.train_state.train_aux_vars,
        )
        self.no_weight = False

        self.train_state.y_pred_test, self.train_state.loss_test = self._outputs_losses(
            False,
            self.train_state.X_test,
            self.train_state.y_test,
            self.train_state.test_aux_vars,
        )

        if isinstance(self.train_state.y_test, (list, tuple)):
            self.train_state.metrics_test = [
                m(self.train_state.y_test[i], self.train_state.y_pred_test[i])
                for m in self.metrics
                for i in range(len(self.train_state.y_test))
            ]
        else:
            self.train_state.metrics_test = [
                m(self.train_state.y_test, self.train_state.y_pred_test)
                for m in self.metrics
            ]


        self.train_state.update_best()
        self.losshistory.append(
            self.train_state.step,
            self.train_state.loss_train,
            self.train_state.loss_test,
            self.train_state.metrics_test,
            self.loss_weights,
        )

        if (
            np.isnan(self.train_state.loss_train).any()
            or np.isnan(self.train_state.loss_test).any()
        ):
            self.stop_training = True
        if config.rank == 0:
            display.training_display(self.train_state)

        SimulationPlot.plot_loss_curve(self.losshistory.loss_train, self.losshistory.steps, os.path.join(self.output_dir, f'Step{self.train_step_num}', 'loss_curve.jpg'), pde_num=self.pde_num)

        if self.train_step_num == 2 and self.meta_learning:
            SimulationPlot.plot_loss_curve(self.losshistory.lambdas, self.losshistory.steps, os.path.join(self.output_dir, f'Step{self.train_step_num}', 'lambdas_curve.jpg'), pde_num=self.pde_num, title='Loss Weight Curve', ylabel='Value')

    def restore(self, save_path, device=None, verbose=0, load_var=None):
        """Restore all variables from a disk file.

        Args:
            save_path (string): Path where model was previously saved.
            device (string, optional): Device to load the model on (e.g. "cpu","cuda:0"...). By default, the model is loaded on the device it was saved from.
        """

        if verbose > 0:
            print("Restoring model from {} ...\n".format(save_path))

        if device is not None:
            checkpoint = torch.load(save_path, map_location=torch.device(device))
        else:
            checkpoint = torch.load(save_path)

        if load_var is not None:
            load_var = replace_group_name(load_var)

            pretrained_dict = checkpoint["model_state_dict"]
            model_dict = self.net.state_dict()
            selected_keys = [k for k in pretrained_dict if f'_{load_var}' in k and k in model_dict]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                if k in selected_keys 
                                and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict, strict=False)
            
        else:
            self.net.load_state_dict(checkpoint["model_state_dict"])

        # self.opt.load_state_dict(checkpoint["optimizer_state_dict"])


class TrainState2(TrainState):
    def __init__(self):
        super().__init__()
        self.X_val = None
        self.y_val = None
        self.val_aux_vars = None

    def set_data_val(self, X_val, y_val, val_aux_vars=None):
        self.X_val = X_val
        self.y_val = y_val
        self.val_aux_vars = val_aux_vars

        

class MDRF_Net(NN):
    def __init__(self, n_layers=[6] * 2 + [8] * 4, layer_widths=[128] * 6, layer_sizes=[4] + [6], activation="sin", kernel_initializer="Glorot uniform"):
        super().__init__()
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)
        
        ########################### NEW ################################

        self.n_layers = n_layers
        self.layer_widths = layer_widths

        self.linears_all = torch.nn.ModuleList()
        layer_sizes_all = [4] + [128]
        for i in range(1, len(layer_sizes_all)):
            self.linears_all.append(
                torch.nn.Linear(
                    layer_sizes_all[i - 1], layer_sizes_all[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears_all[-1].weight)
            initializer_zero(self.linears_all[-1].bias)
            
        self.linears_temp = torch.nn.ModuleList()
        layer_sizes_temp = [128] + [layer_widths[0]] * (n_layers[0] - 1) + [1]
        for i in range(1, len(layer_sizes_temp)):
            self.linears_temp.append(
                torch.nn.Linear(
                    layer_sizes_temp[i - 1], layer_sizes_temp[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears_temp[-1].weight)
            initializer_zero(self.linears_temp[-1].bias)
            
        self.linears_sal = torch.nn.ModuleList()
        layer_sizes_sal = [128] + [layer_widths[1]] * (n_layers[1] - 1) + [1]
        for i in range(1, len(layer_sizes_sal)):
            self.linears_sal.append(
                torch.nn.Linear(
                    layer_sizes_sal[i - 1], layer_sizes_sal[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears_sal[-1].weight)
            initializer_zero(self.linears_sal[-1].bias)
        
        self.linears_w = torch.nn.ModuleList()
        layer_sizes_w = [128] + [layer_widths[2]] * (n_layers[2] - 1) + [1]
        for i in range(1, len(layer_sizes_w)):
            self.linears_w.append(
                torch.nn.Linear(
                    layer_sizes_w[i - 1], layer_sizes_w[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears_w[-1].weight)
            initializer_zero(self.linears_w[-1].bias)
            
        self.linears_v1 = torch.nn.ModuleList()
        layer_sizes_v1 = [128] + [layer_widths[3]] * (n_layers[3] - 1) + [1]
        for i in range(1, len(layer_sizes_v1)):
            self.linears_v1.append(
                torch.nn.Linear(
                    layer_sizes_v1[i - 1], layer_sizes_v1[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears_v1[-1].weight)
            initializer_zero(self.linears_v1[-1].bias)
            
        self.linears_v2 = torch.nn.ModuleList()
        layer_sizes_v2 = [128] + [layer_widths[4]] * (n_layers[4] - 1) + [1]
        for i in range(1, len(layer_sizes_v2)):
            self.linears_v2.append(
                torch.nn.Linear(
                    layer_sizes_v2[i - 1], layer_sizes_v2[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears_v2[-1].weight)
            initializer_zero(self.linears_v2[-1].bias)
            
        self.linears_pres = torch.nn.ModuleList()
        layer_sizes_pres = [128] + [layer_widths[5]] * (n_layers[5] - 1) + [1]
        for i in range(1, len(layer_sizes_pres)):
            self.linears_pres.append(
                torch.nn.Linear(
                    layer_sizes_pres[i - 1], layer_sizes_pres[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears_pres[-1].weight)
            initializer_zero(self.linears_pres[-1].bias)
            

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        
        ############################ NEW ##################################
            
        for j, linear in enumerate(self.linears_all[:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
        x = self.linears_all[-1](x)
        
        x_temp = x_sal = x_w = x_v1 = x_v2 = x_pres = x
        
        for j, linear in enumerate(self.linears_temp[:-1]):
            x_temp = (
                self.activation[j](linear(x_temp))
                if isinstance(self.activation, list)
                else self.activation(linear(x_temp))
            )
        x_temp = self.linears_temp[-1](x_temp)
        
        for j, linear in enumerate(self.linears_sal[:-1]):
            x_sal = (
                self.activation[j](linear(x_sal))
                if isinstance(self.activation, list)
                else self.activation(linear(x_sal))
            )
        x_sal = self.linears_sal[-1](x_sal)
        
        for j, linear in enumerate(self.linears_w[:-1]):
            x_w = (
                self.activation[j](linear(x_w))
                if isinstance(self.activation, list)
                else self.activation(linear(x_w))
            )
        x_w = self.linears_w[-1](x_w)
        
        for j, linear in enumerate(self.linears_v1[:-1]):
            x_v1 = (
                self.activation[j](linear(x_v1))
                if isinstance(self.activation, list)
                else self.activation(linear(x_v1))
            )
        x_v1 = self.linears_v1[-1](x_v1)
        
        for j, linear in enumerate(self.linears_v2[:-1]):
            x_v2 = (
                self.activation[j](linear(x_v2))
                if isinstance(self.activation, list)
                else self.activation(linear(x_v2))
            )
        x_v2 = self.linears_v2[-1](x_v2)
    
        for j, linear in enumerate(self.linears_pres[:-1]):
            x_pres = (
                self.activation[j](linear(x_pres))
                if isinstance(self.activation, list)
                else self.activation(linear(x_pres))
            )
        x_pres = self.linears_pres[-1](x_pres)
                    
        x = torch.cat([x_temp, x_sal, x_w, x_v1, x_v2, x_pres], dim=1)
        
        ####################################################################
        
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
    
    
class MDRF_Net_SIMULATION(NN):
    def __init__(self, n_layers=[4] * 4, layer_width=32, layer_sizes=[5] + [4], activation="tanh", kernel_initializer="Glorot normal", Q=False):
        super().__init__()
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")
        
        ########################### NEW ################################

        self.n_layers = n_layers
        self.Q = Q

        self.linears_all = torch.nn.ModuleList()
        layer_sizes_all = [layer_sizes[0]] + [layer_width]
        for i in range(1, len(layer_sizes_all)):
            self.linears_all.append(
                torch.nn.Linear(
                    layer_sizes_all[i - 1], layer_sizes_all[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears_all[-1].weight)
            initializer_zero(self.linears_all[-1].bias)
            
        self.linears_temp = torch.nn.ModuleList()
        layer_sizes_temp = [layer_width] * n_layers[0] + [1]
        for i in range(1, len(layer_sizes_temp)):
            self.linears_temp.append(
                torch.nn.Linear(
                    layer_sizes_temp[i - 1], layer_sizes_temp[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears_temp[-1].weight)
            initializer_zero(self.linears_temp[-1].bias)
        
        self.linears_w = torch.nn.ModuleList()
        layer_sizes_w = [layer_width] * n_layers[1] + [1]
        for i in range(1, len(layer_sizes_w)):
            self.linears_w.append(
                torch.nn.Linear(
                    layer_sizes_w[i - 1], layer_sizes_w[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears_w[-1].weight)
            initializer_zero(self.linears_w[-1].bias)
            
        self.linears_v = torch.nn.ModuleList()
        layer_sizes_v = [layer_width] * n_layers[2] + [1]
        for i in range(1, len(layer_sizes_v)):
            self.linears_v.append(
                torch.nn.Linear(
                    layer_sizes_v[i - 1], layer_sizes_v[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears_v[-1].weight)
            initializer_zero(self.linears_v[-1].bias)
            
        self.linears_pres = torch.nn.ModuleList()
        layer_sizes_pres = [layer_width] * n_layers[3] + [1]
        for i in range(1, len(layer_sizes_pres)):
            self.linears_pres.append(
                torch.nn.Linear(
                    layer_sizes_pres[i - 1], layer_sizes_pres[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears_pres[-1].weight)
            initializer_zero(self.linears_pres[-1].bias)
            
        if self.Q:
            self.linears_q = torch.nn.ModuleList()
            layer_sizes_q = [layer_width] * n_layers[3] + [1]
            for i in range(1, len(layer_sizes_q)):
                self.linears_q.append(
                    torch.nn.Linear(
                        layer_sizes_q[i - 1], layer_sizes_q[i], dtype=config.real(torch)
                    )
                )
                initializer(self.linears_q[-1].weight)
                initializer_zero(self.linears_q[-1].bias)
            

    def forward(self, inputs):
        x = inputs
        z = inputs[:, 1:2]

        if self._input_transform is not None:
            x = self._input_transform(x)
        
        ############################ NEW ##################################
            
        for j, linear in enumerate(self.linears_all[:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
        x = self.linears_all[-1](x)
        
        x_temp = x_w = x_v = x_pres = x

        if self.Q:
            x_q = x
        
        for j, linear in enumerate(self.linears_temp[:-1]):
            x_temp = (
                self.activation[j](linear(x_temp))
                if isinstance(self.activation, list)
                else self.activation(linear(x_temp))
            )
        x_temp = self.linears_temp[-1](x_temp)
        
        for j, linear in enumerate(self.linears_w[:-1]):
            x_w = (
                self.activation[j](linear(x_w))
                if isinstance(self.activation, list)
                else self.activation(linear(x_w))
            )
        # x_w = self.linears_w[-1](x_w) * (1 - torch.exp(0 - z)) * (1 - torch.exp(z - 1))
        x_w = self.linears_w[-1](x_w)

        x_w *= (1 - torch.exp(0 - z)) * (1 - torch.exp(z - 1))
        
        
        for j, linear in enumerate(self.linears_v[:-1]):
            x_v = (
                self.activation[j](linear(x_v))
                if isinstance(self.activation, list)
                else self.activation(linear(x_v))
            )
        x_v = self.linears_v[-1](x_v)
    
        for j, linear in enumerate(self.linears_pres[:-1]):
            x_pres = (
                self.activation[j](linear(x_pres))
                if isinstance(self.activation, list)
                else self.activation(linear(x_pres))
            )
        x_pres = self.linears_pres[-1](x_pres)
        
        if self.Q:
            for j, linear in enumerate(self.linears_q[:-1]):
                x_q = (
                    self.activation[j](linear(x_q))
                    if isinstance(self.activation, list)
                    else self.activation(linear(x_q))
                )
            x_q = self.linears_q[-1](x_q)
                    
        if self.Q:
            x = torch.cat([x_temp, x_w, x_v, x_pres, x_q], dim=1)
        else:
            x = torch.cat([x_temp, x_w, x_v, x_pres], dim=1)
        
        ####################################################################
        
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
    
    
def saveplot_2(
    loss_history,
    train_state,
    issave=True,
    isplot=True,
    loss_fname="loss.dat",
    train_fname="train.dat",
    test_fname="test.dat",
    output_dir=None,
):
    """Save/plot the loss history and best trained result.

    This function is used to quickly check your results. To better investigate your
    result, use ``save_loss_history()`` and ``save_best_state()``.

    Args:
        loss_history: ``LossHistory`` instance. The first variable returned from
            ``Model.train()``.
        train_state: ``TrainState`` instance. The second variable returned from
            ``Model.train()``.
        issave (bool): Set ``True`` (default) to save the loss, training points,
            and testing points.
        isplot (bool): Set ``True`` (default) to plot loss, metric, and the predicted
            solution.
        loss_fname (string): Name of the file to save the loss in.
        train_fname (string): Name of the file to save the training points in.
        test_fname (string): Name of the file to save the testing points in.
        output_dir (string): If ``None``, use the current working directory.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    if not os.path.exists(output_dir):
        print(f"Warning: Directory {output_dir} doesn't exist. Creating it.")
        os.mkdir(output_dir)

    if issave:
        loss_fname = os.path.join(output_dir, loss_fname)
        train_fname = os.path.join(output_dir, train_fname)
        test_fname = os.path.join(output_dir, test_fname)
        save_loss_history(loss_history, loss_fname)
        save_best_state(train_state, train_fname, test_fname)

    if isplot:
        plot_loss_history(loss_history, fname=os.path.join(output_dir, 'loss_plot.png'))
        plot_best_state(train_state)
        plt.show()


class LossHistory2:
    def __init__(self, meta_learning=False):
        self.steps = []
        self.loss_train = []
        self.loss_test = []
        self.metrics_test = []
        self.lambdas = []
        self.meta_learning = meta_learning

    def append(self, step, loss_train, loss_test, metrics_test, lambdas=None):
        self.steps.append(step)
        self.loss_train.append(loss_train)

        if loss_test is None:
            loss_test = self.loss_test[-1]
        if metrics_test is None:
            metrics_test = self.metrics_test[-1]

        self.loss_test.append(loss_test)
        self.metrics_test.append(metrics_test)

        if lambdas is not None and self.meta_learning:
            if type(lambdas) is not list:
                lambdas = lambdas.tolist()
            self.lambdas.append(lambdas)
