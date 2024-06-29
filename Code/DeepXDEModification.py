import os
from deepxde.model import Model
from deepxde.nn.pytorch.nn import NN
from deepxde.nn import activations, initializers
from deepxde import config
from deepxde.utils.external import save_loss_history, save_best_state, plot_loss_history, plot_best_state
from matplotlib import pyplot as plt
import torch


class Model2(Model):
    def change_data(self, new_data):
        self.data = new_data
        

class MDRF_Net(NN):
    def __init__(self, n_layers=[6] * 2 + [8] * 4, layer_width=128, layer_sizes=[4] + [6], activation="tanh", kernel_initializer="Glorot uniform"):
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

        self.linears_all = torch.nn.ModuleList()
        layer_sizes_all = [4] + [layer_width]
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
            
        self.linears_sal = torch.nn.ModuleList()
        layer_sizes_sal = [layer_width] * n_layers[1] + [1]
        for i in range(1, len(layer_sizes_sal)):
            self.linears_sal.append(
                torch.nn.Linear(
                    layer_sizes_sal[i - 1], layer_sizes_sal[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears_sal[-1].weight)
            initializer_zero(self.linears_sal[-1].bias)
        
        self.linears_w = torch.nn.ModuleList()
        layer_sizes_w = [layer_width] * n_layers[2] + [1]
        for i in range(1, len(layer_sizes_w)):
            self.linears_w.append(
                torch.nn.Linear(
                    layer_sizes_w[i - 1], layer_sizes_w[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears_w[-1].weight)
            initializer_zero(self.linears_w[-1].bias)
            
        self.linears_v1 = torch.nn.ModuleList()
        layer_sizes_v1 = [layer_width] * n_layers[3] + [1]
        for i in range(1, len(layer_sizes_v1)):
            self.linears_v1.append(
                torch.nn.Linear(
                    layer_sizes_v1[i - 1], layer_sizes_v1[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears_v1[-1].weight)
            initializer_zero(self.linears_v1[-1].bias)
            
        self.linears_v2 = torch.nn.ModuleList()
        layer_sizes_v2 = [layer_width] * n_layers[4] + [1]
        for i in range(1, len(layer_sizes_v2)):
            self.linears_v2.append(
                torch.nn.Linear(
                    layer_sizes_v2[i - 1], layer_sizes_v2[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears_v2[-1].weight)
            initializer_zero(self.linears_v2[-1].bias)
            
        self.linears_pres = torch.nn.ModuleList()
        layer_sizes_pres = [layer_width] * n_layers[5] + [1]
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
    def __init__(self, n_layers=[4] * 4, layer_width=32, layer_sizes=[3] + [4], activation="tanh", kernel_initializer="Glorot normal", Q=False):
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
        layer_sizes_all = [3] + [layer_width]
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
        x_w = self.linears_w[-1](x_w)
        
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