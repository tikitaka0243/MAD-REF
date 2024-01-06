class FNN(NN):

    def __init__(self, n_layers, C, layer_sizes, activation, kernel_initializer, HT=False):
        self.HT = HT
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
        # [3] * 2 + [4] * 4 + [2]
        if self.HT:
            self.n_layers = n_layers
            self.C = C
            
            if self.C == 0:
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
                
                self.linears_argo = torch.nn.ModuleList()
                layer_sizes_argo = [128] * n_layers[0] + [4]
                for i in range(1, len(layer_sizes_argo)):
                    self.linears_argo.append(
                        torch.nn.Linear(
                            layer_sizes_argo[i - 1], layer_sizes_argo[i], dtype=config.real(torch)
                        )
                    )
                    initializer(self.linears_argo[-1].weight)
                    initializer_zero(self.linears_argo[-1].bias)

                self.linears_cur = torch.nn.ModuleList()
                layer_sizes_cur = [128] * n_layers[1] + [4]
                for i in range(1, len(layer_sizes_cur)):
                    self.linears_cur.append(
                        torch.nn.Linear(
                            layer_sizes_cur[i - 1], layer_sizes_cur[i], dtype=config.real(torch)
                        )
                    )
                    initializer(self.linears_cur[-1].weight)
                    initializer_zero(self.linears_cur[-1].bias)
                    
            elif self.C == 1:
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
                    
                self.linears_temp_1 = torch.nn.ModuleList()
                layer_sizes_temp_1 = [128] * n_layers[0] + [1]
                for i in range(1, len(layer_sizes_temp_1)):
                    self.linears_temp_1.append(
                        torch.nn.Linear(
                            layer_sizes_temp_1[i - 1], layer_sizes_temp_1[i], dtype=config.real(torch)
                        )
                    )
                    initializer(self.linears_temp_1[-1].weight)
                    initializer_zero(self.linears_temp_1[-1].bias)
                    
                self.linears_sal_1 = torch.nn.ModuleList()
                layer_sizes_sal_1 = [128] * n_layers[1] + [1]
                for i in range(1, len(layer_sizes_sal_1)):
                    self.linears_sal_1.append(
                        torch.nn.Linear(
                            layer_sizes_sal_1[i - 1], layer_sizes_sal_1[i], dtype=config.real(torch)
                        )
                    )
                    initializer(self.linears_sal_1[-1].weight)
                    initializer_zero(self.linears_sal_1[-1].bias)
                
                self.linears_w_1 = torch.nn.ModuleList()
                layer_sizes_w_1 = [128] * n_layers[2] + [1]
                for i in range(1, len(layer_sizes_w_1)):
                    self.linears_w_1.append(
                        torch.nn.Linear(
                            layer_sizes_w_1[i - 1], layer_sizes_w_1[i], dtype=config.real(torch)
                        )
                    )
                    initializer(self.linears_w_1[-1].weight)
                    initializer_zero(self.linears_w_1[-1].bias)
                    
                self.linears_v1_1 = torch.nn.ModuleList()
                layer_sizes_v1_1 = [128] * n_layers[3] + [1]
                for i in range(1, len(layer_sizes_v1_1)):
                    self.linears_v1_1.append(
                        torch.nn.Linear(
                            layer_sizes_v1_1[i - 1], layer_sizes_v1_1[i], dtype=config.real(torch)
                        )
                    )
                    initializer(self.linears_v1_1[-1].weight)
                    initializer_zero(self.linears_v1_1[-1].bias)
                    
                self.linears_v2_1 = torch.nn.ModuleList()
                layer_sizes_v2_1 = [128] * n_layers[4] + [1]
                for i in range(1, len(layer_sizes_v2_1)):
                    self.linears_v2_1.append(
                        torch.nn.Linear(
                            layer_sizes_v2_1[i - 1], layer_sizes_v2_1[i], dtype=config.real(torch)
                        )
                    )
                    initializer(self.linears_v2_1[-1].weight)
                    initializer_zero(self.linears_v2_1[-1].bias)
                    
                self.linears_pres_1 = torch.nn.ModuleList()
                layer_sizes_pres_1 = [128] * n_layers[5] + [1]
                for i in range(1, len(layer_sizes_pres_1)):
                    self.linears_pres_1.append(
                        torch.nn.Linear(
                            layer_sizes_pres_1[i - 1], layer_sizes_pres_1[i], dtype=config.real(torch)
                        )
                    )
                    initializer(self.linears_pres_1[-1].weight)
                    initializer_zero(self.linears_pres_1[-1].bias)
            
            else:
                self.linears_temp = torch.nn.ModuleList()
                layer_sizes_temp = [4] + [128] * n_layers[0] + [1]
                for i in range(1, len(layer_sizes_temp)):
                    self.linears_temp.append(
                        torch.nn.Linear(
                            layer_sizes_temp[i - 1], layer_sizes_temp[i], dtype=config.real(torch)
                        )
                    )
                    initializer(self.linears_temp[-1].weight)
                    initializer_zero(self.linears_temp[-1].bias)
                    
                self.linears_sal = torch.nn.ModuleList()
                layer_sizes_sal = [4] + [128] * n_layers[1] + [1]
                for i in range(1, len(layer_sizes_sal)):
                    self.linears_sal.append(
                        torch.nn.Linear(
                            layer_sizes_sal[i - 1], layer_sizes_sal[i], dtype=config.real(torch)
                        )
                    )
                    initializer(self.linears_sal[-1].weight)
                    initializer_zero(self.linears_sal[-1].bias)
                
                self.linears_w = torch.nn.ModuleList()
                layer_sizes_w = [4] + [128] * n_layers[2] + [1]
                for i in range(1, len(layer_sizes_w)):
                    self.linears_w.append(
                        torch.nn.Linear(
                            layer_sizes_w[i - 1], layer_sizes_w[i], dtype=config.real(torch)
                        )
                    )
                    initializer(self.linears_w[-1].weight)
                    initializer_zero(self.linears_w[-1].bias)
                    
                self.linears_v1 = torch.nn.ModuleList()
                layer_sizes_v1 = [4] + [128] * n_layers[3] + [1]
                for i in range(1, len(layer_sizes_v1)):
                    self.linears_v1.append(
                        torch.nn.Linear(
                            layer_sizes_v1[i - 1], layer_sizes_v1[i], dtype=config.real(torch)
                        )
                    )
                    initializer(self.linears_v1[-1].weight)
                    initializer_zero(self.linears_v1[-1].bias)
                    
                self.linears_v2 = torch.nn.ModuleList()
                layer_sizes_v2 = [4] + [128] * n_layers[4] + [1]
                for i in range(1, len(layer_sizes_v2)):
                    self.linears_v2.append(
                        torch.nn.Linear(
                            layer_sizes_v2[i - 1], layer_sizes_v2[i], dtype=config.real(torch)
                        )
                    )
                    initializer(self.linears_v2[-1].weight)
                    initializer_zero(self.linears_v2[-1].bias)
            
                self.linears_pres = torch.nn.ModuleList()
                layer_sizes_pres = [4] + [128] * n_layers[5] + [1]
                for i in range(1, len(layer_sizes_pres)):
                    self.linears_pres.append(
                        torch.nn.Linear(
                            layer_sizes_pres[i - 1], layer_sizes_pres[i], dtype=config.real(torch)
                        )
                    )
                    initializer(self.linears_pres[-1].weight)
                    initializer_zero(self.linears_pres[-1].bias)
            
            # if self.C != 0:
            #     self.linears_para = torch.nn.ModuleList()
            #     layer_sizes_para = [3] + [128] * n_layers[6] + [2]
            #     for i in range(1, len(layer_sizes_para)):
            #         self.linears_para.append(
            #             torch.nn.Linear(
            #                 layer_sizes_para[i - 1], layer_sizes_para[i], dtype=config.real(torch)
            #             )
            #         )
            #         initializer(self.linears_para[-1].weight)
            #         initializer_zero(self.linears_para[-1].bias)
        
        #######################################################################

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for j, linear in enumerate(self.linears[:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
        x = self.linears[-1](x)
        
        ############################ NEW ###################################
        
        if self.HT:
            if self.C == 0:
                for j, linear in enumerate(self.linears_all[:-1]):
                    inputs = (
                        self.activation[j](linear(inputs))
                        if isinstance(self.activation, list)
                        else self.activation(linear(inputs))
                    )
                inputs = self.linears_all[-1](inputs)
                
                x_cur = x_argo = inputs
                
                for j, linear in enumerate(self.linears_argo[:-1]):
                    x_argo = (
                        self.activation[j](linear(x_argo))
                        if isinstance(self.activation, list)
                        else self.activation(linear(x_argo))
                    )
                x_argo = self.linears_argo[-1](x_argo)
                
                for j, linear in enumerate(self.linears_cur[:-1]):
                    x_cur = (
                        self.activation[j](linear(x_cur))
                        if isinstance(self.activation, list)
                        else self.activation(linear(x_cur))
                    )
                x_cur = self.linears_cur[-1](x_cur)
            
            
            elif self.C == 1:
                # x_para = torch.cat([inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]], dim=1)
                
                for j, linear in enumerate(self.linears_all[:-1]):
                    inputs = (
                        self.activation[j](linear(inputs))
                        if isinstance(self.activation, list)
                        else self.activation(linear(inputs))
                    )
                inputs = self.linears_all[-1](inputs)
                
                x_temp = x_sal = x_w = x_v1 = x_v2 = x_pres = inputs
                
                for j, linear in enumerate(self.linears_temp_1[:-1]):
                    x_temp = (
                        self.activation[j](linear(x_temp))
                        if isinstance(self.activation, list)
                        else self.activation(linear(x_temp))
                    )
                x_temp = self.linears_temp_1[-1](x_temp)
                
                for j, linear in enumerate(self.linears_sal_1[:-1]):
                    x_sal = (
                        self.activation[j](linear(x_sal))
                        if isinstance(self.activation, list)
                        else self.activation(linear(x_sal))
                    )
                x_sal = self.linears_sal_1[-1](x_sal)
                
                for j, linear in enumerate(self.linears_w_1[:-1]):
                    x_w = (
                        self.activation[j](linear(x_w))
                        if isinstance(self.activation, list)
                        else self.activation(linear(x_w))
                    )
                x_w = self.linears_w_1[-1](x_w)
                
                for j, linear in enumerate(self.linears_v1_1[:-1]):
                    x_v1 = (
                        self.activation[j](linear(x_v1))
                        if isinstance(self.activation, list)
                        else self.activation(linear(x_v1))
                    )
                x_v1 = self.linears_v1_1[-1](x_v1)
                
                for j, linear in enumerate(self.linears_v2_1[:-1]):
                    x_v2 = (
                        self.activation[j](linear(x_v2))
                        if isinstance(self.activation, list)
                        else self.activation(linear(x_v2))
                    )
                x_v2 = self.linears_v2_1[-1](x_v2)
            
                for j, linear in enumerate(self.linears_pres_1[:-1]):
                    x_pres = (
                        self.activation[j](linear(x_pres))
                        if isinstance(self.activation, list)
                        else self.activation(linear(x_pres))
                    )
                x_pres = self.linears_pres_1[-1](x_pres)
            
            
            else:
                x_temp = x_sal = x_w = x_v1 = x_v2 = x_pres = inputs
                x_para = torch.cat([inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]], dim=1)
                
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
            
            # if self.C != 0:
            #     for j, linear in enumerate(self.linears_para[:-1]):
            #         x_para = (
            #             self.activation[j](linear(x_para))
            #             if isinstance(self.activation, list)
            #             else self.activation(linear(x_para))
            #         )
            #     x_para = self.linears_para[-1](x_para)
            
            if self.C == 0:
                x = torch.cat([x_argo[:, :2], x_cur, x_argo[:, 2:4]], dim=1)
            else:
                x = torch.cat([x_temp, x_sal, x_w, x_v1, x_v2, x_pres], dim=1)
        
        ####################################################################
        
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
