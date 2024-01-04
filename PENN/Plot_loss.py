import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


print('################ Plot_loss ################')


loss = pd.read_csv('loss.csv', header=None)
loss_va = pd.read_csv('loss_va.csv', header=None)
loss.columns = loss_va.columns = ['x', 'temp', 'sal', 'w', 'v_theta', 'v_phi']

loss_rmse = loss
loss_va_rmse = loss_va

loss_rmse.iloc[:, 1:] = np.sqrt(loss.iloc[:, 1:])
loss_va_rmse.iloc[:, 1:] = loss_va.iloc[:, 1:].apply(np.sqrt)


if not os.path.exists('Plot_loss'):
    os.makedirs('Plot_loss')

for var in loss.columns[1:]:
    print('Minimun train RMSE of ' + var + ':', min(loss_rmse[var]))
    print('Minimun validate RMSE of ' + var + ':', min(loss_va_rmse[var]))
    plt.figure(figsize=(20, 10), dpi=150)
    # plt.plot(loss_rmse['x'], loss_rmse[var], color='blue', label='Train')
    plt.plot(loss_va_rmse['x'], loss_va_rmse[var], color='red', label='Validation')
    plt.legend()
    plt.savefig('Plot_loss/loss_' + var + '.png', bbox_inches='tight')

# print('loss_rmse\n', loss_rmse)
# print('loss_va_rmse\n', loss_va_rmse)