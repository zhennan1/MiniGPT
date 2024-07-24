### TODO: add your import
import matplotlib.pyplot as plt
import os

def visualize_loss(train_loss_list, train_interval, val_loss_list, val_interval, dataset, out_dir):
    ### TODO: visualize loss of training & validation and save to [out_dir]/loss.png
    train_steps = [i * train_interval for i in range(len(train_loss_list))]
    val_steps = [i * val_interval for i in range(len(val_loss_list))]
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss_list, label='train')
    plt.plot(val_steps, val_loss_list, label='validation')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.title(f'Loss of {dataset}')
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()
    ###
