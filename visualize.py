### TODO: add your import
import matplotlib.pyplot as plt
import os
import numpy as np

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

def visualize_rouge_l(rouge_ls, out_dir):
    rouge_ls = np.array(rouge_ls)
    plt.hist(rouge_ls)  # 绘制频数直方图
    plt.xlabel("rouge_l")
    plt.ylabel("frequency")
    plt.title(f"rouge_l evaluation, mean value = {np.mean(rouge_ls)}")
    plt.savefig(out_dir + "/rouge_l.png")
    plt.close()

def visualize_perplexity(perplexities, out_dir):
    perplexities = np.array(perplexities)
    plt.hist(perplexities)  # 绘制频数直方图
    plt.xlabel("perplexity")
    plt.ylabel("frequency")
    plt.title(f"perplexity evaluation, mean value = {np.mean(perplexities)}")
    plt.savefig(out_dir + "/perplexity.png")
    plt.close()