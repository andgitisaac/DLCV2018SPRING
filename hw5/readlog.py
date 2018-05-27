import csv
import numpy as np
from matplotlib import pyplot as plt


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


base_log_path = 'C:/Users/Isaac/Desktop/workstation/hw5/p1_base_log.out'
finetune_log_path = 'C:/Users/Isaac/Desktop/workstation/hw5/p1_finetune_log.out'
p3_log_path = 'C:/Users/Isaac/Desktop/workstation/hw5/p3_log.out'

with open(p3_log_path) as f:
    data = f.readlines()
data = csv.reader(data)
acc = []
loss = []
Nsample = 43
for row in data:
    if row[0][:5] == 'Epoch':
        splited = row[0].split(' ')
        loss.append(float(splited[5]))
        acc.append(float(splited[7]))

loss = loss[::Nsample]
acc = acc[::Nsample]
x = list(range(len(loss)))
x = [i*Nsample for i in x]

fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(1, 1, 1)
ax.plot(x, loss, 'r', label='Training Loss')
ax.plot(x, acc, 'b', label='Training Accuracy')
ax.legend()
ax.set_title('Learning Curve')
ax.set_xlabel('Training Steps')
ax.set_ylabel('Accuracy and Loss')
# ax.set_ylim(-0.1, 2.7)
simpleaxis(ax)
plt.show()




# with open(base_log_path) as f:
#     base = f.readlines()
# base = csv.reader(base)

# with open(finetune_log_path) as f:
#     fine = f.readlines()
# fine = csv.reader(fine)

# base_acc = []
# base_loss = []
# Nsample = 50
# for row in base:
#     if row[0][:5] == 'Epoch':
#         splited = row[0].split(' ')
#         base_loss.append(float(splited[5]))
#         base_acc.append(float(splited[7]))

# fine_acc = []
# fine_loss = []
# for row in fine:
#     if row[0][:5] == 'Epoch':
#         splited = row[0].split(' ')
#         fine_loss.append(float(splited[5]))
#         fine_acc.append(float(splited[7]))

# base_loss = base_loss[::Nsample]
# base_acc = base_acc[::Nsample]
# x1 = list(range(len(base_loss)))
# x1 = [x * Nsample for x in x1]
# fine_loss = fine_loss[:5000]
# fine_acc = fine_acc[:5000]
# x2 = list(range(len(fine_loss)))



# fig = plt.figure(figsize=(15, 6))

# ax = fig.add_subplot(1, 2, 1)
# ax.plot(x1, base_loss, 'r', label='Training Loss')
# ax.plot(x1, base_acc, 'b', label='Training Accuracy')
# ax.legend()
# ax.set_title('Learning Curve of finetuned Resnet50 (frame by frame)')
# ax.set_xlabel('Training Steps')
# ax.set_ylabel('Accuracy and Loss')
# ax.set_ylim(-0.1, 2.7)
# simpleaxis(ax)

# ax = fig.add_subplot(1, 2, 2)
# ax.plot(x2, fine_loss, 'r', label='Training Loss')
# ax.plot(x2, fine_acc, 'b', label='Training Accuracy')
# ax.legend()
# ax.set_title('Learning Curve of finetuned Resnet50 (video by video)')
# ax.set_xlabel('Training Steps')
# ax.set_ylabel('Accuracy and Loss')
# ax.set_ylim(-0.1, 2.7)
# simpleaxis(ax)

# plt.show()