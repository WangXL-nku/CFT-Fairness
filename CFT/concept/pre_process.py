import os
import shutil
basedir = '/media/data0/wzy/dataset/CelebA/'
# 读取分组信息
partition_file = os.path.join(basedir,'Eval/list_eval_partition.txt')
with open(partition_file, 'r') as f:
    partitions = f.readlines()

# 创建目录
os.makedirs(os.path.join(basedir,'train'), exist_ok=True)
os.makedirs(os.path.join(basedir,'val'), exist_ok=True)
os.makedirs(os.path.join(basedir,'test'), exist_ok=True)

# 移动图片到对应的分组文件夹
for line in partitions:
    img_name, partition = line.strip().split()
    partition = int(partition)
    if partition == 0:
        shutil.move(os.path.join(basedir, 'Img', f'img_align_celeba/{img_name}'), os.path.join(basedir, 'train/'))
    elif partition == 1:
        shutil.move(os.path.join(basedir, 'Img', f'img_align_celeba/{img_name}'), os.path.join(basedir, 'val/'))
    elif partition == 2:
        shutil.move(os.path.join(basedir, 'Img', f'img_align_celeba/{img_name}'), os.path.join(basedir, 'test/'))
