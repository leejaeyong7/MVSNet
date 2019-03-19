
from dataset import MVSDataset
d = MVSDataset('/mnt/data/datasets/tnt/train/sets', 3, 128, 640, 512, 'train')
test = iter(d)
for data in test:
  cam = data[1]
  print(cam)
  print(cam.shape)
  print(cam[0, 0])
  print(cam[0, 1])
  break
