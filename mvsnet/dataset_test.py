
from dataset import MVSDataset
d = MVSDataset('/mnt/data/datasets/tnt/train/sets',3,1.5, 256, 640, 512, 'train', None)
test = iter(d)
for data in test:
    img, cam, depth = data
    print(img.shape)
    print(cam.shape)
    print(depth.shape)
    print(cam)
    break
