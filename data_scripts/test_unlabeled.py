from voc_unlabeled_dataset import VOCUnlabeledDataset

root = r"C:/Users/Gheffari Youcef/Downloads/semi-supervised_segmentation_approach-master/datasets/VOC2012_test"

dataset = VOCUnlabeledDataset(root, target_size=(256, 256))
print("Dataset size:", len(dataset))

image = dataset[0]
print("Image shape:", image.shape)
