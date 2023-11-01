from cifar10_tutorial_dataset import MyDataset

dataset = MyDataset()
print(len(dataset))

item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)
# 1098477
# malinois
# (224, 224, 3)
# (224, 224, 3)