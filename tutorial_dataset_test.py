from tutorial_dataset import MyDataset

dataset = MyDataset()
print(len(dataset))

item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)

# 50000
# burly wood circle with orange background
# (512, 512, 3)
# (512, 512, 3)