import cPickle as pkl
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from layers import Flatten
import torch.optim as optim 
from torch.autograd import Variable
from torchvision.models import resnet18


def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pkl.load(fo)
	return dict

def get_image(array):
	array = np.transpose(array, (1,2,0))
	img = Image.fromarray(array, 'RGB')
	return img

def show_image(array):
	img = get_image(array)
	img.show()
	return

def class_accuracy(targets, predictions):
	return np.mean(targets == predictions)



def get_cifar_simple_model():
	model = nn.Sequential(
		nn.Conv2d(3, 16, kernel_size=3, stride=1),
		nn.ReLU(inplace=True),
		nn.BatchNorm2d(16),
		nn.AdaptiveMaxPool2d(16),
		nn.Conv2d(16, 32, kernel_size=3, stride=1),
		nn.ReLU(inplace=True),
		nn.BatchNorm2d(32),
		nn.AdaptiveMaxPool2d(8),
		Flatten(),
		nn.Linear(32*8*8, 256),
		nn.ReLU(inplace=True),
		nn.Linear(256, 10)
	)
	return model

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        # print out.data.numpy().shape
        # print residual.data.numpy().shape

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        # self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = BasicBlock(64, 64)
        self.layer2 = BasicBlock(64, 64)
        self.layer3 = BasicBlock(64, 64)
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(64*4, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    # def _make_layer(self, block, planes, blocks, stride=1):
    #     # downsample = None
    #     # if stride != 1 or self.inplanes != planes * block.expansion:
    #     #     downsample = nn.Sequential(
    #     #         nn.Conv2d(self.inplanes, planes * block.expansion,
    #     #                   kernel_size=1, stride=stride, bias=False),
    #     #         nn.BatchNorm2d(planes * block.expansion),
    #     #     )

    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(self.inplanes, planes))

    #     return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CifarDataset(Dataset):
	def __init__(self, dtype=torch.FloatTensor):
		self.dtype = dtype

		pickle_data = unpickle('cifar-10-batches-py/data_batch_1')
		self.data = pickle_data['data']
		self.data = np.divide(self.data, 255.)
		self.data = np.reshape(self.data, (10000,3,32,32))

		self.labels = np.array(pickle_data['labels'])
		self.labels = self.labels.astype(np.ndarray)

	def __getitem__(self, index):
		return self.data[index,:,:,:], self.labels[index]
		# return self.data[index,:,:,:], torch.from_numpy(self.labels[index]).type(self.dtype)

	def __len__(self):
		return self.data.shape[0]

def generate_train_val_dataloader(dataset, batch_size, num_workers,
								  shuffle=True, split=0.9, use_fraction_of_data=1.):
	"""
	return two Data`s split into training and validation
	`split` sets the train/val split fraction (0.9 is 90 % training data)
	u
	"""
	## this is a testing feature to make epochs go faster, uses only some of the available data
	if use_fraction_of_data < 1.:
		n_samples = int(use_fraction_of_data * len(dataset))
	else:
		n_samples = len(dataset)
	inds = np.arange(n_samples)
	train_inds, val_inds = train_test_split(inds, test_size=1-split, train_size=split)

	train_loader = DataLoader(
		dataset,
		sampler=SubsetRandomSampler(train_inds),
		batch_size=batch_size,
		# shuffle=shuffle,
		num_workers=num_workers
	)
	val_loader = DataLoader(
		dataset,
		sampler=SubsetRandomSampler(val_inds),
		batch_size=batch_size,
		# shuffle=shuffle,
		num_workers=num_workers
	)
	return train_loader, val_loader


class PytorchTrainer():
	def __init__(self, train_loader, val_loader, model, optimizer, loss_fn, acc_fn, dtype=torch.FloatTensor):
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.model = model.type(dtype)
		self.optimizer = optimizer
		self.loss_fn = loss_fn
		self.acc_fn = acc_fn

		self.acc_history = []
		self.loss_history = []
		self.scores = []

		self.val_acc = []
		self.val_loss = []
		self.val_scores = []

	def train_epoch(self, print_every=1):
		print 'training'
		self.model.train()
		epoch_scores = []
		for t, (x, y) in enumerate(self.train_loader):
			x_var = Variable(x.type(dtype))
			y_var = Variable(y.type(torch.LongTensor))

			scores = model(x_var)
			for score in scores:
				epoch_scores.append(score)

			loss = loss_fn(scores, y_var)
			self.scores.append(scores)
			self.loss_history.append(loss.data[0])

			y_pred = np.argmax(scores.data.numpy(), axis=1)

			acc = self.acc_fn(y.numpy(), y_pred)
			self.acc_history.append(acc)

			# if (t + 1) % print_every == 0:
			#	 print('t = %d, loss = %.4f, f2 = %.4f' % (t + 1, loss.data[0], acc))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# print np.mean(self.loss_history)
			print np.round(np.mean(self.acc_history),2), np.round(np.mean(self.loss_history),2)


		self.scores.append(epoch_scores)

	def validate(self):
		print 'validation'
		self.model.eval()
		epoch_scores = []
		for t, (x, y) in enumerate(self.val_loader):
			x_var = Variable(x.type(dtype))
			y_var = Variable(y.type(torch.LongTensor))

			scores = model(x_var)
			for score in scores:
				epoch_scores.append(score)

			loss = loss_fn(scores, y_var)
			self.val_scores.append(scores)
			self.val_loss.append(loss.data[0])

			y_pred = np.argmax(scores.data.numpy(), axis=1)

			acc = self.acc_fn(y.numpy(), y_pred)
			self.val_acc.append(acc)

			print np.round(np.mean(self.acc_history),2), np.round(np.mean(self.loss_history),2)





if __name__ == '__main__':
	mode = 'new'
	# mode = 'continue'

	if mode == 'new':
		dtype = torch.FloatTensor
		dataset = CifarDataset()
		# model = get_cifar_simple_model()
		model = ResNet(num_classes=10)
		optimizer = optim.Adam(model.parameters(), lr=1e-2)
		loss_fn = nn.CrossEntropyLoss()
		train_loader, val_loader = generate_train_val_dataloader(dataset=dataset, batch_size=100, num_workers=4, shuffle=True, split=0.9, use_fraction_of_data=1.)
		acc_fn = class_accuracy
		trainer = PytorchTrainer(train_loader, val_loader, model, optimizer, loss_fn, acc_fn, dtype)

		trainer.train_epoch()
		trainer.validate()
		trainer.optimizer = optim.Adam(model.parameters(), lr=5e-3)
		trainer.train_epoch()
		trainer.validate()
		# pkl.dump(trainer, open('trainer.pkl','wb'))

	elif mode == 'continue':
		trainer = pkl.load(open('trainer.pkl','rb'))
		trainer.optimizer = optim.Adam(model.parameters(), lr=5e-3)
		trainer.train_epoch()


	# for a in train_loader:
	# 	print a.numpy().shape
	# 	show_image(np.squeeze(a.numpy()))
	# 	break

































