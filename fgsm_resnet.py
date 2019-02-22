from __future__ import print_function
import torch, cv2, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "/home/zw119/research/res_cifar/trained_weights/resnet56_cifar10_80000.pt"
use_cuda=True

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

target_dir = '/home/zw119/research/res_cifar/cifar/perturbed'

class LambdaLayer(nn.Module):
	def __init__(self, lambd):
		super(LambdaLayer, self).__init__()
		self.lambd = lambd

	def forward(self, x):
		return self.lambd(x)

def _weights_init(m):
	classname = m.__class__.__name__
	#print(classname)
	if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
		init.kaiming_normal(m.weight)

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, option='A'):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != planes:
			if option == 'A':
				"""
				For CIFAR10 ResNet paper uses option A.
				"""
				self.shortcut = LambdaLayer(lambda x:
											F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
			elif option == 'B':
				self.shortcut = nn.Sequential(
					 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
					 nn.BatchNorm2d(self.expansion * planes)
				)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out

class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(ResNet, self).__init__()
		self.in_planes = 16

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
		self.linear = nn.Linear(64, num_classes)

		self.apply(_weights_init)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion

		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = F.avg_pool2d(out, out.size()[3])
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out

def resnet56():
	return ResNet(BasicBlock, [9, 9, 9])

def resnet110():
	return ResNet(BasicBlock, [18, 18, 18])

def resnet1202():
	return ResNet(BasicBlock, [200, 200, 200])

test_data_dir = '/home/zw119/research/res_cifar/cifar/classified'
data_transforms = transforms.Compose([
			transforms.ToTensor(),
			])
test_dataset = datasets.ImageFolder(test_data_dir,
										  data_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
											 shuffle=True, num_workers=4)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = resnet56().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()



def fgsm_attack(image, epsilon, data_grad):
	# Collect the element-wise sign of the data gradient
	sign_data_grad = data_grad.sign()
	# Create the perturbed image by adjusting each pixel of the input image
	perturbed_image = image + epsilon*sign_data_grad
	# Adding clipping to maintain [0,1] range
	perturbed_image = torch.clamp(perturbed_image, 0, 1)
	# Return the perturbed image
	return perturbed_image




def save_img(img_tensor, class_num, count):
	img_numpy = img_tensor.squeeze().detach().cpu().numpy()
	img_numpy = np.transpose(img_numpy, (1, 2, 0))
	img_numpy *= 255

	class_name = classes[class_num]
	current_dir = os.path.join(target_dir, class_name)
	if not os.path.exists(current_dir):
		os.mkdir(current_dir)
	img_path = os.path.join(current_dir, '{}.png'.format(count))

	cv2.imwrite(img_path, img_numpy)




def transform_label(target):
	num = target.item()
	num_trans = [2, 1, 3, 4, 5, 6, 7, 0, 8, 9]
	target[0] = num_trans[num]
	return target


def reverse_transform_label(num):
	num_trans_rev = [7, 1, 0, 2, 3, 4, 5, 6, 8, 9]
	return num_trans_rev[num]


def test( model, device, test_loader, epsilon ):

	# Accuracy counter
	correct = 0
	adv_examples = []
	count = 0

	# Loop over all examples in test set
	for data, target in test_loader:
		count += 1

		target_ori = target

		target = transform_label(target)

		# Send the data and label to the device
		data, target = data.to(device), target.to(device)

		# Set requires_grad attribute of tensor. Important for Attack
		data.requires_grad = True

		# Forward pass the data through the model
		output = model(data)
		init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

		# If the initial prediction is wrong, dont bother attacking, just move on
		if init_pred.item() != target.item():
			continue

		# Calculate the loss
		loss = F.nll_loss(output, target)

		# Zero all existing gradients
		model.zero_grad()

		# Calculate gradients of model in backward pass
		loss.backward()

		# Collect datagrad
		data_grad = data.grad.data

		# Call FGSM Attack
		perturbed_data = fgsm_attack(data, epsilon, data_grad)

		save_img(perturbed_data, target_ori.item(), count)

		# Re-classify the perturbed image
		output = model(perturbed_data)

		# Check for success
		final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
		if final_pred.item() == target.item():
			correct += 1
			# Special case for saving 0 epsilon examples
			if (epsilon == 0) and (len(adv_examples) < 5):
				adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
				adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
		else:
			# Save some adv examples for visualization later
			if len(adv_examples) < 5:
				adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
				adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

	# Calculate final accuracy for this epsilon
	final_acc = correct/float(len(test_loader))
	print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

	# Return the accuracy and an adversarial example
	return final_acc, adv_examples

def run():
	accuracies = []
	examples = []

	eps = 0.05
	acc, ex = test(model, device, test_loader, eps)
	accuracies.append(acc)
	examples.append(ex)
	'''
	# Run test for each epsilon
	for eps in epsilons:
		acc, ex = test(model, device, test_loader, eps)
		accuracies.append(acc)
		examples.append(ex)
	'''

if __name__ == '__main__':
	run()

