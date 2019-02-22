import os, shutil

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

main_dir = '/home/zw119/research/res_cifar/cifar/classified'
img_dir = '/home/zw119/research/res_cifar/cifar/test'

for clas in classes:
	dir = os.path.join(main_dir, clas)
	if not os.path.exists(dir):
		os.mkdir(dir)

def classify():
	target_path = ''
	for img in os.listdir(img_dir):
		current_path = os.path.join(img_dir, img)
		for clas in classes:
			if clas in img:
				target_path = os.path.join(os.path.join(main_dir, clas), img)
		if 'automobile' in img:
			target_path = os.path.join(os.path.join(main_dir, 'car'), img)
		if target_path == '':
			print(img)
			continue
		shutil.copy2(current_path, target_path)

if __name__ == '__main__':
	classify()
