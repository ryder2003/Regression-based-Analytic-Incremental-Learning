import os
from .utils import DatasetBase, Datum

template = ['a photo of the number: "{}".']

print('preparing MNIST dataset')

class MNISTWrapper(DatasetBase):
    
    dataset_dir = 'mnist'
    
    def __init__(self, root, num_shots):
        from torchvision import datasets
        
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Create image directories
        train_img_dir = os.path.join(self.dataset_dir, 'images', 'train')
        test_img_dir = os.path.join(self.dataset_dir, 'images', 'test')
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(test_img_dir, exist_ok=True)
        
        self.template = template
        
        # Download/load MNIST
        train_dataset = datasets.MNIST(
            self.dataset_dir, train=True, download=True, transform=None
        )
        test_dataset = datasets.MNIST(
            self.dataset_dir, train=False, download=True, transform=None
        )
        
        # Convert to Datum format and save images
        train = []
        print(f"Processing MNIST training data...")
        for idx in range(min(len(train_dataset), 6000)):  # Limit for faster processing
            img, label = train_dataset[idx]
            img_path = os.path.join(train_img_dir, f'{idx:05d}_{label}.png')
            if not os.path.exists(img_path):
                img.save(img_path)
            
            datum = Datum(
                impath=img_path,
                label=int(label),
                classname=str(label)
            )
            train.append(datum)
        
        test = []
        print(f"Processing MNIST test data...")
        for idx in range(len(test_dataset)):
            img, label = test_dataset[idx]
            img_path = os.path.join(test_img_dir, f'{idx:05d}_{label}.png')
            if not os.path.exists(img_path):
                img.save(img_path)
            
            datum = Datum(
                impath=img_path,
                label=int(label),
                classname=str(label)
            )
            test.append(datum)
        
        # Generate few-shot dataset
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        
        super().__init__(train_x=train, val=test[:100], test=test)
        
        # Set classnames
        self.classnames = [str(i) for i in range(10)]
