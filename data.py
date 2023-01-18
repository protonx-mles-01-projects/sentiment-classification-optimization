<<<<<<< HEAD
=======
from datasets import load_dataset, ClassLabel

>>>>>>> 738ff3d (Fix model)
class Dataset:
    def __init__(self):
        pass

<<<<<<< HEAD
    def load_dataset(self):
        pass

    def build_dataset(self):
        pass
=======
    def build_dataset(self, *file_paths):
        dataset = load_dataset("csv", data_files=[file_path for file_path in file_paths])
        label = ClassLabel(num_classes = 2 ,names=["neg", "pos"])
        dataset = dataset.cast_column("label", label)
        dataset = dataset.shuffle()
        return dataset['train']

    '''def concat_dataset(self, *datasets):
        dataset = concatenate_datasets([dataset for dataset in datasets])
        return dataset.shuffle()'''

    def split_dataset(self, dataset, test_size=0.2):
        split_dataset = dataset.train_test_split(test_size=test_size, stratify_by_column='label')
        train_dataset, test_dataset = split_dataset['train'], split_dataset['test']
        return train_dataset, test_dataset
>>>>>>> 738ff3d (Fix model)
