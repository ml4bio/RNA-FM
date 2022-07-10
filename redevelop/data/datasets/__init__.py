# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

from .custom import Custom

def build_dataset(dataset_names, set_name, root_paths, transforms):
    """
    :param dataset_name: the name of dataset
    :param root_path: data is usually located under the root path
    :param set_name: "train", "valid", "test"
    :param transforms:
    :return:
    """
    if isinstance(dataset_names, str) and isinstance(root_paths, str):
        dataset_names = [dataset_names]
        root_paths = [root_paths]
        num_dataset = 1
    elif len(dataset_names) == len(root_paths):
        num_dataset = len(dataset_names)
    else:
        raise Exception("Wrong Dataset Type!")

    datasets = {}
    for dataset_name, root_path in zip(dataset_names, root_paths):
        if "cath-decoy" in dataset_name:
            dataset = CATH_DECOY.create_dataset_with_name(dataset_name, set_name, root_path, transforms)
        elif "cath" in dataset_name:
            dataset = CATH.create_dataset_with_name(dataset_name, set_name, root_path, transforms)
        elif "spot-rna-bprna" in dataset_name:
            dataset = SPOT_RNA_bpRNA.create_dataset_with_name(dataset_name, set_name, root_path, transforms)
        elif "spot-rna-pdb" in dataset_name:
            dataset = SPOT_RNA_PDB.create_dataset_with_name(dataset_name, set_name, root_path, transforms)
        elif "toehold-switch" in dataset_name:
            dataset = Toehold_Switch.create_dataset_with_name(dataset_name, set_name, root_path, transforms)
        elif "human-5-utr-vary-length" in dataset_name:
            dataset = Human_5Prime_UTR_VarLength.create_dataset_with_name(dataset_name, set_name, root_path, transforms)
        elif "human-5-utr" in dataset_name:
            dataset = Human_5Prime_UTR.create_dataset_with_name(dataset_name, set_name, root_path, transforms)
        elif "rna-contact" in dataset_name:
            dataset = RNA_CONTACT.create_dataset_with_name(dataset_name, set_name, root_path, transforms)
        elif "archive2-ufold" in dataset_name:
            dataset = Archive2_UFold393.create_dataset_with_name(dataset_name, set_name, root_path, transforms)
        elif "archive2-all" in dataset_name:
            dataset = Archive2_All600.create_dataset_with_name(dataset_name, set_name, root_path, transforms)
        elif "archive2" in dataset_name:
            dataset = Archive2.create_dataset_with_name(dataset_name, set_name, root_path, transforms)
        elif "rnastralign" in dataset_name:
            dataset = RNAStralign.create_dataset_with_name(dataset_name, set_name, root_path, transforms)
        elif "custom" in dataset_name:
            dataset = Custom.create_dataset_with_name(dataset_name, set_name, root_path, transforms)
        elif "memory-test" in dataset_name:
            dataset = MemoryTest.create_dataset_with_name(dataset_name, set_name, root_path, transforms)
        else:
            raise Exception("Can not build unknown image dataset: {}".format(dataset_name))
        datasets[dataset_name] = dataset

    if num_dataset > 1:
        dataset = Combination(datasets, set_name)

    return dataset