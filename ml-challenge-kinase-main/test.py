from instadeep.utils import reader, build_labels, get_amino_acid_frequencies, build_vocab

data_dir = './random_split'
def prep_inputs(data_dir, partition="train"):
    train_data, train_targets = reader(partition, data_dir)
    fam2label = build_labels(train_targets)
    word2id = build_vocab(train_data)
    return fam2label, word2id
# main high-level test
class TestClass:
    def test_dataset(self):
        for part in ["train","dev","test"]:
            fam2label, word2id = prep_inputs(data_dir, partition=part)
            data, targets = reader(part, data_dir)
            fam2label = build_labels(targets)
            word2id = build_vocab(data)
            assert len(word2id)==22
            if part=="train":
                assert len(fam2label)==17930
            elif part=="dev" or  part=="test":
                assert len(fam2label)==13072         
            set(word2id).issubset(set(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '<pad>', '<unk>']))
            # check data length consistency
            assert len(data)==len(targets)

    def check_imports(self):
        try:
            from instadeep.dataset import SequenceDataset
            from instadeep.model import ProtCNN
            from instadeep.utils import reader, build_labels, get_amino_acid_frequencies, build_vocab
            import pytorch_lightning as pl
            import torch
        except Exception as e:
            assert False, f"e"
        