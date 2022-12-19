import json, yaml, argparse
from types import SimpleNamespace
from pathlib import Path
#
import pandas as pd, os, gc, numpy as np
#
#import pytorch_lightning as pl
import torch

pl.seed_everything(0)

    
def main(args):
    fam2label, word2id = prep_inputs(args.data_dir, args.mode)
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.epochs)
    dataloaders = dict()
    if args.mode=="train":
        train_dataset = SequenceDataset(word2id, fam2label, args.seq_max_len, 
                                        args.data_dir, "train")
        dev_dataset = SequenceDataset(word2id, fam2label, args.seq_max_len, 
                                      args.data_dir, "dev")
        dataloaders['train'] = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
        )
        dataloaders['dev'] = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
        )
        prot_cnn = ProtCNN(len(fam2label))
        trainer.fit(prot_cnn, dataloaders['train'], dataloaders['dev'])
        
    elif args.mode=="test":
        test_dataset = SequenceDataset(word2id, fam2label, args.seq_max_len, 
                                       args.data_dir, "test")
        dataloaders['test'] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
        )
        model = MyLightningModule.load_from_checkpoint(
            checkpoint_path=args.pytorch_checkpoint,
            hparams_file=args.hparams,
            map_location=None,
        )
        # test (pass in the model)
        trainer.validate(model=model, 
                         dataloaders=dataloaders['test'])

    
    
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    subparsers = ap.add_subparsers(help="train or test mode")
    subparsers.required = True
    # 
    ap.add_argument('-d','--data_dir', help='path to data', type=str,
                    required=True, default='./random_split')
    ap.add_argument('-max_seq_len','--seq_max_len',
                    help='max sequence length permitted', type=int, default=120)
    ap.add_argument('-bs','--batch_size', help='batch size', type=int, default=1)
    ap.add_argument('-nw','--num_workers', help='dataloader workers', 
                    type=int, default=0)
    ap.add_argument('--gpus', '-gpu', help='number of gpus used', type=int,
                    default= 1 if torch.cuda.is_available() else 0)
    #
    train_ap = subparsers.add_parser("train")
    train_ap.set_defaults(mode="train") 
    train_ap.add_argument('-e','--epochs', help='number of epochs', type=int,
                          default=100)
    #
    test_ap = subparsers.add_parser("test")
    test_ap.add_argument("--hparams", '-hp', required=True,
                         help='path to hyperparameters file')
    test_ap.add_argument("--chkpt", '-chkpt', required=True,
                         help='path to pytorch checkpoint')
    args = ap.parse_args()
    #print(f"Arguments:{args.__dict__}")
    # run main
    main(args)
    