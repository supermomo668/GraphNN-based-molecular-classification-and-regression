import json, yaml, argparse, pandas as pd
from types import SimpleNamespace
from pathlib import Path
#
from postera_gnn.run import train, evaluate, KFoldCV, infer
from postera_gnn.run import json_to_list, list_to_json 
from postera_gnn.model.models import *
from postera_gnn.settings import (FeaturesArgs, TrainArgs, ModelArgs, DataArgs, InferArgs)

def main(args):
  # controllable variables
  DataArgs.num_workers = args.num_workers
  TrainArgs.batch_size = args.batch_size
  TrainArgs.model = args.model
  
  if args.mode=="train":
    TrainArgs.epochs =  args.epochs
    TrainArgs.model_save_pt.parent.mkdir(exist_ok=True, parents=True)
    df_train = pd.read_csv(args.data_path)
    model, train_hist = train(df_train, DataArgs, TrainArgs, args.model_path)
    return train_hist
  elif args.mode=="evaluate" or args.mode=="infer":
    InferArgs.output_path.mkdir(exist_ok=True, parents=True)
    # select model (GIN)
    if ModelArgs.model=="GIN":
      model = GIN
    elif ModelArgs.model not in ModelArgs.available_models:
      raise Exception(
        f"Model not one of:{ModelArgs.available_models}")
    gnn_model = model(FeaturesArgs.n_node_features, 1)
    gnn_model.load_state_dict(torch.load(args.model_path))
    gnn_model.eval()
    # load data
    df_eval = pd.read_csv(args.data_path)
    if args.mode=="evaluate":
      if args.eval_mode=="cross":
        cv_perf = KFoldCV(df_eval, DataArgs, TrainArgs, k_fold=args.kfold, 
                          target_test_size=0.2)
        list_to_json(InferArgs.output_path/"cross_eval.json", cv_perf)
        return cv_perf
      else:
        eval_perf = evaluate(gnn_model, df_eval)
        list_to_json(InferArgs.output_path/"single_eval.json", eval_perf)
        return eval_perf
    elif args.mode=="infer":  
      InferArgs.model_save_pt.parent.mkdir(exist_ok=True, parents=True)
      input_smiles = json_to_list(args.input_json)  
      prediction = infer(input_smiles, gnn_model)
      list_to_json(args.pred_json, prediction)
      return prediction
  else:
    raise Exception(f"Mode not supported:{args.mode}")
    
    
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    subparsers = ap.add_subparsers(help="train or test mode")
    subparsers.required = True
    # 
    ap.add_argument('-d','--data_path', help='path to data', type=str,
                    required=True, default=DataArgs.data_source)
    ap.add_argument('-m', '--model', default=ModelArgs.model, help='model to be used')
    ap.add_argument('-bs','--batch_size', help='batch size', type=int, 
                    default=TrainArgs.batch_size)
    ap.add_argument('-nw','--num_workers', help='dataloader workers', 
                    type=int, default=DataArgs.num_workers)
    ap.add_argument("--model_path",'-pth', default=TrainArgs.model_save_pt,
                    help='path to model file for load/save(*.pth)')
    
    # train arguments
    train_ap = subparsers.add_parser(name="train")
    train_ap.set_defaults(mode="train")
    train_ap.add_argument('-e','--epochs', help='number of epochs', type=int,
                          default=25)
    # eval arguments
    eval_ap = subparsers.add_parser(name="evaluate")
    eval_ap.set_defaults(mode="evaluate")
    eval_ap.add_argument("--eval_mode", '-eval_mode', default='single',
                         choices=['single','cross'],
                         help='evaluation mode')
    eval_ap.add_argument("--kfold", '-kf', type=int, default=5,
                         help='k for k-fold cross validation')
    eval_ap.add_argument("--eval_path", '-epth', default=InferArgs.output_path,
                         help='output path of evaluations') 
    # infer arguments
    infer_ap = subparsers.add_parser(name="infer")
    infer_ap.set_defaults(mode="infer")
    infer_ap.add_argument("--input_json", '-i', required=True, 
                          default=DataArgs.data_source.parent/'test_input.json',
                          help='input path to json file of SMILES') 
    infer_ap.add_argument("--pred_json", '-p', 
                          default=InferArgs.output_path/'prediction.json',
                          help='output path of predictions') 
    args = ap.parse_args()
    print(f"All used/unused Arguments :{args.__dict__}")
    # run main
    main(args)
    