import time
import yaml
import argparse
from Execute.astgcnExecute import astgcnExecute
from Evaluation.ASTGCN_eval import evalASTGCN as evalASTGCN

def main():
    time_start = time.time()
    print("Timer started for experimentation! :p ")
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='path to YAML config file')
    args = parser.parse_args()
    # Load the YAML config file which contains all the required settings for platform
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    ##############################  Training  ##################################
    if config['train_ast_gcn']['default']:
        print("************* Starting training process for the AST-GCN Model ************* ")
        trainer = astgcnExecute(config)
        trainer.train()
        print("*************  Finished training process for the AST-GCN Model ************* ")
    
    ##############################  Evaluation  ##################################     
    if config['eval_ast_gcn']['default']:
        print("************* Starting eval process for the AST-GCN Model ************* ")
        evalASTGCN(config)
        print("*************  Finished eval process for the AST-GCN Model ************* ")
        
    time_end = time.time()
    print("Time taken for the experimental pipeline :@ " ,time_end-time_start,'s')
if __name__ == '__main__':
    main()