import argparse
import copy
import os

import cogdl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from cogdl import experiment
from cogdl.datasets import build_dataset_from_path
from cogdl.utils import set_random_seed
from cogdl.utils import spmm
from cogdl.wrappers.model_wrapper.node_classification import NodeClfModelWrapper
from custom_models.gat_mlp import GATX

import utils

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10.0, 8.0)
figure_dir = None
log_write = None


def get_model(model_name, dataset, args):
    model = GATX(in_feats=dataset.num_features,
                    hidden_size=args.hidden_size,
                    out_feats=dataset.num_classes,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    attn_drop=args.dropout,
                    alpha=0.2,
                    nhead=8,
                    residual=False,
                    last_nhead=1,
                    gat_first=args.gat_first,
                    activation="relu",
                    norm=None,
                    act_first=False,
                    bias=True)
    return model


def plot_acc(dataset, models_dict, args):
    global log_write
    dataset.data = dataset.data.to(args.device)
    graph = copy.deepcopy(dataset.data)
    if args.remove_self_loop:
        graph.remove_self_loops()
    else:
        graph.add_remaining_self_loops()
    graph.sym_norm()
    plot_labels = []
    train_accs_10, val_accs_10, test_accs_10 = [], [], []
    train_confs_10, val_confs_10, test_confs_10 = [], [], []
    train_entropys_10, val_entropys_10, test_entropys_10 = [], [], []
    train_perps_10, val_perps_10, test_perps_10 = [], [], []
    verbose = False
    ret = None
    for k in models_dict.keys():
        train_accs, val_accs, test_accs = [], [], []
        train_confs, val_confs, test_confs = [], [], []
        train_entropys, val_entropys, test_entropys = [], [], []
        train_perps, val_perps, test_perps = [], [], []
        for seed in range(args.n_train):
            model = models_dict[k]['seed_{}'.format(seed)]['model']
            pred = model.embed(graph)
            if verbose:
                print("*" * 20, models_dict[k]['seed_{}'.format(seed)]['name'], "*" * 20)
            train_acc, val_acc, test_acc = utils.eval_acc(pred, dataset, verbose)
            plot_labels.append(models_dict[k]['seed_{}'.format(seed)]['name'])
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
            train_conf, val_conf, test_conf = utils.eval_confidence(pred, dataset, verbose)
            train_confs.append(train_conf)
            val_confs.append(val_conf)
            test_confs.append(test_conf)
            train_entropy, val_entropy, test_entropy = utils.eval_entropy(pred, dataset, verbose)
            train_entropys.append(train_entropy)
            val_entropys.append(val_entropy)
            test_entropys.append(test_entropy)
            train_perp, val_perp, test_perp = utils.eval_perplexity(pred, dataset, verbose)
            train_perps.append(train_perp)
            val_perps.append(val_perp)
            test_perps.append(test_perp)
        ret = test_accs
        # Acc
        train_accs_10.append([torch.mean(torch.tensor(train_accs)).item(), torch.std(torch.tensor(train_accs)).item()])
        val_accs_10.append([torch.mean(torch.tensor(val_accs)).item(), torch.std(torch.tensor(val_accs)).item()])
        test_accs_10.append([torch.mean(torch.tensor(test_accs)).item(), torch.std(torch.tensor(test_accs)).item()])
        # Confidence
        train_confs_10.append([torch.mean(torch.cat(train_confs)).item(), torch.std(torch.cat(train_confs)).item()])
        val_confs_10.append([torch.mean(torch.cat(val_confs)).item(), torch.std(torch.cat(val_confs)).item()])
        test_confs_10.append([torch.mean(torch.cat(test_confs)).item(), torch.std(torch.cat(test_confs)).item()])
        # Entropy
        train_entropys_10.append(
            [torch.mean(torch.cat(train_entropys)).item(), torch.std(torch.cat(train_entropys)).item()])
        val_entropys_10.append([torch.mean(torch.cat(val_entropys)).item(), torch.std(torch.cat(val_entropys)).item()])
        test_entropys_10.append(
            [torch.mean(torch.cat(test_entropys)).item(), torch.std(torch.cat(test_entropys)).item()])
        # Perplexity
        train_perps_10.append(
            [torch.mean(torch.tensor(train_perps)).item(), torch.std(torch.tensor(train_perps)).item()])
        val_perps_10.append([torch.mean(torch.tensor(val_perps)).item(), torch.std(torch.tensor(val_perps)).item()])
        test_perps_10.append([torch.mean(torch.tensor(test_perps)).item(), torch.std(torch.tensor(test_perps)).item()])
    train_accs = [acc[0] for acc in train_accs_10]
    val_accs = [acc[0] for acc in val_accs_10]
    test_accs = [acc[0] for acc in test_accs_10]
    max_train_acc = np.argmax(np.array(train_accs))
    max_val_acc = np.argmax(np.array(val_accs))
    max_test_acc = np.argmax(np.array(test_accs))
    print("After training: ACC: Train {:.4f}+-{:.4f} | Val {:.4f}+-{:.4f} | Test {:.4f}+-{:.4f}".format(train_accs_10[max_train_acc][0], train_accs_10[max_train_acc][1], val_accs_10[max_val_acc][0], val_accs_10[max_val_acc][1], test_accs_10[max_test_acc][0], test_accs_10[max_test_acc][1]))
    
    log_write += "After training: ACC: Train {:.4f}+-{:.4f} | Val {:.4f}+-{:.4f} | Test {:.4f}+-{:.4f}\n\n".format(train_accs_10[max_train_acc][0], train_accs_10[max_train_acc][1], val_accs_10[max_val_acc][0], val_accs_10[max_val_acc][1], test_accs_10[max_test_acc][0], test_accs_10[max_test_acc][1])

    bar_labels = [args.model_name.upper()]
    n_models = len(bar_labels)
    x = np.arange(3)
    width = 0.25
    for i in range(n_models):
        plt.bar(x + (i - int(n_models / 2)) * width, [train_accs_10[i][0], val_accs_10[i][0], test_accs_10[i][0]],
                width, yerr=[train_accs_10[i][1], val_accs_10[i][1], test_accs_10[i][1]], label=bar_labels[i])
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy', fontsize=32)
    plt.yticks(fontsize=28)
    plt.xticks(ticks=x, labels=["Train", "Val", "Test"], fontsize=32)
    plt.title('Accuracy Comparison ({})'.format(args.dataset_name), fontsize=32)
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(figure_dir, "acc_{}_{}.png".format(args.dataset_name, model_name)), bbox_inches='tight')
    # plt.show()

    for i in range(n_models):
        y = [train_confs_10[i][0], val_confs_10[i][0], test_confs_10[i][0]]
        yerr = [train_confs_10[i][1], val_confs_10[i][1], test_confs_10[i][1]]
        plt.bar(x + (i - int(n_models / 2)) * width, y, width, yerr=yerr, label=bar_labels[i])
    plt.ylim(0, 1.1)
    plt.ylabel('Confidence', fontsize=32)
    plt.yticks(fontsize=28)
    plt.xticks(ticks=x, labels=["Train", "Val", "Test"], fontsize=32)
    plt.title('Confidence Comparison ({})'.format(args.dataset_name), fontsize=32)
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(figure_dir, "confidence_{}_{}.png".format(args.dataset_name, model_name)),
                bbox_inches='tight')
    # plt.show()

    for i in range(n_models):
        y = [train_entropys_10[i][0], val_entropys_10[i][0], test_entropys_10[i][0]]
        yerr = [train_entropys_10[i][1], val_entropys_10[i][1], test_entropys_10[i][1]]
        plt.bar(x + (i - int(n_models / 2)) * width, y, width, yerr=yerr, label=bar_labels[i])
    plt.ylabel('Entropy', fontsize=32)
    plt.yticks(fontsize=28)
    plt.xticks(ticks=x, labels=["Train", "Val", "Test"], fontsize=32)
    plt.title('Entropy Comparison({})'.format(args.dataset_name), fontsize=32)
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(figure_dir, "entropy_{}_{}.png".format(args.dataset_name, model_name)),
                bbox_inches='tight')
    # plt.show()

    for i in range(n_models):
        plt.bar(x + (i - int(n_models / 2)) * width, [train_perps_10[i][0], val_perps_10[i][0], test_perps_10[i][0]],
                width,
                yerr=[train_perps_10[i][1], val_perps_10[i][1], test_perps_10[i][1]], label=bar_labels[i])
    plt.ylabel('Perplexity', fontsize=32)
    plt.yticks(fontsize=28)
    plt.xticks(ticks=x, labels=["Train", "Val", "Test"], fontsize=32)
    plt.title('Perplexity Comparison ({})'.format(args.dataset_name), fontsize=32)
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(figure_dir, "perplexity_{}_{}.png".format(args.dataset_name, model_name)), bbox_inches='tight')
    # plt.show()
    return ret
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("--exp_name", type=str, default="exp_0118")
    # Dataset settings
    parser.add_argument("--dataset_name", type=str, default="cora")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--remove_self_loop", action="store_true")
    # Model settings
    parser.add_argument("--model_name", type=str, default="gatx")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--gat_first", action="store_true")
    # Training settings
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--n_epoch", type=int, default=500, help="Training epoch.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--train_mode", type=str, default="transductive")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=50)
    parser.add_argument("--lr_scheduler", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--training", action="store_true")
    # Post augmentation settings
    parser.add_argument("--n_post", type=int, default=50)

    args = parser.parse_args()

    if args.gpu >= 0:
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"
    
    save_dir = os.path.join(args.save_dir, args.dataset_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_dir = "./save_models/{}/{}".format(args.model_name, args.dataset_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Load dataset
    print("loading dataset {} from {} for {}".format(args.dataset_name, args.data_dir, args.model_name))
    dataset = build_dataset_from_path(data_path=args.data_dir, dataset=args.dataset_name)
    train_nid = dataset.data.train_nid
    val_nid = dataset.data.val_nid
    test_nid = dataset.data.test_nid

    # Load model
    models_dict_1 = {args.model_name: {}}
    models_dict_2 = {args.model_name: {}}
    for seed in range(args.n_train):
        set_random_seed(seed)
        model_name = "model_{}_gat_mlp{}{}_{}_{}_seed_{}".format(args.model_name,  "_res_" if args.residual else "_", args.hidden_size, args.num_layers, args.dropout, seed)
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        args.gat_first = True
        model = get_model(args.model_name, dataset, args)
        models_dict_1[args.model_name]['seed_{}'.format(seed)] = {'name' : model_name, 'model': model, 'path' : checkpoint_path}
        
        set_random_seed(seed)
        model_name = "model_{}_mlp_gat{}{}_{}_{}_seed_{}".format(args.model_name,  "_res_" if args.residual else "_", args.hidden_size, args.num_layers, args.dropout, seed)
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        args.gat_first = False
        model = get_model(args.model_name, dataset, args)
        models_dict_2[args.model_name]['seed_{}'.format(seed)] = {'name' : model_name, 'model': model, 'path' : checkpoint_path}
    figure_dir = "./save_figures_2layer/{}/{}/{}{}_{}_{}".format(args.model_name, args.dataset_name, "res_" if args.residual else "", args.hidden_size, args.num_layers, args.dropout)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    log_dir = "./log_2layer/{}/{}".format(args.model_name, args.dataset_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_write = ""
        
    # Training
    if args.training:
        for k in models_dict_1.keys():
            for seed in range(args.n_train):
                set_random_seed(seed)
                experiment(model=models_dict_1[k]['seed_{}'.format(seed)]['model'], dataset=dataset,
                           checkpoint_path=models_dict_1[k]['seed_{}'.format(seed)]['path'], seed=[seed],
                           device=[args.gpu])
        for k in models_dict_2.keys():
            for seed in range(args.n_train):
                set_random_seed(seed)
                experiment(model=models_dict_2[k]['seed_{}'.format(seed)]['model'], dataset=dataset,
                           checkpoint_path=models_dict_2[k]['seed_{}'.format(seed)]['path'], seed=[seed],
                           device=[args.gpu])
    
    acc_dict = {}
    # Evaluation
    for k in models_dict_1.keys():
        for seed in range(args.n_train):
            model_wrapper = NodeClfModelWrapper(models_dict_1[k]['seed_{}'.format(seed)]['model'], None)
            model_wrapper.load_state_dict(torch.load(models_dict_1[k]['seed_{}'.format(seed)]['path']))
            model = model_wrapper.model
            model = model.to(args.device)
            model.eval()
            models_dict_1[k]['seed_{}'.format(seed)]['model'] = model
    for k in models_dict_2.keys():
        for seed in range(args.n_train):
            model_wrapper = NodeClfModelWrapper(models_dict_2[k]['seed_{}'.format(seed)]['model'], None)
            model_wrapper.load_state_dict(torch.load(models_dict_2[k]['seed_{}'.format(seed)]['path']))
            model = model_wrapper.model
            model = model.to(args.device)
            model.eval()
            models_dict_2[k]['seed_{}'.format(seed)]['model'] = model
    
    # Accuracy, confidence, entropy, perplexity
    acc_dict["GAT+MLP"] = plot_acc(dataset, models_dict_1, args)
    acc_dict["MLP+GAT"] = plot_acc(dataset, models_dict_2, args)
    
    data = list(acc_dict.values())
    print(data)
    fix, ax = plt.subplots()
    ax.boxplot(data)
    plt.yticks(fontsize=30, fontweight='bold')
    plt.xticks(ticks=np.arange(len(acc_dict.keys())) + 1, labels=list(acc_dict.keys()), fontsize=36, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=36, fontweight='bold')
    plt.title('{}, {}'.format("GAT Accuracy", args.dataset_name), fontsize=42, fontweight='bold')
    plt.savefig(os.path.join(figure_dir, "boxplot_{}_{}.png".format(args.dataset_name, args.model_name)), bbox_inches='tight')

    log = open(os.path.join(log_dir, "{}{}_{}_{}.txt".format("res_" if args.residual else "", args.hidden_size, args.num_layers, args.dropout)), 'w')
    log.write(log_write)
    log.close()