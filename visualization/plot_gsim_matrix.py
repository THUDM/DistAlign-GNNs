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

import utils
from custom_models.mlp_cas import MLP_CAS

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10.0, 8.0)
figure_dir = None
log_write = None


def get_model(model_name, dataset, args):
    if model_name == "gcn":
        from custom_models.gcn import GCN

        model = GCN(in_feats=dataset.num_features,
                    out_feats=dataset.num_classes,
                    hidden_size=args.hidden_size,
                    num_layers=args.num_layers,
                    residual=args.residual,
                    dropout=args.dropout)
    elif model_name == "mlp":
        from custom_models.mlp import MLP

        model = MLP(in_feats=dataset.num_features,
                    hidden_size=args.hidden_size,
                    out_feats=dataset.num_classes,
                    num_layers=args.num_layers,
                    dropout=args.dropout)
    elif model_name == "sgc":
        from custom_models.sgc import SGC

        model = SGC(in_feats=dataset.num_features,
                    hidden_size=args.hidden_size,
                    out_feats=dataset.num_classes,
                    num_layers=args.num_layers,
                    order=args.k,
                    dropout=args.dropout)
    elif model_name == "gin":
        from custom_models.gin import GIN
        
        model = GIN(in_feats=dataset.num_features, 
                hidden_dim=256,
                out_feats=dataset.num_classes,
                num_layers=args.num_layers,
                num_mlp_layers=1,
                dropout=0.9)
    elif model_name == "mlp_cas":
        from custom_models.mlp_cas import MLP_CAS

        model = MLP_CAS(in_feats=dataset.num_features,
                    hidden_size=args.hidden_size,
                    out_feats=dataset.num_classes,
                    num_layers=args.num_layers,
                    correct_alpha=0.8,
                    smooth_alpha=0.8,
                    num_correct_prop=30,
                    num_smooth_prop=30,
                    correct_norm="row",
                    smooth_norm="sym",
                    dropout=args.dropout)
    elif model_name == "appnp":
        from custom_models.appnp import APPNP
        
        model = APPNP(nfeat=dataset.num_features, 
                      nhid=args.hidden_size, 
                      nclass=dataset.num_classes, 
                      num_layers=args.num_layers, 
                      dropout=args.dropout, 
                      alpha=0.2,
                      niter=10)
    return model


def plot_acc(dataset, models_dict, args):
    global log_write
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
    for k in models_dict.keys():
        train_accs, val_accs, test_accs = [], [], []
        train_confs, val_confs, test_confs = [], [], []
        train_entropys, val_entropys, test_entropys = [], [], []
        train_perps, val_perps, test_perps = [], [], []
        for seed in range(args.n_train):
            model = models_dict[k]['seed_{}'.format(seed)]['model']
            pred = model.embed(graph)
            if isinstance(model, MLP_CAS):
                pred = model.cas(graph, pred)
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
    plt.savefig(os.path.join(figure_dir, "perplexity_{}_{}.png".format(args.dataset_name, model_name)),
                bbox_inches='tight')
    # plt.show()


def plot_postaug(dataset, models_dict, args, loc, prop, n_post=50):
    global log_write
    remove_self_loop = False
    for k in models_dict.keys():
        model_name = k.upper()
        post_props = np.arange(n_post)
        train_accs_10, val_accs_10, test_accs_10 = [], [], []
        train_conf_means_10, train_conf_stds_10 = [], []
        val_conf_means_10, val_conf_stds_10 = [], []
        test_conf_means_10, test_conf_stds_10 = [], []
        train_entropy_means_10, train_entropy_stds_10 = [], []
        val_entropy_means_10, val_entropy_stds_10 = [], []
        test_entropy_means_10, test_entropy_stds_10 = [], []
        for seed in range(args.n_train):
            model = models_dict[k]['seed_{}'.format(seed)]['model']
            x_post = copy.deepcopy(dataset.data.x)
            graph = copy.deepcopy(dataset.data)
            graph_pred = copy.deepcopy(dataset.data)
            if args.remove_self_loop:
                graph.remove_self_loops()
                graph_pred.remove_self_loops()
            else:
                graph.add_remaining_self_loops()
                graph_pred.add_remaining_self_loops()
            graph.sym_norm()
            graph_pred.sym_norm()
            mid = None
            pred = None
            if loc == 'b':
                if prop:
                    mid = model.embed_layer(graph, 0)
                else:
                    mid = model.embed_without_prop_layer(graph, 0)
            elif loc == 'c':
                if prop:
                    pred = model.embed(graph)
                else:
                    pred = model.embed_without_prop(graph, args.num_layers)
                pred = torch.softmax(pred, dim=1)
            
            train_accs, val_accs, test_accs = [], [], []
            train_conf_means, train_conf_stds = [], []
            val_conf_means, val_conf_stds = [], []
            test_conf_means, test_conf_stds = [], []
            train_entropy_means, train_entropy_stds = [], []
            val_entropy_means, val_entropy_stds = [], []
            test_entropy_means, test_entropy_stds = [], []
            for n_post_prop in post_props:
                if loc == 'a':
                    graph = copy.deepcopy(dataset.data)
                    if remove_self_loop:
                        graph.remove_self_loops()
                    else:
                        graph.add_remaining_self_loops()
                    graph.sym_norm()
                    graph.x = x_post
                    if prop:
                        pred = model.embed(graph)
                    else:
                        pred = model.embed_without_prop(graph, args.num_layers)
                elif loc == 'b':
                    pred = copy.deepcopy(mid.detach())
                    for i in range(n_post_prop):
                        pred = spmm(graph, pred)
                    graph_pred.x = pred
                    if prop:
                        pred = model.embed_layer(graph_pred, 1)
                    else:
                        pred = model.embed_without_prop_layer(graph_pred, 1)
                # Get accuracy
                train_acc, val_acc, test_acc = utils.eval_acc(pred, dataset, verbose=False)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                test_accs.append(test_acc)
                # Get confidence
                train_confidence, val_confidence, test_confidence = utils.eval_confidence(pred, dataset, verbose=False)
                train_conf_means.append(torch.mean(train_confidence).item())
                val_conf_means.append(torch.mean(val_confidence).item())
                test_conf_means.append(torch.mean(test_confidence).item())
                train_conf_stds.append(torch.std(train_confidence).item())
                val_conf_stds.append(torch.std(val_confidence).item())
                test_conf_stds.append(torch.std(test_confidence).item())
                # Get entropy
                train_entropy, val_entropy, test_entropy = utils.eval_entropy(pred, dataset, verbose=False)
                train_entropy_means.append(torch.mean(train_entropy).item())
                val_entropy_means.append(torch.mean(val_entropy).item())
                test_entropy_means.append(torch.mean(test_entropy).item())
                train_entropy_stds.append(torch.std(train_entropy).item())
                val_entropy_stds.append(torch.std(val_entropy).item())
                test_entropy_stds.append(torch.std(test_entropy).item())
                if loc == 'a':
                    x_post = spmm(graph, x_post)
                elif loc == 'b' or loc == 'c':
                    pred = spmm(graph, pred)
            train_accs_10.append(train_accs)
            val_accs_10.append(val_accs)
            test_accs_10.append(test_accs)
            train_conf_means_10.append(train_conf_means)
            train_conf_stds_10.append(train_conf_stds)
            val_conf_means_10.append(val_conf_means)
            val_conf_stds_10.append(val_conf_stds)
            test_conf_means_10.append(test_conf_means)
            test_conf_stds_10.append(test_conf_stds)
            train_entropy_means_10.append(train_entropy_means)
            train_entropy_stds_10.append(train_entropy_stds)
            val_entropy_means_10.append(val_entropy_means)
            val_entropy_stds_10.append(val_entropy_stds)
            test_entropy_means_10.append(test_entropy_means)
            test_entropy_stds_10.append(test_entropy_stds)

        # Accuracy
        train_acc_mean, train_acc_std = torch.tensor(train_accs_10).mean(dim=0).numpy(), torch.tensor(train_accs_10).std(dim=0).numpy()
        val_acc_mean, val_acc_std = torch.tensor(val_accs_10).mean(dim=0).numpy(), torch.tensor(val_accs_10).std(dim=0).numpy()
        test_acc_mean, test_acc_std = torch.tensor(test_accs_10).mean(dim=0).numpy(), torch.tensor(test_accs_10).std(dim=0).numpy()
        acc_max = np.argmax(val_acc_mean)
        log_write += "Augumentation in {} for model {} propagation:\nacc max: {}, max acc mean: {}, std: {}\n".format(loc, "with" if prop else "without", acc_max, test_acc_mean[acc_max], test_acc_std[acc_max])

        fig, axs = plt.subplots(1, 3, figsize=(36, 10))
        axs[0].plot(post_props, train_acc_mean, label="Train")
        axs[0].fill_between(post_props, train_acc_mean - train_acc_std,
                            train_acc_mean + train_acc_std, alpha=0.2)
        axs[0].plot(post_props, val_acc_mean, label="Val")
        axs[0].fill_between(post_props, val_acc_mean - val_acc_std,
                            val_acc_mean + val_acc_std, alpha=0.2)
        axs[0].plot(post_props, test_acc_mean, label="Test")
        axs[0].fill_between(post_props, test_acc_mean - test_acc_std,
                            test_acc_mean + test_acc_std, alpha=0.2)
        axs[0].set_title("{} Accuracy ({})".format(model_name, args.dataset_name), fontsize=32, fontweight='bold')
        axs[0].set_ylim(0, 1.05)
        axs[0].set_ylabel("Accuracy", fontsize=32, fontweight='bold')
        axs[0].set_xlabel("Post Augmentation Step (Loc. {})".format(loc.upper()), fontsize=32, fontweight='bold')
        axs[0].tick_params(axis="x", labelsize=32)
        axs[0].tick_params(axis="y", labelsize=32)
        axs[0].legend(loc=4, fontsize=32)

        # Confidence
        train_conf_mean, train_conf_std = torch.tensor(train_conf_means_10).mean(dim=0).numpy(), torch.tensor(
            train_conf_stds_10).mean(dim=0).numpy()
        val_conf_mean, val_conf_std = torch.tensor(val_conf_means_10).mean(dim=0).numpy(), torch.tensor(
            val_conf_stds_10).mean(dim=0).numpy()
        test_conf_mean, test_conf_std = torch.tensor(test_conf_means_10).mean(dim=0).numpy(), torch.tensor(
            test_conf_stds_10).mean(dim=0).numpy()

        axs[1].plot(post_props, train_conf_mean, label="Train")
        axs[1].fill_between(post_props, train_conf_mean - train_conf_std,
                            train_conf_mean + train_conf_std, alpha=0.2)
        axs[1].plot(post_props, val_conf_mean, label="Val")
        axs[1].fill_between(post_props, val_conf_mean - val_conf_std,
                            val_conf_mean + val_conf_std, alpha=0.2)
        axs[1].plot(post_props, test_conf_mean, label="Test")
        axs[1].fill_between(post_props, test_conf_mean - test_conf_std,
                            test_conf_mean + test_conf_std, alpha=0.2)
        axs[1].set_title("{} Confidence ({})".format(model_name, args.dataset_name), fontsize=32, fontweight='bold')
        axs[1].set_ylim(0, 1.05)
        axs[1].set_ylabel("Confidence", fontsize=32, fontweight='bold')
        axs[1].set_xlabel("Post Augmentation Step (Loc. {})".format(loc.upper()), fontsize=32, fontweight='bold')
        axs[1].tick_params(axis="x", labelsize=32)
        axs[1].tick_params(axis="y", labelsize=32)
        axs[1].legend(loc=1, fontsize=32)

        # Entropy
        train_entropy_mean, train_entropy_std = torch.tensor(train_entropy_means_10).mean(dim=0).numpy(), torch.tensor(
            train_entropy_stds_10).mean(dim=0).numpy()
        val_entropy_mean, val_entropy_std = torch.tensor(val_entropy_means_10).mean(dim=0).numpy(), torch.tensor(
            val_entropy_stds_10).mean(dim=0).numpy()
        test_entropy_mean, test_entropy_std = torch.tensor(test_entropy_means_10).mean(dim=0).numpy(), torch.tensor(
            test_entropy_stds_10).mean(dim=0).numpy()

        axs[2].plot(post_props, train_entropy_mean, label="Train")
        axs[2].fill_between(post_props, train_entropy_mean - train_entropy_std,
                            train_entropy_mean + train_entropy_std, alpha=0.2)
        axs[2].plot(post_props, val_entropy_mean, label="Val")
        axs[2].fill_between(post_props, val_entropy_mean - val_entropy_std,
                            val_entropy_mean + val_entropy_std, alpha=0.2)
        axs[2].plot(post_props, test_entropy_mean, label="Test")
        axs[2].fill_between(post_props, test_entropy_mean - test_entropy_std,
                            test_entropy_mean + test_entropy_std, alpha=0.2)
        axs[2].set_title("{} Entropy ({})".format(model_name, args.dataset_name), fontsize=32, fontweight='bold')
        axs[2].set_ylabel("Entropy", fontsize=32, fontweight='bold')
        axs[2].set_xlabel("Post Augmentation Step (Loc. {})".format(loc.upper()), fontsize=32, fontweight='bold')
        axs[2].tick_params(axis="x", labelsize=32)
        axs[2].tick_params(axis="y", labelsize=32)
        axs[2].legend(loc=4, fontsize=32)

        plt.savefig(os.path.join(figure_dir, "post_aug_{}_{}_{}_prop_loc_{}.png".format(args.dataset_name, model_name, "with" if prop else "without", loc.upper())), bbox_inches='tight')
        # plt.show()

        n_post_optimal = np.argmax(val_acc_mean)
        print("N post optimal: {}.".format(n_post_optimal))
        print("ACC: Val {:.4f}+-{:.4f} | Test {:.4f}+-{:.4f}".format(val_acc_mean[n_post_optimal], val_acc_std[n_post_optimal], test_acc_mean[n_post_optimal], test_acc_std[n_post_optimal]))
        log_write += "N post optimal: {}\n".format(n_post_optimal)
        log_write += "ACC: Val {:.4f}+-{:.4f} | Test {:.4f}+-{:.4f}\n\n".format(val_acc_mean[n_post_optimal], val_acc_std[n_post_optimal], test_acc_mean[n_post_optimal], test_acc_std[n_post_optimal])
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("--exp_name", type=str, default="exp_0118")
    # Dataset settings
    parser.add_argument("--dataset_name", type=str, default="cora")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--remove_self_loop", action="store_true")
    # Model settings
    parser.add_argument("--model_name", type=str, default="gcn")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--residual", action="store_true")
    # Training settings
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--n_train", type=int, default=10)
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
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    save_dir = os.path.join(args.save_dir, args.dataset_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_dir = "./save_models/{}/{}".format(args.model_name, args.dataset_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Load dataset
    print("loading dataset {} from {}".format(args.dataset_name, args.data_dir))
    dataset = build_dataset_from_path(data_path=args.data_dir, dataset=args.dataset_name)
    train_nid = dataset.data.train_nid
    val_nid = dataset.data.val_nid
    test_nid = dataset.data.test_nid

    # Load model
    models_dict = {args.model_name: {}}
    for seed in range(args.n_train):
        set_random_seed(seed)
        model_name = "model_{}{}{}_{}_{}_seed_{}".format(args.model_name, "_res_" if args.residual else "_", args.hidden_size, args.num_layers, args.dropout, seed)
        checkpoint_path = os.path.join(checkpoint_dir, model_name)

        model = get_model(args.model_name, dataset, args)
        models_dict[args.model_name]['seed_{}'.format(seed)] = {'name' : model_name, 'model': model, 'path' : checkpoint_path}
    figure_dir = "./save_figures/{}/{}/{}{}_{}_{}".format(args.model_name, args.dataset_name, "res_" if args.residual else "", args.hidden_size, args.num_layers, args.dropout)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    log_dir = "./log/{}/{}".format(args.model_name, args.dataset_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_write = ""
        
    # Training
    if args.training:
        for k in models_dict.keys():
            for seed in range(args.n_train):
                set_random_seed(seed)
                experiment(model=models_dict[k]['seed_{}'.format(seed)]['model'], dataset=dataset, checkpoint_path=models_dict[k]['seed_{}'.format(seed)]['path'], seed=[seed])

    # Evaluation
    for k in models_dict.keys():
        for seed in range(args.n_train):
            model_wrapper = NodeClfModelWrapper(models_dict[k]['seed_{}'.format(seed)]['model'], None)
            model_wrapper.load_state_dict(torch.load(models_dict[k]['seed_{}'.format(seed)]['path']))
            model = model_wrapper.model
            model.eval()
            models_dict[k]['seed_{}'.format(seed)]['model'] = model

    # Accuracy, confidence, entropy, perplexity
    plot_acc(dataset, models_dict, args)

    # Post Augmentation Test
    for prop in [True, False]:
        if args.model_name == 'gamlp':
            plot_postaug(dataset, models_dict, args, n_post=args.n_post, loc='c', prop=prop)
        else:
            for loc in ['a', 'b', 'c']:
                plot_postaug(dataset, models_dict, args, n_post=args.n_post, loc=loc, prop=prop)
    log = open(os.path.join(log_dir, "{}{}_{}_{}.txt".format("res_" if args.residual else "", args.hidden_size, args.num_layers, args.dropout)), 'w')
    log.write(log_write)
    log.close()