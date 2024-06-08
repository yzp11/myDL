import argparse
import os

def load_configs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--dataset_train_dir', type=str,
                    default="./data/train/",
                    help='The directory containing the train data.')
    parser.add_argument('--dataset_val_dir', type=str,
                    default="./data/validation/",
                    help='The directory containing the val data.')
    parser.add_argument('--summary_dir', type=str, default="./summary/vit_base_patch16_224",
                    help='The directory of saving weights and tensorboard.')

    parser.add_argument('--gpu', type=str, default='0', help='Select gpu device.')
   
    parser.add_argument('--weights', type=str, default='', help='Pre-trained weights.')

    parser.add_argument('--model', type=str, default='ViT-B/16',
                    help='The name of ViT model, Select one to train.')
    parser.add_argument('--seed', type=int, default=2003,
                    help='Set the random seed.')
    parser.add_argument('--label_name', type=list, default=[
        "daisy",
        "dandelion",
        "roses",
        "sunflowers",
        "tulips"
        ], help='The name of class.')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    return args