import argparse
import pathlib

def get_config():

    parser = argparse.ArgumentParser(
        prog='VQVAE MAPPING',
        description='This program maps a chosen modality [image, video, pose, music] to text, utilizing the in-context learning of a LLM to compress sequences',
    )

    # GPU
    parser.add_argument("--device", default="cuda:0")

    # Experiment Settings
    parser.add_argument("--name", default="none", type=str)
    parser.add_argument("--algo", default="base", help="the algorithm used for training")
    
    parser.add_argument("--seq_len", default=256, type=int, help="context length the language model will input")
    parser.add_argument("--bits", default=8, type=int, help="How many bits will be grouped into a token, 8(default) bits is a byte")
    
    parser.add_argument("--exp_type", default="image", help="which modality is used. [Image, Music, Pose, Video]")
    parser.add_argument('--llm', choices=['gpt2', 'llama', 'gemma'], help='The language model to use')
    parser.add_argument("--train_val_split", default=0.8, type=float, help="What percentage of data should be used as training, default=0.8")
    parser.add_argument("--val", default=False, action='store_true')
    
    # Dataset
    parser.add_argument("--dataset_path", type=pathlib.Path)
    parser.add_argument('--modalities', nargs='+', help='List of Modalities you are training on', default=['.jpg'])

    # Reproductivity
    parser.add_argument("--use_seed", default=False, action="store_true")
    parser.add_argument("--seed", default=42, type=int)

    # HyperParameters
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--lr", default=5e-6, type=float)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--gamma", default=0.95, type=float, help="discount factor for REINFORCE")
    parser.add_argument("--decay", default=0.9, type=float, help="how learning rate decays for each epoch")
    parser.add_argument("--alpha", default=1.0, type=int, help="how many times I want to scale down RL Loss")
    parser.add_argument("--temperature", default=1.0, type=float, help="the lower the temperature, the more the logits looks like a one-hot. Range: (0, 1]")

    return parser