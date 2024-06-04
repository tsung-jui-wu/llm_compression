import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from Mapper.mapper import TokenMapper
from process_data import BinaryDataset
from llm import GPT2Model, LlamaModel, GemmaModel

def main(args):
    device = args.device
    batch_size = args.batch
    
    print("===============================================================")
    print(f"       MAPPING FROM ({args.exp_type}) TO {args.llm}")
    print("===============================================================")
    
    print(args.name)
    
    if args.name != "none":
        experiment_name = f"{args.exp_type}/{args.algo}/{args.name}/model={args.llm}"
    else:
        experiment_name = f"{args.exp_type}/{args.algo}/model={args.llm}"
    
    writer = SummaryWriter(log_dir = f'runs/{experiment_name}')
    
    match args.llm:
        case "gpt2" | "gpt2-xl":
            llm = GPT2Model(args)
        case "gemma":
            llm = GemmaModel(args)
        case "llama":
            llm = LlamaModel(args)
    
    
    input_dim = args.bits**2
    output_dim = llm.vocab_size
    mapper = TokenMapper(args, input_dim, output_dim)
    
    dataset = BinaryDataset(args)
    train_size = int(len(dataset)*args.train_val_split)
    val_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])
    
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    print(f"Train Dataset Length: {len(trainloader)}")
    print(f"Validation Dataset Length: {len(valloader)}")
    print("finish loading dataset\n")
    
    if args.algo == 'rl':
        print("USING ALGORITHM: REINFORCE\n")
    elif args.algo == 'base':
        print("USING ALGORITHM: SUPERVISED\n")
    
    print(f"training {args.epochs} epoch(s) with learning rate={args.lr}\n")
    
    optimizer = optim.Adam(mapper.model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay)
    ce_criterion = nn.CrossEntropyLoss()
    rl_criterion = nn.CrossEntropyLoss(reduction='none')
            
    writer.add_hparams(
        {
            "llm_model": args.llm,
            "modality": args.exp_type,
            "learning_rate": args.lr,
        },
        {}
    )

    global_step = 0
    for epoch in range(args.epochs):
        mapper.model.train()
        for i, (data, _) in enumerate(trainloader):
            
            optimizer.zero_grad()
            
            ground_truth_tokens = data.reshape(-1, seq_len).to(device)
            one_hot_tokens = F.one_hot(ground_truth_tokens, num_classes=bits_vocab_len).float()

            # Logits are to be compared with the next ground truth tokens
            ground_truth_tokens = ground_truth_tokens[:,1:]
            inputs_feature_vector = mapper.model(one_hot_tokens)

            # Map tokens and get ground truth from LLM
            translated_feature_vector, translated_logits, translated_text_tokens = llm.translate(
                inputs_feature_vector,
                llm.embeddings.detach(),
                temperature=args.temperature
            )
           
            # Calculate Representation of Last Layer in LLM
            final_layer_fv = llm.generate_next_token_predictions_withfv(translated_feature_vector)

            # Calculate Logits with mapper function
            logits = torch.matmul(final_layer_fv, mapper.model.weight)
            logits = logits[:,:-1]
            logits_ = logits.reshape(-1, bits_vocab_len)
            ground_truth_tokens = ground_truth_tokens.reshape(-1)        
            ce_loss = criterion(logits_, ground_truth_tokens)

            writer.add_scalar("training/cross_entropy", ce_loss.item(), global_step)
            ce_loss.backward()
            optimizer.step()
            if global_step%100==0:
                print(f"Epoch {epoch+1}, Batch {global_step}, CE Loss: {ce_loss.mean().item()}")
            global_step+=1
            # torch.cuda.empty_cache()
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} completed.")
    writer.close()


if __name__=="__main__":
    parser = get_config()
    args = parser.parse_args()
    
    if args.use_seed:
        torch.manual_seed(args.seed)
        
    main(args)
    