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
        experiment_name = f"{args.exp_type}/{args.algo}/{args.name}/model={args.llm}_lr={args.lr}"
    else:
        experiment_name = f"{args.exp_type}/{args.algo}/model={args.llm}_lr={args.lr}"
    
    writer = SummaryWriter(log_dir = f'runs/{experiment_name}')
    
    if "gpt2" in args.llm:
        llm = GPT2(args)
    
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
    
    print(f"training {args.epoch} epoch(s) with learning rate={args.lr}\n")
    
    optimizer = optim.Adam(mapper.model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay)
    ce_criterion = nn.CrossEntropyLoss()
    rl_criterion = nn.CrossEntropyLoss(reduction='none')
            
    writer.add_hparams(
        {
            "llm_model": args.llm,
            "modality": args.exp_type,
            "encoder": args.image_encoder,
            "learning_rate": args.lr,
        },
        {}
    )

    global_step = 0
    for epoch in range(args.epochs):
        mapper.model.train()
        for i, dd in enumerate(dataloader):
            
            optimizer.zero_grad()

            """
            * For IMAGES
            """
            data = dd[0]

            # tokens = []
            # for image_idx, image in enumerate(images):
            #     # Construct a unique filename for each image
            #     filename = os.path.join(temp_dir, f'batch_{i}_image_{image_idx}.jpg')
            #     # Save the image as a .jpg file
            #     save_image(image, filename)
            #     # Convert the saved image file to binary representation
            #     binary_representation = image_to_binary(filename)
            #     tokens.append(binary_representation)
            #     # Optionally, delete the file if it's no longer needed
            #     os.remove(filename)

            # tensor_list = [torch.tensor(sublist) for sublist in tokens]
            # # Pad the sequence of tensors, padding zeros behind each sequence
            # data = pad_sequence(tensor_list, batch_first=True, padding_value=0)        
            # num_chunks = data.shape[1] // seq_len
            
            # data = data[:,:num_chunks*seq_len]
            """
            * For IMAGES
            """
            
            # data = dd
            # round_loss = 0
            
            ground_truth_tokens = data.reshape(-1, seq_len).to(device)
            one_hot_tokens = F.one_hot(ground_truth_tokens, num_classes=bits_vocab_len).float()

            # Logits are to be compared with the next ground truth tokens
            ground_truth_tokens = ground_truth_tokens[:,1:]
            inputs_feature_vector = mapper(one_hot_tokens)
            
            # Add prompt to input
            # prompt_feature_vector = prompt(prompt_inputs)
            # prompt_feature_vector = prompt_feature_vector.unsqueeze(0).repeat(batch_len, 1, 1)
            # inputs_feature_vector = torch.cat((prompt_feature_vector, mapped_feature_vector), dim=1)

            # Map tokens and get ground truth from LLM
            translated_feature_vector, translated_logits, translated_text_tokens = translate(inputs_feature_vector, embeddings.detach(), temperature=temperature)
            # translated_feature_vector, translated_logits, translated_text_tokens = translate(inputs_feature_vector, embeddings.detach(), temperature=temperature)

            # Calculate Representation of Last Layer in LLM
            final_layer_fv = generate_next_token_predictions_withfv(translated_feature_vector)

            # Calculate Logits with mapper function
            logits = torch.matmul(final_layer_fv, reverseMapper.mapper.weight)
            # logits = torch.matmul(final_layer_fv, mapper.mapper.weight)
            logits = logits[:,prompt_len:-1]
            logits_ = logits.reshape(-1, bits_vocab_len)
            ground_truth_tokens = ground_truth_tokens.reshape(-1)        
            ce_loss = criterion(logits_, ground_truth_tokens)
            # round_loss += ce_loss.item()

            writer.add_scalar("training/cross_entropy", ce_loss.item(), global_step)
            ce_loss.backward()
            optimizer.step()
            if global_step%100==0:
                print(f"Epoch {epoch+1}, Batch {global_step}, CE Loss: {ce_loss.mean().item()}")
            global_step+=1

            torch.cuda.empty_cache()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} completed.")
    writer.close()


if __name__=="__main__":
    parser = get_config()
    args = parser.parse_args()
    
    if args.use_seed:
        torch.manual_seed(args.seed)
        
    main(args)
    