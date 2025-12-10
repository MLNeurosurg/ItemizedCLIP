"""
Classification evaluation for RSICD
"""


import torch
import torch.nn.functional as F
from tqdm import tqdm


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

# Perform 30-way classification on RSICD
def classify(model, dataloader, args, tokenizer):
    model.eval()
    
    textsets = dataloader.dataset.cls_texts
    textsets.append('average')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tembs = []
    for texts in textsets:
        if texts == 'average':
            textemb = torch.stack(tembs,dim=0).mean(dim=0)
            textemb = F.normalize(textemb, dim=-1)
            tembs.append(textemb)     
        else:
            tokenized_text = tokenizer(texts).to(device)
            global_text_token, local_text_tokens = unwrap_model(model).encode_text(tokenized_text, normalize=False)
            textemb = unwrap_model(model).text_post(global_text_token)
            textemb = F.normalize(textemb, dim=-1)
            tembs.append(textemb)
            continue # we only inference on average now
          
        total_samples = 0
        correct_predictions = 0
        correct_predictions5 = 0
        total_ranks = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, disable=(args.local_rank != 0)):
                images, label = batch
                images = images.to(device=device)
                label = label.to(device)

                if args.inference_with_flair: # use FLAIR's way of inference (with TCSim)
                    _, image_embeddings = unwrap_model(model).encode_image(images, normalize=False)
                    imageemb = unwrap_model(model).image_post(image_embeddings)
                    visproj = unwrap_model(model).visual_proj
                    imageemb = F.normalize(imageemb, dim=-1)
                    tcout = visproj(textemb.unsqueeze(0).expand(imageemb.size(0), -1, -1),imageemb,imageemb)
                    tcout = F.normalize(tcout, dim=-1)
                    logits = torch.einsum('bik,ik->bi', tcout, textemb)
                else: # Use standard CLIP inference
                    imageemb = unwrap_model(model).encode_image(images, normalize=True)
                    if isinstance(imageemb,tuple):
                        imageemb =unwrap_model(model).image_post(imageemb[0])
                    imageemb = F.normalize(imageemb, dim=-1)
                    logits = torch.einsum('ik,jk->ij', imageemb, textemb)

                ranks = logits.argsort(dim=1, descending=True)
                gt_positions = (ranks == label.unsqueeze(1)).nonzero(as_tuple=True)[1]
                ranks = gt_positions+1

                predictions = torch.argmax(logits, dim=-1)

                total_samples += images.size(0)
                total_ranks += ranks.sum()
                correct_predictions += (predictions == label).sum().item()
                correct_predictions5 += (ranks <= 5).sum().item()

        accuracy = correct_predictions / total_samples
        meanrank = total_ranks / total_samples
        if args.local_rank == 0:
            print(f"Classification Accuracy: {accuracy:.4f}")
            print(f"Mean Rank: {meanrank:.4f}")
            print(f"Top-5 Accuracy: {correct_predictions5 / total_samples:.4f}")

    return accuracy