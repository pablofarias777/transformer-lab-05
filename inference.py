import torch
from transformer import Transformer
from dataset import prepare_dataloader, BATCH_SIZE, MAX_LEN, tokenize_pair

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def autoregressive_decode(model, src_ids, tokenizer, max_len=MAX_LEN):
  
    model.eval()

    with torch.no_grad():

        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)


        encoder_output = model.encode(src)


        start_token = tokenizer.cls_token_id   
        eos_token   = tokenizer.sep_token_id   

        decoded_ids = [start_token]

        for _ in range(max_len):
            tgt = torch.tensor(decoded_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

            decoder_output = model.decode(tgt, encoder_output)

            logits = model.Wo(decoder_output[:, -1, :])

            next_token = logits.argmax(dim=-1).item()
            decoded_ids.append(next_token)

            if next_token == eos_token:
                break

    return decoded_ids

def overfitting_test(model, tokenizer, src_sentence, tgt_expected):
   
    print(f"\n{'='*55}")
    print(f"  Tarefa 4 — Prova de Fogo (Overfitting Test)")
    print(f"{'='*55}")
    print(f"  Frase fonte (EN) : {src_sentence}")
    print(f"  Tradução esperada: {tgt_expected}")


    src_ids, _ = tokenize_pair(tokenizer, src_sentence, tgt_expected)

    generated_ids = autoregressive_decode(model, src_ids, tokenizer)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"  Tradução gerada  : {generated_text}")
    print(f"\n  IDs gerados: {generated_ids}")
    print(f"{'='*55}\n")

    return generated_text

if __name__ == "__main__":
    from train import train, D_MODEL, D_FF, H, N


    model, tokenizer, loss_history = train()


    from datasets import load_dataset
    hf_dataset = load_dataset("Helsinki-NLP/opus_books", "en-pt", split="train")
    hf_dataset = hf_dataset.select(range(1000))


    sample = hf_dataset[0]["translation"]
    src_sentence = sample["en"]
    tgt_expected  = sample["pt"]

    overfitting_test(model, tokenizer, src_sentence, tgt_expected)

    
    print("\n--- Mais exemplos do conjunto de treino ---")
    for i in [1, 2, 3]:
        s = hf_dataset[i]["translation"]
        print(f"\nFrase {i+1} (EN): {s['en'][:60]}...")
        result = overfitting_test(model, tokenizer, s["en"], s["pt"])

