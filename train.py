import torch
import torch.nn as nn
from transformer import Transformer
from dataset import prepare_dataloader, BATCH_SIZE

D_MODEL = 128
D_FF = 512
H = 4        
N = 2        
MAX_LEN = 64
EPOCHS = 15
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    print(f"\n{'='*55}")
    print(f"  Laboratório 05 — Treinamento Fim-a-Fim do Transformer")
    print(f"{'='*55}")
    print(f"  Dispositivo: {DEVICE}")

    loader, vocab_size, pad_idx, tokenizer = prepare_dataloader(
        subset_size=1000, batch_size=BATCH_SIZE
    )

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=D_MODEL,
        d_ff=D_FF,
        h=H,
        N=N,
        max_len=MAX_LEN
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parâmetros treináveis: {total_params:,}\n")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

    print(f"{'Epoch':>6} | {'Loss':>10} | {'Progresso'}")
    print("-" * 45)

    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for src, tgt in loader:
            src = src.to(DEVICE)  
            tgt = tgt.to(DEVICE) 

            tgt_input  = tgt[:, :-1]   
            tgt_target = tgt[:, 1:]    

        
            logits = model(src, tgt_input)
            
            logits_flat  = logits.reshape(-1, logits.size(-1))
            targets_flat = tgt_target.reshape(-1)

            loss = criterion(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()        
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()         
            n_tokens = (targets_flat != pad_idx).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        loss_history.append(avg_loss)

        pct = epoch / EPOCHS
        bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
        print(f"  {epoch:>4} | {avg_loss:>10.4f} | {bar} {pct*100:.0f}%")

    print("-" * 45)
    reducao = ((loss_history[0] - loss_history[-1]) / loss_history[0]) * 100
    print(f"\n  Loss inicial : {loss_history[0]:.4f}")
    print(f"  Loss final   : {loss_history[-1]:.4f}")
    print(f"  Redução total: {reducao:.1f}% ✓\n")

    torch.save(model.state_dict(), "transformer_trained.pt")
    print("  Modelo salvo em: transformer_trained.pt")

    return model, tokenizer, loss_history


if __name__ == "__main__":
    train()
