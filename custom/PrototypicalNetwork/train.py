import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt

from model.CNN import SpeakerCNN
from model.ECAPATDNN import ECAPATDNNBackbone
from utils.data_preprocessing import SpeakerDataset
from utils.general import *

class PrototypicalLoss(nn.Module):
    """
    Prototypical Loss - used in prototypical networks for few-shot learning
    """
    def __init__(self, n_support, n_classes, n_query):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support
        self.n_classes = n_classes
        self.n_query = n_query

    def forward(self, logits):
        log_p_y = F.log_softmax(logits, dim=1).view(self.n_classes, self.n_query, -1)

        target_inds = torch.arange(0, self.n_classes, device=log_p_y.device)
        target_inds = target_inds.view(self.n_classes, 1, 1)
        target_inds = target_inds.expand(self.n_classes, self.n_query, 1).long()

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

        return loss_val,  acc_val
    
def _get_tqdm():
    try:
        from tqdm import tqdm
    except Exception:
        return None
    return tqdm


def sample_episode(dataset_list, n_way, k_shot, n_query):
    """
    Sample one episode for few-shot learning

    Args:
        dataset_list: list of dicts with 'audio_filepath', 'label_id', 'split'
        n_way: number of classes (speakers) per episode
        k_shot: number of support examples per class
        n_query: number of query examples per class

    Returns:
        support_data: list of support samples
        query_data: list of query samples
    """
    # Group samples by speaker
    speaker_dict = {}
    for item in dataset_list:
        label_id = item['label_id']
        if label_id not in speaker_dict:
            speaker_dict[label_id] = []
        speaker_dict[label_id].append(item)

    # Randomly select n_way speakers
    available_speakers = list(speaker_dict.keys())
    selected_speakers = random.sample(available_speakers, n_way)

    support_data = []
    query_data = []

    for new_label, speaker_id in enumerate(selected_speakers):
        samples = speaker_dict[speaker_id]

        # Need at least k_shot + n_query samples
        if len(samples) < k_shot + n_query:
            raise ValueError(f"Speaker {speaker_id} has only {len(samples)} samples, need {k_shot + n_query}")

        # Randomly sample support and query
        selected = random.sample(samples, k_shot + n_query)

        for i, sample in enumerate(selected):
            # Create new dict with remapped label (0 to n_way-1)
            new_sample = sample.copy()
            new_sample['episode_label'] = new_label

            if i < k_shot:
                support_data.append(new_sample)
            else:
                query_data.append(new_sample)

    return support_data, query_data

def train_episode(model, support_data, query_data, dataset, loss_fn, optimizer, device, n_way):
    """
    Train one episode of prototypical network

    Args:
        model: PrototypicalNetwork model
        support_data: list of support samples
        query_data: list of query samples
        dataset: SpeakerDataset for loading audio
        criterion: loss function
        optimizer: optimizer
        device: torch device
        n_way: number of classes in episode
    """
    # Forward pass
    optimizer.zero_grad()
    
    model.train()
    
    filepath_to_idx = {item['audio_filepath']: i for i, item in enumerate(dataset.dataset_list)}

    # Load and process support samples
    support_mels = []
    support_labels = []
    for sample in support_data:
        # Find index in original dataset
        idx = filepath_to_idx[sample['audio_filepath']]
        mel_spec, _ = dataset[idx]
        support_mels.append(mel_spec)
        support_labels.append(sample['episode_label'])

    support_mels = torch.stack(support_mels).to(device)
    support_labels = torch.tensor(support_labels).to(device)

    # Load and process query samples
    query_mels = []
    query_labels = []
    for sample in query_data:
        idx = filepath_to_idx[sample['audio_filepath']]
        mel_spec, _ = dataset[idx]
        query_mels.append(mel_spec)
        query_labels.append(sample['episode_label'])

    query_mels = torch.stack(query_mels).to(device)
    query_labels = torch.tensor(query_labels).to(device)

    # Extract embeddings
    support_embeddings = model(support_mels)
    query_embeddings = model(query_mels)

    # Compute prototypes
    prototypes = compute_prototypes(support_embeddings, support_labels, n_way)

    # Classify query samples
    distance = model.pn_predict(query_embeddings, prototypes)

    # Compute loss
    loss, _ = loss_fn(distance)
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Compute accuracy
    _, predicted = distance.max(1)
    correct = predicted.eq(query_labels).sum().item()
    total = query_labels.size(0)
    accuracy = 100. * correct / total

    return loss.item(), accuracy


def validate_episode(model, support_data, query_data, dataset, criterion, device, n_way):
    """Validate one episode"""
    model.eval()

    with torch.no_grad():
        filepath_to_idx = {item['audio_filepath']: i for i, item in enumerate(dataset.dataset_list)}
        
        # Load support samples
        support_mels = []
        support_labels = []
        for sample in support_data:
            idx = filepath_to_idx[sample['audio_filepath']]
            mel_spec, _ = dataset[idx]
            support_mels.append(mel_spec)
            support_labels.append(sample['episode_label'])

        support_mels = torch.stack(support_mels).to(device)
        support_labels = torch.tensor(support_labels).to(device)

        # Load query samples
        query_mels = []
        query_labels = []
        for sample in query_data:
            idx = filepath_to_idx[sample['audio_filepath']]
            mel_spec, _ = dataset[idx]
            query_mels.append(mel_spec)
            query_labels.append(sample['episode_label'])

        query_mels = torch.stack(query_mels).to(device)
        query_labels = torch.tensor(query_labels).to(device)

        # Extract embeddings
        support_embeddings = model(support_mels)
        query_embeddings = model(query_mels)

        # Compute prototypes
        prototypes = compute_prototypes(support_embeddings, support_labels, n_way)

        # Classify query samples
        distances = model.pn_predict(query_embeddings, prototypes)

        # Compute loss and accuracy
        loss, _ = criterion(distances)
        _, predicted = distances.max(1)
        correct = predicted.eq(query_labels).sum().item()
        total = query_labels.size(0)
        accuracy = 100. * correct / total

    return loss.item(), accuracy

def train_prototypical_network(
    dataset_list: List[dict],
    train_mode: bool = True,
    n_way: int = 5,
    k_shot: int = 5,
    n_query: int = 5,
    n_episodes: int = 1000,
    n_val_episodes: int = 100,
    n_test_episodes: int | None = None,
    sr: int = 16000,
    n_mels: int = 80,
    duration: float = 3.0,
    n_fft: int = 1024,
    hop_length: int = 256,
    apply_filter: bool = True,
    filter_top_db: float = 20.0,
    show_progress: bool = True,
    plot_path: Optional[str | Path] = "ECAPATDNN_protonet_training_curves.png",
):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Split data
    train_data = [item for item in dataset_list if item['split'] == 'train']
    val_data = [item for item in dataset_list if item['split'] == 'valid']
    test_data = [item for item in dataset_list if item['split'] == 'test']

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"\nEpisodic Learning Configuration:")
    print(f"  {n_way}-way {k_shot}-shot learning")
    print(f"  {n_query} query samples per class")
    print(f"  {n_episodes} training episodes")
    print(f"  {n_val_episodes} validation episodes")

    dataset = SpeakerDataset(
        train_data,
        sr=sr,
        n_mels=n_mels,
        duration=duration,
        augment=True,
        n_fft=n_fft,
        hop_length=hop_length,
        apply_filter=apply_filter,
        filter_top_db=filter_top_db,
    )

    val_dataset = SpeakerDataset(
        val_data,
        sr=sr,
        n_mels=n_mels,
        duration=duration,
        augment=True,
        n_fft=n_fft,
        hop_length=hop_length,
        apply_filter=apply_filter,
        filter_top_db=filter_top_db,
    )

    test_dataset = SpeakerDataset(
        test_data,
        sr=sr,
        n_mels=n_mels,
        duration=duration,
        augment=True,
        n_fft=n_fft,
        hop_length=hop_length,
        apply_filter=apply_filter,
        filter_top_db=filter_top_db,
    )

    # Initialize model
    model = ECAPATDNNBackbone(n_mels=n_mels, channels=512, emb_dim=64)
    model = model.to(device)

    # Loss and optimizer
    loss_fn = PrototypicalLoss(n_support=k_shot, n_classes=n_way, n_query=n_query)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    model_path = Path("ECAPATDNN_protonet_model.pth")

    # Training loop
    best_val_loss = float('inf')
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    tqdm = _get_tqdm()

    # Training episodes
    if train_mode is True:
        print("\nStarting training...")
        train_losses = []
        train_accs = []

        iterable = range(n_episodes)
        if show_progress and tqdm is not None:
            iterable = tqdm(range(n_episodes), desc="Training")

        for episode in iterable:
            try:
                # Sample episode
                support_data, query_data = sample_episode(train_data, n_way, k_shot, n_query)

                # Train on episode
                loss, acc = train_episode(
                    model, support_data, query_data, dataset,
                    loss_fn, optimizer, device, n_way
                )

                train_losses.append(loss)
                train_accs.append(acc)

                # Validate every 2 episodes
                if (episode + 1) % 2 == 0:
                    val_losses = []
                    val_accs = []

                    for _ in range(n_val_episodes):
                        try:
                            val_support, val_query = sample_episode(val_data, n_way, k_shot, n_query)
                            val_loss, val_acc = validate_episode(
                                model, val_support, val_query, val_dataset,
                                loss_fn, device, n_way
                            )
                            val_losses.append(val_loss)
                            val_accs.append(val_acc)
                        except ValueError:
                            continue

                    avg_train_loss = np.mean(train_losses[-100:])
                    avg_train_acc = np.mean(train_accs[-100:])
                    avg_val_loss = np.mean(val_losses)
                    avg_val_acc = np.mean(val_accs)

                    scheduler.step(avg_val_loss)

                    print(f"\nEpisode {episode+1}/{n_episodes}")
                    print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%")
                    print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%")

                    history["train_loss"].append(avg_train_loss)
                    history["train_acc"].append(avg_train_acc)
                    history["val_loss"].append(avg_val_loss)
                    history["val_acc"].append(avg_val_acc)

                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(model.state_dict(), model_path)
                        print(f"  ✓ Saved best model (Val Loss: {avg_val_loss:.4f})")

            except ValueError as e:
                # Skip episodes where we can't sample enough data
                continue

        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")

    if plot_path and len(history["train_loss"]) > 0:
        _plot_training_curves(history, plot_path)
        print(f"Saved training curves to: {plot_path}")

    # Evaluate on test episodes using the best checkpoint if available
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

    test_episode_count = n_val_episodes if n_test_episodes is None else n_test_episodes
    test_losses = []
    test_accs = []

    for _ in range(test_episode_count):
        try:
            test_support, test_query = sample_episode(test_data, n_way, k_shot, n_query)
            test_loss, test_acc = validate_episode(
                model, test_support, test_query, test_dataset,
                loss_fn, device, n_way
            )
            test_losses.append(test_loss)
            test_accs.append(test_acc)
        except ValueError:
            continue

    if test_losses:
        avg_test_loss = np.mean(test_losses)
        avg_test_acc = np.mean(test_accs)
        print(f"\nTest Results ({len(test_losses)} episodes):")
        print(f"  Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.2f}%")
    else:
        print("\nTest evaluation skipped: not enough data to sample episodes.")

    return model

def _plot_training_curves(history: dict, output_path: str | Path) -> None:
    """Plot training curves"""
    checkpoints = range(10, len(history["train_loss"]) * 10 + 1, 10)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(checkpoints, history["train_loss"], label="Train Loss", marker='o')
    plt.plot(checkpoints, history["val_loss"], label="Val Loss", marker='s')
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss over Episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(checkpoints, history["train_acc"], label="Train Acc", marker='o')
    plt.plot(checkpoints, history["val_acc"], label="Val Acc", marker='s')
    plt.xlabel("Episode")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"Training curves saved to {output_path}")
