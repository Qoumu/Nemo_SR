from utils.general import *
from PrototypicalNetwork.train import *

# ============================================================================
# AUDIO PREPROCESSING (Your existing function)
# ============================================================================

dataset_list, label_map = build_prototypical_dataset(
    root="/home/d/Projects/Nemo_SR/data/speakerdataset/LibriSpeech",
    num_speakers=40,           # Total number of speakers to select
    train_ratio=0.7,           # 75% of speakers for training (28 speakers)
    val_ratio=0.15,            # 10% of speakers for validation (6 speakers)
    test_ratio=0.15,           # 10% of speakers for testing (6 speakers)
    min_samples_per_speaker=5, # Each speaker must have at least 5 samples
    max_samples_per_speaker=15, # Limit to 15 samples per speaker (None = use all)
    seed=42
)

# Get number of unique speakers
unique_label_ids = set(item['label_id'] for item in dataset_list)
n_speakers = len(unique_label_ids)

print(f"Dataset size: {len(dataset_list)}")
print(f"Number of speakers: {n_speakers}")
print(f"Unique labels: {set(item['label'] for item in dataset_list)}")

# Verify data splits
train_samples = [item for item in dataset_list if item['split'] == 'train']
val_samples = [item for item in dataset_list if item['split'] == 'valid']
test_samples = [item for item in dataset_list if item['split'] == 'test']

print(f"\nData Split Summary:")
print(f"  Train: {len(train_samples)} samples")
print(f"  Valid: {len(val_samples)} samples")
print(f"  Test: {len(test_samples)} samples")

model = train_prototypical_network(
    dataset_list=dataset_list,
    train_mode=False,
    n_way=5,        # 5-way classification
    k_shot=5,       # 5 support examples per class
    n_query=10,      # 5 query examples per class
    n_episodes=200,
    n_val_episodes=2,
    sr=16000,
    n_mels=80,
    duration=3.0, 
    show_progress=True,
    plot_path="ECAPATDNN_protonet_curves.png"
)
