import argparse
from pathlib import Path

from datasets import DownloadMode, load_dataset, load_dataset_builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a specific LibriSpeech split.")
    parser.add_argument(
        "--subset",
        default="clean",
        help="LibriSpeech subset name (e.g. clean, other, clean.360).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to download (e.g. train, validation, test).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination directory. Defaults to data/dataset/librispeech_<subset>_<split>.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Skip loading the dataset into memory and just download it into the Hugging Face cache.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if the dataset is already cached locally.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    default_dir = Path(__file__).resolve().parent / f"librispeech_{args.subset}_{args.split}"
    target_dir = Path(args.output_dir) if args.output_dir else default_dir
    download_mode = DownloadMode.FORCE_REDOWNLOAD if args.force_download else DownloadMode.REUSE_DATASET_IF_EXISTS

    if args.download_only:
        builder = load_dataset_builder("openslr/librispeech_asr", args.subset)
        builder.download_and_prepare(download_mode=download_mode)
        cache_dir = Path(builder.cache_dir)
        print(
            "Dataset assets downloaded to Hugging Face cache at "
            f"{cache_dir}. Use `datasets.load_dataset(..., split='{args.split}')` "
            "to materialize the data when needed."
        )
        return

    # Login using e.g. `huggingface-cli login` to access gated datasets.
    dataset = load_dataset(
        "openslr/librispeech_asr",
        args.subset,
        split=args.split,
        download_mode=download_mode,
    )

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(target_dir.as_posix())
    print(f"Saved LibriSpeech subset='{args.subset}' split='{args.split}' to {target_dir}")


if __name__ == "__main__":
    main()
