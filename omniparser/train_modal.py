import modal
import os
from pathlib import Path

checkpoint = "ustc-community/dfine-medium-obj365"
width, height = 800, 640
COMET_API_KEY = os.environ.get("COMET_API_KEY")
if not COMET_API_KEY:
    raise ValueError(
        "COMET_API_KEY environment variable is not set. Please set it to your Comet API key."
    )
COMET_PROJECT_NAME = "omni-parser"
COMET_WORKSPACE = "francesco-zuppichini"

# # Finetuning OmniParser with Modal
# Useful examples:
# https://modal.com/docs/examples/long-training
# https://modal.com/docs/examples/finetune_yolo
# https://modal.com/docs/examples/flan_t5_finetune

# Modal runs your code in the cloud inside containers. So to use it, we have to define the dependencies
# of our code as part of the container's [image](https://modal.com/docs/guide/custom-container).

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Adds comet api key
    .env({"COMET_API_KEY": COMET_API_KEY})
    .pip_install(  # install python libraries for computer vision
        [
            "albumentations>=2.0.7",
            "datasets>=3.6.0",
            "faster-coco-eval>=1.6.6",
            "pillow>=11.2.1",
            "pycocotools>=2.0.10",
            "scipy>=1.15.3",
            "torchmetrics>=1.7.1",
            "torchvision>=0.21.0",
            "transformers[torch]>=4.52.0",
            "comet-ml>=3.49.10",
        ]
    )
    # adds local modules to the container
    .add_local_python_source("transform")
    .add_local_python_source("dataset")
    .add_local_python_source("metrics")
    .add_local_python_source("train")
)
# We also create a persistent [Volume](https://modal.com/docs/guide/volumes) for storing datasets, trained weights, and inference outputs.

volume = modal.Volume.from_name("dfine-finetune", create_if_missing=True)
volume_path = Path("/root")  # the path to the volume from within the container
datasetPath = volume_path / "dataset"  # where the dataset will be downloaded

# We attach both of these to a Modal [App](https://modal.com/docs/guide/apps).
app = modal.App("dfine-finetune", image=image, volumes={volume_path: volume})


# ## Download the dataset
@app.function()
def download_dataset():
    from datasets import load_dataset

    load_dataset("Francesco/open-omniparser-dataset", cache_dir=datasetPath)


# ## Train a model
# GPU Costs (per hour)
# NVIDIA_B200: $6.25
# NVIDIA_H200: $4.54
# NVIDIA_H100: $3.95
# NVIDIA_A100_80GB: $2.50
# NVIDIA_A100_40GB: $2.10
# NVIDIA_L40S: $1.95
# NVIDIA_A10G: $1.10
# NVIDIA_L4: $0.80
# NVIDIA_T4: $0.59

# CPU Cost (per hour)
# Physical core (2 vCPU equivalent): $0.0473 / core
# Minimum: 0.125 cores per container

# Memory Cost (per hour)
# $0.0080 / GiB

MINUTES = 60

TRAIN_GPU_COUNT = 1
TRAIN_GPU = f"T4:{TRAIN_GPU_COUNT}"
TRAIN_CPU_COUNT = 4


@app.function(
    gpu=TRAIN_GPU,
    cpu=TRAIN_CPU_COUNT,
    timeout=24 * 60 * MINUTES,
)
def train(resume_from_checkpoint=False):
    from comet_ml import start
    from train import CometImageLogger, collate_fn
    from datasets import load_dataset
    from transformers import (
        AutoImageProcessor,
        AutoModelForObjectDetection,
        Trainer,
        TrainingArguments,
    )

    from dataset import OmniparserDataset
    from metrics import MAPEvaluator
    from transform import get_transforms

    # make sure volume is synced
    volume.reload()

    train_transform, val_transform = get_transforms((width, height))

    image_processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        do_resize=True,
        size={"width": width, "height": height},
        use_fast=True,
    )

    dataset = load_dataset("Francesco/open-omniparser-dataset", cache_dir=datasetPath)

    if "validation" not in dataset:
        split = dataset["train"].train_test_split(0.10, seed=1337)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    train_dataset = OmniparserDataset(
        dataset["train"],
        train_transform,
    )
    validation_dataset = OmniparserDataset(
        dataset["validation"],
        val_transform,
    )
    test_dataset = OmniparserDataset(
        dataset["test"],
        val_transform,
    )
    categories = dataset["train"].features["objects"].feature["category"].names
    id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}

    eval_compute_metrics_fn = MAPEvaluator(
        image_processor=image_processor, threshold=0.05, id2label=id2label
    )

    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )

    # create comet dashboard
    experiment = start(
        project_name=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE,
        # online=False,
    )

    experiment.log_parameters({"image_width": width, "image_height": height})

    model_slug = volume_path / f"{checkpoint.split('/')[-1]}-{width}x{height}"

    training_args = TrainingArguments(
        output_dir=model_slug,
        fp16=True,
        num_train_epochs=150,
        max_grad_norm=1.0,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_steps=100,
        per_device_train_batch_size=4,
        dataloader_num_workers=8,
        fp16_full_eval=True,
        gradient_accumulation_steps=2,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=20,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        report_to="comet_ml",
        logging_strategy="epoch",
    )

    # Set the resume_from_checkpoint parameter in train() to resume training from the last checkpoint or a specific checkpoint.
    if resume_from_checkpoint:
        training_args.resume_from_checkpoint = model_slug

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    trainer.add_callback(
        CometImageLogger(experiment, validation_dataset, image_processor)
    )

    trainer.train()


# ## Running the training job
# ```bash
# uvx modal run train_modal.py --detach
# ```


@app.local_entrypoint()
def main(resume_from_checkpoint=False):
    with modal.enable_output():
        download_dataset.remote()
        train.remote(kwargs={"resume_from_checkpoint": resume_from_checkpoint})
