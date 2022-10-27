# AIO3 BPR Baseline

This is a baseline system for AIO3 competition utilizing [Binary Passage Retriever (BPR)](https://github.com/studio-ousia/bpr).

BPR is an efficient passage retrieval model for a large collection of documents.
BPR integrates a learning-to-hash technique into [Dense Passage Retriever (DPR)](https://github.com/facebookresearch/DPR) to represent the passage embeddings using compact binary codes rather than continuous vectors.
It substantially reduces the memory size without a loss of accuracy when tested on several QA datasets (see [the BPR repository](https://github.com/studio-ousia/bpr) for more detail).

This work utilizes the implementation of BPR in [studio-ousia/soseki](https://github.com/studio-ousia/soseki) and is provided as one of the baseline systems for AIO3 competition.

## Installation

```sh
# Clone the repository with soseki submodule.
$ git clone --recursive https://github.com/cl-tohoku/AIO3_BPR_baseline

$ cd AIO3_BPR_baseline

# Upgrade pip and setuptools.
$ pip install -U pip setuptools

# Install the PyTorch package.
# You may want to check the install option for your CUDA environment.
# https://pytorch.org/get-started/locally/
$ pip install 'torch==1.11.0'

# Install other dependencies.
$ pip install -r soseki/requirements.txt

# Install the soseki package.
$ pip install -e ./soseki
```

## Example Usage

Before you start, you need to download the datasets into `$DATA_DIR`.

```sh
$ bash download_data.sh $DATA_DIR
```

In the following experiments, we used 4 GPUs with 12GB memory each.

**1. Build passage database**

```sh
$ python soseki/build_passage_db.py \
    --passage_file $DATA_DIR/wikipedia-split/jawiki-20220404-c400-large.tsv.gz \
    --db_file $WORK_DIR/passages.db \
    --db_map_size 10000000000
```

**2. Train a biencoder**

```sh
$ python soseki/train_biencoder.py \
    --train_file $DATA_DIR/retriever/jawiki-20220404-c400-large/aio_02_train.json.gz \
    --val_file $DATA_DIR/retriever/jawiki-20220404-c400-large/aio_02_dev.json.gz \
    --output_dir $WORK_DIR/biencoder \
    --max_question_length 128 \
    --max_passage_length 256 \
    --num_negative_passages 1 \
    --shuffle_hard_negative_passages \
    --shuffle_normal_negative_passages \
    --base_pretrained_model cl-tohoku/bert-base-japanese-v2 \
    --binary \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 1e-5 \
    --warmup_proportion 0.1 \
    --gradient_clip_val 2.0 \
    --max_epochs 20 \
    --gpus 4 \
    --precision 16 \
    --strategy ddp
```

**3. Build passage embeddings**

```sh
$ python soseki/build_passage_embeddings.py \
    --biencoder_file $WORK_DIR/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --passage_db_file $WORK_DIR/passages.db \
    --output_file $WORK_DIR/passage_embeddings.idx \
    --max_passage_length 256 \
    --batch_size 2048 \
    --device_ids 0 1 2 3
```

**4. Evaluate the retriever and create datasets for reader**

```sh
$ mkdir $WORK_DIR/reader_data

$ python soseki/evaluate_retriever.py \
    --biencoder_file $WORK_DIR/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --passage_db_file $WORK_DIR/passages.db \
    --passage_embeddings_file $WORK_DIR/passage_embeddings.idx \
    --qa_file $DATA_DIR/qas/aio_02_train.tsv \
    --output_file $WORK_DIR/reader_data/aio_02_train.jsonl \
    --batch_size 64 \
    --max_question_length 128 \
    --top_k 1 2 5 10 20 50 100 \
    --binary \
    --binary_k 2048 \
    --answer_match_type simple_nfkc \
    --include_title_in_passage \
    --device_ids 0 1 2 3
# The result should be logged as follows:
# Recall at 1: 0.6649 (14850/22335)
# Recall at 2: 0.7693 (17182/22335)
# Recall at 5: 0.8499 (18982/22335)
# Recall at 10: 0.8815 (19688/22335)
# Recall at 20: 0.9015 (20135/22335)
# Recall at 50: 0.9181 (20505/22335)
# Recall at 100: 0.9281 (20729/22335)

$ python soseki/evaluate_retriever.py \
    --biencoder_file $WORK_DIR/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --passage_db_file $WORK_DIR/passages.db \
    --passage_embeddings_file $WORK_DIR/passage_embeddings.idx \
    --qa_file $DATA_DIR/qas/aio_02_dev.tsv \
    --output_file $WORK_DIR/reader_data/aio_02_dev.jsonl \
    --batch_size 64 \
    --max_question_length 128 \
    --top_k 1 2 5 10 20 50 100 \
    --binary \
    --binary_k 2048 \
    --answer_match_type simple_nfkc \
    --include_title_in_passage \
    --device_ids 0 1 2 3
# The result should be logged as follows:
# Recall at 1: 0.5370 (537/1000)
# Recall at 2: 0.6350 (635/1000)
# Recall at 5: 0.7410 (741/1000)
# Recall at 10: 0.7940 (794/1000)
# Recall at 20: 0.8350 (835/1000)
# Recall at 50: 0.8670 (867/1000)
# Recall at 100: 0.8840 (884/1000)
```

**5. Train a reader**

```sh
$ python soseki/train_reader.py \
    --train_file $WORK_DIR/reader_data/aio_02_train.jsonl \
    --val_file $WORK_DIR/reader_data/aio_02_dev.jsonl \
    --output_dir $WORK_DIR/reader \
    --train_num_passages 16 \
    --eval_num_passages 50 \
    --max_input_length 384 \
    --shuffle_positive_passage \
    --shuffle_negative_passage \
    --num_dataloader_workers 1 \
    --base_pretrained_model cl-tohoku/bert-base-japanese-v2 \
    --answer_normalization_type simple_nfkc \
    --train_batch_size 1 \
    --eval_batch_size 2 \
    --learning_rate 1e-5 \
    --warmup_proportion 0.1 \
    --accumulate_grad_batches 4 \
    --gradient_clip_val 2.0 \
    --max_epochs 10 \
    --gpus 4 \
    --precision 16 \
    --strategy ddp
```

**6. Evaluate the reader**

```sh
$ python soseki/evaluate_reader.py \
    --reader_file $WORK_DIR/reader/lightning_logs/version_0/checkpoints/best.ckpt \
    --test_file $WORK_DIR/reader_data/aio_02_dev.jsonl \
    --test_num_passages 100 \
    --test_max_load_passages 100 \
    --test_batch_size 4 \
    --gpus 4 \
    --strategy ddp
# The result should be printed as follows:
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │   test_answer_accuracy    │    0.5770000219345093     │
# │ test_classifier_precision │    0.7040000557899475     │
# └───────────────────────────┴───────────────────────────┘
```

**7. (optional) Convert the trained models into ONNX format**

```sh
$ python soseki/convert_models_to_onnx.py \
    --biencoder_ckpt_file $WORK_DIR/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --reader_ckpt_file $WORK_DIR/reader/lightning_logs/version_0/checkpoints/best.ckpt \
    --output_dir $WORK_DIR/onnx
```

**8. Perform prediction**

```sh
$ python predict.py \
    --input_file $DATA_DIR/test/aio_03_test_unlabeled.jsonl \
    --output_file $WORK_DIR/aio_03_test_prediction.jsonl \
    --biencoder_ckpt_file $WORK_DIR/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --reader_ckpt_file $WORK_DIR/reader/lightning_logs/version_0/checkpoints/best.ckpt \
    --passage_db_file $WORK_DIR/passages.db \
    --passage_embeddings_file $WORK_DIR/passage_embeddings.idx \
    --device cuda:0
```

or if you have exported the models to ONNX format:

```sh
$ python predict.py \
    --input_file $DATA_DIR/test/aio_03_test_unlabeled.jsonl \
    --output_file $WORK_DIR/aio_03_test_prediction.jsonl \
    --onnx_model_dir $WORK_DIR/onnx \
    --passage_db_file $WORK_DIR/passages.db \
    --passage_embeddings_file $WORK_DIR/passage_embeddings.idx
```

### Subission for AIO3 Competition

**Note:** It is assumed that the trained models are converted into onnx format.

**1. Copy the models and data files into `models/` directory**

The files should be placed and renamed as follows:

```sh
models
├── passage_embeddings.idx      # from $WORK_DIR/passage_embeddings.idx
├── onnx
│   ├── biencoder_hparams.json  # from $WORK_DIR/onnx/biencoder_hparams.json
│   ├── question_encoder.onnx   # from $WORK_DIR/onnx/question_encoder.onnx
│   ├── reader.onnx             # from $WORK_DIR/onnx/reader.onnx
│   └── reader_hparams.json     # from $WORK_DIR/onnx/reader_hparams.json
└── passages.tsv.gz             # from $DATA_DIR/wikipedia-split/jawiki-20220404-c400-large.tsv.gz
```

**Note:** You will not need `$WORK_DIR/onnx/passage_encoder.onnx` since it is not used in the prediction stage.

**2. Build the Docker image**

```sh
$ docker build -t aio3-bpr-baseline .
```

You can find the size of the image by executing the command below:

```sh
$ docker run --rm aio3-bpr-baseline du -h --max-depth=0 /
```

**3. Run the image to perform prediction**

We assume `$TEST_DATA_DIR` contains a test file such as [`aio_03_test_unlabeled.jsonl`](https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_03/aio_03_test_unlabeled.jsonl), which is distributed on the [AIO official website](https://sites.google.com/view/project-aio/competition3).

```sh
$ docker run --rm -v $(realpath $TEST_DATA_DIR):/app/data -it aio3-bpr-baseline \
    bash submission.sh data/aio_03_test_unlabeled.jsonl data/predictions.jsonl
```

The prediction result will be saved to `$TEST_DATA_DIR/predictions.jsonl`.

**4. Save a Docker image to file**

```sh
$ docker save aio3-bpr-baseline | gzip > aio3-bpr-baseline.tar.gz
```

The saved Docker image should be about 2.4GB.

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This
work is licensed under a
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative
Commons Attribution-NonCommercial 4.0 International License</a>.

## Citation

If you find this work useful, please cite the following paper:

[Efficient Passage Retrieval with Hashing for Open-domain Question Answering](https://arxiv.org/abs/2106.00882)

```
@inproceedings{yamada2021bpr,
  title={Efficient Passage Retrieval with Hashing for Open-domain Question Answering},
  author={Ikuya Yamada and Akari Asai and Hannaneh Hajishirzi},
  booktitle={ACL},
  year={2021}
}
```
