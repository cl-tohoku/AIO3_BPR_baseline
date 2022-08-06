#!/bin/bash

set -e

OUTPUT_DIR=$1

if [ -z $OUTPUT_DIR ] ; then
  echo "[ERROR] Please specify the 'output_dir'"
  echo "Usage: bash $0 [output_dir]"
  exit 1
fi

mkdir -p $OUTPUT_DIR

echo "Downloading the retriever input files into $OUTPUT_DIR/retriever/"
mkdir $OUTPUT_DIR/retriever
mkdir $OUTPUT_DIR/retriever/jawiki-20220404-c400-large
curl -o $OUTPUT_DIR/retriever/jawiki-20220404-c400-large/aio_02_train.json.gz -OL https://github.com/cl-tohoku/quiz-datasets/releases/download/v1.0.0/dpr.retriever.jawiki-20220404-c400-large.aio_02_train.json.gz
curl -o $OUTPUT_DIR/retriever/jawiki-20220404-c400-large/aio_02_dev.json.gz -OL https://github.com/cl-tohoku/quiz-datasets/releases/download/v1.0.0/dpr.retriever.jawiki-20220404-c400-large.aio_02_dev.json.gz

echo "Downloading the questions TSV files into $OUTPUT_DIR/qas/"
mkdir $OUTPUT_DIR/qas
curl -o $OUTPUT_DIR/qas/aio_02_train.tsv -OL https://github.com/cl-tohoku/quiz-datasets/releases/download/v1.0.0/dpr.qas.aio_02_train.tsv
curl -o $OUTPUT_DIR/qas/aio_02_dev.tsv -OL https://github.com/cl-tohoku/quiz-datasets/releases/download/v1.0.0/dpr.qas.aio_02_dev.tsv

echo "Downloading the passages TSV file into $OUTPUT_DIR/wikipedia-split/"
mkdir $OUTPUT_DIR/wikipedia-split
curl -o $OUTPUT_DIR/wikipedia-split/jawiki-20220404-c400-large.tsv.gz -OL https://github.com/cl-tohoku/quiz-datasets/releases/download/v1.0.0/dpr.wikipedia_split.jawiki-20220404-c400-large.tsv.gz

echo "Downloading the test dataset into $OUTPUT_DIR/evaluation/"
mkdir $OUTPUT_DIR/test
curl -o $OUTPUT_DIR/test/aio_03_test_unlabeled.jsonl -OL https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_03/aio_03_test_unlabeled.jsonl

echo "Done!"
