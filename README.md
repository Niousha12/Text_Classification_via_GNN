# TextING

Pytorch implementation for the ACL2020 paper [Every Document Owns Its Structure: Inductive Text Classification via Graph Neural Networks](https://arxiv.org/abs/2004.13826).

For official code please refer to [TextING](https://github.com/CRIPAC-DIG/TextING) repository.

Some functions are based on [Official Code](https://github.com/CRIPAC-DIG/TextING) and [Text GCN](https://github.com/yao8839836/text_gcn). Thank for their work.

## Requirements

* Python 3.8.5
* install requirements.txt

## Usage

Download pre-trained word embeddings `glove.6B.300d.txt` from [here](http://nlp.stanford.edu/data/glove.6B.zip) and unzip to the `./pretrained_models`.

Build graphs from the datasets in `data/corpus/` as:

    python build_graph.py [DATASET] [WINSIZE]

Provided datasets include `mr`,`ohsumed`,`R8`and`R52`. The default sliding window size is 3.

<!To use your own dataset, put the text file under `data/corpus/` and the label file under `data/` as other datasets do. Preprocess the text by running `remove_words.py` before building the graphs.
>

Start training and inference as:

    python pre_dataloader [--dataset DATASET]

    python train.py [--dataset DATASET] [--learning_rate LR]
                    [--epochs EPOCHS] [--batch_size BATCHSIZE]
                    [--hidden HIDDEN] [--steps STEPS]
                    [--dropout DROPOUT] [--weight_decay WD]

<!To reproduce the result, large hidden size and batch size are suggested as long as your memory allows. We report our result based on 96 hidden size with 1 batch. For the sake of memory efficiency, you may change according to your hardware.
>
## Citation

    @inproceedings{zhang2020every,
      title={Every Document Owns Its Structure: Inductive Text Classification via Graph Neural Networks},
      author={Zhang, Yufeng and Yu, Xueli and Cui, Zeyu and Wu, Shu and Wen, Zhongzhen and Wang, Liang},
      booktitle="Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
      year={2020}
    }
