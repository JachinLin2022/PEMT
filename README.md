# PEMT: Multi-Task Correlation Guided Mixture-of-Experts Enables Parameter-Efficient Transfer Learning
This repo contains the original implementation of our paper "[PEMT: Multi-Task Correlation Guided Mixture-of-Experts Enables Parameter-Efficient Transfer Learning](https://arxiv.org/abs/2402.15082)".
![architecture](/figure/architecture.png)
# Environment
Please run the command below to install the dependent libraries.
```
conda create -n pemt_env python=3.8
conda activate pemt_env
pip install -r requirements.txt
```
# PEMT
PEMT consists of two-stage training: Source Task Training and Target Task Adaptation.
## Training 

1. **Source Task Training**: The goal of Stage 1 is to capture the task-specific knowledge of each source task. We fine-tune the PLM on multiple source tasks using adapter-based PEFT methods to obtain Source Task Adapter and Task Description Prompts. 


2. **Target Task Adaptation**: In the second stage, PEMT is guided by the correlation between tasks to utilize the distilled knowledge of all source tasks for adaptation to the downstream target task.


### Source Task Training
We use 6 high-resource datasets as the source tasks: MNLI, QNLI, QQP, SST-2, SQuAD and ReCoRD.
```
cd pemt
bash run_pemt_stage1.sh
```
### Checkpoint
We provide source prompts and adapter weights for the six source tasks. You can download all the checkpoint from:
>https://drive.google.com/drive/folders/1s-NHXNJ2XmTEkmaD2dx-sfFd7DegFhh4?usp=sharing

### Target Task Adaptation
We use other tasks from four benchmarks as the target task to validate the effectiveness of PEMT under both full data and few-shot settings. 
#### Full data
```
cd pemt
# remenber to modify the stage1 prompt and adapter loading path 
bash run_pemt_stage2.sh
```
#### Few data
```
cd pemt
# remenber to modify the stage1 prompt and adapter loading path 
bash run_pemt_stage2_few_shot.sh
```

#### Result Example on SuperGLUE-CB
```
***** test metrics *****
  epoch                   =              20.0
  test_accuracy           =           92.8571
  test_average_metrics    = 92.85714285714286
  test_loss               =            0.1278
  test_moe_weight_0       =            0.6418
  test_moe_weight_1       =            0.0734
  test_moe_weight_2       =            0.0514
  test_moe_weight_3       =            0.0623
  test_moe_weight_4       =            0.0425
  test_moe_weight_5       =            0.1285
  test_runtime            =        0:00:00.52
  test_samples_per_second =             53.67
```
# Acknowledgements
The implementations of the baselines are from the [ATTEMPT](https://github.com/AkariAsai/ATTEMPT) repository. Huge thanks to the contributors of those amazing repositories!
# Citation and Contact
If you find this repository helpful, please cite our paper.
```
@article{lin2024pemt,
  title={PEMT: Multi-Task Correlation Guided Mixture-of-Experts Enables Parameter-Efficient Transfer Learning},
  author={Lin, Zhisheng and Fu, Han and Liu, Chenghao and Li, Zhuo and Sun, Jianling},
  journal={arXiv preprint arXiv:2402.15082},
  year={2024}
}
```