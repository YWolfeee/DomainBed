This is the implementation of [Towards a Theoretical Framework of Out-of-Distribution Generalization](https://arxiv.org/abs/2106.04496). Our code is inherted from [DomainBed](https://github.com/facebookresearch/DomainBed).

# Dataset

Downloads VLCS dataset and link it to 
```
/dataset/VLCS
```
The structure of dataset should be same as [DomainBed](https://github.com/facebookresearch/DomainBed). To run our full experiments, one should also download PACS and OfficeHome dataset in the same way.

# Single Experiment

to run a single experiment, run the following command:
```
python -m main --data_dir domainbed/datasets \
--trial_seed 89500 --algorithm ERM --dataset VLCS --test_envs 1 \
--steps 5001 --output_dir logs/ERM_VLCS_test_env1/ \
--extract_feature ERM_exp1  \
--output_result_file result.csv \
--checkpoint_freq 500 \
--save_feature_every_checkpoint \
--hparams "{\"lr\": 0.0001, \"resnet18\": false}" \
```

The final result will be saved in `logs/ERM_VLCS_test_env1/ERM_exp1`

# Model Selection
To reproduce our experiment, one can run
```
python my_launcher.py --command_launcher my_multi_gpu launch
```
to generate different models for model selection. After model generation, one should use the following codes
```
cd logs
python collect_feature.py
```
to collect feature and metric from different models. The result will be record in `renamed/*.csv`

to use our model selection criterion, one can run
```
python model_selection.py
```
This command use the result in `.csv` file to select model according to different criterion and report their test evironment accuracy. `train_acc` means the selection criterion is the average training accuracy, `train_mix` is the selection criterion defined in our [paper](https://arxiv.org/abs/2106.04496)
