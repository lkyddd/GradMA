# GradMA

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==1.9.0

- scipy==1.7.2

- numpy==1.21.2

- sklearn==1.0.1

- matplotlib==3.5.3

- pandas==1.3.4

- mpi4py==3.1.1

You can simply run 

```python
pip install -r requirements.txt
```

## Repository structure
We select some important files for detailed description.

```python
|-- LightFed # experiments for GradMA and datasets
    |-- experiments/ #
        |-- datasets/ 
            |-- data_distributer.py/  # the load datasets,including MNIST, EMNIST, FMNSIT and CIFAR-10
        |-- horizontal/ ## GradMA, GradMA_S and GradMA_W
            |-- GradMA/
            |-- GradMA_S/
            |-- GradMA_W/
        |-- models
            |-- model.py/  ##load backnone architectures
    |-- lightfed/  
        |-- core # important configure
        |-- tools
```

## Run pipeline for Run pipeline for GradMA
1. Entering the GradMA
```python
cd LightFed
cd experiments
cd horizontal
cd GradMA
```

2. You can run any models implemented in `main_gradma_sw.py`. For examples, you can run our model on `CIFAR-10` dataset by the script:
```python
python main_gradma_sw.py --gamma_l_list 0=0.001,100=0.001,200=0.001  --gamma_g_list 0=1.0,100=1.0,200=1.0 --beta_1_list 0=0.5,100=0.5,200=0.5 --beta_2_list 0=0.5,100=0.5,200=0.5 --batch_size 64 --I 5 --comm_round 1000 --data_partition_mode non_iid_dirichlet --non_iid_alpha 0.01 --client_num 100 --selected_client_num 10 --memory_num 100 --seed 0 --model_type Lenet_5 --data_set CIFAR-10 --eval_batch_size 256 --device cuda --log_level INFO
```

## Citation
If you find this useful for your work, please consider citing:

```
@InProceedings{Luo2023GradMA,
  title = {GradMA: A Gradient-Memory-based Accelerated Federated Learning with Alleviated Catastrophic Forgetting},
  author = {Luo, Kangyang and Li, Xiang and Lan, Yunshi and Gao, Ming},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023}
}
```


