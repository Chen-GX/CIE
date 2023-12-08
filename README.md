# Causality and Independence Enhancement for Biased Node Classification (CIKM 2023)

**This repository contains the code for our research paper titled "[Causality and Independence Enhancement for Biased Node Classification](https://dl.acm.org/doi/abs/10.1145/3583780.3614804)".**

> Guoxin Chen, Yongqing Wang, Fangda Guo, Qinglang Guo, Jiangli Shao, Huawei Shen, and Xueqi Cheng. 2023. Causality and Independence Enhancement for Biased Node Classification. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM '23). Association for Computing Machinery, New York, NY, USA, 203–212. https://doi.org/10.1145/3583780.3614804

# Requirements
* Python 3.8
* Ubuntu 22.04
* Python Packages

```
conda create -n cie python=3.8
conda activate cie
pip install -r requirements.txt
```

# Data
The folder `./generate_bias_data` contains the example data for various bias. 
```
cd ./generate_bias_data
unzip *.zip
```

# Training
For a single training:

```
cd CIE_pyg\GCN_CIE

python main_causal.py
```

For searching the optimal parameters:

```
cd CIE_pyg\GCN_CIE

python main_shell.py
```


# Citation
If our work contributes to your research, please acknowledge it by citing our paper. We greatly appreciate your support.

```
@inproceedings{10.1145/3583780.3614804,
    author = {Chen, Guoxin and Wang, Yongqing and Guo, Fangda and Guo, Qinglang and Shao, Jiangli and Shen, Huawei and Cheng, Xueqi},
    title = {Causality and Independence Enhancement for Biased Node Classification},
    year = {2023},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3583780.3614804},
    doi = {10.1145/3583780.3614804},
    booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
    location = {Birmingham, United Kingdom},
    series = {CIKM '23}
}
```
