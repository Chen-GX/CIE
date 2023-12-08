import numpy as np
import os, sys
import shutil
import time

os.chdir(sys.path[0])


def run_code(dataset, model, lr, weight_decay, epochs, dropout, idp, co, idp_type, crite, normal_feature, hidden):
    begin_time = time.time()
    k_times = 2
    result_dict = {}
    seeds = np.arange(20)  # 'unbiased', '0.4', '0.6', '0.8', '1.0' / 'unbiased', '0.2', '0.4', '0.6', '0.8'
    # 'unbiased', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'
    # level_dict = {'node': ['unbiased', '0.4', '0.6', '0.8'], 'struc': ['unbiased', '0.4', '0.6', '0.8']}
    level_dict =  ['0.4', '0.6', '0.8']  #   
    save_file_name = f'mixture_result_{dataset}_{idp}_{co}_{idp_type}_{crite}_{weight_decay}_{dropout}_1-12'
    if os.path.exists(save_file_name) == True:
        shutil.rmtree(save_file_name)
    os.makedirs(save_file_name)

    # for bias_type in ['node', 'struc']:  # , 'struc'
    for bias_type in ['mixture']:  #
        # for level in level_dict[bias_type]:  #
        for select_level in level_dict:  #
            for struc_level in level_dict:
                level = select_level + '+' + struc_level
                for k in range(k_times):  # 跑10次随机种子
                    os.system(
                        f'python main_causal.py'
                        f' --dataset {dataset}'
                        f' --idp {idp}'
                        f' --co {co}'
                        f' --save_file_name {save_file_name}'
                        f' --normal_feature {False}'
                        f' --use_scheduler False'
                        f' --lr {lr}'
                        f' --weight_decay {weight_decay}'
                        f' --epochs {epochs}'
                        f' --dropout {dropout}'
                        f' --seed {int(seeds[k] * 10)}'
                        f' --bias_type {bias_type}'
                        f' --level {level}'
                        f' --save_result True'
                        f' --idp_type {idp_type}'
                        f' --crite {crite}'
                        f' --hidden {hidden}'
                    )
                # 统计结果
                with open(f'./{save_file_name}/{dataset}_{model}_{bias_type}_{level}.result', 'r') as f:
                    key = f'{bias_type}_{level}'
                    result_dict[key] = {}
                    result_dict[key]['o'] = []
                    result_dict[key]['c'] = []
                    result_dict[key]['co'] = []
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip().split('\t')
                        result_dict[key]['o'].append(float(line[0]))
                        result_dict[key]['c'].append(float(line[1]))
                        result_dict[key]['co'].append(float(line[2]))
    with open(f'./{save_file_name}/{dataset}_{model}.meanresult', 'w') as f:
        for key, value in result_dict.items():
            f.write('{}'.format(key))
            print('{}'.format(key), end="")
            for k, v in value.items():
                f.write('\t{}: {:5f}\t{:8f}'.format(k, np.mean(v), np.var(v)))
                print('\t{}: {:5f}\t{:8f}'.format(k, np.mean(v), np.var(v)), end="")
            f.write('\n')
            print()

    print("一共运行{:.3f}min".format((time.time() - begin_time) / 60))


if __name__ == '__main__':
    b_time = time.time()
    # 参数区
    dataset = 'Cora'
    model = 'CausalSAGE'
    lr = 0.01
    weight_decay = [1e-3]  # 1e-3
    epochs = 200
    dropout = 0.6
    idp = [0.1, 0.2, 0.3, 0.4, 0.5]  # 
    co = [0.5, 0.5, 0.5, 0.5, 0.5]  #  
    hiddens = [64]
    # normal_features = [False, True]
    idp_type = ['xo']  # , 'o_logs'
    crite = ['acc', 'loss']
    for idp_i in idp:
        for co_i in co:
            for cri in crite:
                for hid in hiddens:
                    # for normal_feature in normal_features:
                    for wd in weight_decay:
                        run_code(dataset, model, lr, wd, epochs, dropout, idp_i, co_i, idp_type[0], cri, False, hid)
    print("一共运行{:.3f}min".format((time.time() - b_time) / 60))
