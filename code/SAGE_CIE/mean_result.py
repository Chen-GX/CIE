import os, sys
import numpy as np

os.chdir(sys.path[0])
dataset = 'Cora'
model = 'CausalGCN'

result_dict = {}
for bias_type in ['node', 'struc']:  #
    for level in ['unbiased', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']:
        with open(f'./result/{dataset}_{model}_{bias_type}_{level}.result', 'r') as f:
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
with open(f'./result/{dataset}_{model}.meanresult', 'w') as f:
    for key, value in result_dict.items():
        f.write('{}'.format(key))
        print('{}'.format(key), end="")
        for k, v in value.items():
            f.write('\t{}: {:5f}\t{:8f}'.format(k, np.mean(v), np.var(v)))
            print('\t{}: {:5f}\t{:8f}'.format(k, np.mean(v), np.var(v)), end="")
        f.write('\n')
        print()
