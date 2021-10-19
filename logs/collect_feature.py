import pandas as pd
import os

dataset = 'VLCS'
env_list = [0,1,2,3]
for env in env_list:
    dir_list = ['{}_CORAL_test_env{}'.format(dataset, env), '{}_ERM_test_env{}'.format(dataset, env),
                '{}_GroupDRO_test_env{}'.format(dataset, env),
                '{}_IRM_test_env{}'.format(dataset, env),
                '{}_Mixup_test_env{}'.format(dataset, env)
        ]
    before = False
    for before in [True]:
        print(before)
        output_file = 'data_collection_before_{}_env{}.csv'.format(dataset, env) if before \
            else 'data_collection_{}_env{}.csv'.format(dataset, env)
        dt = None

        for dir0 in dir_list:
            print(dir0)
            assert dataset in dir0, 'not correct dataset'
            assert str(env) in dir0, 'not correct env'
            if not os.path.isdir(dir0) or 'env' not in dir0:
                continue
            file_list = sorted(os.listdir(dir0))
            for sub_name in file_list:
                if not os.path.isdir(os.path.join(dir0, sub_name)):
                    continue
                sub_file_list = os.listdir(os.path.join(dir0, sub_name))
                try:
                    fname = 'before_result.csv' if before else 'result.csv'
                    res = pd.read_csv(os.path.join(dir0, sub_name, fname), index_col=False)
                    if dt is None:
                        dt = res
                    else:
                        dt = pd.concat([dt, res], axis=0)
                        #print('count')
                except FileNotFoundError as e:
                    print(e)
                    print(dir0, sub_name)
        if dt is not None:
            dt.to_csv(output_file, index=False)
        print(dt)


