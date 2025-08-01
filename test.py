# python3 test.py meanetA_3 --gpus=1 --weight= --save --val=DUTS_test

import sys
import importlib
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
import time
from progress.bar import Bar
import os
from collections import OrderedDict
import cv2
from PIL import Image
import numpy as np

from base.framework_factory import load_framework
from base.data import Test_Dataset
from base.metric import *
from base.util import *

from PIL import Image
import numpy as np
import pandas as pd
def test_model(model, test_sets, config, saver=None):
    model.eval()
    st = time.time()
    excel_file_path = './test_results_meanet.xlsx'
    if os.path.exists(excel_file_path) and os.path.getsize(excel_file_path) > 0:
        df_existing = pd.read_excel(excel_file_path, engine='openpyxl')
    else:
        df_existing = pd.DataFrame(columns=['Dataset', 'Max-F', 'Mean-F', 'Fbw', 'MAE', 'SM', 'EM'])    
    results = []

    for set_name, test_set in test_sets.items():
        save_folder = os.path.join(config['save_path'], set_name)
        check_path(save_folder)
        
        titer = test_set.size
        MR = MetricRecorder(titer)
        
        test_bar = Bar('Dataset {:10}:'.format(set_name), max=titer)
        for j in range(titer):
            image, gt, name = test_set.load_data(j)
            Y = model(image.cuda())
            pred = Y['final'][0, 0].sigmoid_().cpu().data.numpy()
            
            out_shape = gt.shape

            h, w = gt.shape[-2], gt.shape[-1]
            dsize = (w, h)
            pred = cv2.resize(pred, dsize, interpolation=cv2.INTER_LINEAR)

            pred, gt = normalize_pil(pred, gt)
            pred = np.clip(np.round(pred * 255) / 255., 0, 1)
            gt = np.squeeze(gt)
            MR.update(pre=pred, gt=gt)
            
            if config['save']:
                fnl_folder = os.path.join(save_folder, 'final')
                check_path(fnl_folder)
                im_path = os.path.join(fnl_folder, name + '.png')
                Image.fromarray((pred * 255)).convert('L').save(im_path)
                
                if saver is not None:
                    saver(Y, gt, name, save_folder, config)
                
            Bar.suffix = '{}/{}'.format(j, titer)
            test_bar.next()
        
        mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)

        results.append([set_name, maxf, meanf, wfm, mae, sm, em])

        print(f'Dataset: {set_name:10}, Max-F: {maxf:.3f}, Mean-F: {meanf:.3f}, Fbw: {wfm:.3f}, MAE: {mae:.3f}, SM: {sm:.3f}, EM: {em:.3f}.')

    df_new = pd.DataFrame(results, columns=['Dataset', 'Max-F', 'Mean-F', 'Fbw', 'MAE', 'SM', 'EM'])
    
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
        df_combined.to_excel(writer, index=False, sheet_name='Results')

        workbook = writer.book
        worksheet = writer.sheets['Results']

        num_format = workbook.add_format({'num_format': '0.000'})

        for col_num in range(1, len(df_combined.columns)):
            worksheet.set_column(col_num, col_num, None, num_format)

    print(f'Test results saved to {excel_file_path}.')
    print('Test using time: {}.'.format(round(time.time() - st, 3)))


def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    config, model, _, _, _, saver = load_framework(net_name)
    print(config)

    saved_model = torch.load(config['weight'], map_location='cpu')
    model.load_state_dict(saved_model)

    test_sets = OrderedDict()
    for set_name in config['vals']:
        test_sets[set_name] = Test_Dataset(name=set_name, config=config)
    
    model = model.cuda()
    
    test_model(model, test_sets, config, saver=saver)
        
if __name__ == "__main__":
    main()