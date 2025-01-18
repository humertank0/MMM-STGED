# 神经网络智能调参库NNI
import nni


import time
from tqdm import tqdm
import logging
import sys
# 用于解析命令行参数
import argparse
import pandas as pd
import os

# 设置GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.optim as optim
import numpy as np

# 导入自定义模块
from utils.utils import save_json_data, create_dir, load_pkl_data
from common.mbr import MBR
from common.spatial_func import SPoint, distance
from common.road_network import load_rn_shp

from utils.datasets import Dataset, collate_fn


from models.model_utils import load_rn_dict, load_rid_freqs, get_rid_grid, get_poi_info, get_rn_info
from models.model_utils import get_online_info_dict, epoch_time, AttrDict, get_rid_rnfea_dict

from models.multi_train import evaluate, init_weights, train
# 模型组件
from models.model import MM_STGED, DecoderMulti, Encoder
from build_graph import load_graph_adj_mtx, load_graph_node_features
import warnings
import json
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':

    debugger = True
    parser = argparse.ArgumentParser(description='Multi-task Traj Interp')
    parser.add_argument('--dataset', type=str, default='Porto',help='data set')
    parser.add_argument('--module_type', type=str, default='simple', help='module type')
    parser.add_argument('--keep_ratio', type=float, default=0.125, help='keep ratio in float')
    parser.add_argument('--lambda1', type=int, default=10, help='weight for multi task rate')
    parser.add_argument('--hid_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--grid_size', type=int, default=50, help='grid size in int')
    parser.add_argument('--dis_prob_mask_flag', type=bool, default=True, help='flag of using prob mask')
    parser.add_argument('--pro_features_flag', action='store_true', help='flag of using profile features')
    parser.add_argument('--online_features_flag', action='store_true', help='flag of using online features')
    parser.add_argument('--tandem_fea_flag', action='store_true', help='flag of using tandem rid features')
    parser.add_argument('--no_attn_flag', type=bool, default=True, help='flag of using attention')
    parser.add_argument('--load_pretrained_flag', default=False, help='flag of load pretrained model')
    parser.add_argument('--model_old_path', type=str, default='', help='old model path')
    parser.add_argument('--no_debug', type=bool, default=False, help='flag of debug')
    parser.add_argument('--no_train_flag', type=bool, default=True, help='flag of training')
    parser.add_argument('--test_flag', type=bool, default=True, help='flag of testing')
    parser.add_argument('--top_K', type=int, default=10, help='top K value in the decoder')
    parser.add_argument('--RD_inter', type=str, default='1h', help='路况的时间间隔')
    # 解析参数，存储在opts对象中
    opts = parser.parse_args()

    debug = opts.no_debug
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = AttrDict()
    if opts.dataset == 'Beijing':
        args_dict = {
            'module_type':opts.module_type,
            'debug':debug,
            'device':device,

            # pre train
            'load_pretrained_flag':opts.load_pretrained_flag,
            'model_old_path':opts.model_old_path,
            'train_flag':opts.no_train_flag,
            'test_flag':opts.test_flag,

            # attention
            'attn_flag':opts.no_attn_flag,

            # constranit
            'dis_prob_mask_flag':opts.dis_prob_mask_flag,
            'search_dist':50,
            'beta':15,

            # features
            'tandem_fea_flag':opts.tandem_fea_flag,
            'pro_features_flag':opts.pro_features_flag,
            'online_features_flag':opts.online_features_flag,

            # extra info module
            'rid_fea_dim':8,
            'pro_input_dim':25, # 24[hour] + 5[waether] + 1[holiday]  without weather
            'pro_output_dim':8,
            'poi_num':5,
            'online_dim':5+5,  # poi/roadnetwork features dim
            'poi_type':'company,food,shopping,viewpoint,house',
            'user_num': 442, 
            # MBR
            'min_lat':41.142,
            'min_lng':-8.652,
            'max_lat':41.174,
            'max_lng':-8.578,

            # input data params
            'keep_ratio':opts.keep_ratio,
            'grid_size':opts.grid_size,
            'time_span':15,
            'win_size':25,
            'ds_type':'uniform',
            'split_flag':False,
            'shuffle':True,
            'input_dim':3,

            # model params
            'hid_dim':opts.hid_dim,
            'id_emb_dim':128,
            'dropout':0.5,
            'id_size':2224+1,

            'lambda1':opts.lambda1,
            'n_epochs':opts.epochs,
            'top_K': opts.top_K,
            'batch_size':128,
            'learning_rate':1e-3,
            'tf_ratio':0.5,
            'clip':1,
            'log_step':1
        }
    elif opts.dataset == 'Chengdu':
        args_dict = {
            'module_type':opts.module_type,
            'debug':debug,
            'device':device,

            # pre train
            'load_pretrained_flag':opts.load_pretrained_flag,
            'model_old_path':opts.model_old_path,
            'train_flag':opts.no_train_flag,
            'test_flag':opts.test_flag,

            # attention
            'attn_flag':opts.no_attn_flag,

            # constranit
            'dis_prob_mask_flag':opts.dis_prob_mask_flag,
            'search_dist':50,
            'beta':15,

            # features
            'tandem_fea_flag':opts.tandem_fea_flag,
            'pro_features_flag':opts.pro_features_flag,
            'online_features_flag':opts.online_features_flag,

            # extra info module
            'rid_fea_dim':8,
            'pro_input_dim':25, # 24[hour] + 5[waether] + 1[holiday]  without weather
            'pro_output_dim':8,
            'poi_num':5,
            'online_dim':5+5,  # poi/roadnetwork features dim
            'poi_type':'company,food,shopping,viewpoint,house',
            'user_num': 17675,

            # MBR
            'min_lat':30.655,
            'min_lng':104.043,
            'max_lat':30.727,
            'max_lng':104.129,

            # input data params
            'keep_ratio':opts.keep_ratio,
            'grid_size':opts.grid_size,
            'time_span':15,
            'win_size':25,
            'ds_type':'uniform',
            'split_flag':False,
            'shuffle':True,
            'input_dim':3,

            # model params
            'hid_dim':opts.hid_dim,
            'id_emb_dim':128,
            'dropout':0.5,
            'id_size':2504+1,

            'lambda1':opts.lambda1,
            'n_epochs':opts.epochs,
            'top_K': opts.top_K,
            'RD_inter': opts.RD_inter,
            'batch_size':128,
            'learning_rate':1e-3,
            'tf_ratio':0.5,
            'clip':1,
            'log_step':1
        }
    
    assert opts.dataset in ['Porto', 'Chengdu', 'Beijing'], 'Check dataset name if in [Porto, Chengdu, Beijing]'

    args.update(args_dict)

    print('Preparing data...')

    train_trajs_dir = "./data/{}/train/".format(opts.dataset)
    valid_trajs_dir = "./data/{}/valid/".format(opts.dataset)
    test_trajs_dir = "./data/{}/test/".format(opts.dataset)

    extra_info_dir = "./data/{}/extra_data/".format(opts.dataset)
    rn_dir = extra_info_dir + "road_network/" # 路网文件
    user_dir = json.load(open( extra_info_dir + "uid2index.json")) # 用户信息文件
    SE_file = extra_info_dir + '{}_SE.txt'.format(opts.dataset) # 空间嵌入文件，表示地理位置信息，如经纬度坐标
    condition_file = extra_info_dir + 'flow.npy' # 道路流量文件
    road_file = extra_info_dir + 'TLG/graph_A.csv' # 道路图文件

    # create model save path
    # 使用串联rid特征
    if args.tandem_fea_flag:
        fea_flag = True
    else:
        fea_flag = False

    model_save_path = './results/'+args.module_type+'_kr_'+str(args.keep_ratio)+'_debug_'+str(args.debug)+\
        '_gs_'+str(args.grid_size)+'_lam_'+str(args.lambda1)+\
        '_attn_'+str(args.attn_flag)+'_prob_'+str(args.dis_prob_mask_flag)+\
        '_fea_'+str(fea_flag)+'_'+time.strftime("%Y%m%d_%H%M%S") + '/'
    create_dir(model_save_path)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=model_save_path + 'log.txt',
                        filemode='a+')
    # spatial embedding
    spatial_A = load_graph_adj_mtx(road_file)
    # 创建全0矩阵，形状为(spatial_A.shape[0]+1, spatial_A.shape[1]+1)，相当于放到右下角
    spatial_A_trans = np.zeros((spatial_A.shape[0]+1, spatial_A.shape[1]+1)) + 1e-10
    spatial_A_trans[1:,1:] = spatial_A

    # 交通流量按时间归一化
    road_condition = np.load(condition_file) # T, N, N
    for i in range(road_condition.shape[0]):
        maxn = road_condition[i].max()
        road_condition[i] = road_condition[i] / maxn

    # 从一个文件中读取???数据，并将其存储在一个 NumPy 数组中
    f = open(SE_file, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0])+1, int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index+1] = temp[1 :]
    SE = torch.from_numpy(SE)

    # 加载道路网络和相关数据
    rn = load_rn_shp(rn_dir, is_directed=True) # 加载shapefile格式的道路网络
    raw_rn_dict = load_rn_dict(extra_info_dir, file_name='raw_rn_dict.json') # 加载原始道路网络字典
    new2raw_rid_dict = load_rid_freqs(extra_info_dir, file_name='new2raw_rid.json') # 字符串到整数字典的映射
    raw2new_rid_dict = load_rid_freqs(extra_info_dir, file_name='raw2new_rid.json')
    rn_dict = load_rn_dict(extra_info_dir, file_name='rn_dict.json') # 加载处理后道路网络字典

    # 创建边界框并进行网格划分
    mbr = MBR(args.min_lat, args.min_lng, args.max_lat, args.max_lng) # mbr: 创建一个边界框（Minimum Bounding Rectangle, MBR），
    # 它定义了地图的地理范围，包括最小纬度 (args.min_lat)、最小经度 (args.min_lng)、最大纬度 (args.max_lat)
    # 和最大经度 (args.max_lng)
    grid_rn_dict, max_xid, max_yid = get_rid_grid(mbr, args.grid_size, rn_dict)
    # 对边界框进行网格划分，并返回一个字典，该字典包含每个网格内的道路ID信息。
    # 同时返回网格的最大X坐标和Y坐标（即 max_xid 和 max_yid）。
    args_dict['max_xid'] = max_xid
    args_dict['max_yid'] = max_yid
    args.update(args_dict)
    print(args)
    logging.info(args_dict)
    with open(model_save_path+'logging.txt', 'a+') as f:
        f.write(str(args_dict))
        f.write('\n')
    # 天气数据未提供
    # load features
    weather_dict = None #load_pkl_data(extra_info_dir, 'weather_dict.pkl')

    # 加载在线特征，数据集没提供
    if args.online_features_flag:
        # 加载POI数据
        grid_poi_df = pd.read_csv(extra_info_dir+'poi'+str(args.grid_size)+'.csv',index_col=[0,1])
        norm_grid_poi_dict = get_poi_info(grid_poi_df, args)
        # 读取道路网格特征
        norm_grid_rnfea_dict = get_rn_info(rn, mbr, args.grid_size, grid_rn_dict, rn_dict)
        # 组合特征
        online_features_dict = get_online_info_dict(grid_rn_dict, norm_grid_poi_dict, norm_grid_rnfea_dict, args)
    else:
        norm_grid_poi_dict, norm_grid_rnfea_dict, online_features_dict = None, None, None
    rid_features_dict = None

    # 数据集创建
    train_dataset = Dataset(train_trajs_dir, user_dir, mbr=mbr, norm_grid_poi_dict=norm_grid_poi_dict,
                            norm_grid_rnfea_dict=norm_grid_rnfea_dict, weather_dict=weather_dict,
                            parameters=args, debug=debug)
    valid_dataset = Dataset(valid_trajs_dir, user_dir, mbr=mbr, norm_grid_poi_dict=norm_grid_poi_dict,
                            norm_grid_rnfea_dict=norm_grid_rnfea_dict, weather_dict=weather_dict,
                            parameters=args, debug=debug)
    test_dataset = Dataset(test_trajs_dir, user_dir, mbr=mbr, norm_grid_poi_dict=norm_grid_poi_dict,
                           norm_grid_rnfea_dict=norm_grid_rnfea_dict, weather_dict=weather_dict,
                           parameters=args, debug=debug)
    print('training dataset shape: ' + str(len(train_dataset))) # 67
    print('validation dataset shape: ' + str(len(valid_dataset))) # 55
    print('test dataset shape: ' + str(len(test_dataset))) # 56

    train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                 shuffle=args.shuffle, collate_fn=collate_fn,
                                                num_workers=4, pin_memory=False)
    valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                                 shuffle=args.shuffle, collate_fn=collate_fn,
                                                num_workers=4, pin_memory=False)
    test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                shuffle=args.shuffle, collate_fn=collate_fn,
                                               num_workers=4, pin_memory=False)

    logging.info('Finish data preparing.')
    logging.info('training dataset shape: ' + str(len(train_dataset)))
    logging.info('validation dataset shape: ' + str(len(valid_dataset)))
    logging.info('test dataset shape: ' + str(len(test_dataset)))

    with open(model_save_path+'logging.txt', 'a+') as f:
        f.write('Finish data preparing.' + '\n')
        f.write('training dataset shape: ' + str(len(train_dataset)) + '\n')
        f.write('validation dataset shape: ' + str(len(valid_dataset)) + '\n')
        f.write('test dataset shape: ' + str(len(test_dataset)) + '\n')


    # 模型设置
    enc = Encoder(args)
    dec = DecoderMulti(args)
    model = MM_STGED(enc, dec, args.hid_dim, args.max_xid, args.max_yid, args.top_K,device).to(device)
    model.apply(init_weights)  # learn how to init weights
    print("pretrained_flag",args.load_pretrained_flag)
    if args.load_pretrained_flag:
        model.load_state_dict(torch.load(args.model_old_path + 'val-best-model.pt'))

    print('model', str(model))
    logging.info('model' + str(model))
    with open(model_save_path+'logging.txt', 'a+') as f:
        f.write('model' + str(model) + '\n')

    if args.train_flag:
        ls_train_loss, ls_train_id_acc1, ls_train_id_recall, ls_train_id_precision, \
        ls_train_rate_loss, ls_train_id_loss = [], [], [], [], [], []
        ls_valid_loss, ls_valid_id_acc1, ls_valid_id_recall, ls_valid_id_precision, \
        ls_valid_dis_mae_loss, ls_valid_dis_rmse_loss = [], [], [], [], [], []
        ls_valid_dis_rn_mae_loss, ls_valid_dis_rn_rmse_loss, ls_valid_rate_loss, ls_valid_id_loss = [], [], [], []

        dict_train_loss = {}
        dict_valid_loss = {}
        best_valid_loss = float('inf')  # compare id loss

        # get all parameters (model parameters + task dependent log variances)
        log_vars = [torch.zeros((1,), requires_grad=True, device=device)] * 2  # use for auto-tune multi-task param
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        for epoch in tqdm(range(args.n_epochs)):
            start_time = time.time()

            new_log_vars, train_loss, train_id_acc1, train_id_recall, train_id_precision, \
            train_rate_loss, train_id_loss = train(model, spatial_A_trans, road_condition, SE, train_iterator, optimizer, log_vars,
                                                   rn_dict, grid_rn_dict, rn, raw2new_rid_dict,
                                                   online_features_dict, rid_features_dict, args)

            valid_id_acc1, valid_id_recall, valid_id_precision, valid_dis_mae_loss, valid_dis_rmse_loss, \
            valid_dis_rn_mae_loss, valid_dis_rn_rmse_loss, \
            valid_rate_loss, valid_id_loss = evaluate(model, spatial_A_trans, road_condition, SE, valid_iterator,
                                                      rn_dict, grid_rn_dict, rn, raw2new_rid_dict,
                                                      online_features_dict, rid_features_dict, raw_rn_dict,
                                                      new2raw_rid_dict, args)
            ls_train_loss.append(train_loss)
            ls_train_id_acc1.append(train_id_acc1)
            ls_train_id_recall.append(train_id_recall)
            ls_train_id_precision.append(train_id_precision)
            ls_train_rate_loss.append(train_rate_loss)
            ls_train_id_loss.append(train_id_loss)

            ls_valid_id_acc1.append(valid_id_acc1)
            ls_valid_id_recall.append(valid_id_recall)
            ls_valid_id_precision.append(valid_id_precision)
            ls_valid_dis_mae_loss.append(valid_dis_mae_loss)
            ls_valid_dis_rmse_loss.append(valid_dis_rmse_loss)
            ls_valid_dis_rn_mae_loss.append(valid_dis_rn_mae_loss)
            ls_valid_dis_rn_rmse_loss.append(valid_dis_rn_rmse_loss)
            ls_valid_rate_loss.append(valid_rate_loss)
            ls_valid_id_loss.append(valid_id_loss)
            valid_loss = valid_rate_loss + valid_id_loss
            ls_valid_loss.append(valid_loss)

            dict_train_loss['train_ttl_loss'] = ls_train_loss
            dict_train_loss['train_id_acc1'] = ls_train_id_acc1
            dict_train_loss['train_id_recall'] = ls_train_id_recall
            dict_train_loss['train_id_precision'] = ls_train_id_precision
            dict_train_loss['train_rate_loss'] = ls_train_rate_loss
            dict_train_loss['train_id_loss'] = ls_train_id_loss

            dict_valid_loss['valid_ttl_loss'] = ls_valid_loss
            dict_valid_loss['valid_id_acc1'] = ls_valid_id_acc1
            dict_valid_loss['valid_id_recall'] = ls_valid_id_recall
            dict_valid_loss['valid_id_precision'] = ls_valid_id_precision
            dict_valid_loss['valid_rate_loss'] = ls_valid_rate_loss
            dict_valid_loss['valid_dis_mae_loss'] = ls_valid_dis_mae_loss
            dict_valid_loss['valid_dis_rmse_loss'] = ls_valid_dis_rmse_loss
            dict_valid_loss['valid_dis_rn_mae_loss'] = ls_valid_dis_rn_mae_loss
            dict_valid_loss['valid_dis_rn_rmse_loss'] = ls_valid_dis_rn_rmse_loss
            dict_valid_loss['valid_id_loss'] = ls_valid_id_loss

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), model_save_path + 'val-best-model.pt')

            if (epoch % args.log_step == 0) or (epoch == args.n_epochs - 1):
                logging.info('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
                weights = [torch.exp(weight) ** 0.5 for weight in new_log_vars]
                logging.info('log_vars:' + str(weights))
                logging.info('\tTrain Loss:' + str(train_loss) +
                             '\tTrain RID Acc1:' + str(train_id_acc1) +
                             '\tTrain RID Recall:' + str(train_id_recall) +
                             '\tTrain RID Precision:' + str(train_id_precision) +
                             '\tTrain Rate Loss:' + str(train_rate_loss) +
                             '\tTrain RID Loss:' + str(train_id_loss))
                logging.info('\tValid Loss:' + str(valid_loss) +
                             '\tValid RID Acc1:' + str(valid_id_acc1) +
                             '\tValid RID Recall:' + str(valid_id_recall) +
                             '\tValid RID Precision:' + str(valid_id_precision) +
                             '\tValid Distance MAE Loss:' + str(valid_dis_mae_loss) +
                             '\tValid Distance RMSE Loss:' + str(valid_dis_rmse_loss) +
                             '\tValid Distance RN MAE Loss:' + str(valid_dis_rn_mae_loss) +
                             '\tValid Distance RN RMSE Loss:' + str(valid_dis_rn_rmse_loss) +
                             '\tValid Rate Loss:' + str(valid_rate_loss) +
                             '\tValid RID Loss:' + str(valid_id_loss))
                with open(model_save_path+'logging.txt', 'a+') as f:
                    f.write('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's' + '\n')
                    f.write('\tTrain Loss:' + str(train_loss) +
                             '\tTrain RID Acc1:' + str(train_id_acc1) +
                             '\tTrain RID Recall:' + str(train_id_recall) +
                             '\tTrain RID Precision:' + str(train_id_precision) +
                             '\tTrain Rate Loss:' + str(train_rate_loss) +
                             '\tTrain RID Loss:' + str(train_id_loss) + 
                             '\n')
                    f.write('\tValid Loss:' + str(valid_loss) +
                             '\tValid RID Acc1:' + str(valid_id_acc1) +
                             '\tValid RID Recall:' + str(valid_id_recall) +
                             '\tValid RID Precision:' + str(valid_id_precision) +
                             '\tValid Distance MAE Loss:' + str(valid_dis_mae_loss) +
                             '\tValid Distance RMSE Loss:' + str(valid_dis_rmse_loss) +
                             '\tValid Distance RN MAE Loss:' + str(valid_dis_rn_mae_loss) +
                             '\tValid Distance RN RMSE Loss:' + str(valid_dis_rn_rmse_loss) +
                             '\tValid Rate Loss:' + str(valid_rate_loss) +
                             '\tValid RID Loss:' + str(valid_id_loss) + 
                             '\n')
                    f.write('\n')
                torch.save(model.state_dict(), model_save_path + 'train-mid-model.pt')
                save_json_data(dict_train_loss, model_save_path, "train_loss.json")
                save_json_data(dict_valid_loss, model_save_path, "valid_loss.json")

    if args.test_flag:
        # model.load_state_dict(torch.load(model_save_path + 'val-best-model.pt'))
        model.load_state_dict(torch.load(model_save_path + 'train-mid-model.pt'))
        start_time = time.time()
        test_id_acc1, test_id_recall, test_id_precision, test_dis_mae_loss, test_dis_rmse_loss, \
        test_dis_rn_mae_loss, test_dis_rn_rmse_loss, test_rate_loss, test_id_loss = evaluate(model, spatial_A_trans, road_condition, SE, test_iterator,
                                                                                             rn_dict, grid_rn_dict, rn,
                                                                                             raw2new_rid_dict,
                                                                                             online_features_dict,
                                                                                             rid_features_dict,
                                                                                             raw_rn_dict, new2raw_rid_dict,
                                                                                             args)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        logging.info('Test Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
        logging.info('\tTest RID Acc1:' + str(test_id_acc1) +
                     '\tTest RID Recall:' + str(test_id_recall) +
                     '\tTest RID Precision:' + str(test_id_precision) +
                     '\tTest Distance MAE Loss:' + str(test_dis_mae_loss) +
                     '\tTest Distance RMSE Loss:' + str(test_dis_rmse_loss) +
                     '\tTest Distance RN MAE Loss:' + str(test_dis_rn_mae_loss) +
                     '\tTest Distance RN RMSE Loss:' + str(test_dis_rn_rmse_loss) +
                     '\tTest Rate Loss:' + str(test_rate_loss) +
                     '\tTest RID Loss:' + str(test_id_loss))
        
        with open(model_save_path+'logging.txt', 'a+') as f:
            f.write("\n")
            f.write('Test Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's' + '\n')
            f.write('\tTest RID Acc1:' + str(test_id_acc1) +
                     '\tTest RID Recall:' + str(test_id_recall) +
                     '\tTest RID Precision:' + str(test_id_precision) +
                     '\tTest Distance MAE Loss:' + str(test_dis_mae_loss) +
                     '\tTest Distance RMSE Loss:' + str(test_dis_rmse_loss) +
                     '\tTest Distance RN MAE Loss:' + str(test_dis_rn_mae_loss) +
                     '\tTest Distance RN RMSE Loss:' + str(test_dis_rn_rmse_loss) +
                     '\tTest Rate Loss:' + str(test_rate_loss) +
                     '\tTest RID Loss:' + str(test_id_loss) +
                     '\n')