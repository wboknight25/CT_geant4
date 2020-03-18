# coding: utf-8

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os, sys, pdb, time, numpy as np, pandas as pd, statsmodels.api as sm
from datetime import datetime,tzinfo,timezone,timedelta
import traceback
import math
import statsmodels.api as sm
import copy

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('precision', 12)

EPSILON=1e-10
#m1_dir_path = '/data/ycz/data_use_min_20190214/'
#trade_price = pd.read_pickle(m1_dir_path + 'm1_adjc')
m5_dir_path = '/pcie/20190301/data_use_m5/'
#trade_price = pd.read_pickle(m5_dir_path + 'm5_adjc')


Tem_raw_data_path = '/pcie/rawdata/auto/'
#risk_factor_path = '/pcie/barra/risk factor0/'
risk_factor_path = '/pcie/barra/ts_risk_factor/'

def get_newest_path():
    paths = np.array(os.listdir(Tem_raw_data_path))
    paths = paths[[True if ele[:4] == 'data' and len(ele) == 12 else False for ele in paths]]
    paths = np.array([int(ele[4:]) for ele in paths])
    return 'data' + str(paths.max()) + '/'

raw_data_path = Tem_raw_data_path + get_newest_path()
# raw_data_path = Tem_raw_data_path + 'data20190327/'

def get_newest_m5_path():
    paths = np.array(os.listdir('/data/m5_rawdata/'))
    paths = paths[[True if ele[:11] == 'm5_rawdata_' and len(ele) == 19 else False for ele in paths]]
    paths = np.array([int(ele[11:]) for ele in paths])
    return 'm5_rawdata_' + str(paths.max()) + '/'

m5_dir_path = '/data/m5_rawdata/' + get_newest_m5_path()
trade_price = pd.read_pickle(m5_dir_path + 'm5_adjc')
m5_amnt = pd.read_pickle(m5_dir_path + 'm5_amnt')

# pdb.set_trace()

#risk_factor = ['BETA', 'BTOP', 'EARNYILD', 'GROWTH', 'LEVERAGE', 'LIQUIDTY', 'MOMENTUM', 'RESVOL', 'SIZE', 'SIZENL', 'COUNTRY']
#industry_factor = ['AERODEF', 'AIRLINE', 'AUTO', 'BANKS', 'BEV', 'BLDPROD', 'CHEM', 'CNSTENG', 'COMSERV', 'CONMAT', 'CONSSERV', 'DVFININS',
#            'ELECEQP', 'ENERGY', 'FOODPROD', 'HDWRSEMI', 'HEALTH', 'HOUSEDUR', 'INDCONG', 'LEISLUX', 'MACH', 'MARINE', 'MATERIAL',
#            'MEDIA', 'MTLMIN', 'PERSPRD', 'RDRLTRAN', 'REALEST', 'RETAIL', 'SOFTWARE', 'TRDDIST', 'UTILITIE']
risk_factor = ['SIZE', 'BETA', 'MOMENTUM', 'RESVOL', 'SIZENL', 'BTOP', 'LIQUIDTY', 'EARNYILD', 'GROWTH', 'LEVERAGE',
      'EARNQUAL', 'EARNVAR', 'INDMOM', 'INVQUAL', 'LTRVSAL', 'PROFIT', 'SEASON', 'STREV', 'COUNTRY', 'DIVYLD']
industry_factor = ['PETROCHEMICAL','COAL','NONFERROUS','ELECPUB','STEEL','CHEMENG','BUILD','BDMATERIAL',
            'LIGHTIND','MACHINE','POWEREQUIP','DEFENSE','CAR','COMRETAIL','FOODTOURISM','HOUSEELEC',
            'TEXTILES','MEDICINE','FOODBEV','AGRICULTURE','BANKS','NONBANK','REALESTATE','TRANSPORTATION',
            'ELECCOMP','COMMUNICATION','COMPUTER','MEDIUM','COMPREHENSIVE']

factor_return = pd.read_pickle(risk_factor_path + 'DlyFctret.pkl')
index_ret = pd.read_pickle(raw_data_path + 'index_ret')
C_price = pd.read_pickle(raw_data_path + 'C_price')
amount = pd.read_pickle(raw_data_path + 'amount')
#O_price_m5_all = pd.read_pickle(m5_dir_path + 'm5_adjo')
ST = pd.read_pickle(raw_data_path + 'ST')
NT = pd.read_pickle(raw_data_path + 'NT')
threshold_st = 0.045
threshold_nst = 0.095
newstk = (ST != 'NL').cumsum(axis = 0) > 1
st_trade = (ST == 'S') & newstk
nst_trade = (ST == 'N') & newstk
nst_tradable = nst_trade & (NT != 1)
st_tradable = (nst_trade | st_trade) & (NT != 1)

all_risk_factor = pd.concat([pd.read_pickle(risk_factor_path + 'Factor_Exposure_14_16.pkl'), pd.read_pickle(risk_factor_path + 'Factor_Exposure_17_now.pkl')])
all_risk_factor['T00018.SH'] = np.nan
all_risk_factor = all_risk_factor[ST.columns]

def get_index_weight(index_name):
    try:
        index_weight = pd.read_pickle(raw_data_path + index_name + '_weight')
    except:
        sys.exit('The benchmark you have inputed is ERROR...')
    index_weight.index = pd.to_datetime(index_weight.index.astype(str))
    return index_weight

def cut_res(data, cut_pos=None, cut_side='right'):
    if cut_pos is None:
        return copy.deepcopy(data).dropna()
    else:
        res = copy.deepcopy(data)
        res = res.dropna()
        if isinstance(cut_pos, str):
            if cut_pos == 'mean':
                cut_pos = res['X'].mean()
            elif cut_pos == 'median':
                cut_pos = res['X'].median()
            elif 'top' in cut_pos:
                sp = cut_pos.split('top')[1]
                if '0.' in sp:
                    res['X'][res['X'].rank(ascending=False, pct=True) > float(sp)] = np.nan
                else:
                    res['X'][res['X'].rank(ascending=False,) > int(sp)] = np.nan
                res = res.dropna()
                return res
            elif 'bot' in cut_pos:
                sp = cut_pos.split('bot')[1]
                if '0.' in sp:
                    res['X'][res['X'].rank(ascending=True, pct=True) > float(sp)] = np.nan
                else:
                    res['X'][res['X'].rank(ascending=True) > int(cut_pos.split('bot')[1])] = np.nan
                res = res.dropna()
                return res
        if isinstance(cut_pos, float) or isinstance(cut_pos, int) or isinstance(cut_pos, np.float32):
            if cut_side == 'right':
                res['X'][res['X']>cut_pos] = np.nan
            else:
                res['X'][res['X']<cut_pos] = np.nan
            res = res.dropna()
            return res
        else:
            print('The input cut param is wrong type!!')
            return None

class Result(object):     # 储存回测结果
    def __init__(self):
        pass

class Backtest(object):
    def __init__(self, weight = None, weight_name = 'NA', result_dir='./', RoP = 0, fund = 100000000, fund_rate = 1, impact_balance = 'fixed', sell_itcpt_impact = 0, sell_slope_impact = 0, buy_itcpt_impact = 0, buy_slope_impact = 0, start = None, end = None, freq = 5, stock_num = 100, ext_num = 300, in_num = 0, max_amount_ratio = 1.0, buy_multiplier=1, sel_mom = 1, top_percent = 0.75, adjust_can_sell = False, benchmark = None, trade_ratio = None, max_ratio_per_stock = None, trade_lost_vs_twap = 10, t0_stock_dict=None, delay_minutes = 0, trade_indexes=None, isweight = False, buyST = False, component = None, comp = 'zz500', trademethod = 'vwap', plot_res = True, optimal_stock = '2', test_res = None, res_path = None, add_component = False, stamptax = 0.001, commision = 0.0003, trade_time_rate = 1, enhance_component=None, enhance_component_weight=0, enhance_cap=None, enhance_cap_weight=0,enhance_cap_weight2=0, barra_dict=None, barra_list=None, barra_list_usage='rebalance', barra_rebalance_num=None, barra_rebalance_th=0.0, barra_rebalance_lookback=20, barra_list_weight=0.0, amnt_lookback_days=1, amnt_max_ratio=None, buy_strategy='equal_sell_num', ext_num_adjust_type=None, ext_num_adjust_score='r2', ext_num_adjust_upper=0.0020, ext_num_adjust_lower=0.0010, ext_num_adjust_freq=5, ext_num_adjust_lookback=5):

        assert impact_balance in ['fixed', 'dynamic_fixed', 'dynamic_growth'], 'ERROR: the variable impact_balance must be valued in ["fixed", "dynamic_fixed", "dynamic_growth"] !!!'
        if weight.index[-1].strftime('%Y-%m-%d') != end:
            print('warnning %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  weight date != end date  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')  
        self.weight = weight.reindex(columns = C_price.columns)
        self.result_dir = result_dir

        self.comp= comp
        self.add_component = add_component
        self.optimal_stock = str(optimal_stock)

        self.stock_num = stock_num
        self.in_num = in_num
        self.ext_num = ext_num
        self.sel_mom = sel_mom
        self.max_amount_ratio = max_amount_ratio
        self.buy_multiplier = buy_multiplier
        self.top_percent = top_percent
        self.adjust_can_sell = adjust_can_sell
        if isinstance(self.stock_num, str):
            if self.stock_num == 'All':
                self.stock_num = 10000
                self.ext_num = self.stock_num
                self.sel_mom = 1
            else:
                print('ERROR: if the paras stock_num be str, it must be valued from ["All"] !!!')
                os._exit()
        self.sel_num = int(self.stock_num * self.sel_mom)

        self.remove_stocks = ['601313.SH', '601360.SH', '001979.SZ', '000024.SZ']

        self.weight_name = weight_name
        self.plot_res = plot_res
        self.start = start
        self.end = end

        self.stamptax = stamptax
        self.commision = commision
        self.impact_balance = impact_balance
        self.sell_itcpt_impact = sell_itcpt_impact
        self.sell_slope_impact = sell_slope_impact
        self.buy_itcpt_impact = buy_itcpt_impact
        self.buy_slope_impact = buy_slope_impact
        self.trade_time_rate = trade_time_rate

        self.RoP = RoP
        self.freq = freq
        if 120 % freq != 0:
            print('Error! freq must be the divisor of 120')
            sys.exit(0)

        self.fund = fund
        self.buyST = buyST
        self.isweight = isweight
        self.RoP_daily = RoP / 250
        self.fund_rate = fund_rate
        self.component = component
        self.benchmark = benchmark
        self.trademethod = trademethod
        self.buy_strategy=buy_strategy

        self.dates = C_price.loc[self.start : self.end].index
        self.times = [d.strftime('%H:%M:%S') for d in pd.date_range('09:30:00', '11:30:00',closed='left', freq=('{}min'.format(self.freq)))] + [d.strftime('%H:%M:%S') for d in pd.date_range('13:00:00', '15:00:00',closed='left', freq=('{}min'.format(self.freq)))]
        self.ori_times = [d.strftime('%H:%M:%S') for d in pd.date_range('09:30:00', '11:30:00',closed='right', freq=('{}min'.format(self.freq)))] + [d.strftime('%H:%M:%S') for d in pd.date_range('13:00:00', '15:00:00',closed='right', freq=('{}min'.format(self.freq)))]
        if trade_indexes:
            if len(trade_indexes) <= 0:
                print('trade_indexes length <=0', len(trade_indexes))
                sys.exit(0)
            self.trade_indexes = trade_indexes
            for i in range(len(trade_indexes)):
                if trade_indexes[i] >= len(self.times):
                    print('trade index maximum', len(self.times), ' trade_indexes over maximum at ', i, 'value', trade_indexes[i])
                    sys.exit(0)
        else:
            self.trade_indexes = list(range(len(self.times)))

        if benchmark == 'IC':
            self.index_name = 'zz500'
        elif benchmark == 'IF':
            self.index_name = 'hs300'
        else:
            self.index_name = benchmark
        
        self.trade_ratio = 1.0 # 一天交易240分钟
        if trade_ratio:
            self.trade_ratio = trade_ratio

        self.max_ratio_per_stock = 2.0 / stock_num
        if max_ratio_per_stock:
            self.max_ratio_per_stock = max_ratio_per_stock

        self.trade_lost_vs_twap = trade_lost_vs_twap

        self.t0_rank_list = []
        self.t0_ratio_list = []
        if t0_stock_dict:
            for key in sorted(t0_stock_dict.keys()):
                self.t0_rank_list.append(key)
                self.t0_ratio_list.append(t0_stock_dict[key])

        self.delay_minutes = delay_minutes
        if delay_minutes % 5 != 0:
            print('Error! delay_minutes%5 != 0')  
            sys.exit(0)

        self.weight = self.weight.loc[self.start:self.end].copy()
        # index的成份股
        
        self.index_weight = get_index_weight(self.comp).reindex(index = self.dates)
        # 可交易的非ST股
        self.nst_tradable = nst_tradable.reindex(index = self.dates).fillna(method='ffill')
        # 可交易的股票（含ST）
        self.st_tradable = st_tradable.reindex(index = self.dates).fillna(method='ffill')
        # index的收益
        self.index_ret = index_ret.reindex(index = self.dates)
        # 收盘价
        self.C_price = C_price.reindex(index = self.dates)
        # 成交金额
        self.amount = amount.reindex(index = self.dates)

        # 交易价格
        self.trade_price = trade_price.loc[self.start:self.end].fillna(method='ffill').copy()
        self.times_index = self.trade_price.index
        self.trade_prices = self.trade_price.rolling(int(self.freq/5)).mean().shift(-int(self.freq/5)-int(self.delay_minutes/5) + 1).fillna(method='ffill')
        #self.trade_prices = self.trade_price.rolling(self.freq).mean().shift(-self.freq + 1).fillna(method='ffill')

        self.amnt_lookback_days = amnt_lookback_days
        self.amnt_max_ratio = amnt_max_ratio
        tmp_start = str(int(self.dates[0].strftime('%Y'))-1) + self.dates[0].strftime('%m%d')
        trade_amnt = m5_amnt.loc[tmp_start:self.end].rolling(int(self.freq/5)).sum().shift(-int(self.freq/5)-int(self.delay_minutes/5) + 1 + 48).fillna(method='ffill')
        self.trade_amnt = trade_amnt.loc[self.start:self.end].copy()
        for i in range(1, self.amnt_lookback_days):
            trade_amnt = trade_amnt.shift(48)
            self.trade_amnt += trade_amnt.loc[self.start:self.end]
        self.trade_amnt /= self.amnt_lookback_days 

        minus_5min = - 5 * pd.offsets.Minute()
        #minus_5min = - 1 * pd.offsets.Minute()
        self.trade_prices.index = self.trade_prices.index.map(lambda x: minus_5min.apply(x))
        self.trade_amnt.index = self.trade_amnt.index.map(lambda x: minus_5min.apply(x))
        #print(self.trade_prices.iloc[10:15,:])    

#        O_price_m5 = O_price_m5_all.reindex(index=self.times_index).fillna(method='ffill')
#        day_close = C_price.shift(1).reindex(index = self.dates)
#        add_575min = 575 * pd.offsets.Minute()
#        day_close.index = day_close.index.map(lambda x: add_575min.apply(x))
#        day_close = day_close.reindex(index=self.times_index).fillna(method='ffill')
#
#        ret = O_price_m5 / day_close - 1
#        nst_trade_m5 = nst_trade.loc[self.start:self.end]
#        nst_trade_m5.index = nst_trade_m5.index.map(lambda x: add_575min.apply(x))
#        nst_trade_m5 = nst_trade_m5.reindex(index=self.times_index).fillna(method='ffill')
#        st_trade_m5 = st_trade.loc[self.start:self.end]
#        st_trade_m5.index = st_trade_m5.index.map(lambda x: add_575min.apply(x))
#        st_trade_m5 = st_trade_m5.reindex(index=self.times_index).fillna(method='ffill')
#
#        zt_m5 = ((ret > threshold_nst) & nst_trade_m5) | ((ret > threshold_st) & st_trade_m5)
#        zt_m5.index = zt_m5.index.map(lambda x: minus_5min.apply(x))
#        dt_m5 = ((ret < -threshold_nst) & nst_trade_m5) | ((ret < -threshold_st) & st_trade_m5)
#        dt_m5.index = dt_m5.index.map(lambda x: minus_5min.apply(x))
#        # 涨停
#        self.zt_m5 = zt_m5.fillna(method='ffill')
#        # 跌停
#        self.dt_m5 = dt_m5.fillna(method='ffill')

        day_close = C_price.shift(1).reindex(index = self.dates)
        add_570min = 570 * pd.offsets.Minute()
        day_close.index = day_close.index.map(lambda x: add_570min.apply(x))
        day_close = day_close.reindex(index=self.trade_prices.index).fillna(method='ffill')
        #print(day_close.iloc[10:15,:])

        ret = self.trade_prices / day_close - 1
        #print(ret.iloc[10:15,:])
        nst_trade_m5 = nst_trade.reindex(index=self.dates)
        nst_trade_m5.index = nst_trade_m5.index.map(lambda x: add_570min.apply(x))
        nst_trade_m5 = nst_trade_m5.reindex(index=self.trade_prices.index).fillna(method='ffill')
        st_trade_m5 = st_trade.reindex(index=self.dates)
        st_trade_m5.index = st_trade_m5.index.map(lambda x: add_570min.apply(x))
        st_trade_m5 = st_trade_m5.reindex(index=self.trade_prices.index).fillna(method='ffill')

        zt_m5 = ((ret > threshold_nst) & nst_trade_m5) | ((ret > threshold_st) & st_trade_m5)
        dt_m5 = ((ret < -threshold_nst) & nst_trade_m5) | ((ret < -threshold_st) & st_trade_m5)
        # 涨停
        self.zt_m5 = zt_m5.fillna(method='ffill')
        #print(self.zt_m5.iloc[10:15,:])
        # 跌停
        self.dt_m5 = dt_m5.fillna(method='ffill')
        #print(self.dt_m5.iloc[10:15,:])


        # pdb.set_trace()
        # 如果weight文件给出的是权重，则进行归一化
        if self.isweight:
            self.weight[self.weight <= 0] = np.nan                  # 不考虑负数权重
            self.weight = self.weight.divide(self.weight.sum(axis = 1), axis = 0)   # 归一化

        self.enhance_component = enhance_component
        self.enhance_component_weight = enhance_component_weight

        if self.enhance_component:
            if self.enhance_component == 'add_constant':
                component_index = self.index_weight > 0
                component_index.index = pd.to_datetime(component_index.index.astype(str) + '15:00', format='%Y-%m-%d%H:%M')
                component_index = component_index.reindex(self.weight.index).fillna(method='bfill', axis=0, limit=47)
                self.weight[component_index] = self.weight[component_index] + self.enhance_component_weight
            elif self.enhance_component == 'mul_constant':
                component_index = self.index_weight > 0
                component_index.index = pd.to_datetime(component_index.index.astype(str) + '15:00', format='%Y-%m-%d%H:%M')
                component_index = component_index.reindex(self.weight.index).fillna(method='bfill', axis=0, limit=47)
                self.weight[component_index] = self.weight[component_index] * (1.0 + self.enhance_component_weight)
            else:
                print('Error!!! enhance_component not valid!!!')
                sys.exit(0)
        else:
            print('Info: enhance_component is None.')

        self.enhance_cap = enhance_cap
        self.enhance_cap_weight = enhance_cap_weight
        self.enhance_cap_weight2 = enhance_cap_weight2

        if self.enhance_cap:
            if self.enhance_cap == 'sub_constant':
                size_tot = pd.read_pickle(raw_data_path + 'size_tot')
                size_tot = size_tot.reindex(index = self.dates)
                size_tot_rank = size_tot.rank(axis=1, ascending = False, method = 'first', na_option = 'keep')
                stock_lens = len(self.weight.columns)
                enhance_value = size_tot_rank.applymap(lambda x: - self.enhance_cap_weight * (max(x, 600) - 600) / (stock_lens - 600))

                enhance_value.index = pd.to_datetime(enhance_value.index.astype(str) + '15:00', format='%Y-%m-%d%H:%M')
                enhance_value = enhance_value.reindex(self.weight.index).fillna(method='bfill', axis=0, limit=47)
                self.weight = self.weight + enhance_value
            elif self.enhance_cap == 'sub_constant_bidirection':
                size_tot = pd.read_pickle(raw_data_path + 'size_tot')
                size_tot = size_tot.reindex(index = self.dates)
                size_tot_rank = size_tot.rank(axis=1, ascending = False, method = 'first', na_option = 'keep')
                stock_lens = len(self.weight.columns)
                enhance_value = size_tot_rank.applymap(lambda x: - self.enhance_cap_weight * (x - 600) / (stock_lens - 600) if x >= 600 else self.enhance_cap_weight2 * (600 - x) / (600 - 1))

                enhance_value.index = pd.to_datetime(enhance_value.index.astype(str) + '15:00', format='%Y-%m-%d%H:%M')
                enhance_value = enhance_value.reindex(self.weight.index).fillna(method='bfill', axis=0, limit=47)
                self.weight = self.weight + enhance_value
            else:
                print('Error!!! enhance_cap not valid!!!')
                sys.exit(0)
        else:
            print('Info: enhance_cap is None.')

        if barra_dict:
            load_risk = all_risk_factor.loc[self.dates[0]:self.dates[-1]]
            risk_factor1 = risk_factor.copy()
            risk_factor1.remove('COUNTRY')
            for key in barra_dict:
                if key in risk_factor1:
                    value = barra_dict[key]
                    tem_load = load_risk.xs(key, level=1)
                    index_mean = (tem_load * self.index_weight).sum(1)
                    enhance_value = tem_load.sub(index_mean, axis='index').abs() * value
                    #print(tem_load.head(5), index_mean.head(5), enhance_value.head(5))
    
                    enhance_value.index = pd.to_datetime(enhance_value.index.astype(str) + '15:00', format='%Y-%m-%d%H:%M')
                    enhance_value = enhance_value.reindex(self.weight.index).fillna(method='bfill', axis=0, limit=47)
                    self.weight = self.weight + enhance_value
                else:
                    print('Error!!! barra_dict not valid!!! Should in ' + '.'.join(risk_factor1))
                    sys.exit(0)
        else:
            print('Info: barra_dict is None.')

        self.barra_list = None
        if barra_list:
            risk_factor1 = risk_factor.copy()
            risk_factor1.remove('COUNTRY')
            for key in barra_list:
                if key not in risk_factor1:
                    print('Error!!! barra_list not valid!!! Should in ' + '.'.join(risk_factor1))
                    sys.exit(0)
            if barra_list_usage not in ['rebalance', 'weight']:
                print('Error!!! barra_list_usage not valid!!! Should in [rebalance, weight]')
                sys.exit(0)
            self.barra_list = barra_list
            self.barra_list_usage = barra_list_usage
            if barra_rebalance_num is None:
                self.barra_rebalance_num = self.stock_num
            else:
                self.barra_rebalance_num = barra_rebalance_num
            self.barra_rebalance_th = barra_rebalance_th
            self.barra_rebalance_lookback = barra_rebalance_lookback
            if self.barra_list_usage == 'rebalance':
                self.barra_rebalance_mean = []
                self.barra_rebalance_std = []
                for key in barra_list:
                    barra_rolling = factor_return[key].loc['20160101':].rolling(barra_rebalance_lookback)
                    self.barra_rebalance_mean.append(barra_rolling.mean().loc[self.start:self.end])
                    self.barra_rebalance_std.append(barra_rolling.std().loc[self.start:self.end])
                #for i in range(len(barra_list)):
                #    print(self.barra_rebalance_mean[i])
                #    print(self.barra_rebalance_std[i])

            self.barra_list_weight = barra_list_weight
        else:
            print('Info: barra_list is None.')

        self.ext_num_adjust_type = None
        if ext_num_adjust_type:
            self.ext_num_adjust_type = ext_num_adjust_type
            self.ext_num_adjust_score = ext_num_adjust_score
            self.ext_num_adjust_upper = ext_num_adjust_upper
            self.ext_num_adjust_lower = ext_num_adjust_lower
            self.ext_num_adjust_freq = ext_num_adjust_freq
            self.ext_num_adjust_lookback = ext_num_adjust_lookback

            ret_shift = int(int(480) / 5)
            m5_ret = (trade_price.shift(-ret_shift - 1) / trade_price.shift( -1) - 1).reindex(index=self.weight.index)
            m5_ret[abs(m5_ret) > 0.1] = np.nan
            zz500 = pd.read_pickle(m5_dir_path + '/index_m5_close')['399905.SZ']
            zz500_ret = (zz500.shift(-ret_shift - 1) / zz500.shift( -1) - 1).reindex(index=self.weight.index)
            self.m5_gold = m5_ret.subtract(zz500_ret, axis=0)
        else:
            print('Info:ext_num_adjust_type is None.')

        # return

        # pdb.set_trace()##################

    def get_prop(self, test_day, test_time, trade_index, oldpool, can_sell_pool, first_open): # 得到调仓之后的相对权重
        if first_open: # 首日建仓
            test_datetime = test_day.strftime('%Y-%m-%d') + ' ' + test_time 
            if test_time == '13:00:00':
                wei = self.weight.loc[test_day.strftime('%Y-%m-%d') + ' 11:30:00'].copy()
            else:
                wei = self.weight.loc[test_datetime].copy()

            # pdb.set_trace()
            pool_all = wei.index[self.st_tradable.loc[test_day]]      # 全市场范围内可交易的股票
            zt_stock = pool_all[self.zt_m5.loc[test_datetime, pool_all]]      # 全市场范围内可交易且涨停的股票
            dt_stock = pool_all[self.dt_m5.loc[test_datetime, pool_all]]      # 全市场范围内可交易且跌停的股票
    
            if self.isweight:
                pool_all = pool_all[wei.loc[pool_all] > 0].copy()
                if len(pool_all) == 0:
                    print('Warnning: The target position is null...')
    
            if not self.buyST: # 可选限定为非ST股
                pool_all = pool_all[self.nst_tradable.loc[test_day, pool_all]]
            if self.component: # 可选限定为成分内
                pool_all = pool_all[self.index_weight.loc[test_day, pool_all] > 0]
    
            if self.add_component: # 是否为成份内选股
                pool_all1 = pool_all[self.index_weight.loc[test_day, pool_all] > 0].copy()
    
            pool_all = list(set(pool_all) - set(zt_stock) - set(self.remove_stocks))           # 可交易、未持仓未涨停的股票（可以买入）
    
            if self.add_component: # 是否添加成份内股票
                pool_all1 = list(set(pool_all1) - set(zt_stock))
    
            wei = wei.loc[pool_all].dropna().sort_index()
            wei1 = wei.rank(ascending = False, method = 'first', na_option = 'keep')    # 将相同因子值的股票按出现顺序排序
            wei2 = wei.rank(ascending = False, method = 'max', na_option = 'keep')
            if self.add_component:
                wei3 = wei.loc[pool_all1].dropna().sort_index().rank(ascending = False, method = 'first', na_option = 'keep')
    
            if self.optimal_stock == '2': # 已经持仓的股票中，如果排名在 ext_num 之内，继续持有
                wei1 = wei1[wei1 <= self.ext_num]
                wei2 = wei2[wei2 <= self.ext_num]
                if len(wei1) != len(wei2):
                    print('  Warnning: %3d equal weight stocks within %3d stocks on %s...'%(self.ext_num - len(wei2), self.ext_num, str(self.test_day)[:10]))
                wei = wei1.sort_values()
                #add_stock = list(wei.index[0 : add_num])
                add_stock = []
                if self.freq <= 5:
                    trade_ratio_per_period = 1.0 / (math.ceil(230 / self.freq) - 1) # 第一天的09:30:00不交易.freq比较小时，最后10分钟不交易
                else:
                    trade_ratio_per_period = 1.0 / (240 / self.freq - 1) # 第一天的09:30:00不交易

                add_stock = list(wei.index[self.in_num : self.in_num + self.stock_num])

#                if self.amnt_max_ratio is None:
#                    add_stock = list(wei.index[self.in_num : self.in_num + self.stock_num])
#                else:
#                    candidate_stock = wei[self.trade_amnt.loc[test_datetime, wei.index[self.in_num:]] > self.fund * trade_ratio_per_period / self.amnt_max_ratio / self.stock_num].index   
#                    candidate_set = set(candidate_stock)
#                    #print(len(candidate_stock), trade_ratio_per_period, self.fund * trade_ratio_per_period / self.amnt_max_ratio / self.stock_num)
#            
#                    for stock in wei.index[self.in_num:]:
#                        if stock in candidate_set:
#                            add_stock.append(stock)
#                            if len(add_stock) >= self.stock_num:
#                                break

                #print(len(add_stock))
                new_stock = add_stock # 目标持仓
    
                if len(new_stock) > 0:
                    after_pool = pd.Series(1.0 / len(new_stock), index = new_stock)
                    buy_pool = after_pool.copy()

                    if self.barra_list:
                        if self.barra_list_usage == 'weight':
                            load_risk = all_risk_factor.loc[test_day.strftime('%Y-%m-%d')]
                            current_risk = 0
                            stock_risk = pd.Series(0, index=add_stock)
                            for key in self.barra_list:
                                tem_load = load_risk.xs(key, level=1)
                                current_risk += (tem_load.iloc[0] * self.prop_res).sum()
                                stock_risk += tem_load.iloc[0][add_stock].fillna(0)
                            stock_risk.sort_values(inplace=True)
                            inc_list = []
                            dec_list = []
                            length = len(add_stock)
                            half_size = int(length / 2)
                            #print(length, half_size, current_risk, self.index_risk, stock_risk)
                            for i in range(half_size):
                                if current_risk > self.index_risk:
                                    if stock_risk[i] < self.index_risk and stock_risk[length - 1 - i] > self.index_risk:
                                        inc_list.append(stock_risk.index[i])
                                        dec_list.append(stock_risk.index[length - 1 - i])
                                    else:
                                        break
                                else:
                                    if stock_risk[i] < self.index_risk and stock_risk[length - 1 - i] > self.index_risk:
                                        inc_list.append(stock_risk.index[length - 1 - i])
                                        dec_list.append(stock_risk.index[i])
                                    else:
                                        break
                            #print(inc_list, dec_list)
                            buy_pool.loc[inc_list] += 1.0 / len(add_stock) * self.barra_list_weight
                            buy_pool.loc[dec_list] -= 1.0 / len(add_stock) * self.barra_list_weight
                        elif self.barra_list_usage == 'rebalance':
                            pass

                    buy_toa = (self.fund * self.buy_slope_impact) / (self.amount.loc[self.test_day, buy_pool.index] * 1000 * self.trade_time_rate * self.freq / 240.0) * 0.0001
                    buy_toa = buy_toa.replace(np.nan, 0)
                    p_trade = after_pool * (1 + self.commision + self.buy_itcpt_impact * 0.0001) / (1 - after_pool * buy_toa)
                    p_trade = p_trade / p_trade.sum()
                    # p_trade经过归一化后本质上new_after_pool就是after_pool / (1 + self.commision + self.stamptax + buy_toa + self.buy_itcpt_impact * 0.0001)
                    trade_weight = (p_trade / (1 + self.commision + buy_toa + (self.buy_itcpt_impact + self.trade_lost_vs_twap) * 0.0001)).sum()
                    new_after_pool = after_pool * trade_weight * trade_ratio_per_period
                    new_after_pool = new_after_pool[new_after_pool>0]
                    #print('new_after_pool sum',new_after_pool.sum())
                    return new_after_pool, pd.Series()
                else:
                    return False, False
            else:
                print('ERROR: The para optimal_stock must be valued in ["2"] !!!')
                return False, False
    
        else: # 非首日换仓的情形
            test_datetime = test_day.strftime('%Y-%m-%d') + ' ' + test_time
            if test_time == '09:30:00':
                wei = self.weight.loc[self.last_day.strftime('%Y-%m-%d') + ' 15:00:00'].copy()
            elif test_time == '13:00:00':
                wei = self.weight.loc[test_day.strftime('%Y-%m-%d') + ' 11:30:00'].copy()
            else:
                wei = self.weight.loc[test_datetime].copy()

#            if self.barra_list:
#                load_risk = all_risk_factor.loc[test_day.strftime('%Y-%m-%d')]
#                x = None
#                for key in self.barra_list:
#                    tem_load = load_risk.xs(key, level=1)
#
#                    if x is None:
#                        x = tem_load
#                    else:
#                        x = pd.concat([x, tem_load])
#                x = x.T
#                result = sm.OLS(wei, sm.add_constant(x), missing='drop').fit()
#                wei = result.resid
#                wei = wei.reindex(index=C_price.columns).fillna(0)

            pool_old = oldpool.index
            pool_all = wei.index[self.st_tradable.loc[test_day]]      # 全市场范围内可交易的股票
            fixed_stock = list(set(pool_old) - set(pool_all))    # 已持仓但不可交易的股票
            zt_stock = pool_all[self.zt_m5.loc[test_datetime, pool_all]]      # 全市场范围内可交易且涨停的股票
            dt_stock = pool_all[self.dt_m5.loc[test_datetime, pool_all]]      # 全市场范围内可交易且跌停的股票
    
            if self.isweight:
                pool_all = pool_all[wei.loc[pool_all] > 0].copy()
                if len(pool_all) == 0:
                    print('Warnning: The target position is null...')
    
            if not self.buyST: # 可选限定为非ST股
                pool_all = pool_all[self.nst_tradable.loc[test_day, pool_all]]
            if self.component: # 可选限定为成分内
                pool_all = pool_all[self.index_weight.loc[test_day, pool_all] > 0]
    
            if self.add_component: # 是否为成份内选股
                pool_all1 = pool_all[self.index_weight.loc[test_day, pool_all] > 0].copy()
    
            pool_all = list(set(pool_all) - (set(zt_stock) - set(pool_old)) - set(self.remove_stocks))     
    
            if self.add_component: # 是否添加成份内股票
                pool_all1 = list(set(pool_all1) - (set(zt_stock) - set(pool_old)))
    
            old_zt_trad_stock = list((set(pool_old) & set(zt_stock)) - set(fixed_stock))    # 已持仓、涨停、可交易的股票（但无法增仓）
            old_dt_trad_stock = list((set(pool_old) & set(dt_stock)) - set(fixed_stock))    # 已持仓、跌停、可交易的股票（但无法卖出）
     
            wei = wei.loc[pool_all].dropna().sort_index()
            wei1 = wei.rank(ascending = False, method = 'first', na_option = 'keep')    # 将相同因子值的股票按出现顺序排序
            wei2 = wei.rank(ascending = False, method = 'max', na_option = 'keep')
            if self.add_component:
                wei3 = wei.loc[pool_all1].dropna().sort_index().rank(ascending = False, method = 'first', na_option = 'keep')
    
            if self.optimal_stock == '2': # 已经持仓的股票中，如果排名在 ext_num 之内，继续持有
                wei1 = wei1[wei1 <= self.ext_num]
                wei2 = wei2[wei2 <= self.ext_num]
                if len(wei1) != len(wei2):
                    print('  Warnning: %3d equal weight stocks within %3d stocks on %s...'%(self.ext_num - len(wei2), self.ext_num, str(self.test_day)[:10]))
                wei = wei1.sort_values()
                limit_stock = wei.index[0 : self.ext_num]
                candidate_stock = list(set(wei.index[self.in_num : self.in_num + self.stock_num]) - set(zt_stock))
                keep_stock = list(set(fixed_stock) | set(old_dt_trad_stock) | (set(pool_old) & set(limit_stock)))
                remove_stock = list((set(pool_old) - set(keep_stock)) & set(can_sell_pool.index))
                remove_num = len(remove_stock)
                if remove_num == 0:
                    #print(test_datetime,'remove_pool is 0')
                    return oldpool,can_sell_pool
                trade_ratio_per_period = self.trade_ratio * self.freq / 240.0

                if self.impact_balance == 'dynamic_growth':
                    fund = self.fund * self.net_value.loc[self.last_day] # 记录当前持仓资金量
                else:
                    fund = self.fund

                left_period = len(self.trade_indexes) - self.trade_indexes.index(trade_index)                

                if self.adjust_can_sell:
                    remove_pool = can_sell_pool.loc[remove_stock] / left_period
                else:
                    remove_pool = np.minimum(can_sell_pool.loc[remove_stock], pd.Series(trade_ratio_per_period / remove_num, index=remove_stock))
                if self.amnt_max_ratio is None:
                    remove_pool = np.minimum(remove_pool, self.max_amount_ratio * (self.amount.loc[self.test_day, remove_stock] * 1000 * self.trade_time_rate * self.freq / 240.0) / fund)
                else:
                    remove_pool = np.minimum(remove_pool, self.amnt_max_ratio * self.trade_amnt.loc[test_datetime, remove_stock] / fund) 
                #print('nan', remove_pool[np.isnan(remove_pool)], can_sell_pool[np.isnan(can_sell_pool)])

                # 先假设股票被卖出，然后再买入目标仓位
                sell_pool = remove_pool.copy()
                #print(test_datetime,'remove_pool sum', sell_pool.sum(), len(sell_pool.index), len(pool_old))
                if sell_pool.sum() < EPSILON:
                    return oldpool,can_sell_pool
                sell_toa_impact = (self.sell_slope_impact * (fund * sell_pool.abs()) / (self.amount.loc[self.test_day, sell_pool.index] * 1000 * self.trade_time_rate * self.freq / 240.0) + self.sell_itcpt_impact + self.trade_lost_vs_twap) * 0.0001
                sell_toa_impact = sell_toa_impact.replace(np.nan, 0)
                total_ratio = (sell_pool * (1 - self.commision - self.stamptax - sell_toa_impact.loc[sell_pool.index])).sum()
                #print(test_datetime,'total_ratio', total_ratio)

                after_pool = oldpool.copy()
                #print(test_datetime, after_pool.loc[remove_stock])
                #print(test_datetime, remove_pool)
                #print(test_datetime,'after_pool[remove_stock] sum', after_pool.loc[remove_stock].sum(), 'remove_pool sum', remove_pool.sum())
                after_pool.loc[remove_stock] = after_pool.loc[remove_stock] - remove_pool
                #print(test_datetime,'oldpool sum', oldpool.sum(), 'remove_pool sum', remove_pool.sum(), 'after_pool sum', after_pool.sum(), 'after_pool[remove_stock] sum', after_pool.loc[remove_stock].sum())

                new_can_sell_pool = can_sell_pool.copy()
                new_can_sell_pool.loc[remove_stock] = new_can_sell_pool.loc[remove_stock] - remove_pool
                new_can_sell_pool = new_can_sell_pool[new_can_sell_pool > EPSILON]

                if self.buy_strategy == 'equal_sell_num':
                    # 买入目标仓位
                    add_num = int(remove_num * self.buy_multiplier)
                    if add_num > len(candidate_stock):
                        add_num = len(candidate_stock)
    
                    comm_stock = list(set(candidate_stock) & set(after_pool.index))
                    candidate_pool = pd.Series(total_ratio / add_num, index=candidate_stock)
                    candidate_pool.loc[comm_stock] = candidate_pool.loc[comm_stock] + after_pool.loc[comm_stock]
                    candidate_pool = candidate_pool[candidate_pool < self.max_ratio_per_stock]
                    candidate_set = set(candidate_pool.index)
    
                    add_stock = []
                    if self.barra_list and self.barra_list_usage == 'rebalance':
                        risk_stocks = list(wei.index[self.in_num: self.in_num + self.barra_rebalance_num])
                        risk_score = pd.Series(0.0, index=risk_stocks)
                        for barra_index, key in enumerate(self.barra_list):
                            tem_load = self.load_risk.xs(key, level=1)
                            current_risk = (tem_load.iloc[0] * after_pool).sum()
                            index_risk = (tem_load.iloc[0] * self.index_weight.loc[self.test_day.strftime('%Y-%m-%d')]).sum()

                            if current_risk > index_risk + self.barra_rebalance_th:
                                tmp_risk = tem_load.iloc[0]
                                #risk_score = risk_score + (index_risk - tmp_risk.reindex(index=risk_stocks)) * (self.barra_rebalance_std[barra_index].loc[self.test_day.strftime('%Y-%m-%d')] + self.barra_rebalance_mean[barra_index].loc[self.test_day.strftime('%Y-%m-%d')])* (current_risk - index_risk)
                                risk_score = risk_score + (index_risk - tmp_risk.reindex(index=risk_stocks)) * self.barra_rebalance_std[barra_index].loc[self.test_day.strftime('%Y-%m-%d')] * (current_risk - index_risk)
                            elif current_risk < index_risk - self.barra_rebalance_th:
                                tmp_risk = tem_load.iloc[0]
                                #risk_score = risk_score + (tmp_risk.reindex(index=risk_stocks) - index_risk) * (self.barra_rebalance_std[barra_index].loc[self.test_day.strftime('%Y-%m-%d')] - self.barra_rebalance_mean[barra_index].loc[self.test_day.strftime('%Y-%m-%d')]) * (index_risk - current_risk)
                                risk_score = risk_score + (tmp_risk.reindex(index=risk_stocks) - index_risk) * self.barra_rebalance_std[barra_index].loc[self.test_day.strftime('%Y-%m-%d')] * (index_risk - current_risk)

                        risk_rank = risk_score.rank(ascending = False, method = 'first', na_option = 'keep')
                        risk_rank = risk_rank.sort_values()

                        if self.amnt_max_ratio is None:
                            for stock in risk_rank.index:
                                if stock in candidate_set:
                                    add_stock.append(stock)
                                    if len(add_stock) >= add_num:
                                        break
                        else:
                            amnt_allow_stock = wei[self.trade_amnt.loc[test_datetime, wei.index[self.in_num:]] > self.fund * total_ratio / add_num / self.amnt_max_ratio].index
                            amnt_allow_set = set(amnt_allow_stock)

                            for stock in  risk_rank.index:
                                if stock in candidate_set and stock in amnt_allow_set:
                                    add_stock.append(stock)
                                    if len(add_stock) >= add_num:
                                         break

                        if len(add_stock) == 0:
                            return oldpool, can_sell_pool
                        buy_pool = pd.Series(total_ratio / len(add_stock), index=add_stock)
                    else:
                        if self.amnt_max_ratio is None:
                            for stock in wei.index[self.in_num: self.in_num + self.stock_num]:
                                if stock in candidate_set:
                                    add_stock.append(stock)
                                    if len(add_stock) >= add_num:
                                        break
                        else:
                            amnt_allow_stock = wei[self.trade_amnt.loc[test_datetime, wei.index[self.in_num:]] > self.fund * total_ratio / add_num / self.amnt_max_ratio].index   
                            amnt_allow_set = set(amnt_allow_stock)
                        
                            for stock in wei.index[self.in_num: self.in_num + self.stock_num]:
                                if stock in candidate_set and stock in amnt_allow_set:
                                    add_stock.append(stock)
                                    if len(add_stock) >= add_num:
                                        break
                        if len(add_stock) == 0:
                            return oldpool, can_sell_pool
                        buy_pool = pd.Series(total_ratio / len(add_stock), index=add_stock)
                        #print(test_datetime,'buy_pool sum', buy_pool.sum(), len(add_stock), add_stock)

                elif self.buy_strategy == 'fill_top':
                    add_stock = []
                    fill_weight_list = []

                    candidate_stock = wei.index[self.in_num: self.in_num + self.stock_num]
                    comm_stock = list(set(candidate_stock) & set(after_pool.index))
                    candidate_pool = pd.Series(self.max_ratio_per_stock, index=candidate_stock)
                    candidate_pool.loc[comm_stock] = candidate_pool.loc[comm_stock] - after_pool.loc[comm_stock]

                    if self.amnt_max_ratio is not None:
                        amnt_allow_pool = self.trade_amnt.loc[test_datetime, candidate_stock] * self.amnt_max_ratio / self.fund
                        candidate_pool = np.minimum(candidate_pool, amnt_allow_pool)

                    left_ratio = total_ratio
                    for stock in wei.index[self.in_num: self.in_num + self.stock_num]:
                        fill_weight = candidate_pool[stock]
                        if fill_weight < EPSILON:
                            continue
                        else:
                            if left_ratio < fill_weight:
                                add_stock.append(stock)
                                fill_weight_list.append(left_ratio)
                                break
                            else: 
                                add_stock.append(stock)
                                fill_weight_list.append(fill_weight)
                                left_ratio -= fill_weight

                    if len(add_stock) == 0:
                        return oldpool, can_sell_pool
                    buy_pool = pd.Series(np.array(fill_weight_list), index=add_stock)
                    #print(total_ratio,buy_pool.sum())
                else:
                    print('Error buy_stategy')
                    return oldpool, can_sell_pool
                    
                if buy_pool.sum() < EPSILON:
                    return oldpool,can_sell_pool

                if self.barra_list and self.barra_list_usage == 'weight':
                    current_risk = 0
                    stock_risk = pd.Series(0, index=add_stock)
                    for key in self.barra_list:
                        tem_load = self.load_risk.xs(key, level=1)
                        current_risk += (tem_load.iloc[0] * after_pool).sum()
                        stock_risk += tem_load.iloc[0][add_stock].fillna(0)
                    stock_risk.sort_values(inplace=True)
                    inc_list = []
                    dec_list = []
                    length = len(add_stock)
                    half_size = int(length / 2)
                    for i in range(half_size):
                        if current_risk > self.index_risk:
                            if stock_risk[i] < self.index_risk and stock_risk[length - 1 - i] > self.index_risk:
                                inc_list.append(stock_risk.index[i])
                                dec_list.append(stock_risk.index[length - 1 - i])
                            else:
                                break
                        else:
                            if stock_risk[i] < self.index_risk and stock_risk[length - 1 - i] > self.index_risk:
                                inc_list.append(stock_risk.index[length - 1 - i])
                                dec_list.append(stock_risk.index[i])
                            else:
                                break
                    buy_pool.loc[inc_list] += total_ratio / len(add_stock) * self.barra_list_weight
                    buy_pool.loc[dec_list] -= total_ratio / len(add_stock) * self.barra_list_weight 

                buy_toa_impact = (self.buy_slope_impact *  (fund * buy_pool.abs()) / (self.amount.loc[self.test_day, add_stock] * 1000 * self.trade_time_rate * self.freq / 240.0) + self.buy_itcpt_impact + self.trade_lost_vs_twap) * 0.0001
                buy_toa_impact = buy_toa_impact.replace(np.nan, 0)
                buy_pool = buy_pool / (1 + self.commision + buy_toa_impact.loc[add_stock])
                #print(test_datetime,'buy_pool sum2', buy_pool.sum())
                self.buy_ratio.extend(((fund * buy_pool) / (self.amount.loc[self.test_day, buy_pool.index] * 1000 * self.trade_time_rate * self.freq / 240.0)).tolist())
        
                #print(test_datetime,'after_pool sum', after_pool.sum())
                after_pool = after_pool[after_pool>0]
                #print(test_datetime,'after_pool sum2', after_pool.sum())
                combine_stock = list(set(after_pool.index) | set(buy_pool.index))
                after_pool = after_pool.reindex(index=combine_stock).fillna(0)
                buy_pool = buy_pool.reindex(index=combine_stock).fillna(0)

                prop_res = after_pool + buy_pool
                #print(test_datetime,'prop_res sum', prop_res.sum(), 'oldpool', oldpool.sum())

                self.sell_ratio.extend(((fund * sell_pool) / (self.amount.loc[self.test_day, sell_pool.index] * 1000 * self.trade_time_rate * self.freq / 240.0)).tolist())

                return prop_res, new_can_sell_pool
            else:
                print('ERROR: The para optimal_stock must be valued in ["2"] !!!')
                return False, pd.Series()

    def calc_turnover(self, prop_before, prop_after): # 计算换手率
        if isinstance(prop_before, pd.Series) and isinstance(prop_after, pd.Series):
            _before, _after = prop_before.copy(), prop_after.copy()
            _before = _before.reindex(index = _after.index).fillna(0)
            _after = _after.reindex(index = _before.index).fillna(0)
            return (_after - _before).abs().sum() / 2
        else:
            return 0

    def run(self):
        self.run = True
        self.cost = 0
        self.turnover = []
        self.fail_list = []
        self.net_value = pd.Series([1] * (len(self.dates) + 1), index = ['origin_net'] + [ele for ele in self.dates])
        self.my_hedge_net_value = self.net_value.copy()
        self.weight_list = []
        self.ccost = []
        self.can_sell_pool = pd.Series()
        self.prop_res = pd.Series()
        self.stockpool = None
        self.first_open = True
        self.buy_ratio = []
        self.sell_ratio = []
        self.max_buy_ratio = []
        self.max_sell_ratio = []
        self.top_buy_ratio = []
        self.top_sell_ratio = []
        self.avg_buy_ratio = []
        self.avg_sell_ratio = []

        self.all_delay_trade_days = 0 # 所有在交易日没有成功交易的次数
        self.last_day = self.dates[0]
        # print("**Backtesting: %s yr**" %self.last_day.year)
        self.total_days = 0

        for date_index, date in enumerate(self.dates):
            self.test_day = date
            #print('MarketOpen',self.test_day, self.first_open)
            if (self.test_day.year != self.last_day.year):
                # print("**Backtesting: %s yr  %02d mt**" %(self.dates[day].year, self.dates[day].month))
                print("**Backtesting: %s yr**" %self.test_day.year)

            day_turnover = 0.0
            day_cost = 0.0
            day_ccost = 0.0
            net_val_cur = 0.0
            first_trade = True
            if not self.first_open:
                net_val_cur = self.net_value.loc[self.last_day]
                wei = self.weight.loc[self.last_day.strftime('%Y-%m-%d') + ' 15:00:00'].copy()
                wei = wei.dropna().sort_index()
                wei1 = wei.rank(ascending = False, method = 'first', na_option = 'keep')    # 将相同因子值的股票按出现顺序排序
                wei = wei1.sort_values()
                tmp_index = 0
                for i in range(len(self.t0_rank_list)):
                    common_stock = list(set(wei.index[tmp_index: self.t0_rank_list[i]]) & set(self.can_sell_pool.index))
                    self.can_sell_pool.loc[common_stock] = self.can_sell_pool.loc[common_stock] * (1.0 - self.t0_ratio_list[i])
                    tmp_index = self.t0_rank_list[i]
                self.can_sell_pool = self.can_sell_pool[self.can_sell_pool > 0]

            if self.barra_list:
                self.load_risk = all_risk_factor.loc[self.test_day.strftime('%Y-%m-%d')]
                self.index_risk = 0
                for key in self.barra_list:
                    tem_load = self.load_risk.xs(key, level=1)
                    self.index_risk += (tem_load.iloc[0] * self.index_weight.loc[self.test_day.strftime('%Y-%m-%d')]).sum()
 


            for i in range(len(self.times)):
                time = self.times[i]
                test_datetime = pd.to_datetime(self.test_day.strftime('%Y-%m-%d') + ' ' + time)
                if self.first_open == True: # 首次建仓
                    if i == 0:
                        continue
                    prop_res, can_sell_pool = self.get_prop(self.test_day, time, i, pd.Series(), pd.Series(), self.first_open)
                    #print(self.test_day, i, prop_res)
                    # pdb.set_trace()
                    ret_stk = self.C_price.loc[self.test_day, prop_res.index] / self.trade_prices.loc[test_datetime, prop_res.index] # 今交到今收
                    ret_stk = ret_stk.replace(np.nan, 1)

                    if prop_res is not False:
                        net_val_cur += np.dot(prop_res, ret_stk)
                        #print(test_datetime, net_val_cur)
                        #if np.isnan(net_val_cur):
                        #    nanstocks = list(set(prop_res[np.isnan(prop_res)].index) | set(ret_stk[np.isnan(ret_stk)].index))
                        #    print(nanstocks)
                        #    print(prop_res.loc[nanstocks])
                        #    print(ret_stk.loc[nanstocks])
                        #    print(self.C_price.loc[self.test_day, nanstocks])
                        #    print(self.trade_prices.loc[test_datetime, nanstocks])
                            
                        combine_stock = list(set(prop_res.index) | set(self.prop_res.index))
                        prop_res = prop_res.reindex(index=combine_stock).fillna(0)
                        self.prop_res = self.prop_res.reindex(index=combine_stock).fillna(0)
                        self.prop_res = self.prop_res + prop_res
                    else:
                        print(self.test_day, 'first_open Error!')

                    if time > '14:49:00' or i == len(self.times) - 1:
                        #print(self.test_day, self.prop_res)
                        day_cost = 1.0 - 1.0 / (1 + self.commision + (self.buy_itcpt_impact + self.trade_lost_vs_twap) * 0.0001)
                        day_ccost = 1.0 - 1.0 / (1 + self.commision + (self.buy_itcpt_impact + self.trade_lost_vs_twap) * 0.0001)
                        day_turnover = 0.5
                        self.prop_res = self.prop_res / self.prop_res.sum()
                        self.last_datetime = test_datetime
                        break
                else:
                    if i not in self.trade_indexes:
                        continue
                    if first_trade:
                        ret_stk = self.trade_prices.loc[test_datetime, self.stockpool] / self.C_price.loc[self.last_day, self.stockpool] # 昨收到今交
                        first_trade = False
                    else:
                        #if test_datetime not in self.trade_prices.index:
                        #    print('Error test_datetime', test_datetime)
                        #elif len(set(self.stockpool) - set(self.trade_prices.columns)) != 0:
                        #    print('Error stock!', print(set(self.stockpool) - set(self.trade_prices.columns)))
                        ret_stk = self.trade_prices.loc[test_datetime, self.stockpool] / self.trade_prices.loc[self.last_datetime, self.stockpool] # 昨收到今交
                
                    #isNan = np.isnan(ret_stk)
                    #if isNan.sum() != 0:
                    #    print(test_datetime, 'Nan',np.where(isNan))
                    #dt_stk = ret_stk[ret_stk < 0.95]
                    #if len(dt_stk.index) > 0:
                    #    print(test_datetime, 'dt_stk', dt_stk)

                    #net_val_cur = net_val_cur * np.dot(self.prop_res, ret_stk)
                    #self.prop_res = self.prop_res * ret_stk
                    #prop_sum = self.prop_res.sum()
                    #self.prop_res = self.prop_res / prop_sum

                    #self.can_sell_pool = self.can_sell_pool * ret_stk.loc[self.can_sell_pool.index]
                    #self.can_sell_pool = self.can_sell_pool / prop_sum

                    #print('ret_nan',ret_stk[np.isnan(ret_stk)])
                    ret_stk = ret_stk.replace(np.nan, 1)
                    #print(time, 'oriProp res sum', self.prop_res.sum(), 'can_sell_pool sum', self.can_sell_pool.sum())
                    self.prop_res = self.prop_res * ret_stk
                    self.can_sell_pool = self.can_sell_pool * ret_stk.loc[self.can_sell_pool.index]
                    #print(time, 'ret_stk after prop_res sum', self.prop_res.sum(), 'can_sell_pool sum', self.can_sell_pool.sum())
                    
                    prop_res, can_sell_pool = self.get_prop(self.test_day, time, i, self.prop_res, self.can_sell_pool, self.first_open)

                    if prop_res is not False: # 正常调仓
                        day_cost += net_val_cur * (self.prop_res.sum() - prop_res.sum())
                        day_ccost += self.prop_res.sum() - prop_res.sum()
                        day_turnover += self.calc_turnover(self.prop_res, prop_res) # 记录每日换手率
                        self.prop_res = prop_res.copy()
                        self.can_sell_pool = can_sell_pool.copy()
                        self.stockpool = self.prop_res.index
                    else: # 无法正常调仓，顺延到下一交易日继续调仓
                        print(self.test_day, 'get_prop Error!')

                    if time > '14:49:00' or i == len(self.times) - 1 or i == self.trade_indexes[len(self.trade_indexes) - 1]:
                        ret_stk = self.C_price.loc[self.test_day, self.stockpool] / self.trade_prices.loc[test_datetime, self.stockpool] # 今交到今收
                        ret_stk = ret_stk.replace(np.nan, 1)
                        self.prop_res = self.prop_res * ret_stk
                        net_val_cur = net_val_cur * self.prop_res.sum()
                        self.prop_res = self.prop_res / self.prop_res.sum()
                        self.stockpool = self.prop_res.index
                        self.last_datetime = test_datetime
                        break
                self.last_datetime = test_datetime
                    
            self.turnover.append([self.test_day, day_turnover]) # 记录每日换手率
            if self.first_open:
                self.net_value.loc[self.test_day] = 1.0
                self.my_hedge_net_value.loc[self.test_day] = 1.0
                print(self.test_day, 'MarketClose. net_val 1.0 hedge_net_val 1.0. first_open_val', round(net_val_cur,4))
            else:
                self.net_value.loc[self.test_day] = net_val_cur
                tmp_portfolio_ret = self.net_value.loc[self.test_day] / self.net_value.loc[self.last_day] - 1
                tmp_index_ret = self.index_ret.loc[self.test_day, self.benchmark+'_ret']
                hedge_ret_today =(tmp_portfolio_ret - tmp_index_ret) * self.fund_rate
                hedge_ret = self.my_hedge_net_value.loc[self.last_day] * (1.0 + hedge_ret_today)
                #hedge_ret = self.my_hedge_net_value.loc[self.last_day] * (1.0 + (tmp_portfolio_ret - tmp_index_ret) * self.fund_rate)
                self.my_hedge_net_value.loc[self.test_day] = hedge_ret
                #print(self.test_day, 'MarketClose. net_val', net_val_cur, 'hedge_net_val', hedge_ret)
                print(self.test_day, 'MarketClose. net_val', round(net_val_cur,4), 'hedge_net_val', round(hedge_ret,4), 'hedge_return', '%.2f%%' % (hedge_ret_today*100))
            if self.prop_res is not False:
                if len(self.buy_ratio) != 0:
                    ratio_array = np.array(self.buy_ratio)
                    self.max_buy_ratio.append(np.nanmax(ratio_array))
                    max_size = int(self.top_percent * len(self.buy_ratio))
                    max_array = ratio_array[np.argpartition(ratio_array, -max_size)[-max_size:]]
                    self.top_buy_ratio.append(np.nanmean(max_array))
                    self.avg_buy_ratio.append(np.nanmean(ratio_array))
                else:
                    self.max_buy_ratio.append(0)
                    self.top_buy_ratio.append(0)
                    self.avg_buy_ratio.append(0)
                if len(self.sell_ratio) != 0:
                    ratio_array = np.array(self.sell_ratio)
                    self.max_sell_ratio.append(np.nanmax(ratio_array))
                    max_size = int(self.top_percent * len(self.sell_ratio))
                    max_array = ratio_array[np.argpartition(ratio_array, -max_size)[-max_size:]]
                    self.top_sell_ratio.append(np.nanmean(max_array))
                    self.avg_sell_ratio.append(np.nanmean(ratio_array))
                else:
                    self.max_sell_ratio.append(0)
                    self.top_sell_ratio.append(0)
                    self.avg_sell_ratio.append(0)

                self.cost += day_cost
                self.ccost.append(day_ccost)
                self.weight_list.append(self.prop_res)
                self.stockpool = self.prop_res.index
                self.can_sell_pool = self.prop_res.copy()
                if self.first_open:
                    self.run_start = self.test_day
                    self.first_open = False
            else:
                print('Error Trade Day')
                self.fail_list.append(self.test_day)
                self.all_delay_trade_days += 1
                self.ccost.append(0)

            self.buy_ratio.clear()
            self.sell_ratio.clear()
            self.last_day = self.test_day

            self.total_days += 1
            if self.ext_num_adjust_type:
                if self.total_days % self.ext_num_adjust_freq == 0: 
                    if self.ext_num_adjust_score == 'r2':
                        tmp_datetimes = []
                        min_index = max(0, date_index - self.ext_num_adjust_lookback)
                        for tmp_date in self.dates[min_index:date_index]:
                            tmp_datetimes.extend([tmp_date.strftime('%Y%m%d') + tmp_time for tmp_time in self.ori_times])
                        tmp_index = pd.to_datetime(tmp_datetimes, format='%Y%m%d%H:%M:%S')
    
                        tmp_weight = self.weight.reindex(index=tmp_index)
                        tmp_ret = self.m5_gold.reindex(index=tmp_index)
    
                        tmp_res = pd.DataFrame()
                        tmp_res['X'] = tmp_weight.values.flatten()
                        tmp_res['Y'] = tmp_ret.values.flatten()
                        
                        tmp_res = cut_res(tmp_res, 'top0.5', 'left')
                        model = sm.OLS(tmp_res['Y'], sm.add_constant(tmp_res['X']))
                        ress = model.fit()
                        r2 = ress.rsquared
                        ori_ext_num = self.ext_num
                        if self.ext_num_adjust_type == 'trisection':
                            if r2 >= self.ext_num_adjust_upper:
                                self.ext_num = 2000
                            elif r2 <= self.ext_num_adjust_lower:
                                self.ext_num = 3000
                            else:
                                self.ext_num = 2500
                        elif self.ext_num_adjust_type == 'linear':
                            if r2 >= self.ext_num_adjust_upper:
                                self.ext_num = 2000
                            elif r2 <= self.ext_num_adjust_lower:
                                self.ext_num = 3000
                            else:
                                step =  int(10 * (r2 - self.ext_num_adjust_lower) / (self.ext_num_adjust_upper - self.ext_num_adjust_lower) + 0.5)
                                self.ext_num = 3000 - 100 * step
                        else:
                            print('unsupported ext_num_adjust_type:', self.ext_num_adjust_type)
                            sys.exit(0)
                        print(self.test_day, 'r2:', r2, 'ori ext_num:', ori_ext_num, 'new ext_num:', self.ext_num)
                    else:
                        print('unsupported ext_num_adjust_score:', self.ext_num_adjust_score)
                        sys.exit(0)
                
        self.run_end = self.test_day
        try:
            self.dates = C_price.loc[self.run_start : self.run_end].index
            self.weight_list = pd.DataFrame(self.weight_list, index = self.dates).reindex(columns = C_price.columns)
            self.weight_list.to_csv(self.result_dir + self.weight_name + '_weight.csv')       
        except Exception as e:
            print(traceback.format_exc())
            self.run = False


    def result(self): # 计算回测的一些指标
        if self.run:
            self.Portfolio_ret = (self.net_value / self.net_value.shift(1) - 1).loc[self.dates]
            self.Portfolio_ret.loc[self.dates[0]] = 0.0
            banchmark_ret = self.benchmark + '_ret'
            self.Index_ret = self.index_ret.loc[self.dates, banchmark_ret]
            self.Index_ret.loc[self.dates[0]] = 0.0
            self.Hedge_ret = (self.Portfolio_ret - self.Index_ret + self.RoP_daily) * self.fund_rate

            hedge_net_val = (self.Hedge_ret + 1).cumprod()

            draw_down = 1 - hedge_net_val / hedge_net_val.cummax()
            Max_dd = draw_down.max()

            final_val = hedge_net_val[self.dates[-1]]
            final_val_addcost = final_val + self.cost

            ret_year = np.power(final_val, 250 / len(self.dates)) - 1
            ret_year_addcost = np.power(final_val_addcost, 250 / len(self.dates)) - 1
            self.cost = 1 - (ret_year + 1) / (ret_year_addcost + 1)

            vol = self.Hedge_ret.std() * np.sqrt(250)
            down_vol = self.Hedge_ret[self.Hedge_ret <= 0].std() * np.sqrt(250)

            sharp_ratio = self.Hedge_ret.mean() * 250 / vol
            adj_sharp_ratio = (self.Hedge_ret.mean() * 250 - 0.12) / vol
            sortino_ratio = self.Hedge_ret.mean() * 250 / down_vol

            turnover = pd.DataFrame(self.turnover)
            turnover.index = turnover[0]
            turnover = turnover.drop(0, axis = 1)[1]
            avg_turnover = turnover[1:].mean()
            turnover.columns = ['turnover']
            self.turnover = turnover.reindex(index = self.dates).fillna(0)
            turnover = self.turnover.copy()
            turnover[0] = 0

            res = Result()

            trade_stock_num = self.weight_list.count(1) - (~np.isnan(self.weight_list + self.weight_list.shift(1))).sum(1)
            res.ptf_df = pd.DataFrame({'Portfolio_ret':self.Portfolio_ret, 'Index_ret':self.Index_ret, 'Hedge_ret':self.Hedge_ret, 'Hedge_net_val':hedge_net_val, 'Drawdown':draw_down, 'Trade_stock_num':trade_stock_num, 'Turnover':turnover}, columns=['Portfolio_ret', 'Index_ret', 'Hedge_ret', 'Hedge_net_val', 'Drawdown', 'Trade_stock_num', 'Turnover'], index = self.dates)

            if self.plot_res:
                res.ptf_df[['Hedge_net_val', 'Drawdown']].plot(kind = 'line', color = 'bg', secondary_y = ['Drawdown'], figsize = (16, 6), title = 'Hedge_net_value and Drawdown')
                plt.grid(linestyle = 'dashdot')
                plt.show()
                plt.savefig(self.result_dir + self.weight_name + '_Hedge_net_val.png')
                plt.close()
                plt.figure(figsize = (16, 6))
                res.ptf_df['Turnover'].plot(color = 'y',title = 'Turnover')
                plt.grid(linestyle = 'dashdot')
                plt.show()
                plt.savefig(self.result_dir + self.weight_name + '_Turnover.png')
                plt.close()
                buy_ratio_df = pd.DataFrame({'max_buy_ratio': self.max_buy_ratio, 'top_buy_ratio': self.top_buy_ratio, 'avg_buy_ratio': self.avg_buy_ratio}, columns=['max_buy_ratio', 'top_buy_ratio', 'avg_buy_ratio'], index=self.dates)
                buy_ratio_df[['max_buy_ratio', 'top_buy_ratio', 'avg_buy_ratio']].plot(kind='line', secondary_y = ['top_buy_ratio','avg_buy_ratio'], title='buy_ratio')
                plt.show()
                plt.savefig(self.result_dir + self.weight_name + '_buy_ratio.png')
                plt.close()
                sell_ratio_df = pd.DataFrame({'max_sell_ratio': self.max_sell_ratio, 'top_sell_ratio': self.top_sell_ratio, 'avg_sell_ratio': self.avg_sell_ratio}, columns=['max_sell_ratio', 'top_sell_ratio', 'avg_sell_ratio'], index=self.dates)
                sell_ratio_df[['max_sell_ratio', 'top_sell_ratio', 'avg_sell_ratio']].plot(kind='line', secondary_y = ['top_sell_ratio','avg_sell_ratio'], title='sell_ratio')
                plt.show()
                plt.savefig(self.result_dir + self.weight_name + '_sell_ratio.png')
                plt.close()

            res.weight = self.weight_list
            # pdb.set_trace
            res.fail_list = self.fail_list
            res.turnover = self.turnover
            Data = pd.DataFrame({'Portfolio_ret':self.Portfolio_ret,'Index_ret':self.Index_ret,'Hedge_ret':self.Hedge_ret,'Net_val':self.net_value[1:],'Max_dd':draw_down,'Turnover':turnover},columns=['Portfolio_ret','Index_ret','Hedge_ret','Net_val','Max_dd','Turnover'],index=self.dates)
            Data.to_csv(self.result_dir + self.weight_name + '_data.csv')

            load_risk = all_risk_factor.loc[res.weight.index[0]:res.weight.index[-1]]
            weis = res.weight.replace(np.nan, 0).values
            long_risk = []
            for ele in risk_factor:
                tem_load = load_risk.xs(ele, level = 1).replace(np.nan, 0).values
                long_risk.append((tem_load * weis).sum(1))
            for ele in industry_factor:
                tem_load = load_risk.xs(ele, level = 1)
                long_risk.append((tem_load * weis).sum(1))
            long_risk = pd.DataFrame(long_risk).T
            long_risk.index = self.dates
            long_risk.columns = risk_factor + industry_factor

            self.index_weight = self.index_weight.loc[res.weight.index].loc[res.weight.index]
            weis = self.index_weight.replace(np.nan,0).values
            bench_risk = []
            for ele in risk_factor:
                tem_load = load_risk.xs(ele, level = 1).replace(np.nan, 0).values
                bench_risk.append((tem_load * weis).sum(1))
            for ele in industry_factor:
                tem_load = load_risk.xs(ele, level = 1).replace(np.nan, 0).values
                bench_risk.append((tem_load * weis).sum(1))

            bench_risk = pd.DataFrame(bench_risk).T
            bench_risk.index=self.dates
            bench_risk.columns = risk_factor + industry_factor
            hedge_risk = long_risk - bench_risk
            res.long_risk = long_risk.copy()
            res.bench_risk = bench_risk.copy()
            res.hedge_risk = hedge_risk.copy()

            dis_risk = pd.concat([long_risk.mean(0), bench_risk.mean(0), hedge_risk.mean(0)], axis = 1).T
            dis_risk.index = ['Long', 'Bench', 'Hedge']
            risk_factor1 = risk_factor.copy();risk_factor1.remove('COUNTRY')
            risk_load = dis_risk[risk_factor1]
            industry_load = dis_risk[industry_factor]

            long_risk.index = pd.MultiIndex.from_product([long_risk.index, ['Long']])
            bench_risk.index = pd.MultiIndex.from_product([bench_risk.index, ['Bench']])
            hedge_risk.index = pd.MultiIndex.from_product([hedge_risk.index, ['Hedge']])
            res.all_risk = pd.concat([long_risk, bench_risk, hedge_risk], axis = 0)
            res.all_risk = res.all_risk.reindex(index = pd.MultiIndex.from_product([res.weight.index, ['Long', 'Bench', 'Hedge']]))
            res.risk_load = risk_load
            res.industry_load = industry_load
            res.all_risk.to_csv(self.result_dir + self.weight_name + '_riskload.csv')

            if self.plot_res:
                res.long_risk[risk_factor1].plot(title = 'Risk Load of Long', figsize = (16, 12))
                plt.grid(linestyle = 'dashdot')
                plt.show()
                plt.savefig(self.result_dir + self.weight_name + '_Risk_Load_of_Long.png')
                plt.close()
                res.hedge_risk[risk_factor1].plot(title = 'Risk Load of Hedge', figsize = (16, 12))
                plt.grid(linestyle = 'dashdot')
                plt.show()
                plt.savefig(self.result_dir + self.weight_name + '_Risk_Load_of_Hedge.png')
                plt.close()

            res.risk_return = res.hedge_risk * factor_return.loc[res.hedge_risk.index]
            res.risk_return.to_csv(self.result_dir + self.weight_name + '_riskreturn.csv')
            res.Hedge_ret = self.Hedge_ret
            x = res.risk_return.sum(1)
            x = sm.add_constant(x)
            y = res.Hedge_ret.copy()
            model = sm.OLS(y, x, missing = 'drop')
            ress = model.fit()
            if self.plot_res:
                res.risk_return[risk_factor].cumsum().plot(title = 'Risk Return  (R-squared = %.4f including industry_factor)'%ress.rsquared, figsize = (16, 12))
                plt.grid(linestyle = 'dashdot')
                plt.show()
                plt.savefig(self.result_dir + self.weight_name + '_Risk_Return.png')
                plt.close()

            risk_ret = res.risk_return.sum(1)
            risk_net_val = (risk_ret + 1).cumprod()
            risk_draw_down = 1 - risk_net_val / risk_net_val.cummax()

            pure_ret = self.Hedge_ret - risk_ret
            pure_net_val = (pure_ret + 1).cumprod()
            pure_final_val = pure_net_val[self.dates[-1]]
            pure_ret_year = np.power(pure_final_val, 250 / len(self.dates)) - 1
            pure_vol = pure_ret.std() * np.sqrt(250)
            pure_sharp_ratio = pure_ret.mean() * 250 / pure_vol
            pure_draw_down = 1 - pure_net_val / pure_net_val.cummax()
            pure_max_dd = pure_draw_down.max()

            res.res_df = pd.DataFrame([[ret_year, vol, down_vol, sharp_ratio, adj_sharp_ratio, sortino_ratio, Max_dd, avg_turnover, self.cost, ret_year / abs(Max_dd + 0.0001), self.benchmark, self.Hedge_ret.mean(), self.Hedge_ret.std(), pure_ret_year, pure_sharp_ratio,pure_max_dd]], index = ['Values'], columns = ['return', 'total_vol', 'down_vol', 'sharp', 'adj_sharp', 'sortino', 'Max_dd', 'turnover', 'cost', 'reward-risk', 'benchmark', 'daily_mean', 'daily_std','pure_ret','pure_sharp','pure_max_dd'])

            
            ptf_df = pd.DataFrame({'Risk_net_val':risk_net_val, 'Risk_drawdown':risk_draw_down}, columns=['Risk_net_val','Risk_drawdown'], index=self.dates)
            if self.plot_res:
                ptf_df.plot(kind='line', color='bg', secondary_y=['Risk_drawdown'], figsize=(16,6), title='Hedge_net_value and Drawdown of Risk')
                plt.grid(linestyle = 'dashdot')
                plt.show()
                plt.savefig(self.result_dir + self.weight_name + '_risk_net_val.png')
                plt.close()
        
            ptf_df = pd.DataFrame({'Pure_net_val':pure_net_val, 'Pure_drawdown':pure_draw_down}, columns=['Pure_net_val','Pure_drawdown'], index=self.dates)
            if self.plot_res:
                ptf_df.plot(kind='line', color='bg', secondary_y=['Pure_drawdown'], figsize=(16,6), title='Hedge_net_value and Drawdown of Pure alpha')
                plt.grid(linestyle = 'dashdot')
                plt.show()
                plt.savefig(self.result_dir + self.weight_name + '_pure_net_val.png')
                plt.close()
       
            ptf_df = pd.DataFrame({'Hedge_net_val':hedge_net_val,'Pure_net_val':pure_net_val, 'Risk_net_val':risk_net_val}, columns=['Hedge_net_val','Pure_net_val','Risk_net_val'], index=self.dates)
            if self.plot_res:
                ptf_df.plot(kind='line', color='brg', figsize=(16,6), title='Hedge_net_val of Risk, Pure, Total')
                plt.grid(linestyle = 'dashdot')
                plt.show()
                plt.savefig(self.result_dir + self.weight_name + '_net_val_analysis.png')
                plt.close() 

 
            # pdb.set_trace()
            res.cost = pd.Series(self.ccost, index = self.dates)
            return res
        else:
            return False


def backtest(**kwargs):
    start_time = datetime.now()
    # 新建类，传入参数，运行得到结果
    B_test = Backtest(**kwargs)
    middle_time = datetime.now()
    print('Load Time consuming: %ds\n' %((middle_time - start_time).seconds))
    B_test.run()
    res = B_test.result()
    end_time = datetime.now()
    print('Run Time consuming: %ds\n' %((end_time - middle_time).seconds))
    return res

if __name__ == '__main__': 
#    path = ''
#    weight = pd.read_pickle(path + 'weight.pkl')
    if len(sys.argv) != 1: 
        print("python3 Backtest.py. Set weightFilePath in Backtest.py directly.")
        sys.exit(0)

    date = '20191226'
    model_name = 'ltb_0026'

    weight = pd.read_pickle('/data/m5_weight/ltb/m5_weight_'+model_name+'_'+date+'.pkl')
    result_dir = '/data2/chenxinxiong/delay0/backtest/' + model_name + '/'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    res = backtest(weight = weight, weight_name=model_name+'_'+date+'_ext_num_adjust', result_dir=result_dir, start = '20190701', end = date, benchmark = 'IC', component = False, freq = 15, stock_num = 500, ext_num = 2000, trade_ratio = 1.0, max_ratio_per_stock = None, trade_lost_vs_twap = 10, delay_minutes = 0, optimal_stock = '2', plot_res = True, trade_indexes=[4,5,6,7,8,9,10,11,12,13,14,15], adjust_can_sell=True, fund = 500000000, ext_num_adjust_type='trisection', ext_num_adjust_upper=0.0020, ext_num_adjust_lower=0.0010)
    print(res.res_df)
#    res = backtest(weight = weight, weight_name=model_name+'_'+date, result_dir=result_dir, start = '20190701', end = date, benchmark = 'IC', component = False, freq = 15, stock_num = 500, ext_num = 2000, trade_ratio = 1.0, max_ratio_per_stock = None, trade_lost_vs_twap = 10, delay_minutes = 0, optimal_stock = '2', plot_res = True, trade_indexes=[4,5,6,7,8,9,10,11,12,13,14,15], adjust_can_sell=True, fund = 500000000)
#    print(res.res_df)

'''
回测函数里的参数说明
paras:
        weight（必须）: 所要测试的因子文件（pd.DataFrame）
        weight_name（可选）: 所要测试因子文件的名称（str）
        RoP（必须）：RoP （float） 类似于返佣的一个东西？
        fund（可选）： 测试实盘资金（float）,默认为1个亿
        fund_rate（必须）: 资金使用率（float）,默认资金使用率为100%，注意实际的基金资金使用率只有70%左右（使用期货对冲需要保证金，以及预留5%的资金来调仓）
        impact_balance（必须）： 冲击成本计算类型（str）,有3种取值，分别为fixed固定，dynamic_fixed动态固定、dynamic_growth动态增长
        sell_itcpt_impact（可选）：冲击成本计算参数（float）
        sell_slope_impact（可选）：冲击成本计算参数（float）
        buy_itcpt_impact（可选）：冲击成本计算参数（float）
        buy_slope_impact（可选）：冲击成本计算参数（float）
        start（必须）： 回测开始时间（str）
        end（必须）： 回测结束时间（str）
        freq（必须）： 回测换仓频率，单位为分钟（int）
        stock_num（必须）： 做多买入股票数量（int）
        in_num(必须)：做多股票的起始位置(int)，默认为0
        enhance_component（必须）：增强成分内计算类型(str)。目前有两种取值，1、若为add_constant（固定），则成分内股票权重均加上enhance_component_weight。2、若为mul_constant，则成分内股票权重均乘上(1 + enhance_component_weight)，默认值为None
        enhance_component_weight（必须）：增强成分内股票权重（float），默认值为0.0。
        enhance_cap（必须）：增强市值因子计算类型(str)。目前有两种取值，1、若为sub_constant（固定），则排名600至最后的股票等比例减去enhance_cap_weight，（即市值最小的股票减去enhance_cap_weight，排名600的股票不用减，中间等比例），2、若为sub_constant_bidirection，则排名600至最后的股票等比例减去enhance_cap_weight，排名1至600的股票等比例加上enhance_cap_weight2。默认值为None
        enhance_cap_weight（必须）：增强市值因子股票权重（float），默认值为0.0。
        enhance_cap_weight2（必须）：增强市值因子股票权重（float），默认值为0.0。
        barra_dict（必须）：barra因子字典。key为barra风险因子，取值范围为['BETA', 'BTOP', 'EARNYILD', 'GROWTH', 'LEVERAGE', 'LIQUIDTY', 'MOMENTUM', 'RESVOL', 'SIZE', 'SIZENL'](未加入行业因子，因子具体含义见barra文档)，value为每个标准差（默认为1）惩罚的比例，例如均值为1，因子值为1.5，value为-0.5，则惩罚值为abs(1.5-1) * -0.5 = -0.25。该字典默认值为None
        buy_multiplier(必须)：买入股票相对于卖出股票的倍数，默认为1
        max_amount_ratio（必须）：卖出股票时相对于每个时间段的amount的比例，默认为1.0，即认为这个时间段所有成交量都能被自己的卖盘吃掉
        top_percent（必须）:在画图时指定top_ratio的分位数，默认为0.75，即75分位数
        ext_num（必须）： 已持仓股票继续持有容忍排名（int）
        sel_mom（必须）： 选股数量乘数（int），这里给了一个选股数量乘数，其实可以调整stock_num和ext_num即可
        benchmark（必须）： benchmark（str）,选用的benchmark
        trade_ratio_per_period（必须）：每个时间段可交易的配额比例（float），若不设置则默认设为freq / 240分钟
        max_ratio_per_stock（必须）：每只股票在总市值中的最大比例（float），若不设置则默认设为 2.0 / stock_num
        trade_lost_vs_twap：交易时相对于twap的交易损失（int，以BPs为单位），默认为5
        t0_stock_dict（必须）： 用于指定股票每天用于t0的比例。值为dict，如{rank1: ratio1, rank2:ratio2, ...}，其中rank为weight文件中前一天15:00:00股票排名，ratio为用于t0的比例（相应地can_sell_pool需要减去相应比例），默认值为None。例如{500: 1.0}即为排名前500的股票全部用于t0，{500:1.0, 1000:0.6}即为前500的股票全部用于t0，500至1000的股票60%用于t0
        delay_minutes（必须）：交易延时的分钟数（int），默认为0
        trade_indexes（必须）：允许交易的索引列表，例如freq=30，一天可交易8次，可设置为[0,2,4,6]指定在9:30,10:30,13:00,14:00进行交易，需要注意的是此时列表里不能有8(因为15:00不能交易)或者大于8的数。此外目前设置trade_indexes暂时不会修改每次交易的配额，例如freq=120时，trade_indexes=[1]时，只在下午13点交易，仍然只交易一半的配额。默认为None，会交易所有时间段
        isweight（可选）： 所要测试的文件为股票权重文件（bool），如果是预测值则为False，一般为False
        buyST（必须）： 是否考虑买入 ST 股票（bool），一般为False，不买入ST股票
        component（必须）： 是否为成分股内选股方式（bool）,默认为None
        trademethod（必须）： 计算收益率时以何种价格（str）,默认为vwap
        plot_res（必须）： 是否画图展示回测结果（bool）,默认为True
        optimal_stock（可选）： 使用weight文件的方式，具体见文中注释（str），调仓方式，如果为2，则超过ext_num才退出，如果为1，则只选前stock_num只股票
        add_component（可选）： 选择股票时是否添加成份内股票，添加数目为sel_num，计算方式见代码，当optimal_stock ！= “2” 时生效，默认为False
        stamptax（必须）： stamptax（float），印花税，默认为千1
        commision（必须）： commision（float），手续费，默认为万3
        trade_time_rate（可选）： 每日成交时间占比，取值为 0-1 之间（float），默认为1
        amnt_lookback_days（可选）: 统计成交额时回看的天数(int)，默认为1，即使用昨天这个时间段的成交额作为今天的估计值，若大于1则统计过去连续多天
        amnt_max_ratio(可选)：成交额最大比例（float），默认为None，若不为None，则该段时间内成交额不得超过估计值的amnt_max_ratio
        buy_strategy(必选)：买入股票时的策略(str)，默认为'equal_sell_num'，即买入股票与卖出股票的数量相等；'fill_top'：按排名依次填充至max_ratio_per_stock
        barra_list（必须）：需要控制风险的barra因子列表。取值范围为['BETA', 'BTOP', 'EARNYILD', 'GROWTH', 'LEVERAGE', 'LIQUIDTY', 'MOMENTUM', 'RESVOL', 'SIZE', 'SIZENL']。该列表默认值为None
        barra_list_usage(必须)：barra_list的使用方式(str)，目前支持两种方式'rebalance'和'weight'，1、rebalance模式：在买入股票时会根据当前风险进行调整，当与IC对应的风险超过barra_rebalance_th时，在买入股票时会挑选使风险变小的股票等权买入。2、weight模式：在买入股票时不再等权，会为股票配对，从一只股票上分barra_list_weight给另外一只股票以使得风险暴露变小。
        barra_rebalance_num（必须）：配合barra_list_usage='rebalance'使用的股票数量，进行rebalance的股票将从这里面挑选，默认值为stock_num
        barra_rebalance_th（必须）：配合barra_list_usage='rebalance'使用的阈值(float)，默认值为0
        barra_rebalance_lookback（必须）：配合barra_list_usage='rebalance'使用的回看天数，用于计算risk_return的均值和方差，默认值为20
        barra_list_weight（必须）：配合barra_list_usage='weight'使用的float值，取值为0.0-1.0。默认值为0
        ext_num_adjust_type（必须）：ext_num调整的使用方式(str)，目前支持trisection和linear两种方式，其中1、trisection方式：ext_num有三档，2000，2500和3000，配合ext_num_adjust_upper和ext_num_adjust_lower使用，当指标位于(-inf,ext_num_adjust_lower]时ext_num取3000，位于(ext_num_adjust_lower,ext_num_adjust_upper)时ext_num取2500，位于[ext_num_adjust_upper,inf)时ext_num取2000；2、linear方式：配合ext_num_adjust_upper和ext_num_adjust_lower使用，与trisection模式类似，区别在于(ext_num_adjust_lower,ext_num_adjust_upper)这里会被线性分为9档（每100一档）。默认为None
        ext_num_adjust_score（必须）：ext_num调整的指标(str)，目前支持'r2'。1、r2：计算过去一定天数内(由ext_num_adjust_lookback指定天数)weight值和未来480min 在top 50%股票上的r2值
        ext_num_adjust_upper（必须）：ext_num调整指标的上界(float)，大于或等于该上界时ext_num取2000
        ext_num_adjust_lower（必须）：ext_num调整指标的下界(float)，小于或等于该下界时ext_num取3000
        ext_num_adjust_freq（必须）：ext_num调整的频率(int)，默认为5，即每5天调整一次
        ext_num_adjust_lookback（必须）：ext_num调整时计算指标的回看天数，默认为5，即回看过去5天计算调整指标
'''
