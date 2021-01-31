import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns
import numpy as np

## functions returning time series as pandas.Series
def col_years_to_time_series(df, drop_cols = None):
    drop_cols = [drop_cols] if type(drop_cols)!=list else drop_cols
    
    values = np.concatenate([[e for i, e in enumerate(df.loc[idx]) if i not in drop_cols] for idx in df.index])
    time_idx = pd.period_range('01-' + str(int(df.index[0])), periods=len(values), freq='M')
    return pd.Series(data = values, index = time_idx).dropna()

def extract_time_series(df):
    date_cols = [col_name for col_name in ['YEAR', 'YR', 'MONTH', 'MON'] if col_name in df.columns]
    target_col = [col_name for col_name in ['INDEX', 'ANOM'] if col_name in df.columns][0]
    values = df[target_col].values
    
    time_idx = pd.period_range(
        '/'.join(list(map(lambda x: str(int(x)),df.iloc[0][date_cols]))), 
        periods=len(values), freq='M')
    return pd.Series(data = values, index = time_idx)

## auxiliary functions
def if_missing_values_in_ts(seq, fill_val):
    missing_in_seq = seq == fill_val
    idxs_wrong_values = np.where(missing_in_seq)[0]
    
    idx_last_correct_val = np.where(missing_in_seq==False)[0][-1]

    return True if any(idxs_wrong_values - idx_last_correct_val < 0) else False
    
def truncate_missing_tail(seq, fill_val):
    assert if_missing_values_in_ts(seq, fill_val) == False
    
    return seq[:np.where(seq==fill_val)[0][0]]

## reading raw files and turning them into pandas.Series using previously defined functions
def get_ts_from_df(path, skiprows, skipfooter=0, missing_value=None, drop_cols=None):
    df = pd.read_csv(path, sep='\s+', skiprows=skiprows, skipfooter=skipfooter, header=None, index_col=0, engine='python')
    ts = col_years_to_time_series(df, drop_cols=drop_cols)
    if missing_value is not None:
        ts = truncate_missing_tail(ts, missing_value)
        
    return ts

def get_ts_from_ds(path, skiprows):
    ds = pd.read_csv(path, skiprows=skiprows, sep='\s+')
    return extract_time_series(ds)

def get_ts_from_xlsx(path, skiprows):
    df = pd.read_excel(path, skiprows=skiprows, index_col=0)
    ts = col_years_to_time_series(df)
    ts.index = ts.index + 1
    return ts

def warm_cold_phase(ts, abs_threshold=.5, minlen=5):
    warms = []
    colds = []
    norms = []
    
    sequences = []
    tmp = None
    for idx in ts.index:
        if tmp == None:
            tmp = idx
            sequences.append([tmp])
        elif ts.loc[tmp] > abs(abs_threshold) and ts.loc[idx] > abs(abs_threshold):
            tmp = idx
            sequences[-1].append(tmp)
        elif ts.loc[tmp] < -abs(abs_threshold) and ts.loc[idx] < -abs(abs_threshold):
            tmp = idx
            sequences[-1].append(tmp)
        elif ts.loc[tmp] < abs(abs_threshold) and ts.loc[tmp] > -abs(abs_threshold) and \
                ts.loc[idx] < abs(abs_threshold) and ts.loc[idx] > -abs(abs_threshold):
            tmp = idx
            sequences[-1].append(tmp)
        else:
            tmp = idx
            sequences.append([tmp])
    for seq in sequences:
        values = ts.loc[seq]
        if np.mean(values)>=abs(abs_threshold) and len(values)>=minlen:
            warms.append(values)
        elif np.mean(values)<=-abs(abs_threshold) and len(values)>=minlen:
            colds.append(values)
        else:
            if len(norms)>0 and (norms[-1].index[-1] + 1) == seq[0]:
                norms[-1] = norms[-1].append(ts.loc[seq])
            else:
                norms.append(ts.loc[seq])
            
    return warms, colds, norms

def phase_table(ts, phase="El Nino", abs_threshold=.5, minlen=5):
    df = pd.DataFrame(columns=['# of {}'.format(phase), 
                               'Starting date',
                              'Ending date',
                              'Months since the end of last {} episode'.format(phase),
                              'Duration of episode in months',
                              'Maximum value',
                              'Time at maximum',
                              'Cumulative sum'])
    if phase=='El Nino' or phase=='positive phase':
        episodes, _, _ = warm_cold_phase(ts, abs_threshold, minlen)
    elif phase=='La Nina' or phase=='negative phase':
        _, episodes, _ = warm_cold_phase(ts, abs_threshold, minlen)
    for i, episode in enumerate(episodes):    
        tmp_df = pd.DataFrame({'# of {}'.format(phase):
                               [i + 1], 
                               'Starting date':
                               [episode.index[0]],
                              'Ending date':
                               [episode.index[-1]],
                              'Months since the end of last {} episode'.format(phase):
                               [(episode.index[0] - episodes[i - 1].index[-1]).freqstr[:-1] if i>0 else '-'],
                              'Duration of episode in months':
                               [len(episode)],
                              'Maximum value':
                               [episode.max() if episode.mean()>0 else episode.min()],
                              'Time at maximum':
                               [episode.idxmax() if episode.mean()>0 else episode.idxmin()],
                              'Cumulative sum':
                              [episode.sum()]})
        df = df.append(tmp_df, ignore_index=True, sort=False)
    return df

def plot_phases(ts, name, abs_tsh=.5, minlen=5, save_as=None, figsize=(14.5,6)):
    ts.plot(style='k', figsize=figsize)
    
    warms, colds, _ = warm_cold_phase(ts, abs_tsh, minlen)
    warms_cnct = np.concatenate([e.index for e in warms])
    colds_cnct = np.concatenate([e.index for e in colds])
    
    warms_idxs = np.array([True if e in warms_cnct else False for e in ts.index])
    colds_idxs = np.array([True if e in colds_cnct else False for e in ts.index])

    plt.fill_between(ts.index, abs(abs_tsh), ts.values, where=(ts.values>abs(abs_tsh)) * warms_idxs, 
                 facecolor='red', interpolate=True)
    plt.fill_between(ts.index, -abs(abs_tsh), ts.values, where=(ts.values<-abs(abs_tsh)) * colds_idxs, 
                 facecolor='blue', interpolate=True)
    
    
    red_patch = mpatches.Patch(color='red', label='El Niño phase')
    blue_patch = mpatches.Patch(color='blue', label='La Niña phase')
    plt.legend(handles=[red_patch, blue_patch])
    
    plt.title(name + " - El Niño and La Niña episodes")
    plt.xlabel('Years')
    plt.ylabel('Values')
    plt.grid(True)
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()

def time_between_extremes(data):
    diffs = []
    for i in range(1, len(data['Time at maximum'])):
        diffs.append((data['Time at maximum'][i] - data['Time at maximum'][i-1]).n)    
    return diffs

def data_to_numpy(data):
    d = []
    for i in data:
        d.append(i)
    return(d)

def plot_variability(data, phases_name, name, y_label, save_as=None, figure=(14.5,6)):
    
    x = np.arange(0, len(data))
    coef, b = np.polyfit(x, data,1)    

    fig, ax = plt.subplots(figsize=figure, dpi=80)
    ax.plot(data, "o", linestyle='--')
    ax.plot(x, data, 'yo', x, coef*x+b, '--r')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.title(name)
    plt.ylabel(y_label)
    plt.xlabel(phases_name)

    if save_as is not None:
        plt.savefig(save_as.format(name))
    plt.show()

def statistics_for_month(data, month):
    during_maximum = 0
    for i in data['Time at maximum']:
        if i.month == month:
            during_maximum = during_maximum + 1
        
    during_beginning = 0
    for i in data['Starting date']:
        if i.month == month:
            during_beginning = during_beginning + 1
        
    during_ending = 0
    for i in data['Ending date']:
        if i.month == month:
            during_ending = during_ending + 1
        
    during_el_nino = 0
    maxi = 0
    for index, row in data.iterrows():
        if month < row['Starting date'].month:
            if month - row['Starting date'].month + 13 <= (row['Ending date'] - row['Starting date']).n + 1:
                during_el_nino = during_el_nino + 1
        elif month - row['Starting date'].month <= (row['Ending date'] - row['Starting date']).n + 1:
            during_el_nino = during_el_nino + 1
        
    return during_maximum, during_beginning, during_ending, during_el_nino

def plot_month_statistic(data, title, save_as):
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    width = 0.55
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(14.5,6))
    rects = ax.bar(x, data, width)
    
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylabel('Frequency')
    
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    
    plt.savefig(save_as)
    plt.show()

def plot_phases_december(ts, name, decembers_oni, decembers_nino_3_4, abs_tsh=.5, minlen=5, save_as=None, figsize=(14.5,6)):
    ts.plot(style='k', figsize=figsize)
    
    warms, colds, _ = warm_cold_phase(ts, abs_tsh, minlen)
    warms_cnct = np.concatenate([e.index for e in warms])
    colds_cnct = np.concatenate([e.index for e in colds])
    
    warms_idxs = np.array([True if e in warms_cnct else False for e in ts.index])
    colds_idxs = np.array([True if e in colds_cnct else False for e in ts.index])
    
    plt.fill_between(ts.index, abs(abs_tsh), ts.values, where=ts.values>abs(abs_tsh), 
                 facecolor='red', interpolate=True)
    plt.fill_between(ts.index, -abs(abs_tsh), ts.values, where=ts.values<-abs(abs_tsh), 
                 facecolor='blue', interpolate=True)
    
    
    red_patch = mpatches.Patch(color='red', label='El Niño phase')
    blue_patch = mpatches.Patch(color='blue', label='La Niña phase')
    plt.legend(handles=[red_patch, blue_patch])
    
    plt.title(name + " - marking Decembers with maximum phase value")
    plt.xlabel('Years')
    plt.ylabel('Values')
    if name == 'Oceanic Nino Index':
        for d in decembers_oni:
            plt.axvline(d, ymin=0, ymax=1)
    else:
        for d in decembers_nino_3_4:
            plt.axvline(d, ymin=0, ymax=1)
    plt.grid(True)
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()

def how_many_phases(data, months, threshold):
    print(months, threshold)
    print("Positive: ", sum(data.rolling(window=months).mean().dropna().values > abs(threshold)))
    print("Negative: ", sum(data.rolling(window=months).mean().dropna().values < -abs(threshold)))

def plot_phases_nao(ts, name, months_len, abs_tsh=.5, minlen=5, save_as=None, figsize=(20.5,6)):
    ts.plot(style='k', figsize=figsize)
    
    warms, colds, _ = warm_cold_phase(ts, abs_tsh, minlen)
    warms_cnct = np.concatenate([e.index for e in warms])
    colds_cnct = np.concatenate([e.index for e in colds])
    
    warms_idxs = np.array([True if e in warms_cnct else False for e in ts.index])
    colds_idxs = np.array([True if e in colds_cnct else False for e in ts.index])
    
    plt.fill_between(ts.index, abs(abs_tsh), ts.values, where=(ts.values > abs(abs_tsh)) * warms_idxs,
                     facecolor='red', interpolate=True)
    plt.fill_between(ts.index, -abs(abs_tsh), ts.values, where=(ts.values < -abs(abs_tsh)) * colds_idxs,
                     facecolor='blue', interpolate=True)

    red_patch = mpatches.Patch(color='red', label='Positive phase')
    blue_patch = mpatches.Patch(color='blue', label='Negative phase')
    plt.legend(handles=[red_patch, blue_patch])
    
    plt.title(name + " - positive and negative phases")
    plt.xlabel('Years')
    plt.ylabel('Values')
    plt.grid(True)
    if save_as is not None:
        plt.savefig(save_as.format(months_len))
    plt.show()

def excursion_episodes(ts):
    episodes_idxs = []
    sign = None
    for idx in ts.index:
        if sign==None:
            sign = ts.loc[idx]>0
            episodes_idxs.append([idx])
        elif ts.loc[idx]==0.:
            episodes_idxs[-1].append(idx)
        elif sign == (ts.loc[idx]>0):
            episodes_idxs[-1].append(idx)
        else:
            sign = ts.loc[idx] > 0
            episodes_idxs.append([idx])
    return [ts.loc[idxs] for idxs in episodes_idxs]

def excursion_table(ts, rolling_months):
    df = pd.DataFrame(columns=['# of excursion episode over or below zero', 
                               'Starting date',
                              'Ending date',
                              'Months since the end of last excursion episode in the same direction',
                              'Duration of excursion episode in months',
                              'Maximum value',
                              'Time at maximum',
                              'Cumulative sum'])
    if rolling_months != 1:
        episodes = excursion_episodes(ts.rolling(window=rolling_months).mean().dropna())
    else:
        episodes = excursion_episodes(ts)
    ctr_above = 0
    ctr_below = 0
    for i, episode in enumerate(episodes):
        if episode.mean()>0:
            ctr_above += 1
        else:
            ctr_below += 1
        tmp_df = pd.DataFrame({'# of excursion episode over or below zero':
                               [ctr_above if episode.mean()>0 else ctr_below], 
                               'Starting date':
                               [episode.index[0]],
                              'Ending date':
                               [episode.index[-1]],
                              'Months since the end of last excursion episode in the same direction':
                               [(episode.index[0] - episodes[i - 2].index[-1]).freqstr[:-1] if i>1 else '-'],
                              'Duration of excursion episode in months':
                               [len(episode)],
                              'Maximum value':
                               [episode.max() if episode.mean()>0 else episode.min()],
                              'Time at maximum':
                               [episode.idxmax() if episode.mean()>0 else episode.idxmin()],
                              'Cumulative sum':
                              [episode.sum()]})
        df = df.append(tmp_df, ignore_index=True)
    return df

def excursion_tables_for_group(group, rolling_months, save_as=None):
    for key, item in group.items():
        tbl = excursion_table(item, rolling_months)
        if save_as is not None:
            if rolling_months != 1:
                tbl.to_excel(save_as.format(rolling_months, key+"_"+str(rolling_months)))
            else:
                tbl.to_excel(save_as.format(key))

def plot_single_ts(ts, name, save_as=None, figure=(14.5,6)):
    plt.figure(figsize=figure, dpi=80)
    ax = ts.plot()
    ax.set_title(name)
    ax.set_ylabel('Values')
    ax.set_xlabel('Years')
    plt.grid(True)
    if save_as is not None:
        plt.savefig(save_as.format(name))
    plt.show()

def plot_all(group, save_as=None):
    for key, item in group.items():
        if save_as is not None:
            plot_single_ts(item, key, save_as)
        else:
            plot_single_ts(item, key)

# xcorrelation_plot(enso['SOI from Climatic Research Unit'][-len(enso['Nino 3.4']):],enso['Nino 3.4'], 'Atlantic Meridional Scollation (AMO)')
def xcorrelation_plot(ts_1, ts_2, name, num_years=10, save_as=None, figure=(14.5,6)):
    plt.figure(figsize=figure, dpi=80)
    samples = min(len(ts_1), len(ts_2))
    plt.xcorr(ts_1[ts_1.index.intersection(ts_2.index)], ts_2[ts_2.index.intersection(ts_1.index)], 
              usevlines=True, normed=True, maxlags=12*num_years, lw=1.5)
    plt.grid(True)
    fig = figure

    plt.title(name+" ({} years)".format(num_years))
    plt.xlabel("Shift in months")
    plt.ylabel("Correlation")
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()

def group_acorrelation_plot(group, num_years=10, save_as=None):
    for key, item in  group.items():
        xcorrelation_plot(item, item, key, num_years=num_years, save_as = save_as.format(key) if save_as is not None else None)

def drop_keys(dictionary: dict, *keys: str):
    new_dict = {}
    for key in dictionary.keys():
        if key not in keys:
            new_dict[key] = dictionary[key]
    return new_dict

def select_keys(dictionary: dict, *keys: str):
    new_dict = {}
    for key in dictionary.keys():
        if key in keys:
            new_dict[key] = dictionary[key]
    return new_dict

def crosscorrelation_between_groups(group1, group2, save_as=None):
    used_keys = []
    for key1,item1 in group1.items():
        used_keys.append(key1)
        for key2, item2 in group2.items():
            if key1!=key2 and key2 not in used_keys:
                xcorrelation_plot(item1, item2, 
                          name="Crosscorelation: {} - {}".format(key1, key2),
                         save_as=save_as.format(key1 + '_-_' + key2))

def convolve_series(series, mask):
    return pd.Series(data=np.convolve(series.values, mask, 'same'), index = series.index)

def operation_on_dict(dictionary, mask=np.ones(3)/3, operation=convolve_series):
    return {key: operation(val, mask) for (key, val) in dictionary.items()}
