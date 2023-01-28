import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import logging

RES_Y = 4096
MARGIN_Y = 243.81

# ==========
# Load Files
# ==========
st.title('Sorter Experiment Analysis Tool')
st.header('Files')
raw_path = st.file_uploader(
    'Upload the results file in *.csv format:', 
    type='csv', 
    accept_multiple_files=False,
)

events_path = st.file_uploader(
    'Upload the events file in *.xls or *xlsx format (optional):', 
    type=['xlsx', 'xls'], 
    accept_multiple_files=False,
)

if raw_path is None:
    raw = pd.read_csv('example.csv', parse_dates=['IMAGE_TIMESTAMP'])
else:
    raw = pd.read_csv(raw_path, parse_dates=['IMAGE_TIMESTAMP'])

if events_path is not None:
    try:
        events = pd.read_excel(events_path, sheet_name=0, parse_dates=['TIME'])
    except Exception:
        logging.warning('bad events file')


# =============
# Preprocessing
# =============
df = raw.copy()
df['CAMERA_ID'] = df['CAMERA_ID'].astype('int8')
# df['CAMERA_MOSQUITO_NUM'] = df['CAMERA_MOSQUITO_NUM'].astype('int32')
# df['TOTAL_MOSQUITO_NUM'] = df['TOTAL_MOSQUITO_NUM'].astype('int32')
df['MID_X'] = df['MID_X'].astype('float32')
df['MID_Y'] = df['MID_Y'].astype('float32')

df['XY'] = (df['MID_X'].round(-1).astype('str') + ' ' + 
            df['MID_Y'].round(-1).astype('str'))
df['TIME_DELTA'] = ((df['IMAGE_TIMESTAMP'] - df['IMAGE_TIMESTAMP'].min()) / 
                    pd.Timedelta('1H')) 
df['TIME_DELTA_ROUND'] = df['TIME_DELTA'].round(2)
df = df.merge(pd.Series(df.groupby('TIME_DELTA_ROUND')['MID_X'].count(), 
                        name='COUNT'), on='TIME_DELTA_ROUND')
try:
    events['TIME_DELTA'] = ((events['TIME'] - df['IMAGE_TIMESTAMP'].min()) / 
                            pd.Timedelta('1H')) 
except:
    pass


# =========
# Variables
# =========
cam_n = df['CAMERA_ID'].max()
cam_range = range(1, cam_n+1)
sns.set_style("whitegrid")
colors='brgycmk'
plt.rcParams.update({'font.size': 16})
st.set_option('deprecation.showPyplotGlobalUse', False)


# =========
# Functions
# =========
def quarter(series, label=True):
    # plot vertical lines on quartiles
    for q in [0.25, 0.5, 0.75]:
        plt.axvline(x=series.quantile(q), color='k', linewidth=1)
        if label:
            plt.text(
                x=series.quantile(q), y=0, 
                s=f'{q:.0%}\n{series.quantile(q):.1f}', 
                ha='center', va='bottom',
                bbox={
                    'boxstyle': "square", 
                    'ec': (0.7, 0.7, 0.7), 
                    'fc': (1, 1, 1), 
                    'alpha': 0.7,
                    },
            )

def midday_midnight(timestamp, delta, d_min=0, d_max=-1):
    # plot vertical lines for 12 o'clock in time delta units
    t0 = timestamp.min()
    to_midnight = 24 - (t0.hour + t0.minute/60 + t0.second/3600) 
    if d_max < 0:
        d_max2 = delta.max()
    else:
        d_max2 = d_max
    for offset in range(int(d_min), int(d_max2-to_midnight), 24):
        plt.axvline(x=to_midnight+offset, color=(0.6, 0.6, 0.8), 
                    ls='-', lw=4, alpha=0.6)
    for offset in range(12+int(d_min), int(d_max2-to_midnight), 24):
        plt.axvline(x=to_midnight+offset, color=(1, 1, 0.6), 
                    ls='-', lw=4, alpha=0.6)
    for offset in range(6+int(d_min), int(d_max2-to_midnight), 12):
        plt.axvline(x=to_midnight+offset, color=(1.0, 0.8, 0.7), 
                    ls='-', lw=4, alpha=0.6)


# ===============
# Sidebar Options
# ===============
st.sidebar.title('Settings')
st.sidebar.subheader('Data slice')
percentile_cutoff = st.sidebar.number_input(
    'Display up to percentile:', 0, 100, 100, 1)
count_cutoff = df['COUNT'].quantile(percentile_cutoff/100.)
t_min, t_max = st.sidebar.slider('Time range (hours)):', 
                                 0., df['TIME_DELTA'].max(), 
                                 (0., df['TIME_DELTA'].max()), 0.25
                                )
interval = st.sidebar.slider('Time interval (min):', 5, 120, 60, 5)
st.sidebar.subheader('Heatmap options')
flip_y = st.sidebar.checkbox('Flip Y axis', True)
overlap = st.sidebar.number_input('Overlap (px)', 0, 
                                  RES_Y//2, int(MARGIN_Y*2+0.5), 1)

n_outlier = st.sidebar.number_input('Number of outliers:', 0, 10, 0, 1)
r_outlier = st.sidebar.slider('Outlier radius (px):', 0, 100, 50, 1)
mirror = st.sidebar.checkbox('Mirror Heatmap', False)
st.sidebar.subheader('Simulation parameters')
tpr = st.sidebar.number_input('TPR (true males/all males)', 
                              0., 1., 0.995, 0.000001, format='%.6f')
fpr = st.sidebar.number_input('FPR (false males/all females)', 
                              0., 1., 0.005, 0.000001, format='%.6f')
pipe = st.sidebar.number_input('Suction Efficiency', 0., 1., 
                               1., 0.001, format='%.3f')
crtg = st.sidebar.number_input('Cartridge capacity', value=1400)
user_ratios = st.sidebar.checkbox('Enable data customization', value=False)
user_m = st.sidebar.slider('Male ratio:', min_value=0., max_value=1., 
                           value=0.5, step=0.01, disabled=not user_ratios)
n_pupa = st.sidebar.number_input('Pupae in', value=10**6, 
                                 disabled=not user_ratios)
emerg_ratio = st.sidebar.number_input('Emergence ratio', 0., 1., 1., 0.01,
                                      disabled=not user_ratios)
if not user_ratios:
    user_m = (df['TAG']=='male').sum()/len(df)
    n_pupa = len(df)
    emerg_ratio = 1.

# data slice to be used henceforth
df_slice = df[
    (df['COUNT'] <= count_cutoff) & 
    (df['TIME_DELTA'] <= t_max) & 
    (df['TIME_DELTA'] >= t_min)
]

# ========================
# Detection Frequency Plot
# ========================
st.header('Graphs')
st.subheader('Frequency')
plt.figure(figsize=(16,8))
plt.title('Detection Frequency')
plt.xlim(t_min-0.5, t_max+0.5)
plt.grid(which='minor', visible=True, linestyle=':')
#plt.xticks(ticks=np.arange(round(t_min-0.5), round(t_max+0.5), step=5))
plt.xlabel('Time (hours)')
plt.ylabel(f'mosq. / {interval} min')
sns.histplot(
    data=df_slice['TIME_DELTA'], label='total',
    binwidth=interval/60, color='limegreen', element='bars',
)
sns.histplot(
    data=df_slice.query('TAG=="male"')['TIME_DELTA'], label='male',
    binwidth=interval/60, color='b', element='step', fill=False,
)
sns.histplot(
    data=df_slice.query('TAG=="female"')['TIME_DELTA'], label='female',
    binwidth=interval/60, color='r', element='step', fill=False,
)
plt.legend(loc='upper right')

# info box
plt.text(
    x=t_min, #plt.gca().get_xlim()[1]*0.015, 
    y=plt.gca().get_ylim()[1]*0.975,
    s="Start: {}\nFinish: {}\nParticipants: {}/{} {:.1%}".format(
        df['IMAGE_TIMESTAMP'].min().round('S'), 
        df['IMAGE_TIMESTAMP'].max().round('S'), 
        len(df_slice),
        len(df),
        len(df_slice)/len(df),
    ),
    ha='left', va='top',
    bbox=dict(
        boxstyle="square", 
        ec=(0.75, 0.75, 0.75), 
        fc=(1, 1, 1), 
        alpha=0.75,
    ),
)

# Vertical lines
midday_midnight(df['IMAGE_TIMESTAMP'], df['TIME_DELTA'], 0, t_max)
#quarter(df['TIME_DELTA'])
try:
    for event in events[(events['TIME_DELTA'] <= t_max) &
                        (events['TIME_DELTA'] >= t_min)]['TIME_DELTA']:
        plt.axvline(x=event, color='m', ls=':',lw=3)
except:
    pass

st.pyplot()

# event list
if events_path != None:
    try:
        st.text('Events:')
        st.dataframe(events)
    except:
        st.text('Event file is not correctly formatted')


# ==============
# Male VS Female
# ==============

st.subheader('Male\\Female Ratio')
plt.figure(figsize=(16,8))
plt.title('Male\\Female Ratio')
#plt.xlim(t_min-0.5, t_max+0.5)
plt.grid(which='minor', visible=True, linestyle=':')
#plt.xticks(ticks=np.arange(round(t_min-0.5), round(t_max+0.5), step=5))
plt.xlabel('Time (hours)')
plt.ylabel('Ratio')
sns.histplot(
    data=df_slice, x='TIME_DELTA', hue='TAG', multiple='fill',
   # binwidth=1., 
    bins=50,
    element='poly', legend=True
)

#plt.legend(loc='upper right')
st.pyplot()


# =========================
# Cumulative Detection Plot
# =========================
st.subheader('Total')

# decide scale
if len(df[df['COUNT'] <= count_cutoff]) >= 10**8:
    scale = 10**6
    scale_char = 'M'
elif len(df[df['COUNT'] <= count_cutoff]) >= 10**5:
    scale = 10**3
    scale_char = 'k'
else:
    scale = 10**0
    scale_char = ''
    
# canvas
fig, ax1 = plt.subplots(figsize=(16,8))
ax2 = ax1.twinx()

# cumulative proportion
ax1.hist(
    x=df_slice['TIME_DELTA'], 
    bins=1000, 
    histtype='stepfilled', 
    color='c', 
    alpha=0.2, 
    cumulative=True, 
    density=True,
)
ax1.grid(visible=True, which='major', linestyle='-')
ax1.grid(visible=True, which='minor', linestyle=':')
ax1.set_yticks(np.arange(0, 1.01, 0.1))
ax1.set_yticks(np.arange(0, 1.01, 0.05), minor=True)
ax1.set_xticks(np.arange(0, df['TIME_DELTA'].max()+1, 5))
ax1.set_xticks(np.arange(0, df['TIME_DELTA'].max()+1, 2.5), minor=True)
ax1.set_xlim(t_min, t_max)
ax1.set_title('Cumulative Detection')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Proportion')

# cumulative count
ax2.hist(
    x=df_slice['TIME_DELTA'], 
    bins=1000, 
    histtype='step', 
    color='k', 
    alpha=0.75, 
    cumulative=True,
    label='total'
)
ax2.hist(
    x=df_slice.query('TAG=="male"')['TIME_DELTA'], 
    bins=1000, 
    histtype='step', 
    color='b', 
    alpha=0.75, 
    cumulative=True, 
    label='male',
)
ax2.hist(
    x=df_slice.query('TAG=="female"')['TIME_DELTA'], 
    bins=1000, 
    histtype='step', 
    color='r', 
    alpha=0.75, 
    cumulative=True,
    label='female',
)
ax2.legend(loc='upper left')
ax2.grid(visible=False)
ax2.set_ylabel('Count')

# vertical percentile lines
for hr in range(int(t_min/12)*12+12, int(t_max+0.99), 12):
    answer = (
        len(df[(df['COUNT'] <= count_cutoff) & (df['TIME_DELTA'] <= hr)]) /
        len(df[(df['COUNT'] <= count_cutoff)])
    )
    plt.axvline(x=hr, color='grey', linewidth=1)
    plt.text(
        x=hr, y=plt.gca().get_ylim()[1]*0.5, 
        s="{0:.0f} hr\n{2:.0f}{3}\n{1:.1%}".format(
            hr, 
            answer, 
            len(df[(df['COUNT'] <= count_cutoff) & 
                   (df['TIME_DELTA'] <= hr)])/scale,
            scale_char,
        ), 
        ha='center', va='center',
        bbox={
            'boxstyle': "square", 
            'ec': (0.7, 0.7, 0.7), 
            'fc': (1, 1, 1), 
            'alpha': 0.6
        },
    )

st.pyplot()


# ==============================
# Conveyor Distribution Heatmaps
# ==============================
st.subheader('2D Distribution')

# map Y values to conveyor scale
if flip_y:
    df_slice['MID_Y'] = RES_Y - df_slice['MID_Y'] + \
        (RES_Y - overlap) * (df_slice['CAMERA_ID'] - 1)
else:
    df_slice['MID_Y'] = df_slice['MID_Y'] + \
        (RES_Y - overlap) * (df_slice['CAMERA_ID'] - 1)
    
# remove outliers
df_ = df_slice.copy()
x_list = []
y_list = []
for i in range(n_outlier):    
    mode_xy = df_['XY'].mode()[0]
    mode_y = float(mode_xy.split()[1])
    buff_y = r_outlier
    mode_x = float(mode_xy.split()[0])
    buff_x = r_outlier
    df_ = df_[~((df_['MID_Y'] >= mode_y-buff_y) & 
                (df_['MID_Y'] <= mode_y+buff_y) & 
                (df_['MID_X'] >= mode_x-buff_x) & 
                (df_['MID_X'] <= mode_x+buff_x))]
    x_list.append(mode_x)
    y_list.append(mode_y)

if mirror:
    # plot mirrored heatmap
    df_l = df_.copy()
    df_l['MID_Y'] = (4096-overlap)*cam_n - df_['MID_Y']
    df_mirror = pd.concat([df_, df_l], axis=0)
    sns.jointplot(data=df_mirror,
                  x='MID_X', 
                  y='MID_Y', 
                  kind='hex', 
                  cmap='viridis',
                  alpha=0.8,
                  height=7,
                 )
    plt.title('Conveyor Distribution (Mirrored)', y=1.2)
else:
    # plot regular heatmap
    sns.jointplot(data=df_,
                  x='MID_X', 
                  y='MID_Y', 
                  kind='hex', 
                  cmap='CMRmap',
                  alpha=0.8,
                  height=7,
                  )
    plt.title('Conveyor Distribution', y=1.2)
#plt.ylim(0, 4096*cam_n)
plt.xlabel('Conveyor Length')
plt.ylabel('Conveyor Width')
st.pyplot()

# outlier list
if n_outlier:
    st.text('Outlier coordinates:')
    st.dataframe(pd.DataFrame({'X': x_list, 'Y': y_list}, dtype='int'))
    
# ===================
# Contamination Model
# ===================
st.subheader('Production Simulation')

# calculate ratio factors
if user_ratios:
    m_factor = emerg_ratio * n_pupa * user_m / \
        (df[df['COUNT'] <= count_cutoff]['TAG'] == 'male').sum()
    f_factor = emerg_ratio * n_pupa * (1. - user_m) / \
        (df[df['COUNT'] <= count_cutoff]['TAG'] == 'female').sum()
else:
    m_factor, f_factor = 1., 1.

# calculate contamination rate
n_male = (df_slice['TAG'] == 'male').cumsum().replace(0, 1) * m_factor
n_female = (df_slice['TAG'] == 'female').cumsum() * f_factor
f_crtg = n_female*fpr + n_female*(1-fpr)*(1-pipe)
m_crtg = n_male*tpr + n_male*(1-tpr)*(1-pipe)
cont_rate = f_crtg / (m_crtg + f_crtg)
if user_ratios:
    m_eff = m_crtg.values[-1] / (n_pupa * user_m)
    f_eff = 1 - f_crtg.values[-1] / (n_pupa * (1 - user_m))
    machine_eff = m_eff / emerg_ratio
else:
    machine_eff = m_crtg.values[-1] / \
        (df[df['COUNT'] <= count_cutoff]['TAG'] == 'male').sum()
    m_eff = machine_eff * emerg_ratio
    f_eff = 1 - f_crtg.values[-1] / \
        (df[df['COUNT'] <= count_cutoff]['TAG'] == 'female').sum() \
        * emerg_ratio

# calculate production rate
production = (f_crtg + m_crtg).reset_index(drop=True) / crtg
for i in range(len(production)-1, 0, -1):
    production[i] -= production[i-1]

# plot production rate
plot_data = pd.DataFrame({
    't': df_slice['TIME_DELTA'].reset_index(drop=True), 
    'w': production
})

fig, ax1 = plt.subplots(figsize=(18,10))
ax2 = ax1.twinx()

ax1.set_title('Production Rate')
ax1.set_xlim(t_min, t_max)
ax2.set_xlim(t_min, t_max)
ax1.grid(which='minor', visible=True, linestyle=':')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel(f'cartridges / {interval} min')
ax2.set_ylabel('cumulative')
sns.histplot(data=plot_data, x='t', weights='w', binwidth=interval/60, ax=ax1)
sns.histplot(data=plot_data, x='t', weights='w', cumulative=True, 
             element='step', fill=False, bins=1000, ax=ax2)
ax2.text(
    x=t_min+(t_max-t_min)*0.01,
    y=fig.gca().get_ylim()[1]*0.975,
    s="INPUT"
      "\npupae in: {9:,}\nmale ratio: {10:.2f}"
      "\nemergence: {8}\nmale recall: {6}\nfemale recall: {7}"
      "\nRESULT"
      "\nmales: {0}\nfemales: {1}\ncartridges: {2}"
      "\ncontamination: {3:.4%}\nmale efficiency: {4:.2%}"
      "\nfemale efficiency: {5:.2%}".format(
        int(round(m_crtg.values[-1])),
        int(round(f_crtg.values[-1])),
        int(np.ceil(production.sum())),
        cont_rate.values[-1],
        m_eff, f_eff, tpr, 1-fpr, emerg_ratio, n_pupa, user_m
    ),
    ha='left', va='top',
    bbox=dict(
        boxstyle="square", 
        ec=(0.75, 0.75, 0.75), 
        fc=(1, 1, 1), 
        alpha=0.75,
    ),
)
st.pyplot()

# plot contamination
plt.figure(figsize=(16, 8))
plt.title('Female Contamination')
plt.xlim(t_min, t_max)
plt.grid(which='minor', visible=True, linestyle=':')
plt.xlabel('Time (hours)')
plt.ylabel('Female ratio')
plt.plot(df_slice['TIME_DELTA'], cont_rate)
st.pyplot()