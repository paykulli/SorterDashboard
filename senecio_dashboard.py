import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


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

if raw_path == None:
    raw = pd.read_csv('example.csv', parse_dates=['IMAGE_TIMESTAMP'])
else:
    raw = pd.read_csv(raw_path, parse_dates=['IMAGE_TIMESTAMP'])

if events_path != None:
    try:
        events = pd.read_excel(events_path, sheet_name=0, parse_dates=['TIME'])
    except:
        pass


# =============
# Preprocessing
# =============
df = raw.drop(
    ['IMAGE_NAME', 'XMIN', 'YMIN', 'XMAX', 'YMAX', 'SCORE'], axis=1)
df['CAMERA_ID'] = df['CAMERA_ID'].astype('int8')
df['CAMERA_MOSQUITO_NUM'] = df['CAMERA_MOSQUITO_NUM'].astype('int32')
df['TOTAL_MOSQUITO_NUM'] = df['TOTAL_MOSQUITO_NUM'].astype('int32')
df['MID_X'] = df['MID_X'].astype('float32')
df['MID_Y'] = df['MID_Y'].astype('float32')
df['MID_Y'] = 4096 - df['MID_Y'] + 4096 * (df['CAMERA_ID'] - 1)
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
        plt.axvline(x=to_midnight+offset, color=(0.7, 0.7, 1.0), ls='-', lw=2)
    for offset in range(12+int(d_min), int(d_max2-to_midnight), 24):
        plt.axvline(x=to_midnight+offset, color=(1, 1, 0.6), ls='-', lw=2)


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
st.sidebar.subheader('Heatmap')
n_outlier = st.sidebar.number_input('Number of outliers:', 0, 10, 0, 1)
r_outlier = st.sidebar.slider('Outlier radius (px):', 0, 100, 50, 1)
mirror = st.sidebar.checkbox('Mirror Heatmap', False)


# ========================
# Detection Frequency Plot
# ========================
st.header('Graphs')
st.subheader('Frequency')
plt.figure(figsize=(16,8))
plt.title('Detection Frequency')
plt.xlim(t_min-0.5, t_max+0.5)
plt.grid(which='minor', visible=True, linestyle=':')
plt.xticks(ticks=np.arange(round(t_min-0.5), round(t_max+0.5), step=5))
plt.xlabel('Time (hours)')
plt.ylabel('mosq. / hr')
sns.histplot(
    data=df[(df['COUNT'] <= count_cutoff) & 
            (df['TIME_DELTA'] <= t_max) & 
            (df['TIME_DELTA'] >= t_min)]['TIME_DELTA'], 
    binwidth=1, color='g', alpha=0.5,
)

# info box
plt.text(
    x=t_min, #plt.gca().get_xlim()[1]*0.015, 
    y=plt.gca().get_ylim()[1]*0.975,
    s="Start: {}\nFinish: {}\nParticipants: {}".format(
        df['IMAGE_TIMESTAMP'].min().round('S'), 
        df['IMAGE_TIMESTAMP'].max().round('S'), 
        len(df),
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
midday_midnight(df['IMAGE_TIMESTAMP'], df['TIME_DELTA'], t_min, t_max)
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



# =========================
# Cumulative Detection Plot
# =========================
st.subheader('Total')

# decide scale
if len(df[df['COUNT'] <= count_cutoff]) > 10**7:
    scale = 10**6
    scale_char = 'M'
elif len(df[df['COUNT'] <= count_cutoff]) > 10**4:
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
    x=df[df['COUNT'] <= count_cutoff]['TIME_DELTA'], 
    bins=1000, 
    histtype='stepfilled', 
    color='c', 
    alpha=0.2, 
    cumulative=True, 
    density=True
)
ax1.grid(visible=True, which='major', linestyle='-')
ax1.grid(visible=True, which='minor', linestyle=':')
ax1.set_yticks(np.arange(0, 1.01, 0.1))
ax1.set_yticks(np.arange(0, 1.01, 0.05), minor=True)
ax1.set_xticks(np.arange(0, df['TIME_DELTA'].max()+1, 5))
ax1.set_xticks(np.arange(0, df['TIME_DELTA'].max()+1, 2.5), minor=True)
ax1.set_xlim(0, df['TIME_DELTA'].max())
ax1.set_title('Cumulative Detection')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Proportion')

# cumulative count
ax2.hist(
    x=df[df['COUNT'] <= count_cutoff]['TIME_DELTA'], 
    bins=1000, 
    histtype='step', 
    color='k', 
    alpha=0.75, 
    cumulative=True)
ax2.grid(visible=False)
ax2.set_ylabel('Count')

# vertical percentile lines
for hr in range(12, int(df['TIME_DELTA'].max()), 12):
    answer = (
        len(df[(df['COUNT'] <= count_cutoff) & (df['TIME_DELTA'] <= hr)]) /
        len(df[(df['COUNT'] <= count_cutoff)])
    )
    plt.axvline(x=hr, color='b', linewidth=1)
    plt.text(
        x=hr, y=plt.gca().get_ylim()[1]*0.5, 
        s="{0:.0f} hr\n{2:.1f}{3}\n{1:.1%}".format(
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

# remove outliers
df_ = df.copy()
x_list = []
y_list = []
for i in range(n_outlier):    
    mode_xy = df_['XY'].mode()[0]
    mode_y = float(mode_xy.split()[1])
    buff_y = r_outlier
    mode_x = float(mode_xy.split()[0])
    buff_x = r_outlier
    df_ = df_[~((df_['MID_Y'] > mode_y-buff_y) & 
                (df_['MID_Y'] < mode_y+buff_y) & 
                (df_['MID_X'] > mode_x-buff_x) & 
                (df_['MID_X'] < mode_x+buff_x))]
    x_list.append(mode_x)
    y_list.append(mode_y)

if mirror:
    # plot mirrored heatmap
    df_l = df_.copy()
    df_l['MID_Y'] = 4096*cam_n - df['MID_Y']
    df_mirror = pd.concat([df_, df_l], axis=0)
    sns.jointplot(data=df_mirror[(df_mirror['COUNT'] <= count_cutoff) & 
                                 (df_mirror['TIME_DELTA'] <= t_max) & 
                                 (df_mirror['TIME_DELTA'] >= t_min)],
                  x='MID_X', 
                  y='MID_Y', 
                  kind='hex', 
                  cmap='CMRmap',
                  alpha=0.8,
                  height=7,
                 )
    plt.title('Conveyor Distribution (Mirrored)', y=1.2)
else:
    # plot regular heatmap
    sns.jointplot(data=df_[(df_['COUNT'] <= count_cutoff) & 
                           (df_['TIME_DELTA'] <= t_max) & 
                           (df_['TIME_DELTA'] >= t_min)],
                  x='MID_X', 
                  y='MID_Y', 
                  kind='hex', 
                  cmap='CMRmap',
                  alpha=0.8,
                  height=7,
                  )
    plt.title('Conveyor Distribution', y=1.2)
plt.ylim(0, 4096*cam_n)
plt.xlabel('Conveyor Length')
plt.ylabel('Conveyor Width')
st.pyplot()

# outlier list
if n_outlier:
    st.text('Outlier coordinates:')
    st.dataframe(pd.DataFrame({'X': x_list, 'Y': y_list}, dtype='int'))
