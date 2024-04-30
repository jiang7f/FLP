
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
if __name__ == '__main__':
    
    import pandas as pd
    df = pd.read_csv('data/decompose_result.csv',index_col=False)
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    scale = 0.009
    fig = plt.figure(figsize=(2500*scale , 800*scale))
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 62
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Times New Roman'
    mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
    mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    # set axes linewidth
    mpl.rcParams['axes.linewidth'] = 5
    ## set ticks linewidth
    mpl.rcParams['xtick.major.size'] = 20
    mpl.rcParams['xtick.major.width'] = 5
    mpl.rcParams['ytick.major.size'] = 20
    mpl.rcParams['ytick.major.width'] = 5
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    axes = plt.axes([0,0,1,1])
    for qi,gr in df.groupby('num_qubits'):
        gr = gr.sort_values('max',ascending=False)
        depth = gr['depth'].values
        for j,dep in enumerate(depth):
            if j == len(depth)-1:
                axes.bar(qi+0.1*j,dep,width=0.1,label='unitary',color=colors[-1],edgecolor='black')
            else:
                axes.bar(qi+0.1*j,dep,width=0.1,label='max_control_qubits='+str(qi-(j)),color=colors[gr['max'].values[j]-2])
    axes.legend(frameon=False,bbox_to_anchor=(2.5,1.3),ncol=4)
    axes.tick_params(axis='x',which='major',length=20)
    # axes.set_xticklabels(allnames,rotation=90)
    axes.set_xlabel('# qubits')
    axes.set_yscale('log')
    axes.set_xticks(df.num_qubits.unique()+0.5)
    axes.set_xticklabels(df.num_qubits.unique())
    axes.grid(axis='y',linestyle='--',linewidth=3,color='#B0B0B0')
    axes.set_ylabel('time (s)')
    fig.savefig('figures/decompose_depth.svg',dpi=600,bbox_inches='tight')

    
