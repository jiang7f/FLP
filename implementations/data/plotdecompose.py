
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
if __name__ == '__main__':
    step = 1
    import pandas as pd
    df = pd.read_csv('decompose_result.csv',index_col=False)
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
    colors = ['#0077b6', '#0096c7', '#00b4d8', '#48cae4', '#90e0ef', '#ade8f4', '#caf0f8', '#eff6fb']
    # 橘色系颜色列表
    colors = ['#ff5733', '#ff6e40', '#ff874d', '#ff9f5a', '#ffb867', '#ffd074', '#ffe781', '#fff08e']



    axes = plt.axes([0,0,1,1])
    road = -1
    road_list = []
    for qi,gr in df.groupby('num_qubits'):
        gr = gr.sort_values('max',ascending=False)
        depth = gr['time'].values
        for j,dep in enumerate(depth):
            road+=step
            if qi == 8:
                if j == len(depth)-1:
                    axes.bar(road,dep,width=step,label='unitary',color=colors[0],edgecolor='black')
                else:
                    print(qi, j, qi - j, dep)
                    axes.bar(road,dep,width=step,label='max_control_qubits='+str(qi-j),color=colors[qi-j-1],edgecolor='black')
            else:
                if j == len(depth)-1:
                    axes.bar(road,dep,width=step,color=colors[0],edgecolor='black')
                else:
                    print(qi, j, qi - j, dep)
                    axes.bar(road,dep,width=step,color=colors[qi-j-1],edgecolor='black')
        road+=step
        road_list.append(road)

    axes.legend(frameon=False,bbox_to_anchor=(1.56,1.5),ncol=4)
    axes.tick_params(axis='x',which='major',length=20)
    # axes.set_xticklabels(allnames,rotation=90)
    axes.set_xlabel('# qubits')
    axes.set_yscale('log')
    road_list = [b - (a + 4) * step / 2 for a, b in enumerate(road_list)]
    axes.set_xticks(road_list)
    axes.set_xticklabels(df.num_qubits.unique())
    axes.grid(axis='y',linestyle='--',linewidth=3,color='#B0B0B0')
    axes.set_ylabel('time (s)')
    fig.savefig('../figures/decompose_time_gp.svg',bbox_inches='tight')

    
