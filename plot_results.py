"""Optional static plots; raw metrics do not depend on matplotlib."""
import argparse
import json
from pathlib import Path


def plot(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=Path(path)
    report=json.loads((path/'metrics.json').read_text())
    if not report.get('paired_test',{}).get('llm',{}).get('n'):
        print('No paired predictions to plot.')
        return
    fig, axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    labels=['≤128','129–256','257–512','513–1024','>1024']
    matrix=report['paired_test']['llm']['confusion_matrix_actual_rows_predicted_columns']
    ax=axes[0,0]
    ax.imshow(matrix,cmap='Blues')
    maximum=max(max(row) for row in matrix)
    for i,row in enumerate(matrix):
        for j,value in enumerate(row):
            ax.text(j,i,str(value),ha='center',va='center',color='white' if value > maximum/2 else 'black')
    ax.set(xticks=range(5),yticks=range(5),xticklabels=labels,yticklabels=labels,
           xlabel='Predicted bucket',ylabel='Actual bucket',title='Qwen forecast: confusion matrix')
    ax.tick_params(axis='x',rotation=45)
    for k,ax in enumerate([axes[0,1],axes[0,2],axes[1,0],axes[1,1]]):
        for method,metrics in report['paired_test'].items():
            bins=[b for b in metrics['thresholds'][k]['reliability'] if b['n']]
            ax.plot([b['mean_probability'] for b in bins],[b['observed_frequency'] for b in bins],
                    marker='o',label=method)
            if method=='llm':
                for b in bins:
                    ax.annotate(f"n={b['n']}",(b['mean_probability'],b['observed_frequency']),xytext=(3,4),textcoords='offset points',fontsize=7)
        threshold=report['paired_test']['llm']['thresholds'][k]['threshold']
        ax.plot([0,1],[0,1],'--',color='gray',linewidth=1)
        ax.set(xlim=(-.03,1.06),ylim=(-.03,1.07),xlabel='Mean predicted probability',
               ylabel='Observed frequency',title=f'P(L > {threshold}): 5-bin reliability')
    axes[0,1].legend(fontsize=8)
    ax=axes[1,2]
    methods=list(report['paired_test'])
    ax.barh(methods,[report['paired_test'][m]['macro_threshold_brier'] for m in methods],color='#357f96')
    ax.set(xlabel='Mean threshold Brier score (lower is better)',title='Same paired test rows for every method')
    banner='SYNTHETIC FIXTURE — NO MODEL RUN' if report['synthetic'] else f"{report['mode'].upper()} · {report['counts']['paired_test']} paired test prompts · cap {report['cap']}"
    fig.suptitle(banner+'\n'+report['verdict'],fontsize=15)
    fig.savefig(path/'evaluation.png',dpi=160)
    plt.close(fig)
    print(path/'evaluation.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run_directory')
    plot(parser.parse_args().run_directory)
