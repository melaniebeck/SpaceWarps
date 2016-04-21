import machine_utils as ml
import swap
import numpy as np
from numpy import random as rand
from astropy.table import Table 
import cPickle
import pdb
import matplotlib.pyplot as plt
import metrics
#from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
import sklearn.metrics as mx
from optparse import OptionParser
from figure_styles import set_big

def find_nearest(a, val):
    idx = (np.abs(a-val)).argmin()
    return idx

def read_pickle(filename):
    F = open(filename,'rb')
    contents = cPickle.load(F)
    F.close()
    return contents

def get_answers(gzfile, sample):
    answers = np.array([])
    for s in sample:
        category,kind,flavor,truth = Nair_or_Not(s)
        if truth == 'SMOOTH':
            answers = np.append(answers,1)
        else: 
            answers = np.append(answers,0)
    return answers

def plot_singlemetric_samplesize(metrics, method, thresh=None, 
                                 keys=None, labels=None, name_modifier=None):

    set_big()

    colors = ['red','purple', 'green','blue','orange','cyan','pink']
    un = np.unique(metrics['k'])
    unique_k = un[np.where(un % 5 == 0)]

    fig = plt.figure(figsize=(20,12))

    j=0
    for k in unique_k:
        if k==10:
            ss = np.where((metrics['k']==k) | (metrics['k']==9))
        elif k==50:
            ss = np.where((metrics['k']==k) | (metrics['k']==49))
        else:
            ss = np.where(metrics['k']==k)
            
        ax = fig.add_subplot(2,5,1+j)
            
        vals = np.where(metrics['thresh']==thresh)
        
        if thresh:
            for idx, key in enumerate(keys):
                y = []
                for i in range(len(ss[0])):
                    l = find_nearest(metrics['thresh'][ss][i], thresh)
                    #print key, k, metrics['n'][ss][i], l
                    #print metrics['acc_thresh'][ss][i][l], thresh
                    #print len(metrics['acc_thresh'][ss][i]), len(metrics[key][ss][i])
                    if key == 'precision': y.append(1-metrics[key][ss][i][l])
                    else: y.append(metrics[key][ss][i][l])
                    
                ax.semilogx(metrics['n'][ss], y, color=colors[idx], 
                            marker='^', label=labels[idx])
        else:
            ax.semilogx(metrics['n'][ss], metrics['accuracy_score'][ss], 
                        color='red', marker='^', label='Accuracy')
            ax.semilogx(metrics['n'][ss], metrics['precision_score'][ss], 
                        color='green', marker='^', label='Precision')
            ax.semilogx(metrics['n'][ss], metrics['recall_score'][ss],
                        color='blue', marker='^', label='Recall')
            ax.semilogx(metrics['n'][ss], metrics['f1_score'][ss],
                        color='orange', marker='^', label='F1 score')
            ax.semilogx(metrics['n'][ss], metrics['roc_auc_score2'][ss],
                        c='cyan', marker='^', label='ROC AUC')
            print "plotted everything for K=%i"%k
            
        ax.set_title("K=%i"%k)
        ax.set_ylim(0.,1.)
        x_max = np.max(metrics['n'])
        ax.set_xlim(5,10**round(np.log10(x_max)))
        j+=1

    ax.legend(loc='best')

    if thresh:
        fig.suptitle("Thresh = %0.1f"%thresh, fontsize=20, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, .97])
        plt.savefig('%s_%.2f_metric_samplesize_%s.png'%(method,thresh, 
                                                        name_modifier))
    else:
        plt.tight_layout()
        plt.savefig('%s_metric_samplesize_%s.png'%(method, name_modifier))
        print '%s_metric_samplesize_%s.png'%(method, name_modifier)

    plt.show()
    plt.close()

def plot_metriccurves(eval_mx, metric, method, **kwargs):
    

    def curveplot(eval_mx, keys, colors, labels, filename, 
                  complement_x=False, one_to_one=True, show=True):
        fig = plt.figure(figsize=(15,15))

        j,m = 0,0
        for i,k in enumerate(eval_mx['k']):
            if i > 0 and eval_mx['n'][i]>eval_mx['n'][i-1]: j+=1
            if k==5: m=0

            ax = fig.add_subplot(3,2,1+j)

            if one_to_one:
                ax.plot([0,1],[0,1], lw=3,ls='--',color='black')
            
            if complement_x: 
                x = 1-eval_mx[keys['x']][i]
                y = eval_mx[keys['y']][i]
                score = 1-eval_mx[keys['score']][i]
            else: 
                x = eval_mx[keys['x']][i]
                y = eval_mx[keys['y']][i]
                score = eval_mx[keys['score']][i]

            ax.plot(x, y, lw=2, color=colors[m], label='K=%i (%.3f)'%(k, score))

            ax.set_title('N=%i'%eval_mx['n'][i])
            ax.legend(loc=labels['leg'])

            if j % 2 == 0:
                ax.set_ylabel(labels['y'], fontsize=16)
            if j == 4 or j == 5:
                ax.set_xlabel(labels['x'], fontsize=16)
            m+=1

        fig.tight_layout()
        fig.savefig(filename)
        if show: plt.show()
        plt.close()


    if metric == 'ROC':

        keys = {'x':'falsepos', 'y':'truepos', 'score':'roc_auc_score2'}

        colors = ['aquamarine','aqua','cyan','deepskyblue', 'cornflowerblue',
                  'cadetblue','darkcyan', 'blue','darkblue','darkslateblue']

        labels = {'x': 'FPR', 'y':'TPR', 'leg':'lower right'}

        filename = '%s_ROC_%s.png'%(method, name_modifier)

        curveplot(eval_mx, keys, colors, labels, filename, **kwargs)

    elif metric == 'PR':

        keys = {'x':'recall', 'y':'precision', 'score':'recall_score'} 
       
        colors = ['lavenderblush','lavender','thistle','violet','fuchsia',
                 'blueviolet','darkviolet','darkmagenta','purple','indigo']

        labels = {'x': 'Recall (TPR)', 'y':'Precision', 'leg':'lower left'}

        filename = '%s_PR_%s.png'%(method, name_modifier)

        curveplot(eval_mx, keys, colors, labels, filename, **kwargs)

    elif metric == 'CC':

        keys = {'x':'precision', 'y':'recall', 'score':'precision_score'}

        colors = ['palegreen','lime','lawngreen','lightgreen', 'springgreen',
                 'limegreen','olivedrab','green','darkolivegreen','darkgreen']

        labels = {'x': 'Contamination', 'y':'Completeness', 'leg':'lower right'}

        filename = '%s_CC_%s.png'%(method, name_modifier)

        curveplot(eval_mx, keys, colors, labels, filename, 
                  complement_x=True, one_to_one=False)

    elif metric == 'all':

        plot_metriccurves(eval_mx, metric='ROC', method=method, **kwargs)
        plot_metriccurves(eval_mx, metric='PR', method=method, **kwargs)
        plot_metriccurves(eval_mx, metric='CC', method=method, **kwargs)

    else:
        print 'Not a valid  metric!'
        exit()


def plot_the_shits(method, metric='all',  **kwargs):
    F = open('%s_eval_%s.pickle'%(method, kwargs['name_modifier']), 'rb')
    eval_mx = cPickle.load(F)
    F.close()
 
    #if len(methods) > 1:
    #   F = open('%s_eval.pickle'%methods[1],'rb')
    #   eval_mx2 = cPickle.load(F)
    #   F.close()      
    #   eval_mx = [eval_mx, eval_mx2]

    plot_singlemetric_samplesize(eval_mx, method, **kwargs)
    #plot_metriccurves(eval_mx, metric, method)




def main():

    parser = OptionParser()
    parser.add_option("-w", dest="weight", default='uniform', 
                      help="Run KNC with preferred weighting.")
    parser.add_option("-t", dest="thresh", default=None, help="Set threshold")
    parser.add_option("-p", dest="plotonly", action='store_true', default=False,
                      help="Skip machine learning and go straight to plotting")
    parser.add_option("-n", dest="name_modifier", default=None, 
                      help="Additional naming identification for output files")
    (options, args) = parser.parse_args()

    if options.thresh:  thresh = float(options.thresh)
    else: thresh = None

    if options.plotonly:
        #metrics_uni = read_pickle('KNC_uniform_eval.pickle')
        #metrics_dist = read_pickle('KNC_distance_eval.pickle')

        ###################### JUST PLOT THE SHITS  ###########################
        kwargs = {'thresh':thresh, 
                  'keys':['accuracy','contamination', 'completeness', 
                          'falseomis','trueneg'],
                  'labels':['Accuracy', 'Contamination (S)', 'Completeness (S)',
                            'Contamination (F)', 'Completeness (F)'], 
                  'name_modifier':options.name_modifier}
        print options.weight
        plot_the_shits(metric='all', method='KNC_%s'%options.weight, **kwargs)
        #explore_accuracy(method='KNC_uniform')
        exit()


    ################### READ IN TRAINING / VALIDATION DATA #############   
    
    filename = 'GZ2_testML2_metadata.pickle'
    data = swap.read_pickle(filename, 'metadata')

    # This is the "New" validation sample -- Expertly classified
    valid_idx = np.where((data['MLsample']=='valid') & (data['GZ2_label']!=-1)
                         & (data['Nair_label']!=-1) 
                         & (data['Expert_label']!=-1))
    valid = data[valid_idx]
    print len(valid)
    valid_meta, valid_features = ml.extract_training(valid)
    valid_labels_ex = valid_meta['Expert_label'].filled()
    valid_labels_gz = valid_meta['GZ2_label'].filled()
    valid_labels_nr = valid_meta['Nair_label'].filled()

    # Let's try to recreate what I had before. 
    valid2_idx = np.where((data['Nair_label']!=-1))
    valid2 = data[valid2_idx]
    valid2_meta, valid2_features = ml.extract_training(valid2)
    valid2_labels = valid2_meta['Nair_label'].filled()

    # Which validation sample do I want to use? BLAH.
    # --> Used this to try to replicate what I had a month ago (Nair "truth")
    valid_features = valid2_features
    valid_labels = valid2_labels
    
    # Now test on the new, smaller validation sample
    # --> first, still with Nair "truth"
    #valid_labels = valid_labels_nr
    # --> second, using GZ2 user "truth"
    #valid_labels = valid_labels_gz
    # --> finally, using Expert "truth"
    #valid_labels = valid_labels_ex

    # Load up the training set (ALL GZ labels)
    train_idx = np.where((data['MLsample']!='valid') & (data['GZ2_label']!=-1))
    train = data[train_idx]
    train_meta, train_features = ml.extract_training(train)
    train_labels = train_meta['GZ2_label'].filled()
    
   
    # select various and increasing size training samples
    # -------------------------------------------------------------------
    N = [50,100,500,1000,5000,10000,50000]#
    K = [5,10,15,20,25,30,35,40,45,50]
    
    evaluation_metrics = {'precision':[], 'recall':[], 'pr_thresh':[], 
                          'falsepos':[], 'truepos':[], 'roc_thresh':[],
                          'accuracy':[], 'thresh':[], 'falseomis':[],
                          'falseneg':[], 'trueneg':[], 
                          'contamination':[], 'completeness':[],
                          'precision_score':[], 'recall_score':[], 
                          'accuracy_score':[], 'roc_auc_score1':[],
                          'roc_auc_score2':[], 'f1_score':[], 'k':[],'n':[]}
    
    ################### RUN CLASSIFIERS WITIH VARIOUS PARAMS ############
        
    for j,n in enumerate(N):
        train_features_sub = train_features[:n]
        train_labels_sub = train_labels[:n]
        
        #ratio = float(np.sum(train_labels_sub==1))/len(train_labels_sub)
        #print "Ratio of Smooth / Total for training sample (%i): %f"\
        #    %(n, ratio)
        
        for i,k in enumerate(K):
                    
            # Adjust k because it can't be => sample size
            if n <= k: k = n-1
            
            preds, probs, machine = ml.runKNC(train_features_sub, 
                                              train_labels_sub, 
                                              valid_features, 
                                              N=k, weights=options.weight)

            #preds = ml.runRNC(train_sample, labels, valid_sample, R=k, 
            #                  weights='distance', outlier=0)
            
            fps, tps, thresh=metrics._binary_clf_curve(valid_labels,probs[:,1])

            metrics_list = metrics.compute_binary_metrics(fps, tps)
            [acc, tpr, fpr, fnr, tnr, prec, fdr, fomis, npv] = metrics_list

            evaluation_metrics['completeness'].append(tpr)
            evaluation_metrics['contamination'].append(fdr)
            evaluation_metrics['falseneg'].append(fnr)
            evaluation_metrics['trueneg'].append(tnr)
            evaluation_metrics['falseomis'].append(fomis)
            evaluation_metrics['accuracy'].append(acc)
            evaluation_metrics['thresh'].append(thresh)
            
            # Curves -- for plotting ROC and PR
            pp, rr, thresh2 = mx.precision_recall_curve(valid_labels,probs[:,1])
            evaluation_metrics['precision'].append(pp)
            evaluation_metrics['recall'].append(rr)
            evaluation_metrics['pr_thresh'].append(thresh2)
            
            fpr, tpr, thresh3=mx.roc_curve(valid_labels, probs[:,1],pos_label=1)
            evaluation_metrics['falsepos'].append(fpr)
            evaluation_metrics['truepos'].append(tpr)
            evaluation_metrics['roc_thresh'].append(thresh3)
            
            # Single value metrics -- for plotting against N? K? whatever...
            evaluation_metrics['roc_auc_score1'].append(mx.auc(fpr, tpr))
            evaluation_metrics['roc_auc_score2'].append(mx.roc_auc_score(
                valid_labels,preds))
            evaluation_metrics['precision_score'].append(mx.precision_score(
                valid_labels,preds))
            evaluation_metrics['recall_score'].append(mx.recall_score(
                valid_labels,preds))
            evaluation_metrics['accuracy_score'].append(mx.accuracy_score(
                valid_labels,preds))
            evaluation_metrics['f1_score'].append(mx.f1_score(
                valid_labels,preds))
            
            # current k and n so I don't have to backstrapolate
            evaluation_metrics['k'].append(k)
            evaluation_metrics['n'].append(n)
            
                
    for key, val in evaluation_metrics.iteritems():
        evaluation_metrics[key] = np.array(evaluation_metrics[key])
        
    # If everything works... Let's save this huge structure as a pickle
    filename = 'KNC_%s_eval_%s.pickle'%(options.weight, options.name_modifier)
    F = open(filename,'wb')
    cPickle.dump(evaluation_metrics, F, protocol=2)
    print "Saved evaluation metrics %s"%filename

    ######################### PLOT THE SHITS  #############################
    #kwargs = {'thresh':.5, 'keys':['accuracy','precision','recall']}
    #plot_the_shits(method='KNC_uniform', metric='all', **kwargs)
    #explore_accuracy(method='KNC_uniform')
        
    exit()



if __name__ == '__main__':
    main()


# Not sure exactly how GridSearch works...
# Second Test: try running KNC using a GridSearch 
for j,n in enumerate(N):
    train = sample[:n]
    train_data, train_sample = ml.extract_training(train)
    labels, not_found = get_truth(gztruth,train_data)

    tuned_parameters = [{'n_neighbors':K, 'weights':['uniform','distance']}]
    #scores = ['precision', 'recall']
    #for score in scores:
    clf = GridSearchCV(KNeighborsClassifier(n_neighbors=5), 
                       tuned_parameters, cv=5, scoring='precision_weighted',
                       error_score=np.nan)
    
    clf.fit(train_sample,labels)
    predictions = clf.predict(valid_sample)
    probas_ = clf.predict_proba(valid_sample)
    
    pp,rr,thresh = mx.precision_recall_curve(answers,probas_[:,1])
    precision.append(pp)
    recall.append(rr)
    thresholds.append(thresh)
    
    fpr, tpr, thresh = mx.roc_curve(answers, probas_[:,1], pos_label=1)
    falsepos.append(fpr)
    truepos.append(tpr)
    roc_thresh.append(thresh)
    
    print "Best params for N=%i sample size:"%n, clf.best_params_
    params.append(clf.best_params_)
    print "Accuracy score:", mx.accuracy_score(answers,predictions)
    accuracy.append(mx.accuracy_score(answers,predictions))
    print mx.classification_report(answers,predictions)


