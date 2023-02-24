import pandas as pd
import numpy as np
from sklearn.ensemble._forest import _get_n_samples_bootstrap,_generate_sample_indices

def lu_fit(forest,X,y):

    forest.fit(X, y)

    # Apply trees in the forest to X to get leaf node (terminal nodes) indices
    train_terminal_nodes = forest.apply(X)

    # Out of Bag errors
    oob_error = forest.predict(X)-y

    # Concatenate terminal nodes w/ oob_errors
    df_oe = pd.DataFrame(train_terminal_nodes)
    df_oe['oob_e'] = oob_error
    colnames = list(df_oe.columns)[0:forest.get_params()['n_estimators']]

    # keep Out Of Bag observations
    n_samples = X.shape[0]
    n_trees = forest.get_params()['n_estimators']
    for i, (tree) in enumerate(forest.estimators_):
        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples,max_samples=None)
        sampled_indices = _generate_sample_indices(tree.random_state, n_samples,n_samples_bootstrap)
        df_oe.iloc[sampled_indices,i] = np.NaN

    # melt the data frame to gather all the errors corresponding to identical pairs of {tree,terminal_node}
    df_oe = pd.melt(df_oe,id_vars='oob_e',value_vars=colnames,var_name='tree', value_name='terminal_node')
    df_oe = df_oe.dropna(how='any', axis=0)
    df_oe = df_oe.astype({"terminal_node": int})
    df_oe = df_oe.groupby(['tree','terminal_node']).agg({'oob_e': list})

    def tmp(x):
        if (len(np.asarray(x['oob_e']).shape)<1):
            return [x['oob_e']]
        else:
            return x['oob_e']

    df_oe['oob_e'] = df_oe.apply(lambda x: tmp(x),axis=1)

    return forest, df_oe

def lu_pred(forest,oob_error,X,alpha,keep_oob_error=False):

    # Apply trees in the forest to X, return leaf node indices
    terminal_nodes = forest.apply(X)

    # retrieve the oob_error given the sequence of the terminal nodes
    df_pe = pd.DataFrame(terminal_nodes)

    colnames = list(df_pe.columns)[0:forest.get_params()['n_estimators']]
    df_pe['pred'] = forest.predict(X)
    df_pe['id'] = np.arange(0,df_pe.shape[0])

    df_pe = pd.melt(df_pe,id_vars=['id','pred'],value_vars=colnames,var_name='tree', value_name='terminal_node')
    df_pe = df_pe.set_index(['tree','terminal_node'])

    df_pe = df_pe.join(oob_error)

    # some predictors use some terminal nodes that has never been used by a tree to provide a prediction
    # No OOB error is thus registered for this {tree,terminal node}. We remove these rows.
    df_pe = df_pe.dropna(how='any', axis=0)

    df_pe = df_pe.groupby(['id','pred']).agg({'oob_e': list})

    def flatten(l):
        return [item for sublist in l for item in sublist]

    df_pe['oob_e'] = df_pe.apply(lambda x: np.sort(flatten(x['oob_e'])),axis=1)

    df_pe = df_pe.reset_index()
    df_pe = df_pe.set_index('id').sort_index()

    p = [alpha/2,1-alpha/2]

    df_pe['N'+str(p[0])] = df_pe.apply(lambda x: x['pred'] + np.asarray(x['oob_e'])[np.floor(len(x['oob_e'])*p[0]).astype(np.int64)],axis=1)
    df_pe['N'+str(p[1])] = df_pe.apply(lambda x: x['pred'] + np.asarray(x['oob_e'])[np.floor(len(x['oob_e'])*p[1]).astype(np.int64)],axis=1)

    if (not keep_oob_error): df_pe = df_pe.drop(columns=['oob_e'])

    return df_pe

def lu_pred_prime(forest,oob_error,X,y_obs,alpha,keep_oob_error=False):

    # Apply trees in the forest to X, return leaf node indices
    terminal_nodes = forest.apply(X)

    # retrieve the oob_error given the sequence of the terminal nodes
    df_pe = pd.DataFrame(terminal_nodes)

    colnames = list(df_pe.columns)[0:forest.get_params()['n_estimators']]
    df_pe['pred'] = forest.predict(X)
    df_pe['obs'] = np.ravel(y_obs)#y_obs
    df_pe['id'] = np.arange(0,df_pe.shape[0])

    df_pe = pd.melt(df_pe,id_vars=['id','pred','obs'],value_vars=colnames,var_name='tree', value_name='terminal_node')
    df_pe = df_pe.set_index(['tree','terminal_node'])

    df_pe = df_pe.join(oob_error)

    # some predictors use some terminal nodes that has never been used by a tree to provide a prediction
    # No OOB error is thus registered for this {tree,terminal node}. We remove these rows.
    df_pe = df_pe.dropna(how='any', axis=0)

    # for each {id, prediction, obs} we gather the oob errors
    df_pe = df_pe.groupby(['id','pred','obs']).agg({'oob_e': list})

    def flatten(l):
        return[item for sublist in l for item in sublist]

    df_pe['oob_e'] = df_pe.apply(lambda x: np.sort(flatten(x['oob_e'])),axis=1)

    df_pe = df_pe.reset_index()
    df_pe = df_pe.set_index('id').sort_index()

    p = [alpha/2,1-alpha/2]

    df_pe['N'+str(p[0])] = df_pe.apply(lambda x: x['pred'] + np.asarray(x['oob_e'])[np.floor(len(x['oob_e'])*p[0]).astype(np.int64)],axis=1)
    df_pe['N'+str(p[1])] = df_pe.apply(lambda x: x['pred'] + np.asarray(x['oob_e'])[np.floor(len(x['oob_e'])*p[1]).astype(np.int64)],axis=1)
    df_pe['p-value'] = df_pe.apply(lambda x: np.mean((x['pred']+np.asarray(x['oob_e']))<=x['obs']),axis=1)

    if (not keep_oob_error): df_pe = df_pe.drop(columns=['oob_e'])

    return df_pe
