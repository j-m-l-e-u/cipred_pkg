import polars as pl
import numpy as np
from sklearn.ensemble._forest import _get_n_samples_bootstrap,_generate_sample_indices


def lu_fit(forest,X,y):
    '''
    The function builds a Random Forest with its {tree,terminal node} OOB errors.
    The outputs are the fitted Random Forest model and a 'polar' DataFrame of the
    {tree,terminal node} OOB errors.

    Parameters:
    -----------
    forest: random forest regressor from the `scikit-learn` package
        A not-yet-fitted random forest regressor.

    X: {array-like} of shape (n_samples, n_features).
        The training input samples.

    y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values.
    '''

    # Fit the RF model
    forest.fit(X, y)

    # Apply trees in the forest to X to get leaf node (terminal nodes) indices
    train_terminal_nodes = forest.apply(X)

    # Out of Bag errors
    oob_error = forest.predict(X)-y

    # Concatenate terminal nodes w/ oob_errors
    df_oe = pl.DataFrame(train_terminal_nodes)
    df_oob_e = pl.DataFrame({'oob_e' : oob_error})
    df_oe = pl.concat((df_oe,df_oob_e),how= 'horizontal')
    colnames = list(df_oe.columns)[0:forest.get_params()['n_estimators']]

    # keep Out Of Bag observations
    n_samples = X.shape[0]
    n_trees = forest.get_params()['n_estimators']
    for i, (tree) in enumerate(forest.estimators_):
        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples,max_samples=None)
        sampled_indices = _generate_sample_indices(tree.random_state, n_samples,n_samples_bootstrap)
        df_oe[sampled_indices,i] = np.NaN

    # melt the data frame to gather all the errors corresponding to identical pairs of {tree,terminal_node}
    df_oe = df_oe.melt(id_vars='oob_e',value_vars=colnames,variable_name='tree', value_name='terminal_node')
    df_oe = df_oe.drop_nulls()
    df_oe = df_oe.groupby(['tree','terminal_node']).agg(pl.col("oob_e"))

    return forest, df_oe

def lu_pred(forest,oob_error,X,alpha,keep_oob_error=False):
    '''
    The function predicts values and their confidence interval (CI) based on
    the Random Forest and its {tree,terminal node} oob errors.
    The output is a `polar` DataFrame with a column providing the prediction,
    and two columns describing the lower and upper limit of the CI.

    Parameters:
    -----------
    forest: random forest regressor from the `scikit-learn` package
        A fitted random forest regressor.

    oob_error: `polars` DataFrame provided by the function `fit_lu`

    X: array-like of shape (n_samples, n_features).
        The input samples.

    alpha: float.
        Significance level desired for the conditional prediction intervals.
        E.g. alpha = 0.05 or 5% gives a CI of 95%.

    keep_oob_error: boolean, default=False
        Concatenate the list of oob errors to the resulting `polar` DataFrame
    '''

    # Apply trees in the forest to X, return leaf node indices
    terminal_nodes = forest.apply(X)

    # retrieve the oob_error given the sequence of the terminal nodes
    df_pe = pl.DataFrame(terminal_nodes)
    df_pe_tmp = pl.DataFrame({
                     'pred' : forest.predict(X),
                     'id'   : np.arange(0,X.shape[0])})

    df_pe = pl.concat((df_pe,df_pe_tmp),how= 'horizontal')

    colnames = list(df_pe.columns)[0:forest.get_params()['n_estimators']]

    df_pe = df_pe.melt(id_vars=['id','pred'],value_vars=colnames,variable_name='tree', value_name='terminal_node')

    df_pe = df_pe.join(oob_error,on = ['tree','terminal_node'], how="left")

    # some predictors use some terminal nodes that has never been used by a tree to provide a prediction
    # No OOB error is thus registered for this {tree,terminal node}. We remove these rows.
    df_pe = df_pe.drop_nulls()

    # for each {id, prediction, obs} we gather the oob errors, and flatten the list.
    df_pe = df_pe.groupby(['id','pred']).agg(pl.col("oob_e").flatten().sort())

    df_pe = df_pe.sort("id")

    p = [alpha/2,1-alpha/2]

    df_pe = df_pe.with_columns(
        pl.struct(["pred", "oob_e"]).apply(lambda x: x['pred'] + np.asarray(x['oob_e'])[np.floor(len(x['oob_e'])*p[0]).astype(np.int64)]).alias('N'+str(p[0])),
        pl.struct(["pred", "oob_e"]).apply(lambda x: x['pred'] + np.asarray(x['oob_e'])[np.floor(len(x['oob_e'])*p[1]).astype(np.int64)]).alias('N'+str(p[1]))
    )

    if (not keep_oob_error): df_pe = df_pe.drop('oob_e')

    return df_pe

def lu_pred_prime(forest,oob_error,X,y_obs,alpha,keep_oob_error=False):
    '''
    The function predicts values, their confidence interval (CI) and the
    p-values of the observations based on the Random Forest
    and its {tree,terminal node} oob errors.
    The output is a `polar` DataFrame with one column providing the prediction,
    one column providing the value to be tested, two columns providing
    the lower and upper limit of the CI, and one column providing the p-value
    of the value to be tested.

    Parameters:
    -----------
    forest: random forest regressor from the `scikit-learn` package
        A fitted random forest regressor.

    oob_error: `polars` DataFrame provided by the function `fit_lu`

    X: array-like of shape (n_samples, n_features).
        The input samples.

    y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        The values to be tested.

    alpha: float.
        Significance level desired for the conditional prediction intervals.
        E.g. alpha = 0.05 or 5% gives a CI of 95%.

    keep_oob_error: boolean, default=False
        Concatenate the list of oob errors to the resulting `polar` DataFrame
    '''


    # Apply trees in the forest to X, return leaf node indices
    terminal_nodes = forest.apply(X)

    # retrieve the oob_error given the sequence of the terminal nodes
    df_pe = pl.DataFrame(terminal_nodes)
    df_pe_tmp = pl.DataFrame({
                     'pred' : forest.predict(X),
                     'obs'  : np.ravel(y_obs),
                     'id'   : np.arange(0,X.shape[0])})

    df_pe = pl.concat((df_pe,df_pe_tmp),how= 'horizontal')

    colnames = list(df_pe.columns)[0:forest.get_params()['n_estimators']]

    df_pe = df_pe.melt(id_vars=['id','pred','obs'],value_vars=colnames,variable_name='tree', value_name='terminal_node')

    df_pe = df_pe.join(oob_error,on = ['tree','terminal_node'], how="left")

    # some predictors use some terminal nodes that has never been used by a tree to provide a prediction
    # No OOB error is thus registered for this {tree,terminal node}. We remove these rows.
    df_pe = df_pe.drop_nulls()

    # for each {id, prediction, obs} we gather the oob errors, and flatten the list.
    df_pe = df_pe.groupby(['id','pred','obs']).agg(pl.col("oob_e").flatten().sort())

    df_pe = df_pe.sort("id")

    p = [alpha/2,1-alpha/2]

    df_pe = df_pe.with_columns(
        pl.struct(["pred", "oob_e"]).apply(lambda x: x['pred'] + np.asarray(x['oob_e'])[np.floor(len(x['oob_e'])*p[0]).astype(np.int64)]).alias('N'+str(p[0])),
        pl.struct(["pred", "oob_e"]).apply(lambda x: x['pred'] + np.asarray(x['oob_e'])[np.floor(len(x['oob_e'])*p[1]).astype(np.int64)]).alias('N'+str(p[1])),
        pl.struct(["pred", "oob_e",'obs']).apply(lambda x: np.mean((x['pred']+np.asarray(x['oob_e']))<=x['obs'])).alias('p-value')
    )

    if (not keep_oob_error): df_pe = df_pe.drop('oob_e')

    return df_pe
