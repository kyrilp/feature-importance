import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# Finds the spear correlation for the input features
def spear(df, target_col):
    corrs = {}
    y = df[target_col]
    y_mean = df[target_col].mean()
    
    for col in df.drop(columns=target_col).columns:
        x = df[col]
        x_mean = df[col].mean()
        corrs[col] = abs(((x-x_mean)*(y-y_mean)).sum()/np.sqrt(((x-x_mean)**2).sum()*((y-y_mean)**2).sum()))
    return pd.DataFrame.from_dict(corrs, orient='index', columns=['Spearmans']).sort_values(by='Spearmans', ascending=False)

# Ranks the columns in the dataframe for spearmans
def rank_col(df):
    df_new = df.copy()
    for col in df_new.columns:
        counts = df_new[col].value_counts().to_dict()
        values = np.sort(np.unique(df_new[col].values))
        ranks = {}
        i = 1
        for val in values:
            ranks[val] = i + (counts[val]-1)/2
            i += counts[val]
        df_new[col] = df_new[col].apply(lambda x: ranks[x])
    return df_new

# This function takes a dataframe and decomposes the features into a few dimensions
# Then finds the 'load' of each component and the feature's importance
def PCA(df, target='price', components=2):
    # Reference: https://www.askpython.com/python/examples/principal-component-analysis
    # Convert dataframe to numpy matrix
    X = df.drop(columns=target).to_numpy()
    # Subtract means from each col to standardize
    X_mean = X - np.mean(X , axis=0)
    # Find the covariance matrix
    cov = np.cov(X_mean, rowvar=False)
    # Find eigenval and eigenvectors
    eig_val, eig_vec = np.linalg.eigh(cov)
    # Sort vectors and values
    sort_val = eig_val[np.argsort(eig_val)[::-1]]
    sort_vec = eig_vec[:,np.argsort(eig_val)[::-1]]
    vec_sub = sort_vec[:,0:components]
    X_red = np.dot(vec_sub.T, X_mean.T)
    PCA_vars = pd.DataFrame(X_red.T).var()
    PCA_sumvars = pd.DataFrame(X_red.T).var().sum()
    Exp = pd.DataFrame(PCA_vars / PCA_sumvars, columns=['Explained Variance'])
    Exp['PC'] = ['PC_1','PC_2']
    Exp.set_index('PC')

    PCA_cols = pd.DataFrame(X_red.T, columns=['PC_1','PC_2'])
    loads = pd.DataFrame(df.drop(columns='price').corrwith(PCA_cols['PC_1']).abs(), columns=['PC_1'])
    loads['PC_2'] = pd.DataFrame(df.drop(columns='price').corrwith(PCA_cols['PC_2']).abs(), columns=['PC_2'])
    
    return Exp, loads

# Implementation of mrmr to select features
def mrmr(X, y, k_features):
    # Keep track of selected and not selected columns
    not_selected = list(X.columns)
    selected = []
    f_scores = []
    for feat in not_selected:
        # Calculate F stat
        f = np.var(X[feat].to_numpy(), ddof=1)/np.var(y.to_numpy(), ddof=1)
        f_scores.append(f)
    selected_feat = not_selected[np.argmax(np.array(f_scores))]
    selected.append(selected_feat)
    not_selected.remove(selected_feat)

    for k in range(k_features-1):
        mrmr_scores = []
        for feat in not_selected:
            f = np.var(X[feat].to_numpy(), ddof=1)/np.var(y.to_numpy(), ddof=1)
            corrs = []
            for sel in selected:
                corrs.append(abs(X[feat].corr(X[sel])))
            corrs_mean = sum(corrs) / len(corrs)
            mrmr_scores.append(f/corrs_mean)
        selected_feat = not_selected[np.argmax(np.array(mrmr_scores))]
        selected.append(selected_feat)
        not_selected.remove(selected_feat)

    return selected

# Implementation of drop column importance
def drop_col_importance(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

    forest = RandomForestRegressor()
    forest.fit(X_train, y_train)
    baseline = r2_score(y_test, forest.predict(X_test))
    importances = []
    for col in X.columns:
        forest_ = RandomForestRegressor(n_estimators=10)
        forest_.fit(X_train.drop(columns=col), y_train)
        r2 = r2_score(y_test, forest_.predict(X_test.drop(columns=col)))
        importances.append(baseline-r2)
    return pd.DataFrame(list(zip(X.columns, importances)), columns=['Feature','Importance']).sort_values(by='Importance', ascending=False).set_index('Feature')

# Implementation of permutation importance
def perm_importance(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

    forest = RandomForestRegressor()
    forest.fit(X_train, y_train)
    baseline = r2_score(y_test, forest.predict(X_test))
    importances = []
    for col in X.columns:
        forest = RandomForestRegressor(n_estimators=10)
        save_col = X[col]
        X_train[col] = np.random.permutation(X_train[col])
        forest.fit(X_train, y_train)
        r2 = r2_score(y_test, forest.predict(X_test))
        importances.append(baseline-r2)
        X_train[col] = save_col
    return pd.DataFrame(list(zip(X.columns, importances)), columns=['Feature','Importance']).sort_values(by='Importance', ascending=False).set_index('Feature')

# Compare strategies using MAE and random forest
def compare_top_k(X, y, df, top_k=8):
    
    strategies = ['Spearman','PCA','MRMR','Drop Column','Permutation']
    # Build a data frame
    result = pd.DataFrame(strategies, columns=['Strategies'])
    
    # Run strategies
    spearmans = spear(rank_col(df), target_col='price')
    Exp, Loads = PCA(df, target='price', components=2)
    MRMR = mrmr(X, y, k_features=8)
    drop = drop_col_importance(X, y)
    perm = perm_importance(X,y)
    
    # Columns for each strategy sorted by importance
    top_k_strats = [spearmans.index[:top_k],
                    Loads.sort_values(by='PC_1', ascending=False).index[:top_k],
                    MRMR[:top_k],
                    drop.index[:top_k],
                    perm.index[:top_k]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

    for k in range(top_k):
        MAEs = []
        # Train a rf on the k-th column
        for strat in top_k_strats:
            # Get columns to use
            cols = strat[:k+1]
            # Train rf
            rf = RandomForestRegressor(n_estimators=10)
            rf.fit(X_train[cols], y_train)
            # Calculate error and add to MAEs
            MAEs.append(mean_absolute_error(y_test, rf.predict(X_test[cols])))
        # Add MAEs to dataframe
        result['Top_'+str(k+1)] = MAEs

    return result.set_index('Strategies')

def drop_analysis(X, y, df):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
    spearmans = spear(rank_col(df), target_col='price')
    
    # Pick a feature selection method
    sorted_features = list(spearmans.index)
    # Sort by least to most important
    sorted_features.reverse()
    # Get a baseline r2 score
    rf = RandomForestRegressor(n_estimators=10)
    rf.fit(X_train, y_train)
    baseline = r2_score(y_test, rf.predict(X_test))
    # Collect results
    drops = {'Drop_0':baseline}
    feature_drops = len(sorted_features)-1
    for feat in range(feature_drops):
        # Get the feature to drop
        keep_feature = sorted_features[1:]
        rf = RandomForestRegressor(n_estimators=10)
        rf.fit(X_train[keep_feature], y_train)
        # Get new r2_score
        new_score = r2_score(y_test, rf.predict(X_test[keep_feature]))
        drops['Drop_'+str(feat)] = new_score
        sorted_features = sorted_features[1:]
    return pd.DataFrame.from_dict(drops, orient='index', columns=['r2_score'])

def automated_feature_selection(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
    
    perm = perm_importance(X,y)

    # Pick a feature selection method
    sorted_features = list(perm.index)
    # Sort by least to most important
    sorted_features.reverse()
    # Get a baseline r2 score
    rf = RandomForestRegressor(n_estimators=10)
    rf.fit(X_train, y_train)
    old_score = r2_score(y_test, rf.predict(X_test))
    
    new_score = 1
    while new_score >= old_score:
        # Don't reassign upon first iteration
        if new_score != 1:
            old_score = new_score
        # Recalculate importances
        perm = perm_importance(X[sorted_features], y)
        sorted_features = list(perm.index)
        sorted_features.reverse()
        # Get the lowest importance feature to drop
        drop_feature = sorted_features[1:]
        rf = RandomForestRegressor(n_estimators=10)
        rf.fit(X_train[drop_feature], y_train)
        # Get new r2_score
        new_score = r2_score(y_test, rf.predict(X_test[drop_feature]))
        if new_score >= old_score:
            sorted_features = sorted_features[1:]
    
    return sorted_features


# Gets the distributions and targets for all inputs
def get_distribution_targets(df, iterations=100):
    
    targets = spear(rank_col(df), target_col='price')
    
    # Create a DataFrame to start
    # Copy the dataframe
    shuffle_df = df.copy()
    shuffle_df['price'] = np.random.permutation(shuffle_df['price'])
    # Calculate importance
    results = spear(rank_col(shuffle_df), 'price').rename(columns={'Spearmans':'0'})

    for n in range(iterations):
        # Shuffle again
        shuffle_df['price'] = np.random.permutation(shuffle_df['price'])
        new_df = spear(rank_col(shuffle_df), 'price').rename(columns={'Spearmans':str(n+1)})
        results = results.join(new_df)
    return targets, results.T