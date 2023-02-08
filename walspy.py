###WALSpy

import pandas as pd
pd.options.mode.chained_assignment = None
import math

###you need to fix the id code for Nandi (it defaults to nan)
###Those times in which you create a dictionary, you could use df[[A,B]].to_dict("something")

raw_idvalues = pd.read_csv(r"raw\domainelement.csv")
raw_data = pd.read_csv(r"raw\value.csv")
raw_id_list = pd.read_csv(r"raw\parameter.csv")
raw_languages = pd.read_csv(r"raw\language.csv", na_filter = False)
raw_lang_gen = pd.read_csv(r"raw\walslanguage.csv")
raw_genus = pd.read_csv(r"raw\genus.csv")
raw_family = pd.read_csv(r"raw\family.csv")

#%%

def search_features(keyword = None):
    """
    Given a keyword, the function retrieves a collection of features associated to it.
    If no keyword is provided, the function retrieves all linguistic features in WALS
    
    Args:
        keyword (str): a keyword that describes a feature (e.g., "vowel")
    
    Returns:
        pd.Series: a series of features with id codes as indexes
    """
    feature_names = raw_id_list[["id", "name"]]
    feature_names.set_index("id", inplace = True)
    if keyword == None:
        return feature_names
    feature_names["name"] = feature_names["name"].apply(lambda x: x.lower())
    return feature_names[feature_names["name"].str.contains(pat = f"{keyword}".lower())]




def feature_to_values(feat = None):
    """
    Given a feature, the function retrieves a collection of values associated to it.
    If no feature is provided, the function returns all features with their corresponding values.
    
    Args:
        feat (str): a feature in WALS (e.g., "17A")
    
    Returns:
        pd.DataFrame: a df with values for the provided feature; feature id codes are indexes
        
    """
    value_names = raw_idvalues[["id", "name", "description"]]
    value_names["feature"] = value_names["id"].apply(lambda x: x.split("-")[0]) 
    value_names.set_index("feature", inplace=True)
    value_names.sort_index(inplace=True)
    if feat == None:
        return value_names
    return value_names[value_names["id"].str.contains(pat = f"^{feat}", regex=True)]


def list_languages():
    """
    Retrieves the list of all languages in WALS

    Returns:
        pd.Series: a series of languages with id codes as indexes

    """
    lang = raw_languages[["id", "name"]]
    lang.set_index("id", inplace=True)
    lang.sort_index(inplace=True)
    return lang

#%%

def get_feature(stringa):
    """
    Retrieves all values for a linguistic feature
    
    Args:
        stringa (str): a string specifying a feature (e.g., "83A")

    Returns:
        tuple: a tuple with the name of the feature (str) and a series with the values per language
    """
    feature = raw_id_list.name[raw_id_list["id"] == stringa].iloc[0]
    relevant_values = raw_idvalues[raw_idvalues["id"].str.contains(pat = f"^{stringa}", regex=True)]
    relevant_data = raw_data[raw_data["id"].str.contains(pat = f"^{stringa}", regex=True)]
    relevant_data = relevant_data[["id", "domainelement_pk"]]        
    values = dict(zip(relevant_values['pk'], relevant_values['name']))
    relevant_data[feature] = relevant_data["domainelement_pk"].map(values)
    relevant_data.drop("domainelement_pk", axis=1, inplace=True)
    languages = dict(zip(raw_languages['id'], raw_languages['name']))
    relevant_data["lang_id"] = relevant_data["id"].apply(lambda x: x.split("-")[1])
    relevant_data["languages"] = relevant_data["lang_id"].map(languages)
    relevant_data.set_index("lang_id", inplace = True)
    return feature, relevant_data

#%%

def get_many_features(lista):
    """
    Retrieves a dataframe with values for a number n of features

    Args:
        lista (list): a list of lenght n specifying linguistic features (e.g., ["83A", "79A"])

    Returns:
        pd.DataFrame: a df with n number of columns
        
    """
    
    f, df = get_feature(lista[0])
    df = df[f]
    for i in lista[1:]:
        f1, df1 = get_feature(i)
        df = pd.merge(df, df1[f1], how="outer", left_index=True, right_index=True)
    lang_dict = dict(zip(raw_languages['id'], raw_languages['name']))
    languages = df.index.map(lang_dict)
    df.insert(0, "languages", languages)
    #it is still necesary to add a column with the languages
    return df 

#%%

def get_totals(stringa):
    """
    Retrieves the number of languages exhibiting values for a certain feature
    
    Args:
        stringa (str): an id code for a feature (e.g., "4A") 
    
    Returns:
        pd.Series: a series of integers, with values as indexes
    
    """
    feature, df = get_feature(stringa)
    return df.groupby(feature).count()["languages"]

#%%
def features_to_table(feat1, feat2):
    """
    Retrieves a table with counts for the intersections of the values of two features
    
    Args:
        feat1 (str): an id feature in WALS (e.g., ("81A")
        feat2 (str): an id feature in WALS (e.g., ("84A")
    
    Returns:
        pd.Dataframe: a df with the values of one feature as indexes, and the values of the other as columns
    """

    return get_many_features([feat1, feat2]).groupby([get_feature(feat1)[0],get_feature(feat2)[0]]).count().reset_index().pivot(index=get_feature(feat1)[0], columns=get_feature(feat2)[0], values="languages")


#%%
def search_by_value(value):
    """
    Retrieves a set of languages exhibiting a certain value
    
    Args:
        value (str): a string corresponding to a value (e.g., "4A-1")

    Returns:
        pd.Series: a series with a collection of languages
    """
    if "-" not in value:
        return print("Make sure you introduced a value (e.g., '4A-1') and not a feature (e.g., '4A')")
    info_val = raw_idvalues[raw_idvalues["id"] == value]
    lang_df = raw_data[raw_data["domainelement_pk"] == info_val.iloc[0, 0]]
    lang_dictio = dict(zip(raw_languages['id'], raw_languages['name']))
    lang_df["lang_id"] = lang_df["id"].apply(lambda x: x.split("-")[1])
    lang_df["languages"] = lang_df["lang_id"].map(lang_dictio)
    lang_df.set_index("lang_id", inplace = True)
    return lang_df["languages"]

#%%

def search_many_values(val_lst):
    """
    Retrieves a set of languages that coincide in n provided values
    
    Args:
        val_lst (list): a list expressing two or more values (e.g., ["4A-1", "6A-1", "22A-2"])

    Returns:
        pd.Series: a series with a collection of languages
    """
    s = search_by_value(val_lst[0])
    for val in val_lst[1:]:
        s1 = search_by_value(val)
        s = pd.merge(s, s1, how="inner", left_index=True, right_index=True)
    s = s.iloc[:,0].rename("languages", inplace=True)
    return s

#%%
def get_genus(id_lang):
    """
    Retrieves the genus for a given language

    Args:
        id_lang (str): the id code for a language in WALS (e.g., 'arn')

    Returns
        string: the name of the genus corresponding to the language

    """
    pk = raw_languages.pk[raw_languages["id"] == id_lang].iloc[0]
    genus_dict = dict(zip(raw_lang_gen["pk"],raw_lang_gen["genus_pk"]))
    genus_pk = genus_dict[pk]
    genus = raw_genus.name[raw_genus["pk"] == genus_pk].iloc[0]
    return genus

#%%

def get_family(id_lang):
    """
    Retrieves the family for a given language

    Args:
        id_lang (str): the id code for a language in WALS (e.g., 'arn')

    Returns
        string: the name of the family corresponding to the language

    """
    family_pk = raw_genus.family_pk[raw_genus["name"] == get_genus(id_lang)].iloc[0]
    family = raw_family.name[raw_family["pk"] == family_pk].iloc[0]
    
    return family

#%%

def get_correlation(stringa, stringb):
    """
    Applies a linear regression on two linguistic features.
    
    Args:
        stringa (str): a string specifying a feature (e.g., "83A")
        stringb (str): a string specifying a feature (e.g., "85A")

    Returns:
        tuple: a tuple with two floats, the correlation coefficient and the p-value

    """
    
    f1, df1 = get_feature(stringa)
    f2, df2 = get_feature(stringb)
    to_corr = pd.merge(df1, df2, how="inner", left_index=True, right_index=True)
    table = to_corr.apply(lambda x: x.factorize()[0])#.corr()
    from scipy import stats
    return stats.pearsonr(table[f1], table[f2])

#%%

def search_correlations(c = 0.5, p = 0.05):
    """
    (Experimental)
    
    Searches for pairs of features in WALS in linear correlation.
    
    (Since running this can take a while, it saves the output as two csv files)
    
    Args:
        c (float, optional): the minimum correlation coefficient magnitude to report (0.5 by default)
        p (float, optional): the maximum p-value to report (0.05 by default)

    Returns:
        pd.DataFrame: a df with pairs of features, a correlation coefficient and a p-value
    """
    correlations = []
    to_return = []
    errors = []
    total_len = len(list(raw_id_list["id"]))
    counter = 0
    for x in raw_id_list["id"]:
        counter += 1
        percent = (counter*100)//total_len
        for y in raw_id_list["id"]:
            print(f"Calculating correlation for {x} and {y}. {percent}% completed.")
            print(f"Correlations found: {len(correlations)} - Errors found: {len(errors)}.")
            set_xy = {x,y}
            if x != y and set_xy not in correlations and set_xy not in errors:
                try:
                    corr, p_value = get_correlation(x, y)
                    if p_value < p and (corr > c or corr < (0-c)):
                        correlations.append(set_xy)
                        str_x = f"{raw_id_list.loc[raw_id_list['id'] == x, 'name'].iloc[0]} ({x})"
                        str_y = f"{raw_id_list.loc[raw_id_list['id'] == y, 'name'].iloc[0]} ({y})"
                        tup = (str_x, str_y, corr, p_value)
                        to_return.append(tup)
                except ValueError:
                    errors.append(set_xy)
    df_return = pd.DataFrame(map(tuple, to_return), columns=["feature_x", "feature_y", "correlation", "p-value"])
    df_error = pd.DataFrame(map(tuple, errors))
    df_return.to_csv("correlations.csv", index=False)
    df_error.to_csv("errors.csv", index=False)
    return df_return

#%%
def features_to_heatmap(stringa, stringb):
    """
    Plots a heatmap showing the relation between two features

    Args:
        stringa (str): a string specifying a feature (e.g., "83A")
        stringb (str): a string specifying a feature (e.g., "85A")

    Returns:
        plot: a two dimensional plot
    """
    import seaborn as sns
    f1, df1 = get_feature(stringa)
    f2, df2 = get_feature(stringb) 
    return sns.heatmap(pd.crosstab(df1[f1], df2[f2]), cmap="crest") # add annot=True for numbers in each bin

#%%

def predict_value(feature_to_predict, value_lst):
    """
    (Experimental)
    
    Predicts the most likely value for a linguistic feature from a set of values for other features

    Args:
        feature_to_predict (str): a feature to be predicted (e.g., "85A")
        value_lst (list): a list with values used to make the prediction (e.g., ["81A-3", "86A-2", "88A-1"])

    Returns:
        tuple: the id value for the input feature (str), its description (str), and the estimated accuracy of the prediction (float)

    """
    
    def raw_feature(feat):
        data = raw_data[raw_data["id"].str.contains(pat = f"^{feat}", regex=True)]       
        value_dict = dict(zip(raw_idvalues['pk'], raw_idvalues['id']))
        data[feat] = data["domainelement_pk"].map(value_dict)
        data[feat] = data[feat].apply(lambda x: x.split("-")[1])
        data.set_index(data["id"].apply(lambda x: x.split("-")[1]), inplace=True)
        data = data[feat]
        return data
    
    df = raw_feature(feature_to_predict)
    feat_lst = list(map(lambda x: x.split("-")[0], value_lst))
    for f in feat_lst:
        val_df = raw_feature(f)
        df = pd.merge(df, val_df, how="inner", left_index=True, right_index=True)
    df = df.apply(pd.to_numeric)
    
    from sklearn.model_selection import train_test_split
    X = df[feat_lst].values
    y = df[feature_to_predict].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
    
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = int(2/3*math.sqrt(len(df))))
    knn.fit(X_train, y_train)
    values = list(map(lambda x: float(x.split("-")[1]), value_lst))
    prediction = knn.predict([values])[0]
    return_value = f"{feature_to_predict}-{prediction}"
    description = raw_idvalues.name[raw_idvalues["id"] == return_value].iloc[0]
    accuracy = knn.score(X_test, y_test)
    
    return return_value, description, accuracy


