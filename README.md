# WALSpy

WALSpy is a collection of Python functions to explore the _World Atlas of Linguistic Structures_ (WALS). It allows to retrieve information from the database as Pandas objects. It also includes some "experimental" functionality for plotting heatmaps, finding correlations between linguistic features, and predicting unreported properties of a language.

#### Dataset
The included _raw_ directory contains the original WALS dataset in csv format. These files (or newer versions of them) are required by WALSpy.

For more information about WALS, visit https://wals.info/. The dataset must be cited as follows.

- Dryer, Matthew S. & Haspelmath, Martin (eds.) 2013. WALS Online (v2020.3) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7385533



### Summary of functions

This is a succint review of the functions included in WALSpy. All arguments for these are (i) the _id_ for a language (e.g., `'map'` for Mapudungun), (ii) the _id_ for a linguistic feature in WALS (e.g., `'33A'` for Coding of Nominal Plurality), (iii) the _id_ for a certain feature-value (e.g., `'33A-2'` for Plural suffix), or (iv) a combination of these.

| Function |	Description |	Example | 
| ------- | ------- | -------- |
| get_features | Retrieves all languages with values for a given linguistic feature	| `get_feature('81A')` |
| get_many_features | Retrieves a dataframe with values for a number _n_ of features | `get_many_features(['81A', '82A'])` |
| get_totals | Retrieves the number of languages exhibiting values for a given feature | `get_totals('81A')` |
| features_to_table |	Retrieves a table with values for a feature as rows and values for another feature as columns |	`features_to_table('81A', '84A')` |
| search_by_value |	Retrieves a set of languages exhibiting a certain value |	`search_by_value('4A-1')` |
| search_many_values |	Retrieves a set of languages that coincide in _n_ provided values	| `search_many_values(['4A-1', '5A-1', '8A-1'])` |
| get_genus |	Retrieves the genus for a given language |	`get_genus('map')` |
| get_family |	Retrieves the family for a given language |	`get_family('map')` |
| get_macroarea |	Retrieves the macroarea for a given language | `get_macroarea('map')` |
| search_features |	Given a keyword, the function retrieves a collection of features associated to it. If no keyword is provided, the function retrieves all linguistic features in WALS |	`search_features('vowel')` |
| feature_to_values |	Given a feature, the function retrieves a collection of values associated to it. If no feature is provided, the function returns all features with their corresponding values |	`feature_to_values('17A')` |
| list_languages |	Retrieves the list of all languages in WALS with their id codes |	`list_languages()` |
| features_to_heatmap |	Plots a heatmap showing the relation between two features |	`features_to_heatmap('83A', '85A')` |


Besides these "basic" functions, there are three that should be considered experimental:

| Function |	Description |	Example |
| -------- | ------- | ------- |
| get_correlation |	Checks whether there is a linear correlation between two linguistic features |	`get_correlation('83A', '85A')` |
| search_correlations |	Searches for pairs of features in WALS in linear correlation |	`search_correlations()` |
| predict_value |	Predicts the most likely value for a linguistic feature based on a set of values for other features |	`predict_value('85A', ['81A-3', '86A-2'])` |

Chech the docstrings in walspy.py for further details.
