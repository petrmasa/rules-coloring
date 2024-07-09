import json
import os

import pandas as pd
from sklearn.impute import SimpleImputer


class datasets:
    version = "0.1"
    dataset_name=['Accidents','Adults','Loans','Pistachio','BMI','Marketing_campaign','Titanic',"Iris",'Diabetes']
    path = None

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

    def _getlabels(self,s):
        lst = []
        for i in range(len(s)-1):
            j=i+1
            if (j==1):
                item = '<' + str(s[i]) + ','+ str(s[i+1]) + '>'
            else:
                item = '(' + str(s[i]) + ','+ str(s[i+1]) + '>'
            lst.append(item)

        return lst


    def get_dataset_names(self):
        return self.dataset_name

    #todo struktury, ktere definuji umisteni souboru, jejich zdroj a popis

    def load_dataset(self,idx=None,name=None):
        """
        Loads a dataset with a given index or name

        :param i: index of a dataset
        :param name: name of a dataset

        """

        i=idx
        dsname=name

        if isinstance(i,str):
            dsname=i
            i = None


        if (i is None and dsname is None):
            print("ERROR: either index or name of dataset to be provided")
            exit(1)
        df = None

        if i is not None:
            if i >=0 and i<= len(self.dataset_name):
                dsname = self.dataset_name[i]
            #     idx=dataset_name.index(name)
            else:
                print(f"Dataset with index {i} not exists. Valid range is 0-{len(self.dataset_name)}")
                exit(1)

        fname = os.path.join(self.path, dsname + ".zip")


        if dsname.upper() == 'ACCIDENTS':
            df = pd.read_csv(fname, encoding='cp1250', sep='\t')

            df = df[
                ['Driver_Age_Band', 'Driver_IMD', 'Sex', 'Journey', 'Hit_Objects_in', 'Hit_Objects_off', 'Casualties','Road_Type',  'Vehicle_Location', 'Vehicle_Type', 'Vehicle_Age',
                 'Severity','Speed_limit','Light','Area']]

            imputer = SimpleImputer(strategy="most_frequent")
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        elif dsname.upper() == 'ADULTS':
            features = ["Age", "Workclass", "fnlwgt", "Education", "Education_Num", "Martial_Status",
                        "Occupation", "Relationship", "Race", "Sex", "Capital_Gain", "Capital_Loss",
                        "Hours_per_week", "Country", "Target"]

            edu_cat = ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad",
                       "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"]

            train_url = os.path.join(self.path, 'adult.data')
            test_url = os.path.join(self.path, 'adult.test')

            original_train = pd.read_csv(train_url, names=features, sep=r'\s*,\s*',
                                         engine='python', na_values="?")
            original_test = pd.read_csv(test_url, names=features, sep=r'\s*,\s*',
                                        engine='python', na_values="?", skiprows=1)

            original_test.Target = original_test.Target.str.replace('.', '')

            original = pd.concat([original_train, original_test])

            original['Education'] = original['Education'].astype('category').cat.reorder_categories(edu_cat,
                                                                                                    ordered=True)

            age_bins = [10, 20, 30, 40, 50, 60, 70, 90]
            original['Age_b'] = pd.cut(original['Age'], include_lowest=True, bins=age_bins, labels=self._getlabels(age_bins),
                                       ordered=True)
            hpw_bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
            original['Hours_per_week_b'] = pd.cut(original['Hours_per_week'], include_lowest=True, bins=hpw_bins,
                                                  labels=self._getlabels(hpw_bins), ordered=True)
            cl_bins = [-1, 0, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 4400]
            original['Capital_Loss_b'] = pd.cut(original['Capital_Loss'], include_lowest=True, bins=cl_bins,
                                                labels=self._getlabels(cl_bins), ordered=True)
            cg_bins = [-1, 0, 2000, 3000, 4000, 5000, 7000, 10000, 20000, 99000, 100000]
            original['Capital_Gain_b'] = pd.cut(original['Capital_Gain'], include_lowest=True, bins=cg_bins,
                                                labels=self._getlabels(cg_bins), ordered=True)

            original['Income'] = original['Target']

            df = original[
                ['Income', 'Capital_Gain_b', 'Capital_Loss_b', 'Hours_per_week_b', 'Occupation', 'Martial_Status',
                 'Relationship', 'Age_b', 'Education', 'Sex', 'Country', 'Race', 'Workclass', 'Target']]

        elif dsname.upper() == 'IRIS':

            original = pd.read_csv(fname)

            to_qcut = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

            for varname in to_qcut:
                original[varname] = pd.qcut(original[varname], q=5, duplicates='drop')

            df = original

        elif dsname.upper() == 'TITANIC':

            original = pd.read_csv(fname)

            to_qcut = ['Age', 'Fare']

            for varname in to_qcut:
                original[varname] = pd.qcut(original[varname], q=5, duplicates='drop')

            df = original

        elif dsname.upper() == 'MARKETING_CAMPAIGN':

            original = pd.read_csv(fname, sep='\t')

            to_qcut = ['Year_Birth', 'Income', 'Kidhome',
                       'Teenhome', 'Recency', 'MntWines', 'MntFruits',
                       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
                       'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
                       'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue']

            for varname in to_qcut:
                original[varname] = pd.qcut(original[varname], q=5, duplicates='drop')

            df = original

        elif dsname.upper() == 'PISTACHIO':
            original = pd.read_csv(fname)


            to_qcut = ['AREA', 'PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS', 'ECCENTRICITY',
                       'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA', 'EXTENT', 'ASPECT_RATIO',
                       'ROUNDNESS', 'COMPACTNESS', 'SHAPEFACTOR_1', 'SHAPEFACTOR_2',
                       'SHAPEFACTOR_3', 'SHAPEFACTOR_4']

            for varname in to_qcut:
                original[varname] = pd.qcut(original[varname], q=5)

            df = original

        elif dsname.upper() == 'DIABETES':
            df = pd.read_csv(fname)

            for col in df.columns:
                dfc = df[col]
                cnt = len(dfc.unique())
                if cnt >= 20:
                    df[col] = pd.qcut(df[col], 10, duplicates="drop")


        elif dsname.upper() == 'LOANS':
            original = pd.read_csv(fname)


            age_bins = [0, 0.1, 1, 2, 3, 100000]
            original[' no_of_dependents'] = pd.cut(original[' no_of_dependents'], include_lowest=True, bins=age_bins,
                                                   labels=self._getlabels(age_bins), ordered=True)

            original[' income_annum'] = pd.qcut(original[' income_annum'], q=5)
            original[' loan_amount'] = pd.qcut(original[' loan_amount'], q=5)
            original[' loan_term'] = pd.qcut(original[' loan_term'], q=5)
            original[' cibil_score'] = pd.qcut(original[' cibil_score'], q=5)
            original[' residential_assets_value'] = pd.qcut(original[' residential_assets_value'], q=5)
            original[' commercial_assets_value'] = pd.qcut(original[' commercial_assets_value'], q=5)
            original[' luxury_assets_value'] = pd.qcut(original[' luxury_assets_value'], q=5)
            original[' bank_asset_value'] = pd.qcut(original[' bank_asset_value'], q=5)


            df = original

        elif dsname == 'BMI':

            original = pd.read_csv(fname)


            to_qcut = ['Height', 'Weight']

            for varname in to_qcut:
                original[varname] = pd.qcut(original[varname], q=5)

            df = original

        else:
            print(f"DATASET {dsname} is not supported now.")


        return df



