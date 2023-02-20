import os
import pandas as pd
import numpy as np

np.random.seed(42)

DATASET_DIR = os.path.join(os.path.curdir, "datasets")


def import_folkstables():
    filename_d = os.path.join(
        os.path.curdir, "datasets", "ACSPUMS", "adult_dataset_income_tax.csv"
    )
    dfI = pd.read_csv(filename_d)

    attributes = list(dfI.columns.drop("income"))

    continuous_attributes = ["AGEP", "WKHP"]
    target = "income"

    dfI = dfI[attributes + [target]]
    return dfI, target, continuous_attributes


def import_process_adult(inputDir=DATASET_DIR):
    education_map = {
        "10th": "Dropout",
        "11th": "Dropout",
        "12th": "Dropout",
        "1st-4th": "Dropout",
        "5th-6th": "Dropout",
        "7th-8th": "Dropout",
        "9th": "Dropout",
        "Preschool": "Dropout",
        "HS-grad": "High School grad",
        "Some-college": "High School grad",
        "Masters": "Masters",
        "Prof-school": "Prof-School",
        "Assoc-acdm": "Associates",
        "Assoc-voc": "Associates",
    }
    occupation_map = {
        "Adm-clerical": "Admin",
        "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar",
        "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar",
        "Handlers-cleaners": "Blue-Collar",
        "Machine-op-inspct": "Blue-Collar",
        "Other-service": "Service",
        "Priv-house-serv": "Service",
        "Prof-specialty": "Professional",
        "Protective-serv": "Other",
        "Sales": "Sales",
        "Tech-support": "Other",
        "Transport-moving": "Blue-Collar",
    }
    married_map = {
        "Never-married": "Never-Married",
        "Married-AF-spouse": "Married",
        "Married-civ-spouse": "Married",
        "Married-spouse-absent": "Separated",
        "Separated": "Separated",
        "Divorced": "Separated",
        "Widowed": "Widowed",
    }

    country_map = {
        "Cambodia": "SE-Asia",
        "Canada": "British-Commonwealth",
        "China": "China",
        "Columbia": "South-America",
        "Cuba": "Other",
        "Dominican-Republic": "Latin-America",
        "Ecuador": "South-America",
        "El-Salvador": "South-America",
        "England": "British-Commonwealth",
        "France": "Euro_1",
        "Germany": "Euro_1",
        "Greece": "Euro_2",
        "Guatemala": "Latin-America",
        "Haiti": "Latin-America",
        "Holand-Netherlands": "Euro_1",
        "Honduras": "Latin-America",
        "Hong": "China",
        "Hungary": "Euro_2",
        "India": "British-Commonwealth",
        "Iran": "Other",
        "Ireland": "British-Commonwealth",
        "Italy": "Euro_1",
        "Jamaica": "Latin-America",
        "Japan": "Other",
        "Laos": "SE-Asia",
        "Mexico": "Latin-America",
        "Nicaragua": "Latin-America",
        "Outlying-US(Guam-USVI-etc)": "Latin-America",
        "Peru": "South-America",
        "Philippines": "SE-Asia",
        "Poland": "Euro_2",
        "Portugal": "Euro_2",
        "Puerto-Rico": "Latin-America",
        "Scotland": "British-Commonwealth",
        "South": "Euro_2",
        "Taiwan": "China",
        "Thailand": "SE-Asia",
        "Trinadad&Tobago": "Latin-America",
        "United-States": "United-States",
        "Vietnam": "SE-Asia",
    }
    # as given by adult.names
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income-per-year",
    ]

    # check_dataset_availability("credit-g.csv", inputDir=inputDir)
    train = pd.read_csv(
        os.path.join(inputDir, "adult.data"),
        header=None,
        names=column_names,
        skipinitialspace=True,
        na_values="?",
    )

    # check_dataset_availability("adult.test", inputDir=inputDir)

    test = pd.read_csv(
        os.path.join(inputDir, "adult.test"),
        header=0,
        names=column_names,
        skipinitialspace=True,
        na_values="?",
    )
    dt = pd.concat([test, train], ignore_index=True)
    dt["education"] = dt["education"].replace(education_map)
    dt.drop(columns=["education-num", "fnlwgt"], inplace=True)
    dt["occupation"] = dt["occupation"].replace(occupation_map)
    dt["marital-status"] = dt["marital-status"].replace(married_map)
    dt["native-country"] = dt["native-country"].replace(country_map)

    dt.rename(columns={"income-per-year": "class"}, inplace=True)
    dt["class"] = (
        dt["class"].astype("str").replace({">50K.": ">50K", "<=50K.": "<=50K"})
    )
    dt.dropna(inplace=True)
    dt.reset_index(drop=True, inplace=True)

    dt.drop(columns=["native-country"], inplace=True)
    continuous_attributes = list(dt.describe().columns)
    return dt, {"N": "<=50K", "P": ">50K"}, continuous_attributes


def import_compas():

    risk_class_type = True

    from import_datasets import import_process_compas

    dfI, class_map = import_process_compas(
        risk_class=risk_class_type, continuous_col=True
    )
    dfI.reset_index(drop=True, inplace=True)

    dfI["predicted"] = dfI["predicted"].replace({"Medium-Low": 0, "High": 1})
    true_class_name, pred_class_name = "class", "predicted"
    class_and_pred_names = [true_class_name, pred_class_name]
    attributes = list(dfI.columns.drop(class_and_pred_names))

    dfI = dfI[attributes + class_and_pred_names]
    continuous_attributes = ["priors_count", "length_of_stay", "age"]
    return dfI, class_map, continuous_attributes


def import_process_wine(input_dir=DATASET_DIR):
    df_all = []

    for d in ["winequality-white", "winequality-white"]:
        if os.path.isfile(os.path.join(input_dir, f"{d}.csv")):
            df = pd.read_csv(os.path.join(input_dir, f"{d}.csv"), sep=";")
        else:
            df = pd.read_csv(
                f"https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/{d}.csv",
                sep=";",
            )

        df["quality"] = df["quality"].apply(lambda x: "good" if x > 5 else "bad")
        class_map = {"P": "good", "N": "bad"}
        df.rename(columns={"quality": "class"}, inplace=True)
        df_all.append(df)

    df = pd.concat(df_all)
    df.reset_index(drop=True, inplace=True)
    continuous_attributes = list(df.describe().columns)

    return df, class_map, continuous_attributes


def train_classifier_kv(
    df,
    k_cv=10,
    type_clf="rf",
    class_name="class",
    encoding=False,
    categorical_attributes=None,
):

    if type_clf != "rf":
        raise ValueError(f"{type_clf} not currently supported")

    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_predict
    from sklearn.ensemble import RandomForestClassifier

    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    attributes = df.columns.drop(class_name)
    X = df[attributes].copy()
    y = df[class_name].copy()

    if encoding:
        encoders = {}
        if categorical_attributes is None:
            categorical_attributes = attributes
        for column in categorical_attributes:
            if df.dtypes[column] == np.object_:
                le = LabelEncoder()
                X[column] = le.fit_transform(df[column])
                encoders[column] = le
            elif df.dtypes[column] == np.bool_:
                X[column] = X[column].astype(int)

    clf = RandomForestClassifier(random_state=42)

    cv = StratifiedKFold(
        n_splits=k_cv, random_state=42, shuffle=True
    )  # Added to fix the random state  #Added shuffle=True for new version sklearn, Value Error

    y_predicted = cross_val_predict(clf, X, y.values, cv=cv)

    df["predicted"] = y_predicted

    return df


def check_dataset_availability(dataset_name, inputDir=DATASET_DIR):
    """
    Check if the dataset is available

    """
    import os

    if os.path.isdir(inputDir):
        filename = os.path.join(inputDir, dataset_name)
        if os.path.isfile(filename):
            return
        else:
            raise ValueError(
                f"Dataset {filename} does not exist. Please add in {inputDir} the dataset of interest ({dataset_name})"
            )
    else:
        raise ValueError(
            f"Dataset folder {inputDir} does not exist. Please add in {inputDir} the dataset of interest ({dataset_name})"
        )


def import_process_german(inputDir=DATASET_DIR):
    check_dataset_availability("credit-g.csv", inputDir=inputDir)
    df = pd.read_csv(os.path.join(inputDir, "credit-g.csv"))
    # print(df['personal_status'].value_counts())
    gender_map = {
        "'male single'": "male",
        "'female div/dep/mar'": "female",
        "'male mar/wid'": "male",
        "'male div/sep'": "male",
    }
    status_map = {
        "'male single'": "single",
        "'female div/dep/mar'": "married/wid/sep",
        "'male mar/wid'": "married/wid/sep",
        "'male div/sep'": "married/wid/sep",
    }
    df["sex"] = df["personal_status"].replace(gender_map)
    df["civil_status"] = df["personal_status"].replace(status_map)
    df.drop(columns=["personal_status"], inplace=True)
    df.rename(columns={"credit": "class"}, inplace=True)
    continuous_attributes = list(df.describe())
    return df, {"P": "good", "N": "bad"}, continuous_attributes


def import_process_online_shoppers_intention(inputDir=DATASET_DIR):

    if os.path.isfile(os.path.join(DATASET_DIR, "online_shoppers_intention.csv")):

        df = pd.read_csv(
            os.path.join(DATASET_DIR, "online_shoppers_intention.csv"), sep=","
        )
    else:
        df = pd.read_csv(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv",
            sep=",",
        )

    class_map = {"P": True, "N": False}
    df.rename(columns={"Revenue": "class"}, inplace=True)

    df["Month"] = df["Month"].replace(
        {
            "May": 5,
            "Nov": 11,
            "Mar": 3,
            "Dec": 12,
            "Oct": 10,
            "Sep": 9,
            "Aug": 8,
            "Jul": 7,
            "June": 6,
            "Feb": 2,
        }
    )

    continuous_attributes = list(df.describe().columns)
    continuous_attributes.remove("SpecialDay")
    return df, class_map, continuous_attributes


def import_process_real_estate(inputDir=DATASET_DIR):

    check_dataset_availability("Daegu_Real_Estate_data.csv", inputDir=inputDir)

    df = pd.read_csv(os.path.join(DATASET_DIR, "Daegu_Real_Estate_data.csv"), sep=",")

    continuous_attributes = list(df.describe())

    target = "SalePrice"

    continuous_attributes.remove(target)

    return df, target, continuous_attributes


def generate_artificial_gaussian_error(n_attributes=3, n=10000):
    from scipy.stats import multivariate_normal
    from sklearn.preprocessing import MinMaxScaler

    X = np.random.uniform(low=-5, high=5, size=(n, n_attributes))

    mean = np.arange(n_attributes)
    cov = np.ones(n_attributes)

    f_g = multivariate_normal(mean, cov)  # , [1, 1, 1])
    g = f_g.pdf(X)

    scaler = MinMaxScaler()
    g_sc = np.round_(scaler.fit_transform(g.reshape(-1, 1))[:, 0], 15)

    import string

    attributes = list(string.ascii_lowercase)[:n_attributes]

    g_attrs = []

    for id_attr in range(X.shape[1]):
        g_attr = multivariate_normal.pdf(
            X[:, id_attr], mean=mean[id_attr], cov=cov[id_attr]
        )
        g_attr_norm = scaler.fit_transform(g_attr.reshape(-1, 1))[:, 0]
        g_attrs.append(g_attr_norm)

    g_attrs = np.array(g_attrs)

    classes = np.random.choice([0, 1], size=X.shape[0], p=[0.5, 0.5])
    opposed = 1 - classes

    values = np.vstack((classes, opposed)).T

    predicted_classes = [
        np.random.choice(values[i : i + 1][0], 1, p=[1 - g_sc[i], g_sc[i]])[0]
        for i in range(0, g_sc.shape[0])
    ]
    predicted_classes = np.asarray(predicted_classes)

    df_analysis = pd.DataFrame(
        np.hstack((X, classes.reshape(-1, 1), predicted_classes.reshape(-1, 1))),
        columns=attributes + ["class", "predicted"],
    ).round(5)

    # min_max_vals = {attributes[e] : (min(X[:, 0]),max(X[:, 0])) for e in range(X.shape[1])}

    class_map = {"P": 1, "N": 0}
    return df_analysis, class_map, attributes


def import_process_default_payment(input_dir=DATASET_DIR):
    if os.path.isfile(os.path.join(input_dir, "default_payment.csv")):
        df = pd.read_excel(os.path.join(input_dir, "default_payment.csv"), skiprows=1)
    else:

        df = pd.read_excel(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
            skiprows=1,
        )
    df = df.drop(columns=["ID"])
    df = df.rename(columns={"default payment next month": "class"})

    from import_process_dataset import train_classifier_kv

    continuous_attributes = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]
    class_map = {"P": 1, "N": 0}
    return df, class_map, continuous_attributes
