import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.utils import resample
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score, roc_curve, auc, \
    log_loss, jaccard_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.filterwarnings("ignore")
###MENJAJ STA HOCES DA PRIKAZES, ima i u meniju###
print_histogram = False
print_boxplot = False
print_scatterplot = False
print_heatmap = False
print_normalizacija_abnormalija = False
tune_stackig_parameters = False
tune_bagging_parameters = False
tune_boosting_parameters = False
tune_knn_parameters = False
make_new_data = False

best_params = {}
classifiers = {}


def toggle(parameter):
    return not parameter


def masinsko_ucenje(X_train, X_test, y_train, y_test):
    ### Stacking
    global classifiers
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svr', make_pipeline(StandardScaler(),
                              LinearSVC(random_state=42)))
    ]

    stacking_clf = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression()
    )
    '''
    za stacking optimalni parametri  {'final_estimator__C': 5, 'rf__max_depth': 10, 'rf__n_estimators': 10, 'svr__linearsvc__C': 0.1}
    '''
    if tune_stackig_parameters == True:
        param_grid = {
            'rf__n_estimators': [1, 10, 50, 100],
            'rf__max_depth': [None, 10, 20, 50],
            'svr__linearsvc__C': [0.1, 1, 10, 20],
            'final_estimator__C': [0.1, 1, 5, 10]
        }

        # Namestanje GridSearchCV-a
        grid_search = GridSearchCV(estimator=stacking_clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Get the best parameters and the best score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
    else:
        best_params = {'final_estimator__C': 5, 'rf__max_depth': 10, 'rf__n_estimators': 10, 'svr__linearsvc__C': 0.1}

    # print("Best Parameters stacking:", best_params)
    # print("Best Score:", best_score)
    best_rf_params = {key.replace('rf__', ''): value for key, value in best_params.items() if key.startswith('rf__')}
    stacking_clf.named_estimators['rf'].set_params(**best_rf_params)

    best_svr_params = {key.replace('svr__linearsvc__', ''): value for key, value in best_params.items() if
                       key.startswith('svr__linearsvc__')}
    stacking_clf.named_estimators['svr'].named_steps['linearsvc'].set_params(**best_svr_params)

    best_final_params = {key.replace('final_estimator__', ''): value for key, value in best_params.items() if
                         key.startswith('final_estimator__')}
    stacking_clf.final_estimator.set_params(**best_final_params)

    # print("Updated RandomForest parameters:", stacking_clf.named_estimators['rf'].get_params())
    # print("Updated LinearSVC parameters:", stacking_clf.named_estimators['svr'].named_steps['linearsvc'].get_params())
    # print("Updated Final Estimator parameters:", stacking_clf.final_estimator.get_params())

    stacking_clf.fit(X_train, y_train)
    print("Stacking rezultat: ", stacking_clf.score(X_test, y_test))

    # bagging

    # Create a Bagging classifier with KNN as the base estimator
    # bagging_clf = BaggingClassifier(estimator=knn, random_state=42) #ovaj nije dobro funkcionisao

    dt = DecisionTreeClassifier(random_state=42)
    # pipeline.fit(X_train, y_train)

    bagging_clf = BaggingClassifier(estimator=dt, n_estimators=50, random_state=42)

    if tune_bagging_parameters == True:

        param_grid = {
            'bootstrap': [True, False],
            'bootstrap_features': [True, False],
            'max_features': [0.5, 0.7, 1.0],
            'max_samples': [0.5, 0.7, 1.0],
            'n_estimators': [10, 50, 100],
        }
        # Set up GridSearchCV
        grid_search = GridSearchCV(estimator=bagging_clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1,
                                   verbose=20)

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Get the best parameters and the best score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print("Best Parameters Bagging:", best_params)
        print("Best Score:", best_score)
    else:
        best_params = {'bootstrap': True, 'bootstrap_features': True, 'max_features': 0.7, 'max_samples': 1.0,
                       'n_estimators': 100}

    # Update bagging classifier with the best parameters
    bagging_clf.set_params(**best_params)
    bagging_clf.fit(X_train, y_train)
    y_pred = bagging_clf.predict(X_test)
    print("bagging rezultat :", accuracy_score(y_test, y_pred))

    # boosting

    boosting_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                              max_depth=1, random_state=0)
    if tune_boosting_parameters == True:
        param_grid = {
            'n_estimators': [10, 20, 100, 150],
            'learning_rate': [0.01, 0.1, 1.0],
            'max_depth': [1, 3, 5],
            'subsample': [0.6, 0.8, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Set up GridSearchCV
        grid_search = GridSearchCV(estimator=boosting_clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Get the best parameters and the best score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print("Best Parameters:", best_params)
        print("Best Score:", best_score)
    else:
        best_params = {'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.1,
                       'loss': 'log_loss', 'max_depth': 3, 'max_features': None, 'max_leaf_nodes': None,
                       'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 10,
                       'min_weight_fraction_leaf': 0.0, 'n_estimators': 150, 'n_iter_no_change': None,
                       'random_state': 0,
                       'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}

    '''za boosting smo dobili 
     {'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.1, 'loss': 'log_loss', 'max_depth': 3, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 150, 'n_iter_no_change': None, 'random_state': 0, 'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
    '''
    # Update boosting classifier with the best parameters
    boosting_clf.set_params(**best_params)

    # Verify the updated parameters
    # print("Updated GradientBoostingClassifier parameters:", boosting_clf.get_params())
    boosting_clf.fit(X_train, y_train)
    print("boosting score: ", boosting_clf.score(X_test, y_test))

    # Create a KNeighborsClassifier

    knn_clf = KNeighborsClassifier()

    # Create the GridSearchCV object
    if tune_knn_parameters == True:
        param_grid = {
            'n_neighbors': [3, 5, 7, 10, 100],  # Number of neighbors
            'weights': ['uniform', 'distance'],  # Weight function used in prediction
            'p': [1, 2, 4]  # Power parameter for the Minkowski metric
        }

        grid_search = GridSearchCV(estimator=knn_clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

        # Perform the grid search
        grid_search.fit(X_train, y_train)

        # Best parameters found by GridSearchCV
        best_params = grid_search.best_params_
        print(f"Best parameters found: {best_params}")
        best_params = grid_search.best_params_
    else:
        best_params = {'n_neighbors': 10, 'p': 1, 'weights': 'distance'}
        # Evaluate the best model on the test set
    knn_clf.set_params(**best_params)
    knn_clf.fit(X_train, y_train)

    error_rate = []

    ###KNN preko elbow metode
    '''
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))
    plt.figure(figsize=(10,6))
    plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error rate')
    plt.show()
    input("coutinue?")
    #k=14
    '''
    y_pred = knn_clf.predict(X_test)
    print("KNN rezultat: ", accuracy_score(y_test, y_pred))

    decision_tree_clf = DecisionTreeClassifier(criterion='log_loss')
    decision_tree_clf.fit(X_train, y_train)
    y_pred = decision_tree_clf.predict(X_test)
    print("Decision tree accuracy ", accuracy_score(y_test, y_pred))

    logistic_regression_clf = LogisticRegression(C=0.5)
    logistic_regression_clf.fit(X_train, y_train)
    y_pred = logistic_regression_clf.predict(X_test)
    print("Logistic regression accuracy ", accuracy_score(y_test, y_pred))

    # krosvalidacija

    classifiers = {
        'Stacking': stacking_clf,
        'Boosting': boosting_clf,
        'Bagging': bagging_clf,
        'KNN': knn_clf,
        'Decision Tree': decision_tree_clf,
        'Logistic regression': logistic_regression_clf
    }

    # Dictionary to store mean cross-validation scores
    scores = {}

    # Perform cross-validation for each classifier
    for name, clf in classifiers.items():
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        scores[name] = np.mean(cv_scores)
        print(f'{name} Classifier Mean CV Accuracy: {scores[name]:.4f}')
    # perform confusion matrix
    for name, clf in classifiers.items():
        predictions = clf.predict(X_test)
        cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
        print("***" * 10)
        print(name, ": ")
        print(cm)

    print("***" * 10)
    print("F1 score: ")
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average=None)
        print("***" * 10)
        print(name, ": ")
        print(f1)

    print("***" * 10)
    print("recall score: ")
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)
        recall = recall_score(y_test, y_pred, average=None)
        print("***" * 10)
        print(name, ": ")
        print(recall)
    print("***" * 10)
    print("accuracy score: ")
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("***" * 10)
        print(name, ": ")
        print(accuracy)
    print("logistic loss: ")
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)
        LL = log_loss(y_test, y_pred)
        print("***" * 10)
        print(name, ": ")
        print(LL)
    print("jaccard score: ")
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)
        JS = jaccard_score(y_test, y_pred)
        print("***" * 10)
        print(name, ": ")
        print(JS)
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        # Plot the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        title = "ROC curve for " + name
        plt.title(title)
        plt.legend()
        plt.show()

    # Fetch the importances
    return X_train, X_test, y_train, y_test


def menu():  # ovo je uradjeno kako ne bih morao da zakomentarisem odredjene stvari pri testiranju
    global print_histogram
    global print_boxplot
    global print_scatterplot
    global print_heatmap
    global print_normalizacija_abnormalija
    global tune_stackig_parameters
    global tune_bagging_parameters
    global tune_boosting_parameters
    global tune_knn_parameters
    global make_new_data
    while True:
        print("\nCurrent parameter states, if you want print functions to work you need to make_new_data=True:")
        print(f"1. print_histogram: [{'x' if print_histogram else ' '}]")
        print(f"2. print_boxplot: [{'x' if print_boxplot else ' '}]")
        print(f"3. print_scatterplot: [{'x' if print_scatterplot else ' '}]")
        print(f"4. print_heatmap: [{'x' if print_heatmap else ' '}]")
        print(f"5. print_normalizacija_abnormalija: [{'x' if print_normalizacija_abnormalija else ' '}]")
        print(f"6. tune_stackig_parameters: [{'x' if tune_stackig_parameters else ' '}]")
        print(f"7. tune_bagging_parameters: [{'x' if tune_bagging_parameters else ' '}]")
        print(f"8. tune_boosting_parameters: [{'x' if tune_boosting_parameters else ' '}]")
        print(f"9. tune_knn_parameters: [{'x' if tune_knn_parameters else ' '}]")
        print(f"10. make_new_data: [{'x' if make_new_data else ' '}]")

        print("11. Start learning")

        choice = input("Enter the number of the parameter to toggle (or 11 to start learning): ")

        if choice == '1' and make_new_data:
            print_histogram = toggle(print_histogram)
        elif choice == '2' and make_new_data:
            print_boxplot = toggle(print_boxplot)
        elif choice == '3' and make_new_data:
            print_scatterplot = toggle(print_scatterplot)
        elif choice == '4' and make_new_data:
            print_heatmap = toggle(print_heatmap)
        elif choice == '5' and make_new_data:
            print_normalizacija_abnormalija = toggle(print_normalizacija_abnormalija)
        elif choice == '6':
            tune_stackig_parameters = toggle(tune_stackig_parameters)
        elif choice == '7':
            tune_bagging_parameters = toggle(tune_bagging_parameters)
        elif choice == '8':
            tune_boosting_parameters = toggle(tune_boosting_parameters)
        elif choice == '9':
            tune_knn_parameters = toggle(tune_knn_parameters)
        elif choice == '10':
            make_new_data = toggle(make_new_data)
            if not make_new_data:
                print_histogram = False
                print_boxplot = False
                print_scatterplot = False
                print_heatmap = False
                print_normalizacija_abnormalija = False
        elif choice == '11':
            print("Starting learning")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 11.")


def make_new_data_func():
    global print_histogram
    global print_boxplot
    global print_scatterplot
    global print_heatmap
    global print_normalizacija_abnormalija
    global tune_stackig_parameters
    global tune_bagging_parameters
    global tune_boosting_parameters
    global tune_knn_parameters
    global make_new_data
    if make_new_data == True:
        podaci = pd.read_csv("diabetes_prediction_dataset.csv")

        print(podaci.shape)  # velicina tabele
        print("broj podataka: ")
        print(podaci.info())
        print(podaci.isnull().sum())  # iz ovog zakljucujemo da ne fale ni jedni podaci
        # pronalazenje duplikata

        print("broj duplikata: ", podaci.duplicated().sum())
        # identifiying garbage values
        for i in podaci.select_dtypes(include="object").columns:
            print(podaci[i].value_counts())
            print(" ***" * 10)

        ##Zakljucujemo da nema smeca

        ###DESKRIPTIVNA ANALIA
        pd.set_option('display.max_columns', None)
        print(podaci.describe().T)
        print("***" * 10)
        print(podaci.describe(include="object").T)  ### zene su nebalansirane ima ih 58552 ali ih ostavljam
        # histogrami
        if print_histogram:
            for i in podaci.select_dtypes(include="number").columns:
                sb.histplot(data=podaci, x=i)
                plt.show()  # odkomentarisi za analizu
        # boxplot
        if print_boxplot:
            for i in podaci.select_dtypes(include="number").columns:
                sb.boxplot(data=podaci, x=i)
                plt.show()  # odkomentarisi za analizu

        # scatter plot
        # print(podaci.select_dtypes(include="number").columns)
        if print_scatterplot:
            kolone = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
            for i in kolone:
                sb.scatterplot(data=podaci, x=i, y='diabetes')
                plt.show()

        # korelacije
        if print_heatmap:
            print("***" * 10)
            print("KORELACIJE")
            print("***" * 3)
            korelacije = podaci.select_dtypes(include="number").corr()
            print(korelacije)
            plt.figure(figsize=(15, 15))
            sb.heatmap(korelacije, annot=True, xticklabels=1, yticklabels=1)
            plt.show()

        ###Normalizacija anomalija
        def wisker(col):
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            lw = q1 - 1.5 * iqr
            uw = q3 + 1.5 * iqr
            return lw, uw

        for i in ['bmi', 'HbA1c_level', 'blood_glucose_level']:
            lw, uw = wisker(podaci[i])
            podaci[i] = np.where(podaci[i] < lw, lw, podaci[i])
            podaci[i] = np.where(podaci[i] > uw, uw, podaci[i])
        if print_normalizacija_abnormalija:
            for i in ['bmi', 'HbA1c_level', 'blood_glucose_level']:
                sb.boxplot(podaci[i])
                plt.show()

        ###BRISANJE DUPLIKATA
        podaci.drop_duplicates(inplace=True)
        ###Brisanje smece vrednosti, buduci da ima oko 16/100000 koji se izjasnjavaju kao drugi pol, njih brisemo
        mask = podaci['gender'] == 'Other'
        podaci = podaci[~mask]
        for i in podaci.select_dtypes(include="object").columns:
            print(podaci[i].value_counts())
            print(" ***" * 10)

        # balansirali smo skup da imamo podjednako ljudi sa dijabetesom i bez dijabetesa
        # metodom Undersampeling, oversampling nije mogucnost jer imamo odnos 80/20
        minority_class = podaci[podaci['diabetes'] == 1]
        majority_class = podaci[podaci['diabetes'] == 0]
        majority_downsampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=42)
        balanced_data = pd.concat([minority_class, majority_downsampled])
        print(balanced_data)
        print(balanced_data.describe().T)
        print(balanced_data.describe(include="object").T)  ### pol je nebalansirane ima ih 9500/7500, nije strasno

        ###NORMALIZACIJA

        skalirani_i_balansirani = balanced_data.copy()
        kolone = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
                  'blood_glucose_level']  # KOLONE KOJE NISU NORMALIZOVANE
        # apply normalization techniques
        for column in kolone:
            skalirani_i_balansirani[column] = skalirani_i_balansirani[column] / skalirani_i_balansirani[
                column].abs().max()
        # view normalized data
        # pretvorili smo objekte u int-ove
        finalni_podaci = pd.get_dummies(data=skalirani_i_balansirani, columns=["gender", "smoking_history"],
                                        drop_first=True)
        finalni_podaci.to_csv("finalni_podaci.csv", index=False)  # pisemo podatke u poseban fajl


def izbaci_manje_bitne_podatke(X_train, X_test, y_train, y_test):
    global classifiers
    '''

        ovo je bilo preko metode permutacija, ali sam se na kraju odlucio preko random forset classifiera

        print("Feature importance preko permutacija")
        for name, clf in classifiers.items():
            r = permutation_importance(clf, X_test, y_test, n_repeats=30, random_state=0)
            print("****"*10)
            print(name)
            print("****" * 10)
            for i in r.importances_mean.argsort()[::-1]:
                print(f"{X_train.columns.values[i]: <8}"
                      f"{r.importances_mean[i]:.3f}"
                      f" +/- {r.importances_std[i]:.3f}")

        '''

    finalni_podaci = pd.read_csv('finalni_podaci.csv')
    print("Feature importance preko random forest classifier-a")

    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)

    importances = forest.feature_importances_
    res = {X_train.columns.values[i]: importances[i] for i in range(len(importances))}
    new_res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
    keys_list = list(new_res.keys())
    # print(new_res)
    # print(keys_list)
    novi_podaci = finalni_podaci.copy()
    for i in range(4, len(keys_list)):
        novi_podaci.drop(labels=keys_list[i], axis=1, inplace=True)
        X_train.drop(labels=keys_list[i], axis=1, inplace=True)
        X_test.drop(labels=keys_list[i], axis=1, inplace=True)

    novi_podaci.to_csv("podaci_sa_izbacenim_vrednostima.csv", index=False)
    return X_train, X_test


menu()
make_new_data_func()
finalni_podaci = pd.read_csv("finalni_podaci.csv")
# splitdata
X = finalni_podaci.loc[:, finalni_podaci.columns != 'diabetes']
y = finalni_podaci["diabetes"]
# delimo skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)
masinsko_ucenje(X_train, X_test, y_train, y_test)
X_train, X_test = izbaci_manje_bitne_podatke(X_train, X_test, y_train, y_test)
masinsko_ucenje(X_train, X_test, y_train, y_test)
