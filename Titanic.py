import pandas as pd

# Look at whole picture of the dataset
titanic = pd.read_csv('~/Coding/Dataset/ML/Titanic/train.csv')
titanic.head()
titanic.columns
titanic.describe()
titanic.shape
titanic.info()

titanic.duplicated().sum()
titanic.isnull().sum()

# Visualize coor
titanic.corr()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
sns.heatmap(titanic.corr(), annot=True)
# plt.show()

# Count unique value of categorical features and plot them
cate_features = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Survived', 'Pclass']
for i in cate_features:
    print(i, len(pd.unique(titanic[i])))

for i in ['Sex', 'Embarked', 'Survived', 'Pclass']:
    titanic[i].value_counts().plot(kind='bar')
    plt.title(i)
    plt.ylabel('Nums')
    # plt.show()

# Delete uninfomative cols
for i in ['Ticket', 'PassengerId', 'Cabin', 'Name']:
    print(i, titanic[i].head(10))

titanic_after = titanic.drop(columns=['Ticket', 'PassengerId', 'Cabin', 'Name'])

titanic_after.head()
titanic_after.isnull().sum()

# Imputation and dummy coding
titanic_after['Embarked'].fillna(titanic_after['Embarked'].mode()[0], inplace = True)

titanic_after = pd.get_dummies(titanic_after, prefix=['Sex', 'Embarked'])

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors = 5)
titanic_cleaned = pd.DataFrame(imputer.fit_transform(titanic_after), columns=titanic_after.columns)

titanic_cleaned.head()
titanic_cleaned.isnull().sum()

label = titanic_cleaned['Survived']
titanic_train = titanic_cleaned.drop(columns='Survived')

# Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np

n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 100, 10)]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,'bootstrap': bootstrap}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(rf, random_grid, cv=10, random_state=42, n_jobs=-1)
rf_random.fit(titanic_train, label)
rf_random.best_params_
rf_random.best_score_
# 0.82

from sklearn.model_selection import cross_val_score
base_model = RandomForestClassifier(n_estimators=10, random_state=42)
score = cross_val_score(base_model, titanic_train, label, cv = 10, scoring='accuracy')
score.mean()
# 0.81

# Logistic Regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


pipe = Pipeline([('scale', StandardScaler()),
    ('logreg', LogisticRegression())])

pipe.fit(titanic_train, label)
pipe.score(titanic_train, label)
# 0.80

# Select random forest as final model
test_df = pd.read_csv('~/Coding/Dataset/ML/Titanic/test.csv')

test_df = test_df.drop(columns=['Ticket', 'Cabin', 'Name'])
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace = True)
test_df = pd.get_dummies(test_df, prefix=['Sex', 'Embarked'])
test_df = pd.DataFrame(imputer.fit_transform(test_df), columns=test_df.columns)

X = test_df.drop(columns='PassengerId')

result = rf_random.predict(X)
final_result = {'PassengerId': test_df['PassengerId'], 'Survived': result}
re = pd.DataFrame(final_result)
re.head()

re['PassengerId'] = re['PassengerId'].astype(int)
re['Survived'] = re['Survived'].astype(int)

re.to_csv(r'~/Coding/Dataset/ML/Titanic/result.csv', index=False, header=True)

