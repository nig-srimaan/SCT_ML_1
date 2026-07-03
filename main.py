import pandas as pd
from sklearn.linear_model import LinearRegression

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']

train_df[features] = train_df[features].fillna(train_df[features].median())
test_df[features] = test_df[features].fillna(train_df[features].median())

X_train = train_df[features]
y_train = train_df['SalePrice']
X_test = test_df[features]

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': predictions
})

submission.to_csv('submission.csv', index=False)
print("submission.csv created successfully!")
