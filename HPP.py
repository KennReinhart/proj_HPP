import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('data/housing.csv')
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df)

'''
step 2
EDA, exploratory data analysis
    - understand what drives house prices
    - validate linear regression assumptions
    - identify relationships between variables
'''
#how price is distributed

plt.figure(figsize = (8,5))
sns.histplot(df["Price"], bins=20)
plt.title("Distribution of house prices")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("output/figures/price_distribution.png")
# plt.show()

#features correlate with price
plt.figure(figsize = (8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("output/figures/correlation_matrix.png")
# plt.show()

#price change with size
plt.figure(figsize = (8,5))
sns.scatterplot(data=df, x="Size", y="Price")
for i, (x_val, y_val) in enumerate(zip(df["Size"], df["Price"])):
    plt.annotate(
        f"(Size:{x_val} , Price:{y_val})",
        (x_val, y_val),
        textcoords="offset points",
        xytext=(5, -2.5),
        fontsize=8,
        color="black"
    )
plt.title("House Price vs Size")
plt.tight_layout()
plt.savefig("output/figures/price_vs_size.png")
# plt.show()

plt.figure(figsize = (8,5))
sns.boxplot(data=df, x="LocationScore", y="Price")
plt.title("House Price by Location Score")
plt.tight_layout()
plt.savefig("output/figures/price_by_location.png")
plt.show()

print(df.corr())

'''
Step 3, Train test split
'''
X = df.drop("Price", axis=1)
y = df["Price"]

from sklearn.model_selection import train_test_split
#separate data to evaluate generalization
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#train linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
#predictions
y_pred = model.predict(X_test)

#evaluate model performance
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

#inspect model coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_,
}).sort_values(by="Coefficient", ascending=False)

print(coefficients)

plt.figure(figsize = (8,5))
sns.scatterplot(x=y_test, y=y_pred, hue="Coefficient")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("output/figures/actual_vs_predicted.png")
plt.show()

'''
Step 4, business insights, limitations and reporting
'''

