import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler


housing=fetch_california_housing()

df=pd.DataFrame(housing.data, columns=housing.feature_names)

df["Price"]=housing.target
print(df.head())

X=df.drop("Price",axis=1)
y=df["Price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model=LinearRegression()
model.fit(X_train,y_train)

predictions=model.predict(X_test)

mae=mean_absolute_error(y_test,predictions)
mse=mean_squared_error(y_test,predictions)
rmse=np.sqrt(mse)
r2=r2_score(y_test,predictions)

print(f"Mean Absolute Error is {mae}")
print(f"Mean Squared Error is {mse}")
print(f"Root mean Squared Error is {rmse}")
print(f"r2 score is {r2}")

x=input("Do you want to predict the price of your house based on some factors of your district (yes/no):").lower()

if x == "yes":
    try:
        medinc_input = float(input("Enter your median income in dollars: "))
        medinc = medinc_input / 10000

        House_age = float(input("Enter the age of your house: "))
        Ave_Rooms = float(input("Average number of rooms per household: "))
        Ave_bedrooms = float(input("Average number of bedrooms per household: "))
        population = float(input("Enter the population of the district: "))
        occupants = float(input("Enter average occupants per household: "))
        latitude = float(input("Enter the latitude (degrees): "))
        longitude = float(input("Enter the longitude (degrees): "))

        sample = [[medinc,House_age,Ave_Rooms,Ave_bedrooms,population,occupants,latitude,longitude]]

        sample_scaled = scaler.transform(sample)
        new_prediction = model.predict(sample_scaled)[0]

        print(f"The predicted house price is approximately ${new_prediction * 100000:.2f}")

    except ValueError:
        print("You can only input numeric values.")
    except Exception:
        print("Something went wrong:")

elif x=="no":
    print()

else:
    print("Invalid input")