import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd


sp500=yf.Ticker("^GSPC")
sp500=sp500.history(period="max")
print(sp500.head())
print(sp500.tail())
print(sp500.columns)
print(sp500.info) 

print(sp500.index)

#cleaning and visuallizing data

sp500.plot.line(y="Close",use_index=True)
plt.show()

#setting up our target for machine learning

del sp500["Dividends"]
del sp500["Stock Splits"]

sp500["Tomorrow"]=sp500['Close'].shift(-1)

sp500["Target"]=(sp500["Tomorrow"]>sp500['Close'].astype(int))

sp500=sp500.loc['1990-01-01':].copy()

#Training an initial Machine learning model

model=RandomForestClassifier(n_estimators=100,min_samples_split=100,random_state=1)

train=sp500.iloc[:-100] #specific rows of exclusive last 100 rows
test=sp500.iloc[-100:]

predictors=["Close","Volume","Open","High","Low"]

model.fit(train[predictors],train["Target"]) 

preds=model.predict(test[predictors])

preds=pd.Series(preds,index=test.index)

precision_score(test["Target"],preds)

combined=pd.concat([test["Target"],preds],axis=1)

#combined.plot()

plt.plot(combined.iloc[:, 1])  # Plot the second column (predicted probabilities)
plt.xlabel("Index")
plt.ylabel("Probability")
plt.title("Predicted Probabilities")
plt.show()

#Building a backtesting system

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# def predict(train,test,predictors,model):
#     model.fit(train[predictors],train["Target"])
#     preds=model.predict(test[predictors])
#     preds=pd.series(preds,index=test.index,name="Predictions")
#     combined=pd.concat([test["Target"],preds],axis=1)
#     return combined

def backtest(data,model,predictors,start=2500,step=250):
    all_predictions=[]

    for i in range(start,data.shape[0],step):
        train=data.iloc[0:i].copy()
        test=data.iloc[i:(i+step)].copy()
        predictions=predict(train,test,predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)   

predictions=backtest(sp500,model,predictors)

predictions["Predictions"].value_counts()

precision_score(predictions["Target"],predictions["Predictions"])

predictions["Target"].value_counts()/predictions.shape[0]

#Adding additional predictors to our model

horizons=[2,5,60,250,1000]

new_predictors=[]

for horizon in horizons:
    rolling_averages=sp500.rolling(horizon).mean()

    ratio_column=f"Close_Ratio_{horizon}"

    sp500[ratio_column]=sp500["Close"]/rolling_averages["Close"]

    trend_column=f"Trend_{horizon}"

    sp500[trend_column]=sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column,trend_column]

    sp500=sp500.dropna()

#improving our model

model=RandomForestClassifier(n_estimators=200,min_samples_split=50,random_state=1)


def predict(train,test,predictors,model):
    model.fit(train[predictors],train["Target"])
    preds=model.predict_proba(test[predictors])[:,1]  # you want to extract all rows from the second column. You can use the [ :, 1] indexing to achieve this in numpy
    preds[preds>=.6]=1
    preds[preds<.6]=0
    preds=pd.Series(preds,index=test.index,name="Predictions")
    combined=pd.concat([test["Target"],preds],axis=1)
    return combined

predictions=backtest(sp500,model,new_predictors)

counts=predictions["Predictions"].value_counts() #error here

print(counts)

precision=precision_score(predictions["Target"],predictions["Predictions"])

print(precision)

#summary and next steps with the model









print()

print()
