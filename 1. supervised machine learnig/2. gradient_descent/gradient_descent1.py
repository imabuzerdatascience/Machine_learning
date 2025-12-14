import numpy as np 
import pandas as pd 

# Expected answer. m = 0.05168176, b=18.0465

def gradient_descent(x , y , lr=0.1 , epochs=3000) :
    
    # scaling the value 
    x_min , x_max = x.min() , x.max() 
    y_min , y_max = y.min() , y.max() 

    x_scaled = (x-x_min) / (x_max - x_min)
    y_scaled = (y-y_min) / (y_max - y_min) 

    # Initialize parameters
    b = 0.0  # Intercept
    m = 0.0  # Slope
    n = len(y_scaled)  # Number of data points

    # run for loop 
    for epoch in range(epochs):
        y_predict = b + m * x_scaled 
        error = y_scaled - y_predict 
        cost = np.mean(error**2) 

         # partial derivative 
        dm = -2 * np.mean(error * x_scaled)
        db = - np.mean(error)
 
         #Update parameters
        b -= lr * db
        m -= lr * dm

         # Optional: Print cost every 100 iterations to monitor progress
        if epoch % 100 == 0:
          print(f"Epoch {epoch}: Cost = {cost}, b = {b}, m = {m}")

    m_original = m * (y_max - y_min) / (x_max - x_min)
    b_original = b * (y_max - y_min) + y_min - m_original * x_min

    return b_original , m_original 

if __name__ == "__main__" : 
    # export excel file 
    df = pd.read_csv("home_prices.csv")

    x = df["area_sqr_ft"].to_numpy()
    y = df["price_lakhs"].to_numpy()
    
    b, m = gradient_descent(x, y) 
    print(f"Final Results: m={m}, b={b}")
 

    