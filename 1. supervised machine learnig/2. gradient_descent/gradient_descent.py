import numpy as np 

def gradient_descent(x, y, lr=0.01, epoch=3000):
    m, b = 0.0, 0.0 
    
    for i in range(epoch):

        # Prediction
        y_predict = m * x + b 

        # Error
        error = y - y_predict 

        # Cost (MSE)
        cost = np.mean(error**2)

        # Partial derivatives (Gradient)
        dm = -2 * np.mean(x * error)
        db = -2 * np.mean(error)

        # Update formula
        m -= dm * lr
        b -= db * lr

        # PRINT EVERY STEP
        print(f"epoch={i}, m={m}, b={b}, cost={cost}")

    return m, b


if __name__ == "__main__":
    x = np.array([1,2,3,4,5])
    y = np.array([5,7,9,11,13])
    gradient_descent(x, y)
