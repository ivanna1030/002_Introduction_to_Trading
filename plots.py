import matplotlib.pyplot as plt

def plot_portfolio_value(portfolio_values):
    plt.plot(portfolio_values)
    plt.title('Portfolio value over time')
    plt.xlabel('Time')
    plt.ylabel('Portfolio value')
    plt.show()