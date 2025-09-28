import matplotlib.pyplot as plt

def plot_portfolio_value(historic):
    plt.plot(historic.index, historic['Portfolio Value'])
    plt.title('Portfolio value over time')
    plt.xlabel('Time')
    plt.ylabel('Portfolio value')
    plt.show()