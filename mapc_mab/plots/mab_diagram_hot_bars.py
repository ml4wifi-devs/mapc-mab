import matplotlib.pyplot as plt
import matplotlib.cm as cm

MCS_11_DATA_RATE = 143


def get_color(value):
    cmap = cm.get_cmap('hot')
    color = cmap(value * 0.5 + 0.25)
    return color


def get_hot_bar(value, fontsize=22):
    plt.figure(figsize=(5, 1))
    plt.barh(["thr"], 2 * MCS_11_DATA_RATE * value, color=get_color(value))
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.xticks(fontsize=fontsize)
    plt.xlim(0, 2 * MCS_11_DATA_RATE)
    plt.xticks(range(0, 2 * MCS_11_DATA_RATE + 1, 80))
    plt.savefig(f"hotbar-{value}.svg", bbox_inches='tight')


if __name__ == "__main__":
    plt.rcParams["font.family"] = "serif"
    get_hot_bar(0.2)
    get_hot_bar(0.9)
    get_hot_bar(0.5)
    get_hot_bar(0.87)
    get_hot_bar(0.92)
