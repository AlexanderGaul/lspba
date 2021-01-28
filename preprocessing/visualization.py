


def plot_patches(patches_stack, robust_mean, mean, closest) :
    num_images = len(patches_stack) + 3
    vmax = 0.6
    
    plot_height = int(np.sqrt(num_images))
    plot_width = int(np.ceil(num_images / plot_height))
    current_plot = 1
    
    fig = plt.figure()
    
    ax = fig.add_subplot(plot_height, plot_width, current_plot)
    ax.imshow(robust_mean.reshape(4,4), vmin=-vmax, vmax=vmax)
    ax.set_axis_off()
    ax.set_title("Robust Mean", fontsize=8)
    current_plot += 1
    ax = fig.add_subplot(plot_height, plot_width, current_plot)
    ax.imshow(mean.reshape(4,4), vmin=-vmax, vmax=vmax)
    ax.set_axis_off()
    ax.set_title("Mean", fontsize=8)
    current_plot += 1
    ax = fig.add_subplot(plot_height, plot_width, current_plot)
    ax.imshow(closest.reshape(4,4), vmin=-vmax, vmax=vmax)
    ax.set_axis_off()
    ax.set_title("Closest", fontsize=8)
    current_plot += 1
    
    for i in range(len(patches_stack)) :
        ax = fig.add_subplot(plot_height, plot_width, current_plot)
        ax.imshow(patches_stack[i, :].reshape(4,4), vmin=-vmax, vmax=vmax)
        ax.set_axis_off()
        current_plot += 1
    
    plt.show()


def combined_plot(images, patches_stack, robust_mean, mean, closest) :
    pass
