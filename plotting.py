import matplotlib.pyplot as plt
import torch


class MetricsPlotter:
    def __init__(self):
        self.is_ipython = "inline" in plt.get_backend()
        if self.is_ipython:
            from IPython import display

            self.display = display
        else:
            self.display = None

        plt.ion()

    def plot_metrics(self, episode_durations, rewards, policy_losses, value_losses, show_result=False, save_path=None):
        # Create a figure with a 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
        fig.suptitle("Training Metrics" if not show_result else "Results", fontsize=16)

        # Plot Episode Durations
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        axes[0, 0].set_title("Episode Durations")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Duration")
        axes[0, 0].plot(durations_t.cpu().numpy(), label="Duration")

        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            axes[0, 0].plot(means.cpu().numpy(), label="100-Episode Avg", linestyle="--")
        axes[0, 0].legend()

        # Plot Rewards
        rewards_t = torch.tensor(rewards, dtype=torch.float)
        axes[0, 1].set_title("Rewards")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Reward")
        axes[0, 1].plot(rewards_t.cpu().numpy(), label="Reward")

        if len(rewards_t) >= 100:
            reward_means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            reward_means = torch.cat((torch.zeros(99), reward_means))
            axes[0, 1].plot(reward_means.cpu().numpy(), label="100-Episode Avg", linestyle="--")
        axes[0, 1].legend()

        # Plot Policy Loss
        policy_t = torch.tensor(policy_losses, dtype=torch.float)
        axes[1, 0].set_title("Policy Loss")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].plot(policy_t.cpu().numpy(), label="Policy Loss", color="orange")
        axes[1, 0].legend()

        # Plot Value Loss
        value_t = torch.tensor(value_losses, dtype=torch.float)
        axes[1, 1].set_title("Value Loss")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].plot(value_t.cpu().numpy(), label="Value Loss", color="green")
        axes[1, 1].legend()

        # Adjust layout and save/show
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for the suptitle
        if save_path:
            plt.savefig(save_path + ".png", dpi=300)
            print(f"Metrics figure saved to {save_path}")

        if self.is_ipython:
            if not show_result:
                self.display.clear_output(wait=True)
                self.display.display(fig)
            else:
                self.display.display(fig)
        else:
            plt.show()

        plt.close(fig)
