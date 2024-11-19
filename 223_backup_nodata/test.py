import wandb

if __name__=="__main__":

    wandb.init(
            project="Fed_Learning",
            entity="aprkal12",
            config={
                "learning_rate": 0.001,
                "architecture": "Resnet18",
                "dataset": "CIFAR-10",
            }
        )
    wandb.run.name = "Resnet18_CIFAR-10_D=100%_oneline"

    for i in range(50):
        wandb.log({"test_acc" : 0.883, "val_acc" : 0.8824})
