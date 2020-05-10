from efficientnet_pytorch import EfficientNet
import torch.nn as nn

class MyEfficientNet(nn.Module):
    def __init__(self, trial, config, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.config = config

        model = EfficientNet.from_pretrained('efficientnet-b0')

        head = list(model.children())[0:2]
        self.head = nn.Sequential(*head)

        for p in self.head.parameters():
            p.requires_grad = False


        # bottom
        
        body_no_grad = model._blocks[:config.EFFICIENTNET_B0_LAYER]
        self.body_no_grad = nn.Sequential(*body_no_grad)

        for p in self.body_no_grad.parameters():
            p.requires_grad = False

        body_grad =  model._blocks[config.EFFICIENTNET_B0_LAYER:]
        self.body_grad = nn.Sequential(*body_grad)

        bottom_1 = list(model.children())[-6:-2]
        self.bottom_1= nn.Sequential(*bottom_1)

        bottom_2 = list(model.children())[-2:]
        self.bottom_2= nn.Sequential(*bottom_2)

        self.linear_block1 = self.linear_block()

    def linear_block(self):
        block = nn.Sequential(
            nn.Linear(1000, self.config.N_CLASSES),
            nn.Sigmoid()
        )
        return block

    def forward(self, image):

        x = self.head(image)
        print(x.shape)
        x = self.body_no_grad(x)
        print(x.shape)
        x = self.body_grad(x)
        x = self.bottom_1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.bottom_2(x)
        x = self.linear_block1(x)


        return x

if __name__ == "__main__":
    import torch
    import optuna

    from ..datasets import SampleDataset2D, Transforms
    from .. import config
    from torchsummary import summary

    model_name = 'efficientnet-b0'
    image_size = EfficientNet.get_image_size(model_name)

    config = config.Config
    transforms = Transforms()
    datasets = SampleDataset2D(transforms, width=image_size, height=image_size, channels=3)

    train_loader = torch.utils.data.DataLoader(
            datasets.data,
            batch_size= config.BATCHSIZE,
            shuffle=False,
        )
    
    for i in train_loader:
        test_data = i.cuda()

        break


    def objective(trial):


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MyEfficientNet(trial, config).to(device)


        try:
            model.calculate_linear_input(test_data)
        except AttributeError:
            print('no linear input')

        summary(model, input_size=(3, image_size, image_size))
        
        print(model(test_data))
        print(model)
        return 5

    study = optuna.create_study()
    study.optimize(objective, n_trials= config.N_TRIALS)

