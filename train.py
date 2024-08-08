from pytorch_lightning.cli import LightningCLI
import yaml
import torch
def cli_main():
    config = yaml.load(open("/data3/tansongbin/vocoder_projects/my_vocosfinetune/configs/vocos.yaml", "r"), Loader=yaml.FullLoader)
    cli = LightningCLI(run=False, args=config)
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)

if __name__ == "__main__":
    cli_main()
