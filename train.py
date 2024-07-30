from pytorch_lightning.cli import LightningCLI
import yaml
def cli_main():
    config = yaml.load(open("/data3/tansongbin/vocoder_projects/vocos/configs/vocos.yaml", "r"), Loader=yaml.FullLoader)
    # config["trainer"]["devices"]=[0]
    cli = LightningCLI(run=False, args=config)
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)

if __name__ == "__main__":
    cli_main()
