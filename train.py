from pytorch_lightning.cli import LightningCLI
import yaml
import torch
def cli_main_vocos_spec():
    config = yaml.load(open("/data3/tansongbin/vocoder_projects/my_vocosfinetune/configs/vocos_spec.yaml", "r"), Loader=yaml.FullLoader)
    # config = yaml.load(open("/data3/tansongbin/vocoder_projects/my_vocosfinetune/configs/vocos.yaml", "r"), Loader=yaml.FullLoader)
    cli = LightningCLI(run=False, args=config)
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule,
                    ckpt_path='/data3/tansongbin/vocoder_projects/my_vocosfinetune/logs/vocos_spec/version_2/checkpoints/vocos_checkpoint_epoch=13_step=105084_val_loss=5.9048.ckpt')

def cli_main_vocos():
    # config = yaml.load(open("/data3/tansongbin/vocoder_projects/my_vocosfinetune/configs/vocos_spec.yaml", "r"), Loader=yaml.FullLoader)
    config = yaml.load(open("/data3/tansongbin/vocoder_projects/vocos_finetune_projects/my_vocosfinetune/configs/vocos.yaml", "r"), Loader=yaml.FullLoader)
    cli = LightningCLI(run=False, args=config)
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)

if __name__ == "__main__":
    cli_main_vocos()
