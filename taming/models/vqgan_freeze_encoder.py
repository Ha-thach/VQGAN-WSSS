import torch
from taming.models.vqgan import GumbelVQ


class GumbelVQFreezeEncoder(GumbelVQ):
    """
    GumbelVQ with frozen encoder.
    Only trains: decoder, quantize, quant_conv, post_quant_conv, and discriminator
    """

    def __init__(self, *args, freeze_encoder=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_encoder = freeze_encoder

        if self.freeze_encoder:
            # Freeze encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen - parameters will not be updated during training")

    def forward(self, input, return_pred_indices=False):
        """
        Override forward to accept return_pred_indices argument from GumbelVQ validation
        Note: return_pred_indices doesn't change output, just allows the parameter for compatibility
        """
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def validation_step(self, batch, batch_idx):
        """
        Override to fix duplicate logging issue with val/rec_loss
        """
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # Remove val/rec_loss from log_dict_ae to avoid duplicate logging
        log_dict_ae_filtered = {k: v for k, v in log_dict_ae.items() if k != "val/rec_loss"}
        self.log_dict(log_dict_ae_filtered)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate

        # Build parameter list excluding encoder if frozen
        params_to_optimize = []

        if not self.freeze_encoder:
            params_to_optimize += list(self.encoder.parameters())

        params_to_optimize += list(self.decoder.parameters())
        params_to_optimize += list(self.quantize.parameters())
        params_to_optimize += list(self.quant_conv.parameters())
        params_to_optimize += list(self.post_quant_conv.parameters())

        opt_ae = torch.optim.Adam(params_to_optimize,
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
