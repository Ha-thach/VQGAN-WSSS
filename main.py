import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

from taming.utils import get_obj_from_str, instantiate_from_config
from taming.data.utils import custom_collate




def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n", # short name of this flag
        "--name", # long name of this flag. Name means name to save for logs dir
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume", # continue training from this logdir or checkpoint
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base", # base config files
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train", # whether to train the model
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )

    return parser


def nondefault_trainer_args(opt): # find all arguments which are different from default 
    parser = argparse.ArgumentParser() # create a new parser
    parser = Trainer.add_argparse_args(parser) # add all default trainer args
    args = parser.parse_args([]) #Thêm tất cả tham số mặc định của Trainer
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config): 
    """
    function instantiate_from_config(cfg):
    assert "target" in cfg
    ClassRef = get_obj_from_str(cfg["target"])
    kwargs   = cfg.get("params", {})
    return ClassRef(**kwargs)
    """

    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset): # dataset list-like
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule): # Pytorch Lightning DataModule helps to initiate data loaders from file config yaml
    """
    It is a LightningDataModule that separates the data configuration (dataset and dataloader) from the code, allowing you to easily switch datasets just by editing the YAML config file, without modifying the code.
    
    class DataModuleFromConfig:
    init(batch_size, train=None, validation=None, test=None, wrap=False, num_workers=None):
        store dataset_configs for provided splits
        set loader funcs to _train/_val/_test if split exists
        self.wrap = wrap
        self.num_workers = num_workers or batch_size * 2

    prepare_data():
        for each split cfg in dataset_configs:
            instantiate_from_config(cfg)   # trigger downloads/prep if dataset does it

    setup(stage=None):
        self.datasets = {k: instantiate_from_config(cfg) for k,cfg in dataset_configs}
        if wrap: self.datasets[k] = WrappedDataset(self.datasets[k])

    _train_dataloader():
        return DataLoader(datasets["train"], batch_size, num_workers, shuffle=True, collate_fn=custom_collate)

    _val_dataloader():
        return DataLoader(datasets["validation"], batch_size, num_workers, collate_fn=custom_collate)

    _test_dataloader():
        return DataLoader(datasets["test"], batch_size, num_workers, collate_fn=custom_collate)

    """
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, collate_fn=custom_collate)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)


class SetupCallback(Callback):
    """
    Callback to be called at the beginning of training.
    Creates log directories and saves config files.
    class SetupCallback(Callback):
    init(resume, now, logdir, ckptdir, cfgdir, config, lightning_config)

    on_pretrain_routine_start(trainer, pl_module):
        if global_rank == 0:
            mkdir logdir, ckptdir, cfgdir
            print + save project config        -> cfgdir/<now>-project.yaml
            print + save lightning config      -> cfgdir/<now>-lightning.yaml
        else:
            if not resume and exists(logdir):
                move logdir -> child_runs/<name>

    """
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume # continue training from this logdir or checkpoint
        self.now = now # current date/time for naming logdir
        self.logdir = logdir # main log directory
        self.ckptdir = ckptdir # checkpoint directory
        self.cfgdir = cfgdir # config directory
        self.config = config # project config: model, data, losses, transforms, any experiment-specific hyperparams
        self.lightning_config = lightning_config # lightning config: trainer, logger, model checkpoint, callbacks

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(OmegaConf.to_yaml(self.config)) # Convert config to YAML format for printing
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now))) # save config file vs self.now chính là mốc thời gian (timestamp) hoặc tên phiên chạy hiện tại, dùng để tạo tên file config có gắn thời gian

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    """
    class ImageLogger(Callback):
    init(batch_frequency, max_images, clamp=True, increase_log_steps=True):
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [1,2,4,8,..., batch_freq] if increase_log_steps else [batch_freq]
        self.logger_log_images = {WandbLogger: _wandb, TestTubeLogger: _testtube}

    check_frequency(batch_idx) -> bool:
        return (batch_idx % batch_freq == 0) or (batch_idx in log_steps and pop_first(log_steps))

    log_img(pl_module, batch, batch_idx, split):
        if not (check_frequency & pl_module.has_log_images & max_images>0): return
        was_training = pl_module.training; set eval
        with no_grad: images = pl_module.log_images(batch, split, pl_module)
        trim to N<=max_images, detach+cpu, clamp to [-1,1] if needed
        log_local(save_dir, split, images, step, epoch, batch_idx)
        pick logger-specific func and log
        restore train if needed

    log_local(save_dir, split, images, step, epoch, batch_idx):
        for each key -> grid(images[key], nrow=4)
        map [-1,1]→[0,1], convert to uint8, save PNG to save_dir/images/split/...

    """
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency # frequency of logging
        self.max_images = max_images
        self.logger_log_images = { 
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TestTubeLogger: self._testtube,
        } # `logger_log_images` is a dictionary that maps specific logger classes to their corresponding image logging methods. This allows the ImageLogger callback to support multiple logging backends by associating each backend with its own method for logging images.
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps: # increase the logging steps to check more frequently at the beginning of training 
            self.log_steps = [self.batch_freq]
        self.clamp = clamp # whether to clamp the images to be in the range [-1, 1] before converting to (0, 1) for logging

    @rank_zero_only # decorator ensures that the decorated method is only executed on the process with rank 0 in a distributed training setup. This is important for logging to avoid duplicate entries from multiple processes.
    def _wandb(self, pl_module, images, batch_idx, split): # wandb: weights and biases
        raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")

def get_num_gpus(trainer_config):
        """Parse number of GPUs from trainer config"""
        if "gpus" not in trainer_config:
            return 0
        gpus = trainer_config["gpus"]
    
        if gpus is None:
            return 0
        elif isinstance(gpus, int):
            return gpus
        elif isinstance(gpus, str):
        # Handle "0,1" or "0," formats
            gpu_list = [x.strip() for x in gpus.strip(",").split(',') if x.strip()]
            return len(gpu_list)
        elif isinstance(gpus, (list, tuple)):
            return len(gpus)
        else:
            return 0

if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") # current date/time for naming logdir

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)


    opt, unknown = parser.parse_known_args()
    # resume means continuing training from this logdir or checkpoint
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume): # if resume is a file, then it is a checkpoint file
            paths = opt.resume.split("/") # split path into components by "/" and save into a list 
            idx = len(paths)-paths[::-1].index("logs")+1 # find the index of "logs" in the reversed list and calculate its index in the original list
            logdir = "/".join(paths[:idx]) # path of log directory
            ckpt = opt.resume # path of checkpoint file (whole path) 
        else: # if resume is a directory, then it is a log directory, and the checkpoint file is "last.ckpt" in the "checkpoints" subdirectory
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt") # path of checkpoint file

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml"))) # find all config files in the "configs" subdirectory of the log directory => sort them alphabetically
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")    
        nowname = _tmp[_tmp.index("logs")+1]
    else:
        if opt.name: # if name is specified, use it as the name of the log directory
            name = "_"+opt.name
        elif opt.base: #
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = "" #
        nowname = now+name+opt.postfix
        logdir = os.path.join("logs", nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base] # load all config files specified in opt.base into a list of OmegaConf objects
        cli = OmegaConf.from_dotlist(unknown) # convert command line arguments of the form `--key value` or `--nested.key value` into an OmegaConf object and save it into cli
        config = OmegaConf.merge(*configs, cli) # merge all config files and command line arguments into a single OmegaConf object. Left-most configs take precedence over right-most ones
        lightning_config = config.pop("lightning", OmegaConf.create()) # pop the "lightning" key from config if it exists, otherwise create an empty OmegaConf object
        # trainer: gpu or cpu
        trainer_config = lightning_config.get("trainer", OmegaConf.create()) 
        # default to ddp
        trainer_config["distributed_backend"] = "ddp" 
        for k in nondefault_trainer_args(opt): # find all arguments which are different from default
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config: #
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        # NOTE wandb < 0.10.0 interferes with shutdown
        # wandb >= 0.10.0 seems to fix it but still interferes with pudb
        # debugging (wrongly sized pudb ui)
        # thus prefer testtube for now
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        logger_cfg = lightning_config.logger or OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_top_k": -1,
                "save_last": True,
                "monitor": None,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        modelckpt_cfg = lightning_config.modelcheckpoint or OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        ckpt_callback = instantiate_from_config(modelckpt_cfg)    
        
        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    #"log_momentum": True
                }
            },
        }
        callbacks_cfg = lightning_config.callbacks or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        #trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        callbacks_list = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        callbacks_list.append(ckpt_callback)
        trainer_kwargs["callbacks"] = callbacks_list    
        trainer_kwargs["checkpoint_callback"] = True
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
    
        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = get_num_gpus(lightning_config.trainer)
            if ngpu == 0:
                print("Warning: GPU requested but gpus config is 0 or invalid. Using CPU.")
                ngpu = 1
                cpu = True
        else:
            ngpu = 1
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
