from src.utils.logger import configure, logkvs, log, dumpkvs
import torch as th
from os.path import exists


class Trainer:
    """
    A generic training loop.
    """
    def __init__(self, model, objective, dataloader, optimizer, scheduler, logger_dict, checkpointing_dict, device, rank=0, config=None):
        self.model = model
        self.objective = objective
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self._set_model_to_device()

        self.checkpointing_dict = checkpointing_dict
        self.log_print_freq = logger_dict.pop("log_print_freq")
        configure(**logger_dict, rank=rank, config=config)

        if self.checkpointing_dict.get("restart"):
            self._resume_training()
        else:
            log("Training the model from scratch")
            self.start_epoch = 0

    def _set_model_to_device(self):
        return self.model.to(self.device)

    def _resume_training(self):
        assert exists(self.checkpointing_dict["save_path"]+f"/checkpoint_{self.checkpointing_dict['restart_epoch']}.pt"), "No checkpoint for resuming the training!"
        restart_epoch = self.checkpointing_dict["restart_epoch"]
        log(f"Loading state from checkpoint epoch : {restart_epoch}")

        state = th.load(self.checkpointing_dict["save_path"]+f"/checkpoint_{restart_epoch}.pt", weights_only=True)

        self.model.network.load_state_dict(state['model_state_dict'])

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.start_epoch = restart_epoch + 1

        if self.scheduler is not None and state.get('sched_state_dict') is not None:
            self.scheduler.load_state_dict(state.get('sched_state_dict'))

    def _save_training(self, epoch):
        state = {
                'epoch': epoch,
                'model_state_dict': self.model.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'sched_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None
                }
        th.save(state, f"{self.checkpointing_dict['save_path']}/checkpoint_{epoch}.pt")

    def train(self, num_epochs):
        for epoch in range(self.start_epoch, num_epochs):
            if hasattr(self.dataloader.sampler, 'set_epoch'):
                self.dataloader.sampler.set_epoch(epoch)

            log(f"Starting epoch {epoch}")
            train_loss = 0.0
            num_batches = 0

            for batch in self.dataloader:
                loss = self.objective(self.model, batch, device=self.device)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                loss_val = loss.item()

                train_loss += loss_val
                num_batches += 1

                if num_batches % self.checkpointing_dict["log_batch_int"] == 0:
                    logkvs({"batch": num_batches, "Minibatch loss": loss_val})
                    log(f"Within Epoch {epoch}, Batch {num_batches}: Training loss: {loss_val}")
                    dumpkvs()

            train_loss = train_loss / num_batches

            log_info = {
                        "epoch": epoch,
                        "Epoch loss": train_loss,
                       }

            logkvs(log_info)

            if epoch % self.log_print_freq == 0:
                log(f"Average Training loss at epoch {epoch}: {train_loss}")

            if epoch % self.checkpointing_dict["save_epoch_int"] == 0:
                log(f"Saving state at epoch {epoch}...")
                self._save_training(epoch)

            dumpkvs()

        log("Training complete")
