import logging

import torch
from torch.optim.optimizer import Optimizer

# wandb_logger and wandb removed: optional logging only used when use_detector=True
# If wandb logging is needed, pass a wandb_run object to EmbeddingSpaceAttack.__init__.

INIT_TYPES = ["instruction", "suffix"]


class NoAttack:
    def __init__(
        self,
        embed_weights,
    ) -> None:
        """
        No Attack
        """
        self.embed_weights = embed_weights
        self.vocab_size = self.embed_weights.shape[0]
        self.embedding_size = self.embed_weights.shape[1]

    def attack(self, model, input_ids, target_ids, attention_mask):
        all_losses = []
        affirmative_responses = []

        adv_perturbation, adv_perturbation_mask = self.init_perturbation(input_ids, target_ids, attention_mask)
        input_embeds = self.get_embeddings(input_ids)

        return (
            input_embeds.detach(),
            adv_perturbation.detach(),
            adv_perturbation_mask.detach(),
            all_losses,
            affirmative_responses,
        )

    def get_adv_embeddings(self, input_embeds, adv_perturbation, adv_perturbation_mask):
        return input_embeds

    def init_perturbation(self, input_ids, target_ids, attention_mask):
        target_mask = target_ids > 0
        input_mask = (~target_mask * attention_mask).to(bool)
        batch_size, num_input_tokens = input_ids.shape
        dtype = self.embed_weights.dtype

        adv_perturbation = torch.zeros(
            (batch_size, num_input_tokens, self.embedding_size), device=input_ids.device, dtype=dtype
        )

        adv_perturbation_mask = torch.zeros(
            (batch_size, num_input_tokens), device=input_ids.device, dtype=input_ids.dtype
        )
        adv_perturbation_mask[input_mask] = 1

        return adv_perturbation, adv_perturbation_mask.unsqueeze(-1)

    def get_one_hot(self, ids):
        device = self.embed_weights.device
        batch_size, num_tokens = ids.shape

        # Adjusting IDs less than 0 to 0
        ids = torch.where(ids < 0, torch.tensor(0, device=device, dtype=ids.dtype), ids)

        one_hot = torch.zeros(batch_size, num_tokens, self.vocab_size, device=device, dtype=self.embed_weights.dtype)
        one_hot.scatter_(2, ids.unsqueeze(2), 1)
        return one_hot

    def get_embeddings(self, ids):
        one_hot = self.get_one_hot(ids)
        embeddings = (one_hot @ self.embed_weights).data
        return embeddings


class EmbeddingSpaceAttack:
    def __init__(
        self,
        embed_weights,
        response_key,
        tokenizer,
        hidden_state_detector_index,
        iters=8,
        opt_config=None,
        eps=1.0,
        init_type="instruction",
        suffix_tokens=10,
        relative_lr=False,
        detector_loss_coeff=0.5,
        wandb_run=None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the EmbeddingAttack class.

        Args:
            embed_weights (torch.Tensor): The token embedding matrix of the respective model.
            response_key (str): The response key of the datacollator used to separate instructions and targets.
            tokenizer (Tokenizer): The tokenizer object.
            use_detector (bool): Whether to use a detector model.
            hidden_state_detector_index (int): The index of the hidden layer to use for the detector.
            iters (int, optional): The number of iterations.
            opt_config (dict, optional): The optimizer configuration.
            eps (float, optional): The epsilon value.
            init_type (str, optional): The initialization type
            suffix_tokens (int, optional): The number of suffix tokens
            debug (int, optional): The debug level with increasing verbosity. (0 None, 1 print final loss, 2 include loss at every iteration, 3 include norms and shapes)
        """

        self.embed_weights = embed_weights
        self.tokenizer = tokenizer
        self.vocab_size = self.embed_weights.shape[0]
        self.embedding_size = self.embed_weights.shape[1]
        self.embedding_norm = torch.norm(embed_weights, p=2, dim=-1).mean()
        self.iters = iters
        self.opt_config = opt_config
        self.detector_loss_coeff = detector_loss_coeff
        if relative_lr:
            self.opt_config["lr"] = self.opt_config["lr"] * self.eps
        self.eps = eps * self.embedding_norm
        self.wandb_step = 0
        self.use_detector = False
        self.hidden_state_detector_index = hidden_state_detector_index
        self.current_iter_log = 0
        self.wandb_run = wandb_run

        print("\nEpsilon calculation:")
        print(f"  Original eps: {eps}")
        print(f"  Embedding norm: {self.embedding_norm}")
        print(f"  Final eps: {self.eps}\n")

        logging.info(
            f"L2 norm of embedding weights equals {self.embedding_norm} eps multiplier is: {eps} using eps: {self.eps}"
        )

        if init_type not in INIT_TYPES:
            ValueError(f"init_type must be in {INIT_TYPES} and not {self.init_type}")
        self.init_type = init_type

        self.suffix_tokens = suffix_tokens
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def attack(self, model, input_ids, target_ids, attention_mask, detector, use_detector, global_step=0):
        """
        Args:
            -model: target model
            -input_ids: input_and_harmfulTarget inputs_ids
            -target_ids: target inputs_ids, it has the shape of input_ids but the non-target tokens are set to 0
            -attention_mask: attention mask for the input_ids which is used to not attend to the padding tokens
            -detector: detector model, if None no detector is used
        """
        print(f"\n\n======================= Starting attack - Global Step {global_step} ========================")
        self.use_detector = use_detector

        # save losses and responses
        best_loss = torch.inf
        all_total_losses = []
        all_losses = []
        all_detector_losses = []

        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        print(f"Attack input text (first element):\n{input_text}\n")

        # init embeddings of input instruction and target and initialize adversarial perturbation
        adv_perturbation, adv_perturbation_mask = self.init_perturbation(input_ids, target_ids, attention_mask)

        input_embeds = self.get_embeddings(input_ids)
        target_one_hot = self.get_one_hot(target_ids)

        # loss targets
        loss_mask = self.get_loss_mask(target_ids)

        # init opt
        opt = self.init_opt([adv_perturbation])

        # optimization loop
        for i in range(self.iters):
            # Clear gradients at the start of each iteration
            opt.zero_grad()

            adv_embeds = self.get_adv_embeddings(input_embeds, adv_perturbation, adv_perturbation_mask)
            logits, total_loss, loss, detector_loss, hidden_states = self.calc_loss(
                i, model, adv_embeds, target_one_hot, target_ids, attention_mask, loss_mask, detector=detector
            )

            if self.use_detector:
                # Use global step if provided, otherwise use local counter
                self.current_iter_log += 1

                log_dict = {
                    "attack_perturb_loss_total": total_loss.item(),
                    "attack_perturb_loss_comparison": loss.item(),
                    "attack_perturb_loss_detector": detector_loss.item() if self.use_detector else 0,
                    "step_attack": self.current_iter_log,
                }
                if self.wandb_run is not None:
                    self.wandb_run.log(log_dict)

            total_loss.backward()
            opt.step()

            all_total_losses.append(total_loss.detach().item())
            all_losses.append(loss.detach().item())
            all_detector_losses.append(detector_loss.detach().item() if self.use_detector else 0)

            # perturbations that are applied directly to the instruction are constraint to a small lp ball
            if self.init_type == "instruction":
                adv_perturbation = self.project_l2(adv_perturbation)
            # perturbations that are added as a suffix should be projected to the simplex
            elif self.init_type == "suffix":
                adv_perturbation = self.project_simplex(adv_perturbation)

            # save best perturbation
            if total_loss < best_loss:
                best_loss = total_loss.detach()
                self.best_adv_perturbation = adv_perturbation.clone().detach()

        self.debug_output_indexed(target_ids, logits, attention_mask, index=0)

        perturbed_embeds = self.get_adv_embeddings(input_embeds, adv_perturbation, adv_perturbation_mask)
        return (
            input_embeds.detach(),
            adv_perturbation.detach(),
            adv_perturbation_mask.detach(),
            perturbed_embeds.detach(),
            hidden_states.detach(),
            all_total_losses,
            all_losses,
            all_detector_losses,
        )

    def calc_loss(
        self, i, model, input_embeds, target_one_hot, target_ids, attention_mask, loss_mask, detector, log_debug=True
    ):

        output = model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True)
        logits = output.logits
        all_hidden_states = output.hidden_states
        hidden_states = all_hidden_states[self.hidden_state_detector_index]

        # ------- Compute target loss -------
        # 1. shift targets by 1
        targets = target_ids[:, 1:].clone()  # shape: (B, T‑1)
        # 2. wipe out the tokens we do not want to learn from
        #    loss_mask == 1  → keep
        #    loss_mask == 0  → set to sentinel value  (‑100)
        targets = targets.masked_fill(loss_mask == 0, -100)
        # 3. flatten batch+time so CrossEntropyLoss sees (N, C)
        logits_flat = logits[:, :-1].reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)
        # 4. compute loss
        target_loss = self.loss_fct(logits_flat, targets_flat)

        # Add detector loss if needed: the attacker wants the detector to score the adversarial
        # example as benign. Each detector defines evasion_loss in its own convention.
        if self.use_detector:
            detector_loss = detector.evasion_loss(hidden_states, target_ids, attention_mask)
        else:
            detector_loss = torch.tensor(0.0, device=logits.device)

        total_loss = (1 - self.detector_loss_coeff) * target_loss + self.detector_loss_coeff * detector_loss

        if i == 0:  # save the logits and loss for the first iteration (ie with no perturbation)
            self.logits_no_perturbation = logits.clone().detach()
            self.loss_no_perturbation = target_loss.clone().detach()

        return logits, total_loss, target_loss, detector_loss, hidden_states

    def project_l2(self, adv_perturbation):
        """
        Constrain the adversarial perturbation to the L2 ball of radius eps.
        Keeps the direction of the perturbation but scales it down if its norm exceeds eps.
        """
        with torch.no_grad():
            norm = torch.norm(adv_perturbation, p=2, dim=-1, keepdim=True)
            mask = (norm > self.eps).squeeze()

            debug_L2_projection = False
            if debug_L2_projection:
                print("L2 Projection Debug:")
                print(f"  perturbation norm: {norm.max().item():.6f}")
                print(f"  epsilon: {self.eps:.6f}")
                print(f"  mask (norm > eps): {mask.sum().item()} out of {mask.numel()} elements")

            if torch.any(mask):
                with torch.no_grad():
                    if len(mask.shape) == 1:  # batch size 1
                        mask = mask.unsqueeze(0)
                    # print(f"  Projecting {mask.sum().item()} elements")
                    adv_perturbation[mask, :] = adv_perturbation[mask, :] / norm[mask] * self.eps

        return adv_perturbation

    def project_simplex(self, adv_perturbation):
        # TODO --> suffix perturbations should be optimized in the one-hot space and not at the embedding space
        raise NotImplementedError("Simplex projection not implemented yet")

    def get_one_hot(self, ids):
        device = self.embed_weights.device
        batch_size, num_tokens = ids.shape

        # Adjusting IDs less than 0 to 0
        ids = torch.where(ids < 0, torch.tensor(0, device=device, dtype=ids.dtype), ids)

        one_hot = torch.zeros(batch_size, num_tokens, self.vocab_size, device=device, dtype=self.embed_weights.dtype)
        one_hot.scatter_(2, ids.unsqueeze(2), 1)
        return one_hot

    def get_embeddings(self, ids):
        one_hot = self.get_one_hot(ids)
        embeddings = one_hot @ self.embed_weights
        return embeddings

    def get_adv_embeddings(self, input_embeds, adv_perturbation, adv_perturbation_mask):
        masked_perturbation = adv_perturbation * adv_perturbation_mask
        adv_embeds = input_embeds + masked_perturbation
        # Verify gradient flow
        # print(f"\nget_adv_embeddings debug:")
        # print(f"  input_embeds.requires_grad: {input_embeds.requires_grad}")
        # print(f"  adv_perturbation.requires_grad: {adv_perturbation.requires_grad}")
        # print(f"  masked_perturbation.requires_grad: {masked_perturbation.requires_grad}")
        # print(f"  adv_embeds.requires_grad: {adv_embeds.requires_grad}\n")
        return adv_embeds

    def get_loss_slice_start_and_end(self, input_embeds):
        input_len = input_embeds.shape[1]

        if self.init_type == "instruction":
            start = input_len - 1
            end = -1
        elif self.init_type == "suffix":
            start = input_len + self.suffix_tokens
            end = -1
        return start, end

    def get_attention_mask(self, input_ids, attention_mask):
        if self.init_type == "instruction":
            return attention_mask
        elif self.init_type == "suffix":
            len_input = input_ids.shape[1]
            input_mask = attention_mask[:, :len_input]
            adversarial_mask = torch.ones(
                (attention_mask.shape[0], self.suffix_tokens),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            target_mask = attention_mask[:, len_input:]
            attention_mask = torch.hstack([input_mask, adversarial_mask, target_mask])
            return attention_mask

    def get_loss_mask(self, target_ids):
        """
        loss_mask is the mask on the input that corresponds to the target in the output of the model

        If "instruction":
            We ignore the first token because it is not in the prediction.
        """
        target_mask = target_ids > 0
        if self.init_type == "instruction":
            return target_mask[:, 1:]  # ignore first token (not in predicted)
        elif self.init_type == "suffix":
            padding_for_suffix = torch.zeros(
                (target_mask.shape[0], self.suffix_tokens), dtype=target_ids.dtype, device=target_ids.device
            )
            loss_mask = torch.hstack([padding_for_suffix, target_mask])
        return loss_mask

    def init_perturbation(self, input_ids, target_ids, attention_mask):
        """
        Initializes the adversarial perturbation based on the input IDs, target IDs, attention mask, and image mask.

        In the case of init_type == instruction:
            adv_perturbation is initialized to zero --> (batch_size, num_input_tokens, embedding_size)
            adv_perturbation mask corresponds to tokens that are part of the input (not target, not padding, not image tokens)

        Returns:
            adv_perturbation (torch.Tensor): The initialized adversarial perturbation (batch_size, num_input_tokens, embedding_size)
            adv_perturbation_mask (torch.Tensor): The mask for the adversarial perturbation (batch_size, num_input_tokens, 1)
        """

        target_mask = target_ids > 0

        # mask is True for input tokens that are not part of the target, not padding and not image tokens
        input_mask = (~target_mask * attention_mask).to(bool)
        batch_size, num_input_tokens = input_ids.shape
        dtype = self.embed_weights.dtype

        if self.init_type == "instruction":
            adv_perturbation = torch.zeros(
                (batch_size, num_input_tokens, self.embedding_size), device=input_ids.device, dtype=dtype
            )

            adv_perturbation_mask = torch.zeros(
                (batch_size, num_input_tokens), device=input_ids.device, dtype=input_ids.dtype
            )
            adv_perturbation_mask[input_mask] = 1
        elif self.init_type == "suffix":
            adv_perturbation = torch.randn(
                (batch_size, num_input_tokens + self.suffix_tokens, self.embedding_size),
                device=input_ids.device,
                dtype=dtype,
            )
            adv_perturbation = self.project_simplex(adv_perturbation)

            adv_perturbation_mask = torch.zeros(
                (batch_size, num_input_tokens + self.suffix_tokens),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            # get indexes of suffix tokens and set them to one
            num_false = torch.sum(input_mask, dim=1)  # TODO FIX as input mask changed
            row_indices = torch.arange(adv_perturbation_mask.shape[0])
            col_indices = num_false[:, None] + torch.arange(self.suffix_tokens)
            col_indices = torch.clip(col_indices, 0, adv_perturbation_mask.shape[1] - 1)
            adv_perturbation_mask[row_indices[:, None], col_indices] = True

        adv_perturbation.requires_grad = True
        adv_perturbation_mask = adv_perturbation_mask.unsqueeze(2)

        # save best adv_perturbation
        self.best_adv_perturbation = adv_perturbation

        return adv_perturbation, adv_perturbation_mask

    def init_opt(self, parameters):
        if self.opt_config is None:
            self.opt_config = {"type": "sign", "lr": 0.01}
            logging.info(f"No opt_config specified using default opt_config: {self.opt_config}")

        optimizer_type = self.opt_config["type"]
        if optimizer_type == "adam":
            opt = torch.optim.Adam(parameters, lr=self.opt_config["lr"])
        elif optimizer_type == "sign":
            opt = SignSGD(parameters, lr=self.opt_config["lr"])
        elif optimizer_type == "rms":
            opt = torch.optim.RMSprop(parameters, lr=self.opt_config["lr"])
        return opt

    def get_num_affirmative_responses(self, target_ids, logits):  # TODO Fix for suffix attack
        target_ids_clone = target_ids.clone()
        target_ids_clone = target_ids_clone[:, 1:]  # ignore first token (not in predicted)
        output_ids = torch.argmax(logits, dim=-1)
        output_ids = output_ids[:, :-1]  # ignore predicted token (not in target_ids)

        input_mask = target_ids_clone < 0
        output_ids[input_mask] = 0
        target_ids_clone[input_mask] = 0
        affirmative_responses = output_ids == target_ids_clone
        affirmative_responses_sum = affirmative_responses.all(dim=-1).sum().item()

        return affirmative_responses_sum

    def debug_norm(self, adv_perturbation):
        if self.init_type == "instruction":
            norm = torch.norm(adv_perturbation, p=2, dim=-1).max()
            logging.info(f"Debugging ESA | L2 Norm max adversarial perturbation: {norm}")
        # TODO print norm for simplex projection?

    def debug_loss(self, loss_adversarial):
        logging.info(
            f"Debugging ESA | Benign loss: {self.loss_benign.item()} | Best Adversarial loss {loss_adversarial.item()}"
        )

    def debug_iter_loss(self, i, loss_adversarial):
        logging.info(f"Debugging ESA | i: {i} | Adversarial loss: {loss_adversarial.item()}")

    def debug_shapes(
        self, input_embeds, target_one_hot, adv_perturbation, adv_perturbation_mask, attention_mask, loss_mask
    ):
        logging.info(
            "====== Debugging ESA Adversarial Attack Shapes ======\n"
            f"input_embeds: {input_embeds.shape}\n"
            f"adv_perturbation: {adv_perturbation.shape}\n"
            f"adv_perturbation_mask: {adv_perturbation_mask.shape}\n"
            f"target_one_hot: {target_one_hot.shape}\n"
            f"attention_mask: {attention_mask.shape}\n"
            f"loss_mask: {loss_mask.shape}"
        )

    #         )

    def debug_output_indexed(self, target_ids, logits, attention_mask, index):
        with torch.no_grad():
            attention_mask = attention_mask.to(dtype=torch.bool)

            # Select the sample at the given index
            target_ids_sample = target_ids[index][1:]
            target_mask = target_ids_sample > 0

            # Original target
            original_target_ids = target_ids_sample[target_mask]
            original_target_text = self.tokenizer.decode(original_target_ids, skip_special_tokens=False)

            # LLM output without perturbation
            no_perturbation_target_ids = torch.argmax(self.logits_no_perturbation, dim=-1)
            no_perturbation_target_ids = no_perturbation_target_ids[index][:-1][target_mask]
            no_perturbation_target_text = self.tokenizer.decode(no_perturbation_target_ids, skip_special_tokens=False)

            # LLM output with perturbation
            perturbed_target_ids = torch.argmax(logits, dim=-1)
            perturbed_target_ids = perturbed_target_ids[index][:-1][target_mask]
            perturbed_target_text = self.tokenizer.decode(perturbed_target_ids, skip_special_tokens=False)

            original_target_text_clean = original_target_text.replace('\n', '\\n')
            no_perturbation_text_clean = no_perturbation_target_text.replace('\n', '\\n')
            perturbed_text_clean = perturbed_target_text.replace('\n', '\\n')

            logging.info(f"===== Debugging attack output for index {index} =====")
            logging.info(f"--> Target in dataset: {original_target_text_clean}")
            logging.info(f"--> Generated text without the adv_perturbation: {no_perturbation_text_clean}")
            logging.info(f"--> Generated with adv perturbation: {perturbed_text_clean}")


class SignSGD(Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        print("\n")
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    print("\n-------")
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    sign = torch.sign(grad)
                    # print(f"SignSGD lr: {group['lr']} | p: {p[0, 0, :5]} |\n grad: {grad[0, 0, :5]} |\n sign: {sign[0, 0, :5]}")
                    # print(f"\nSignSGD lr: {group['lr']} | p: {p.shape} |\n grad: {grad.shape} |\n sign: {sign.shape}")

                    p.add_(other=sign, alpha=-group["lr"])
        print("\n\n")
        return loss
