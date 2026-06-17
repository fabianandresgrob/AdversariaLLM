import torch
from torch.optim.optimizer import Optimizer
import logging
from torch import nn
import time
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def get_chat_template(model_name):
    found = 0

    if "gemma" in model_name:
        found += 1
        first_user_msg = "<start_of_turn>user\n{instruction}<end_of_turn>\n"
        first_user_msg_first_part = "<start_of_turn>user\n{instruction}"
        first_user_msg_second_part = "<end_of_turn>\n"
        user_chat_template = "<start_of_turn>user\n{instruction}<end_of_turn>\n"
        response_key = "<start_of_turn>model\n"
        response_template = response_key + "{target}<end_of_turn>\n"

    #Llama-3.1-8B-Instruct
    elif "meta-llama/Llama-3.1-8B-Instruct" == model_name:
        found += 1
        first_user_msg = """<|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}"""

        first_user_msg_first_part = """<|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}"""
        first_user_msg_second_part = """<|eot_id|>"""

        response_key = """<|start_header_id|>assistant<|end_header_id|>

"""
        response_template = response_key + "{target}<|eot_id|>"

    #Llama2
    elif "llama2" == model_name or "llama-2" == model_name:
        found += 1
        first_user_msg = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

{instruction} """
        user_chat_template = "<s>[INST] {instruction} "
        response_key = "[/INST]"
        first_user_msg_first_part = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

{instruction}"""
        first_user_msg_second_part = " "

        # Llama2 tokenizer does not satisfy enc(a+b) = enc(a) + enc(b)
        # Llama2 expects there to be a space token after [/INST]
        # Since we tokenize the prompt plus response in one go, we need two spaces
        response_template = response_key + "  {target} </s>"
    elif "safe-llama2" == model_name:
        found += 1
        first_user_msg = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} """
        first_user_msg_first_part = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction}"""
        first_user_msg_second_part = " "
        user_chat_template = "<s>[INST] {instruction} "
        response_key = "[/INST]"
        response_template = response_key + " {target} </s>"
    elif "mistral-instruct" == model_name or ("mistral" in model_name.lower() and "instruct" in model_name.lower()):
        found += 1
        first_user_msg = "[INST] {instruction} "
        first_user_msg_first_part = "[INST] {instruction}"
        first_user_msg_second_part = " "
        user_chat_template = "[INST] {instruction} "
        response_key = "[/INST]"
        response_template = response_key + " {target} </s>"
    elif "mistral" == model_name:
        found += 1
        first_user_msg = """<|user|>\n{instruction}</s>"""
        first_user_msg_first_part = "<|user|>\n{instruction}"
        first_user_msg_second_part = "</s>"
        user_chat_template = "\n<|user|>\n{instruction}</s>"
        response_key = "\n<|assistant|>\n"
        response_template = response_key + "{target}</s>"
    elif "phi" in model_name:
        found += 1
        first_user_msg = "<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\n{instruction}<|end|>"
        first_user_msg_first_part = "<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\n{instruction}"
        first_user_msg_second_part = "<|end|>"
        user_chat_template = "<|user|>\n{instruction}<|end|>"
        response_key = "\n<|assistant|>\n"
        response_template = response_key + "{target}<|end|>\n"

    if found == 0:
        raise NotImplementedError(f"Model {model_name} not supported")
    if found > 1:
        raise ValueError(f"Model {model_name} is ambiguous")

    return first_user_msg, response_template, response_key, first_user_msg_first_part, first_user_msg_second_part


def get_embed_weights(model):
    return model.get_input_embeddings().weight


class Detector(nn.Module):
    """
    Detector model that classifies whether the input prompt is harmful or not.

    -->Structure of the final prompt:
    self.first_user_msg_first_part (eg: "<start_of_turn>user\n" for Gemma)
    detector_prompt: "Is the following prompt harmful?"
    projected hidden states
    self.first_user_msg_second_part (eg: "<end_of_turn>\n" for Gemma)
    response_key: (eg: "<start_of_turn>model\n" for Gemma)
    """

    def __init__(self, model, tokenizer, config, model_name):
        super(Detector, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.initial_prompt = config["prompt"]
        self.embeddings_weights = get_embed_weights(model)

        (
            self.first_user_msg,
            self.response_template,
            self.response_key,
            self.first_user_msg_first_part,
            self.first_user_msg_second_part,
        ) = get_chat_template(model_name)

        #tokenize and get the embeddings for (self.first_user_msg_first_part + detector_prompt)
        self.formatted_initial_prompt = self.first_user_msg_first_part.format(instruction=self.initial_prompt)
        self.formatted_initial_prompt_ids = self.tokenizer.encode(self.formatted_initial_prompt, add_special_tokens=True)
        with torch.no_grad():
            self.formatted_initial_prompt_ids = torch.tensor(self.formatted_initial_prompt_ids, device=self.embeddings_weights.device).unsqueeze(0)
            self.formatted_initial_prompt_embeddings = self.model.get_input_embeddings()(self.formatted_initial_prompt_ids)

        #tokenize and get the embeddings for the (self.first_user_msg_second_part + self.response_key)
        with torch.no_grad():
            response_key_ids = tokenizer.encode(self.first_user_msg_second_part + self.response_key, add_special_tokens=False)
            self.response_key_embeddings = self.embeddings_weights[response_key_ids].unsqueeze(0)

        #compute the "yes" and "no" token ids
        self.yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
        self.no_id = tokenizer.encode("no", add_special_tokens=False)[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yes_no_ids = [self.yes_id, self.no_id]

        # Initialize the projector
        self.input_shape = config["input_shape"] #shape of the embeddings before projection
        self.projected_shape = config["projected_shape"]  #shape of the embeddings after projection
        model_dtype = self.embeddings_weights.dtype
        self.projector = nn.Linear(self.input_shape, self.projected_shape, bias=False, dtype=model_dtype)



    def decode_embeddings(self, embeddings, sample_index=0):
        sample_embeddings = embeddings[sample_index]
        # Normalize
        sample_embeddings_norm = F.normalize(sample_embeddings, dim=1)
        vocab_embeddings_norm = F.normalize(self.embeddings_weights, dim=1)

        # Compute cosine similarity: [seq_len, vocab_size]
        cos_sim = sample_embeddings_norm @ vocab_embeddings_norm.T

        # Closest = argmax of similarity
        closest_token_ids = torch.argmax(cos_sim, dim=1)

        #Decode the closest token ids
        decoded_tokens_input = self.tokenizer.batch_decode([[id] for id in closest_token_ids.tolist()], skip_special_tokens=False)
        return decoded_tokens_input


    def debug_output_tokens_yes_no_position(self, outputs, position_of_yes_no_token):
        # Print the top 10 tokens with highest probability for the yes/no position in the first sample
        with torch.no_grad():
            # Get logits for the first sample at the yes/no position
            idx = position_of_yes_no_token[0].item()
            logits_at_idx = outputs.logits[0, idx]  # (vocab_size,)
            top_k = 10
            topk_vals, topk_indices = torch.topk(logits_at_idx, k=top_k)
            topk_tokens = self.tokenizer.batch_decode([[i] for i in topk_indices.tolist()], skip_special_tokens=False)
            print(f"\nTop {top_k} tokens at position_of_yes_no_token[0] (index {idx}):")
            for token, logit in zip(topk_tokens, topk_vals.tolist()):
                print(f"Token: {repr(token)} | Logit: {logit:.4f}")


    def compute_loss(self, logits_yes_no, ground_truth):
        """ Compute the loss for the detector model.
        Args:
            logits_yes_no (torch.Tensor): Logits for the yes/no classification.
            ground_truth (string): "yes" meaning the detector should classify as harmful (index 0), "no" meaning the detector should classify as not harmful (index 1).
        Returns:
            torch.Tensor: Computed loss.
        """
        if ground_truth not in ["yes", "no"]:
            raise ValueError("detector class: in the loss function, the ground_truth must be 'yes' or 'no'.")

        B = logits_yes_no.size(0)
        if ground_truth == "yes":
            ground_truth = torch.zeros((B,), dtype=torch.long, device=logits_yes_no.device).requires_grad_(False)
        else:
            ground_truth = torch.ones((B,), dtype=torch.long, device=logits_yes_no.device).requires_grad_(False)

        # Compute the loss using CrossEntropyLoss
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits_yes_no, ground_truth)


    def debug_tensor(
        self,
        concatenated_inputs_embeddings,
        hidden_states_projected,
        hidden_states_projected_with_response_key,
        target_ids,
        attention_mask,
        first_non_zero_indices,
        position_of_yes_no_token,
        sample_index=0,
        decoded_tokens_input=None,
        output_decoded=None,
    ):
        """ Debugging function to print the concatenated inputs embeddings, projected hidden states, and the response key embeddings.
        If decoded_tokens_input or output_decoded is provided, print the decoded token for each position.
        """
        i = sample_index  # Index of the sample to debug in the batch
        prompt_len = self.formatted_initial_prompt_embeddings.shape[1]
        response_len = self.response_key_embeddings.shape[1]
        insert_pos = first_non_zero_indices[i].item()
        vals_concat = concatenated_inputs_embeddings[i, :, 0]  # [total_seq_len]
        vals_projected = hidden_states_projected[i, :, 0]  # [seq_len]
        vals_with_key = hidden_states_projected_with_response_key[i, :, 0]  # [seq_len + L']
        token_ids = target_ids[i]  # [seq_len]
        resp_key_emb = self.response_key_embeddings[0, :, 0]  # [L']

        max_len = vals_concat.shape[0]

        print(f"\nDEBUG: Concatenated Embeddings")
        header = f"{'idx':>4} {'concat':>12} {'projected':>12} {'with_key':>12} {'target_id':>10} {'resp_key_emb':>12} {'insert?':>8} {'attn_mask':>10} {'yes_no?':>8}"
        if decoded_tokens_input is not None:
            header += f" {'token':>12}"
        if output_decoded is not None:
            header += f" {'output_dec':>12}"
        print(header)
        for idx in range(max_len):
            is_prompt = idx < prompt_len
            is_insert = prompt_len + insert_pos <= idx < prompt_len + insert_pos + response_len
            in_proj = (idx - prompt_len) >= 0 and (idx - prompt_len) < vals_projected.shape[0]
            in_with_key = (idx - prompt_len) >= 0 and (idx - prompt_len) < vals_with_key.shape[0]
            in_target = (idx - prompt_len) >= 0 and (idx - prompt_len) < token_ids.shape[0]
            in_resp_key = prompt_len + insert_pos <= idx < prompt_len + insert_pos + response_len
            is_yes_no = idx == position_of_yes_no_token[i].item()
            concat_val = vals_concat[idx].item()
            proj_val = vals_projected[idx - prompt_len].item() if in_proj else float('nan')
            with_key_val = vals_with_key[idx - prompt_len].item() if in_with_key else float('nan')
            tgt_id = token_ids[idx - prompt_len].item() if in_target else -1
            resp_val = resp_key_emb[idx - (prompt_len + insert_pos)] if in_resp_key else float('nan')
            attn_val = attention_mask[i, idx].item()
            marker = "<PROMPT>" if is_prompt else ("<RESP>" if is_insert else "")
            yes_no_marker = "<YESNO>" if is_yes_no else ""
            line = f"{idx:4d} {concat_val:12.6f} {proj_val:12.6f} {with_key_val:12.6f} {tgt_id:10d} {resp_val:12.6f} {marker:>8} {attn_val:10d} {yes_no_marker:>8}"
            if decoded_tokens_input is not None and idx < len(decoded_tokens_input):
                line += f" {repr(decoded_tokens_input[idx]):>12}"
            if output_decoded is not None and idx < len(output_decoded):
                line += f" {repr(output_decoded[idx]):>12}"
            print(line)


    def debug_outputs(self, outputs):
        #decode from the logits to the tokens (find the token with the highest logit for each position and decode it)
        output_logits = outputs.logits[0]  # (seq_len, vocab_size)
        predicted_token_ids = torch.argmax(output_logits, dim=1)  # (seq_len,)
        decoded_output_tokens = self.tokenizer.batch_decode([[id] for id in predicted_token_ids.tolist()], skip_special_tokens=False)
        return decoded_output_tokens


    def insert_tensor(self, x, insert, position):
        """ Insert a tensor `insert` into another tensor `x` at specified positions.
        x:        (B, L, D)
        insert:   (B, L', D)
        position: (B, 1), each entry in [0, L]

        Returns: (B, L + L', D)
        """
        B, L, D = x.shape
        L_prime = insert.shape[1]
        L_new = L + L_prime
        position = position.squeeze(-1)  # (B,)

        # Prepare output
        out = torch.empty((B, L_new, D), dtype=x.dtype, device=x.device)

        # For each batch, compute indices where data will be written
        arange = torch.arange(L_new, device=x.device).unsqueeze(0)  # (1, L + L')
        pos = position.unsqueeze(1)  # (B, 1)

        # Create masks for where to put the original x and where to put insert
        insert_mask = (arange >= pos) & (arange < (pos + L_prime))  # (B, L + L')  # True where insert should go
        x_mask = ~insert_mask  # (B, L + L')   # True where original x should go

        # Compute the indices
        x_indices = torch.cumsum(x_mask.int(), dim=1) - 1  # (B, L + L')
        insert_indices = torch.cumsum(insert_mask.int(), dim=1) - 1  # (B, L + L')

        # Expand dimensions to (B, L+L', D) for scatter

        x_indices = x_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, D)  # (B, L + L', D)
        insert_indices = insert_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, D)  # (B, L + L', D)

        # Fill the output tensor
        #dim=1 so for each batch, we take the element at the index x_indices[batch, :]
        #we then apply mask to only take the elements where x_mask is True
        out[x_mask] = x.gather(1, x_indices)[x_mask]
        out[insert_mask] = insert.gather(1, insert_indices)[insert_mask]

        return out


    def forward(self, hidden_states, target_ids, attention_mask):
        """ Forward pass through the detector model.
        Starts by projecting the input hidden states to the input shape of the detector model.


        #hidden_states:  [----HS prompt----,----HS response----,----HS padding = 0----]
        #target_ids:     [----    0     ---,----ID response----,----ID padding = 0----]
        #attention_mask: [----    1     ---,----     1     ----,----ID padding = 0----]

        Final tensor we construct:
        [--- FirstUserMsgFirstPart ---, --- Detector Prompt ---, --- HS prompt ---, --- FirstUserMsgSecondPart ---, --- Response Key ---, --- padding ---]
        Which is:
        [--- self.formatted_initial_prompt_embeddings ---, --- HS prompt ---,--- self.response_key_embeddings ---, ---  padding ---]

        Then for the detector we take the logits (for "yes" and "no") for the token generated at the index of the last token of Response Key


        Args:
            hidden_states (torch.Tensor): Input hidden states of the target model
            target_ids (torch.Tensor): Target ids for the input sequence
            attention_mask (torch.Tensor): Attention mask for the input sequence
        Returns:
        """
        if target_ids is None or attention_mask is None:
            raise ValueError("target_ids and attention_mask must be provided for the detector model.")

        # ============= Project the hidden states =============
        hidden_states_projected = self.projector(hidden_states)


        # ============= Insert response key in the hidden states tensor =============
        #find the index of the first token in target_ids that is not 0
        non_zero_mask_target_ids = target_ids != 0  # shape: (B, seq_len)
        first_non_zero_indices = non_zero_mask_target_ids.float().argmax(dim=1, keepdim=True)  # shape: (B, 1)

        hidden_states_projected_with_response_key = self.insert_tensor(
            hidden_states_projected,
            self.response_key_embeddings.expand(hidden_states_projected.shape[0], -1, -1),  # (B, L', D) where L' is the length of the response key embeddings
            first_non_zero_indices
        )  # shape: (B, seq_len + L', D) where L' is the length of the response key embeddings



        # ============= Concatenate the initial prompt embeddings with the projected hidden states =============
        concatenated_inputs_embeddings = torch.cat(
            [
                self.formatted_initial_prompt_embeddings.expand(hidden_states_projected.shape[0], -1, -1),
                hidden_states_projected_with_response_key,
            ],
            dim=1,
        )


        # ============= Prepare the attention mask =============
        B, seq_len = concatenated_inputs_embeddings.shape[:2]
        # Set the attention mask to 0 for the tokens after [--- self.formatted_initial_prompt_embeddings ---, --- HS prompt ---,--- self.response_key_embeddings ---
        new_attention_mask = torch.ones((B, seq_len), dtype=torch.long, device=hidden_states_projected_with_response_key.device)  # (B, seq_len)
        index_of_the_next_element_after_input = first_non_zero_indices + self.formatted_initial_prompt_embeddings.shape[1] + self.response_key_embeddings.shape[1]  # shape: (B, 1)

        # Set the attention mask to 0 for the tokens after the response key
        for i in range(B):
            start_idx = index_of_the_next_element_after_input[i].item()
            new_attention_mask[i, start_idx:] = 0


        # ============= Compute the index of the yes_no token in the output =============
        position_of_yes_no_token = (
            self.formatted_initial_prompt_embeddings.shape[1]
            + first_non_zero_indices - 1 # -1 because we want the last token before the assistant turn, which is the yes/no token
            + self.response_key_embeddings.shape[1]
        ) # shape: (B, 1)


        # ============= Forward pass through the detector LLM =============
        prompt_length = self.formatted_initial_prompt_embeddings.shape[1]
        response_key_length = self.response_key_embeddings.shape[1]
        max_prompt_length = first_non_zero_indices.max().item()
        max_length_input_batch = prompt_length + max_prompt_length + response_key_length #truncate the input to the maximum length of the batch

        # Force the math SDPA backend: the memory-efficient kernel requires a
        # 16-byte-aligned attention bias and hard-errors ("p.attn_bias_ptr is not
        # correctly aligned") on some (batch, seq_len) shapes with our custom mask.
        with sdpa_kernel(SDPBackend.MATH):
            outputs = self.model(inputs_embeds=concatenated_inputs_embeddings[:, :max_length_input_batch, :], attention_mask=new_attention_mask)

        # ============= Extract the logits for the yes/no token =============
        all_logits_yes_no = outputs.logits[:, :, self.yes_no_ids]

        logits_yes_no = torch.gather(
            all_logits_yes_no,                      # (B, seq_len, 2)
            dim=1,
            index=position_of_yes_no_token.unsqueeze(-1).expand(-1, -1, 2)  # (B, 1, 2)
        ).squeeze(1)  # final shape: (B, 2)


        # ============= Debug =============
        if self.config["debug"]:
            print(f"\n\n------------------------------------------------------")
            concatenated_inputs_embeddings_decoded = self.decode_embeddings(concatenated_inputs_embeddings, sample_index=0)  # Debugging: decode the embeddings of the first sample
            outputs_decoded = self.debug_outputs(outputs)
            self.debug_tensor(
                concatenated_inputs_embeddings,
                hidden_states_projected,
                hidden_states_projected_with_response_key,
                target_ids,
                new_attention_mask,
                first_non_zero_indices,
                position_of_yes_no_token,
                sample_index=0,
                decoded_tokens_input=concatenated_inputs_embeddings_decoded,
                output_decoded=outputs_decoded,
            )
            self.debug_output_tokens_yes_no_position(outputs, position_of_yes_no_token)
            print("\nlogits_yes_no:\n", logits_yes_no.detach().cpu())

        return outputs, logits_yes_no
